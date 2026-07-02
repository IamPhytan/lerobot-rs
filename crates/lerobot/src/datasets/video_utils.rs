use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct VideoFrames {
    /// Shape: [num_frames, channels, height, width]
    /// Values are float32 in [0.0, 1.0]
    pub data: Vec<f32>,
    pub num_frames: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum VideoBackend {
    Ffmpeg,
}

impl TryFrom<&str> for VideoBackend {
    type Error = VideoDecodeError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ffmpeg" => Ok(VideoBackend::Ffmpeg),
            _ => Err(VideoDecodeError::UnsupportedBackend(value.to_string())),
        }
    }
}

#[derive(Debug, Error)]
pub enum VideoDecodeError {
    #[error("unsupported video backend: {0}")]
    UnsupportedBackend(String),

    #[error(
        "one or more query timestamps violate tolerance: min distances={min_distances:?}, tolerance_s={tolerance_s}, video={video_path}"
    )]
    FrameTimestamp {
        min_distances: Vec<f64>,
        tolerance_s: f64,
        video_path: String,
    },

    #[error("ffmpeg error: {0}")]
    Ffmpeg(String),

    #[error("no frames decoded from video: {0}")]
    NoFrames(String),
}

pub fn decode_video_frames(
    video_path: &PathBuf,
    timestamps: &Vec<f64>,
    tolerance_s: f64,
    backend: Option<VideoBackend>,
) -> Result<VideoFrames, VideoDecodeError> {
    let backend = backend.unwrap_or(VideoBackend::Ffmpeg);

    match backend {
        VideoBackend::Ffmpeg => decode_video_frames_ffmpeg(video_path, timestamps, tolerance_s),
    }
}

pub fn decode_video_frames_ffmpeg(
    video_path: &PathBuf,
    timestamps: &Vec<f64>,
    tolerance_s: f64,
) -> Result<VideoFrames, VideoDecodeError> {
    if timestamps.is_empty() {
        return Ok(VideoFrames {
            data: Vec::new(),
            num_frames: 0,
            channels: 3,
            height: 0,
            width: 0,
        });
    }

    ffmpeg::init().map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

    let mut input = ffmpeg::format::input(video_path)
        .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

    let stream = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| {
            VideoDecodeError::Ffmpeg(format!(
                "could not find video stream in {}",
                video_path.display()
            ))
        })?;

    let stream_index = stream.index();
    let time_base = stream.time_base();

    let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;
    let mut decoder = context
        .decoder()
        .video()
        .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

    let width = decoder.width() as usize;
    let height = decoder.height() as usize;

    let mut scaler = Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        Flags::BILINEAR,
    )
    .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

    let first_ts = timestamps.iter().copied().fold(f64::INFINITY, f64::min);
    let last_ts = timestamps.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Get stream timestamp for first ts
    let seek_target = seconds_to_stream_ts(first_ts, time_base);

    input
        .seek(seek_target, ..seek_target)
        .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;
    decoder.flush();

    let mut loaded_frames: Vec<Vec<f32>> = Vec::new();
    let mut loaded_ts: Vec<f64> = Vec::new();

    // Get frames between first and last ts
    for (packet_stream, packet) in input.packets() {
        if packet_stream.index() != stream_index {
            continue;
        }

        decoder
            .send_packet(&packet)
            .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

        receive_decoded_frames(
            &mut decoder,
            &mut scaler,
            time_base,
            width,
            height,
            last_ts,
            &mut loaded_frames,
            &mut loaded_ts,
        )?;

        if loaded_ts.last().is_some_and(|&ts| ts >= last_ts) {
            break;
        }
    }

    decoder
        .send_eof()
        .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

    receive_decoded_frames(
        &mut decoder,
        &mut scaler,
        time_base,
        width,
        height,
        last_ts,
        &mut loaded_frames,
        &mut loaded_ts,
    )?;

    if loaded_frames.is_empty() {
        return Err(VideoDecodeError::NoFrames(video_path.display().to_string()));
    }

    // Retrieve frames for each value in timestamps
    let mut closest_indices = Vec::with_capacity(timestamps.len());
    let mut min_distances = Vec::with_capacity(timestamps.len());

    for &query_ts in timestamps {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;

        for (idx, &ts) in loaded_ts.iter().enumerate() {
            let dist = (query_ts - ts).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        closest_indices.push(best_idx);
        min_distances.push(best_dist);
    }

    let bad_distances: Vec<f64> = min_distances
        .iter()
        .copied()
        .filter(|&dist| dist >= tolerance_s)
        .collect();

    if !bad_distances.is_empty() {
        return Err(VideoDecodeError::FrameTimestamp {
            min_distances,
            tolerance_s,
            video_path: video_path.display().to_string(),
        });
    }

    let mut output = Vec::with_capacity(timestamps.len() * 3 * height * width);

    for idx in closest_indices {
        output.extend_from_slice(&loaded_frames[idx]);
    }

    Ok(VideoFrames {
        data: output,
        num_frames: timestamps.len(),
        channels: 3,
        height,
        width,
    })
}

fn seconds_to_stream_ts(seconds: f64, time_base: ffmpeg::Rational) -> i64 {
    let numerator = time_base.numerator() as f64;
    let denominator = time_base.denominator() as f64;

    // stream timestamp = seconds / time_base
    (seconds * denominator / numerator).round() as i64
}

fn stream_ts_to_seconds(ts: i64, time_base: ffmpeg::Rational) -> f64 {
    ts as f64 * time_base.numerator() as f64 / time_base.denominator() as f64
}

fn receive_decoded_frames(
    decoder: &mut ffmpeg::decoder::Video,
    scaler: &mut Context,
    time_base: ffmpeg::Rational,
    width: usize,
    height: usize,
    last_ts: f64,
    loaded_frames: &mut Vec<Vec<f32>>,
    loaded_ts: &mut Vec<f64>,
) -> Result<(), VideoDecodeError> {
    let mut decoded = Video::empty();

    while decoder.receive_frame(&mut decoded).is_ok() {
        let Some(pts) = decoded.pts() else {
            continue;
        };

        let ts = stream_ts_to_seconds(pts, time_base);

        let mut rgb_frame = Video::empty();
        scaler
            .run(&decoded, &mut rgb_frame)
            .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

        let frame_chw = rgb24_frame_to_chw_f32(&rgb_frame, width, height);

        loaded_frames.push(frame_chw);
        loaded_ts.push(ts);

        if ts >= last_ts {
            break;
        }
    }

    Ok(())
}

fn rgb24_frame_to_chw_f32(frame: &Video, width: usize, height: usize) -> Vec<f32> {
    let data = frame.data(0);
    let stride = frame.stride(0);

    let mut output = vec![0.0_f32; 3 * height * width];

    for y in 0..height {
        let row = &data[y * stride..y * stride + width * 3];

        for x in 0..width {
            let r = row[x * 3] as f32 / 255.0;
            let g = row[x * 3 + 1] as f32 / 255.0;
            let b = row[x * 3 + 2] as f32 / 255.0;

            let hw_idx = y * width + x;

            output[hw_idx] = r;
            output[height * width + hw_idx] = g;
            output[2 * height * width + hw_idx] = b;
        }
    }

    output
}

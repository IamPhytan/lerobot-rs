use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use ndarray::{ArrayView3, ShapeError};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct VideoFrames {
    /// Each frame is RGB24, HWC, row-major.
    ///
    /// Shape per frame: [height, width, 3]
    /// Values are u8 in [0, 255].
    pub frames: Vec<VideoFrameRgb>,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

#[derive(Debug, Clone)]
pub struct VideoFrameRgb {
    /// Shape: [height, width, 3]
    /// Layout: interleaved RGB24, row-major.
    pub data: Vec<u8>,

    /// Actual decoded frame timestamp that was selected.
    pub timestamp_s: f64,

    /// Requested timestamp this frame matched.
    pub requested_timestamp_s: f64,

    /// Distance between requested timestamp and selected frame timestamp.
    pub distance_s: f64,
}

impl VideoFrameRgb {
    pub fn as_hwc_array(
        &self,
        height: usize,
        width: usize,
    ) -> Result<ArrayView3<'_, u8>, ShapeError> {
        ArrayView3::from_shape((height, width, 3), &self.data)
    }
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

#[derive(Debug, Clone, Copy)]
struct QueryTimestamp {
    original_index: usize,
    ts: f64,
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
    video_path: &Path,
    timestamps: &[f64],
    tolerance_s: f64,
    backend: Option<VideoBackend>,
) -> Result<VideoFrames, VideoDecodeError> {
    let backend = backend.unwrap_or(VideoBackend::Ffmpeg);

    match backend {
        VideoBackend::Ffmpeg => decode_video_frames_ffmpeg(&video_path, timestamps, tolerance_s),
    }
}

pub fn decode_video_frames_ffmpeg(
    video_path: &Path,
    timestamps: &[f64],
    tolerance_s: f64,
) -> Result<VideoFrames, VideoDecodeError> {
    if timestamps.is_empty() {
        return Ok(VideoFrames {
            frames: Vec::new(),
            height: 0,
            width: 0,
            channels: 3,
        });
    }

    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Error);
    ffmpeg::init().map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

    let mut input = ffmpeg::format::input(&video_path)
        .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

    let stream = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| {
            VideoDecodeError::Ffmpeg(format!(
                "could not find video stream in {}",
                &video_path.display()
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
    let seek_start_s = (first_ts - tolerance_s).max(0.0);
    let decode_until_s = last_ts + tolerance_s;
    let seek_target = seconds_to_stream_ts(seek_start_s, time_base);

    input
        .seek(seek_target, ..seek_target)
        .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

    decoder.flush();

    let mut queries: Vec<QueryTimestamp> = timestamps
        .iter()
        .copied()
        .enumerate()
        .map(|(original_index, ts)| QueryTimestamp { original_index, ts })
        .collect();

    queries.sort_by(|a, b| a.ts.total_cmp(&b.ts));

    // Retrieve frames for each value in timestamps
    let mut selected_frames: Vec<Option<VideoFrameRgb>> = vec![None; timestamps.len()];
    let mut best_distances = vec![f64::INFINITY; timestamps.len()];
    let mut found_frames = vec![false; timestamps.len()];

    // Get frames between first and last ts
    for (packet_stream, packet) in input.packets() {
        if packet_stream.index() != stream_index {
            continue;
        }

        decoder
            .send_packet(&packet)
            .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

        let reached_end = receive_decoded_frames_rgb24(
            &mut decoder,
            &mut scaler,
            time_base,
            width,
            height,
            decode_until_s,
            tolerance_s,
            &queries,
            timestamps,
            &mut selected_frames,
            &mut best_distances,
            &mut found_frames,
        )?;

        if reached_end {
            break;
        }
    }

    decoder
        .send_eof()
        .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

    receive_decoded_frames_rgb24(
        &mut decoder,
        &mut scaler,
        time_base,
        width,
        height,
        decode_until_s,
        tolerance_s,
        &queries,
        timestamps,
        &mut selected_frames,
        &mut best_distances,
        &mut found_frames,
    )?;

    let has_missing = found_frames.iter().any(|&found| !found);
    let has_bad_distances = best_distances.iter().any(|&dist| dist > tolerance_s);

    if has_missing || has_bad_distances {
        return Err(VideoDecodeError::FrameTimestamp {
            min_distances: best_distances,
            tolerance_s,
            video_path: video_path.display().to_string(),
        });
    }

    let frames = selected_frames
        .into_iter()
        .map(|frame| {
            frame.ok_or_else(|| VideoDecodeError::FrameTimestamp {
                min_distances: best_distances.clone(),
                tolerance_s,
                video_path: video_path.display().to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if frames.is_empty() {
        return Err(VideoDecodeError::NoFrames(video_path.display().to_string()));
    }

    Ok(VideoFrames {
        frames,
        height,
        width,
        channels: 3,
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

fn receive_decoded_frames_rgb24(
    decoder: &mut ffmpeg::decoder::Video,
    scaler: &mut Context,
    time_base: ffmpeg::Rational,
    width: usize,
    height: usize,
    decode_until_s: f64,
    tolerance_s: f64,
    queries: &[QueryTimestamp],
    requested_timestamps: &[f64],
    selected_frames: &mut [Option<VideoFrameRgb>],
    best_distances: &mut [f64],
    found_frames: &mut [bool],
) -> Result<bool, VideoDecodeError> {
    let mut decoded = Video::empty();
    let mut rgb_frame = Video::empty();
    let mut improved_indices: Vec<(usize, f64)> = Vec::new();

    while decoder.receive_frame(&mut decoded).is_ok() {
        let Some(pts) = decoded.pts() else {
            continue;
        };

        let ts = stream_ts_to_seconds(pts, time_base);

        if ts > decode_until_s {
            return Ok(true);
        }

        let lower_ts = ts - tolerance_s;
        let upper_ts = ts + tolerance_s;

        let lower = queries.partition_point(|q| q.ts < lower_ts);
        let upper = queries.partition_point(|q| q.ts <= upper_ts);

        if lower == upper {
            continue;
        }

        improved_indices.clear();

        // Determine whether frame is closest to some query
        for query in &queries[lower..upper] {
            let original_index = query.original_index;
            let dist = (query.ts - ts).abs();

            if dist < best_distances[original_index] {
                improved_indices.push((original_index, dist));
            }
        }

        if improved_indices.is_empty() {
            continue;
        }

        scaler
            .run(&decoded, &mut rgb_frame)
            .map_err(|err| VideoDecodeError::Ffmpeg(err.to_string()))?;

        let frame_data = rgb24_frame_to_hcw_u8(&rgb_frame, width, height);

        for &(original_index, dist) in &improved_indices {
            selected_frames[original_index] = Some(VideoFrameRgb {
                data: frame_data.clone(),
                timestamp_s: ts,
                requested_timestamp_s: requested_timestamps[original_index],
                distance_s: dist,
            });

            best_distances[original_index] = dist;
            found_frames[original_index] = true;
        }
    }

    Ok(false)
}

fn rgb24_frame_to_hcw_u8(frame: &Video, width: usize, height: usize) -> Vec<u8> {
    let data = frame.data(0);
    let stride = frame.stride(0);

    let row_bytes = width * 3;
    let mut output = vec![0; height * row_bytes];

    for y in 0..height {
        let src_start = y * stride;
        let src_end = src_start + row_bytes;

        let dst_start = y * row_bytes;
        let dst_end = dst_start + row_bytes;

        output[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
    }

    output
}

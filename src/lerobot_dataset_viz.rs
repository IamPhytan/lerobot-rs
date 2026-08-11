use crate::lerobot::{DatasetItem, DatasetItemValue, VideoFrames};
use crate::lerobot_cli::DatasetVizMode;
use anyhow;
use lerobot::LeRobotDataset;
use num_traits::cast::NumCast;
use pl::polars_utils::float::IsFloat;
use polars as pl;
use polars::datatypes::{AnyValue, DataType};
use polars::error::{PolarsError, PolarsResult};
use rerun as rr;
use tqdm::Iter;

pub fn visualize_dataset(
    dataset: &LeRobotDataset,
    episode_index: usize,
    mode: DatasetVizMode,
) -> anyhow::Result<()> {
    rr::external::re_log::setup_logging();

    let repo_id = dataset.repo_id.to_string();
    let _spawn_local_viewer = mode == DatasetVizMode::Local;

    let rec_builder = rr::RecordingStreamBuilder::new(format!("{repo_id}/episode_{episode_index}"));

    let rec = match mode {
        DatasetVizMode::Local => rec_builder.spawn()?,
        DatasetVizMode::Distant => rec_builder.connect_grpc()?,
    };

    if mode == DatasetVizMode::Distant {
        // GRPC server
        // let server_uri = rec.serve_grpc(ServerOptions::default());
    }

    let mut first_index: Option<i64> = None;

    for (idx, item) in dataset.iter().enumerate().tqdm().total(Some(dataset.len())) {
        let item = match item {
            Ok(item) => item,
            Err(err) => {
                eprintln!("Failed to load item {idx}: {err}");
                break;
            }
        };

        let index = get_value::<i64>(&item, "index")?;
        let timestamp = get_value::<f64>(&item, "timestamp")?;

        let first = *first_index.get_or_insert(index);

        rec.set_time_sequence("frame_index", index - first);
        rec.set_timestamp_secs_since_epoch("timestamp", timestamp);

        // Display each camera image
        for key in dataset.meta.camera_keys() {
            match item.get(&key) {
                Some(DatasetItemValue::VideoFrames(v)) => {
                    log_video_frames(&rec, &key, v)?;
                }
                Some(DatasetItemValue::Polars(v)) => {
                    let bytes = polars_image_to_encoded_bytes(v)?;
                    rec.log(key, &rr::EncodedImage::from_file_contents(bytes))?;
                }
                v => {
                    eprintln!("Key {key} does not contain a valid image item: {:?}", v);
                }
            }
        }

        // Log scalar vectors
        log_vector(&rec, &item, "action", "action")?;
        log_vector(&rec, &item, "observation.state", "state")?;

        // Log simple scalar fields.
        log_scalar(&rec, &item, "done", "done")?;
        log_scalar(&rec, &item, "reward", "reward")?;
        log_scalar(&rec, &item, "next.success", "next.success")?;

        // Log task as text.
        if let Some(DatasetItemValue::String(task)) = item.get("task") {
            rec.log("task", &rr::TextDocument::new(task.clone()))?;
        }
    }
    Ok(())
}

fn get_value<T: NumCast + IsFloat>(item: &DatasetItem, key: &str) -> anyhow::Result<T> {
    match item.get(key) {
        Some(DatasetItemValue::Polars(v)) => Ok(v.try_extract::<T>()?),
        _ => anyhow::bail!("could not unpack field {key}"),
    }
}

fn log_vector(
    rec: &rr::RecordingStream,
    item: &DatasetItem,
    item_key: &str,
    prefix: &str,
) -> anyhow::Result<()> {
    let Some(value) = item.get(item_key) else {
        return Ok(());
    };

    match value {
        DatasetItemValue::Polars(v) => {
            let values = polars_list_to_vec(v)?;

            for (dim_idx, &val) in values.iter().enumerate() {
                rec.log(format!("{prefix}/{dim_idx}"), &rr::Scalars::single(val))?;
            }
        }

        DatasetItemValue::DataFrame(df) => {
            // Useful when delta_timestamps makes get_item return a mini sequence.
            let series = df.column(item_key)?;
            for row_idx in 0..series.len() {
                let value = series.get(row_idx)?.try_extract::<f64>()?;
                rec.log(format!("{prefix}/{row_idx}"), &rr::Scalars::single(value))?;
            }
        }

        _ => {}
    }

    Ok(())
}

fn log_scalar(
    rec: &rr::RecordingStream,
    item: &DatasetItem,
    item_key: &str,
    entity_path: &str,
) -> anyhow::Result<()> {
    let Some(DatasetItemValue::Polars(v)) = item.get(item_key) else {
        return Ok(());
    };

    if let Ok(x) = v.try_extract::<f64>() {
        rec.log(entity_path, &rr::Scalars::single(x))?;
    } else if let Some(x) = v.extract_bool() {
        rec.log(entity_path, &rr::Scalars::single(if x { 1.0 } else { 0.0 }))?;
    }

    Ok(())
}

fn log_video_frames(
    rec: &rr::RecordingStream,
    entity_path: &str,
    video: &VideoFrames,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        video.channels == 3,
        "expected RGB video frames with 3 channels, got {}",
        video.channels
    );

    anyhow::ensure!(
        video.frames.len() == 1,
        "expected one video frame per dataset item, got {}",
        video.frames.len()
    );

    let frame = &video.frames[0];

    anyhow::ensure!(
        frame.data.len() == video.height * video.width * 3,
        "invalid RGB frame buffer length: got {}, expected {}",
        frame.data.len(),
        video.height * video.width * 3
    );

    // rec.set_time_sequence("video_frame_index", frame_idx as i64);
    // rec.set_timestamp_secs_since_epoch("video_timestamp", frame.requested_timestamp_s);

    let image = rr::Image::from_color_model_and_bytes(
        frame.data.clone(),
        [video.width as u32, video.height as u32],
        rr::ColorModel::RGB,
        rr::ChannelDatatype::U8,
    );

    rec.log(entity_path, &image)?;

    Ok(())
}

/// Convert Polars image string to a vector of bytes
fn polars_image_to_encoded_bytes(value: &AnyValue) -> anyhow::Result<Vec<u8>> {
    match value {
        // Direct binary value
        AnyValue::Binary(bytes) => Ok(bytes.to_vec()),

        // Owned binary value
        AnyValue::BinaryOwned(bytes) => Ok(bytes.clone()),

        // struct { bytes: binary, path: null }
        AnyValue::StructOwned(fields) => {
            let (values, _field_defs) = fields.as_ref();

            for v in values {
                match v {
                    AnyValue::Binary(bytes) => {
                        return Ok(bytes.to_vec());
                    }
                    AnyValue::BinaryOwned(bytes) => {
                        return Ok(bytes.clone());
                    }
                    _ => {}
                }
            }
            anyhow::bail!("image struct did not contain a binary field");
        }

        // Borrowed struct variant
        AnyValue::Struct(_, _, _) => {
            let mut values = Vec::new();
            value._materialize_struct_av(&mut values);

            for v in values {
                match v {
                    AnyValue::Binary(bytes) => return Ok(bytes.to_vec()),
                    AnyValue::BinaryOwned(bytes) => return Ok(bytes),
                    _ => {}
                }
            }

            anyhow::bail!("image struct did not contain a binary field")
        }

        other => {
            anyhow::bail!("expected image bytes, got {other:?}")
        }
    }
}

/// Convert a Polars List value into a Rust Vec
fn polars_list_to_vec(value: &AnyValue) -> PolarsResult<Vec<f64>> {
    match value {
        AnyValue::List(series) => {
            let series = series.cast(&DataType::Float64)?;
            let ca = series.f64()?;
            Ok(ca.into_no_null_iter().collect())
        }
        _ => Err(PolarsError::ComputeError("expected AnyValue::List".into())),
    }
}

use std::collections::HashMap;
use std::collections::HashSet;
use std::io;
use std::path::PathBuf;

use crate::datasets::feature_utils::{check_delta_timestamps, get_delta_indices};
use crate::datasets::utils::FileError;
use crate::datasets::video_utils::VideoBackend;
use crate::datasets::video_utils::{VideoFrames, decode_video_frames};
use crate::lerobot_dataset::LeRobotDatasetMetadata;
use crate::types::DatasetItem;
use crate::types::{DeltaIndices, DeltaTimestamps, PaddingMask, QueryIndices};
use polars as pl;
use polars::error::{PolarsError, PolarsResult};
use polars::lazy::prelude::LazyFrame;
use polars::prelude::{AnyValue, DataType, PlPath, UnionArgs, col, concat, lit};

#[derive(Debug)]
pub enum DatasetItemValue {
    Polars(AnyValue<'static>),
    DataFrame(pl::frame::DataFrame),
    BoolVec(Vec<bool>),
    String(String),
    VideoFrames(VideoFrames),
}

#[derive(Debug)]
pub struct DatasetReader {
    meta: LeRobotDatasetMetadata,
    pub hf_dataset: Option<pl::frame::DataFrame>,
    episodes: Option<Vec<usize>>,
    tolerance_s: f64,
    video_backend: Option<VideoBackend>,
    pub delta_indices: Option<DeltaIndices>,
    absolute_to_relative_idx: Option<HashMap<usize, usize>>,
}

impl DatasetReader {
    pub fn new(
        meta: LeRobotDatasetMetadata,
        episodes: Option<Vec<usize>>,
        tolerance_s: f64,
        video_backend: Option<&str>,
        delta_timestamps: Option<DeltaTimestamps>,
    ) -> Self {
        let delta_indices: Option<DeltaIndices> = match delta_timestamps {
            Some(delta_timestamps) => {
                check_delta_timestamps(&delta_timestamps, meta.fps(), tolerance_s)
                    .expect("Invalid delta_timestamps");

                Some(get_delta_indices(&delta_timestamps, meta.fps()))
            }
            None => None,
        };

        // TODO: Determine whether we want to propagate this parsing error further
        let video_backend = match video_backend {
            Some(name) => match VideoBackend::try_from(name) {
                Ok(backend) => Some(backend),
                Err(err) => {
                    eprintln!("Warning: {err}. Falling back to ffmpeg.");
                    Some(VideoBackend::Ffmpeg)
                }
            },
            None => None,
        };

        Self {
            meta,
            hf_dataset: None,
            episodes,
            tolerance_s,
            video_backend,
            delta_indices,
            absolute_to_relative_idx: None,
        }
    }

    pub fn len(&self) -> usize {
        self.hf_dataset.as_ref().map(|df| df.height()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn try_load(&mut self) -> bool {
        self.hf_dataset = match self.load_hf_dataset() {
            Ok(value) => Some(value),
            _ => None,
        };

        return self.check_cached_episodes_sufficient();
    }

    pub fn load_hf_dataset(&self) -> Result<pl::frame::DataFrame, FileError> {
        println!("Reading data in: {:?}", self.meta.root.join("data"));

        let requested_episodes = pl::series::Series::from_iter(match &self.episodes {
            Some(episodes) => episodes.iter().map(|&ep| ep as u32).collect::<Vec<u32>>(),
            None => (0..self.meta.info.total_episodes)
                .map(|ep| ep as u32)
                .collect::<Vec<u32>>(),
        });

        let files = self
            .get_episodes_file_paths()
            .into_iter()
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "parquet")
                    .unwrap_or(false)
            })
            .map(|path| self.meta.root.join(path))
            .collect::<Vec<PathBuf>>();

        if files.is_empty() {
            return Err(FileError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                "No parquet files found for requested_episodes",
            )));
        }

        let data = files
            .iter()
            .map(|fpath| {
                LazyFrame::scan_parquet(
                    PlPath::new(fpath.to_str().expect("Polars Path error")),
                    Default::default(),
                )
                .map_err(FileError::from)
                .unwrap_or_else(|_| panic!("Error while scanning parquet file {:?}", fpath))
                .with_column(col("episode_index").cast(DataType::UInt32))
                .with_column(col("task_index").cast(DataType::UInt32))
            })
            .collect::<Vec<LazyFrame>>();

        let all_data = concat(data, UnionArgs::default())?
            .filter(col("episode_index").is_in(lit(requested_episodes), false))
            .collect()?;

        Ok(all_data)
    }

    fn check_cached_episodes_sufficient(&self) -> bool {
        if (self.hf_dataset == None) && (self.len() == 0) {
            return false;
        }

        let available_episodes = self
            .hf_dataset
            .as_ref()
            .expect("Could not find hf_dataset")
            .column("episode_index")
            .expect("No column 'episode_index")
            .unique()
            .expect("Issue getting unique values")
            .u32()
            .expect("Episode indices cannot be converted to u32")
            .iter()
            .filter_map(|x| match x {
                Some(v) => Some(v as usize),
                None => None,
            })
            .collect::<Vec<usize>>();

        let requested_episodes: HashSet<usize> = match &self.episodes {
            Some(episodes) => HashSet::from_iter(episodes.clone()),
            None => HashSet::from_iter((0..self.meta.info.total_episodes).into_iter()),
        };

        if !requested_episodes.is_subset(&HashSet::from_iter(available_episodes)) {
            return false;
        }

        for ep_index in requested_episodes {
            for vid_key in self.meta.video_keys() {
                let video_path = self.meta.root.join(
                    self.meta
                        .get_video_file_path(ep_index, vid_key.as_str())
                        .expect("Could not get video file path"),
                );
                if !video_path.exists() {
                    panic!("Missing video file {}", video_path.display());
                    return false;
                }
            }
        }

        return true;
    }

    fn collect_relatives_indices(&self, absolute_indices: &Vec<usize>) -> Vec<usize> {
        if let Some(abs_to_rel) = &self.absolute_to_relative_idx {
            absolute_indices
                .iter()
                .map(|idx| {
                    *abs_to_rel.get(idx).expect(
                        format!(
                            "absolute index {} missing from absolute_to_relative_idx",
                            idx
                        )
                        .as_str(),
                    )
                })
                .collect()
        } else {
            absolute_indices.clone()
        }
    }

    pub fn get_episodes_file_paths(&self) -> Vec<PathBuf> {
        let requested_episodes: HashSet<usize> = match &self.episodes {
            Some(episodes) => HashSet::from_iter(episodes.clone()),
            None => HashSet::from_iter((0..self.meta.info.total_episodes).into_iter()),
        };

        let mut fpaths = requested_episodes
            .iter()
            .filter_map(|&ep_idx| self.meta.get_data_file_path(ep_idx))
            .collect::<Vec<PathBuf>>();

        let video_fpaths = self
            .meta
            .video_keys()
            .iter()
            .map(|vid_key| {
                requested_episodes
                    .iter()
                    .filter_map(|&ep_idx| self.meta.get_video_file_path(ep_idx, vid_key))
                    .collect::<Vec<PathBuf>>()
            })
            .flatten()
            .collect::<Vec<PathBuf>>();

        fpaths.extend(video_fpaths);

        return fpaths;
    }

    fn get_query_indices(
        &self,
        abs_idx: usize,
        ep_idx: usize,
    ) -> PolarsResult<(QueryIndices, PaddingMask)> {
        let Some(delta_indices) = &self.delta_indices else {
            return Ok((HashMap::new(), HashMap::new()));
        };

        let ep = self
            .meta
            .get_episode(ep_idx)
            .ok_or_else(|| {
                PolarsError::ComputeError(format!("Could not find episode {ep_idx}").into())
            })?
            .select([col("dataset_from_index"), col("dataset_to_index")])
            .collect()?;

        if ep.height() == 0 {
            return Err(PolarsError::ComputeError(
                format!("Could not find episode metadata for episode_index {ep_idx}").into(),
            ));
        }

        let ep_start = ep
            .column("dataset_from_index")?
            .get(0)?
            .try_extract::<u32>()? as usize;
        let ep_end = ep
            .column("dataset_to_index")?
            .get(0)?
            .try_extract::<u32>()? as usize;

        let mut query_indices: QueryIndices = HashMap::new();
        let mut padding: PaddingMask = HashMap::new();

        for (key, delta_idx) in delta_indices {
            let mut indices = Vec::with_capacity(delta_idx.len());
            let mut pad = Vec::with_capacity(delta_idx.len());

            for &delta in delta_idx {
                let raw_idx = abs_idx as isize + delta;
                let is_pad = raw_idx < ep_start as isize || raw_idx >= ep_end as isize;

                let clamped = raw_idx.max(ep_start as isize).min(ep_end as isize - 1) as usize;

                indices.push(clamped);
                pad.push(is_pad);
            }

            query_indices.insert(key.clone(), indices);
            padding.insert(format!("{key}_is_pad"), pad);
        }

        Ok((query_indices, padding))
    }

    fn get_query_timestamps(
        &self,
        current_ts: f64,
        query_indices: Option<&QueryIndices>,
    ) -> PolarsResult<HashMap<String, Vec<f64>>> {
        let dataset = self
            .hf_dataset
            .as_ref()
            .expect("hf_dataset must be loaded before querying timestamps");

        let mut query_timestamps: HashMap<String, Vec<f64>> = HashMap::new();
        for key in self.meta.video_keys() {
            let Some(indices) = query_indices.and_then(|q| q.get(&key)) else {
                query_timestamps.insert(key, vec![current_ts]);
                continue;
            };

            let relative_indices: Vec<usize> = self.collect_relatives_indices(indices);

            let mut timestamps = Vec::with_capacity(relative_indices.len());
            let ts_col = dataset.column("timestamp")?;
            for rel_idx in relative_indices {
                let ts = ts_col.get(rel_idx)?.try_extract::<f64>()?;
                timestamps.push(ts);
            }

            query_timestamps.insert(key, timestamps);
        }

        Ok(query_timestamps)
    }

    fn query_hf_dataset(
        &self,
        query_indices: &QueryIndices,
    ) -> PolarsResult<HashMap<String, pl::frame::DataFrame>> {
        let dataset = self
            .hf_dataset
            .as_ref()
            .expect("hf_dataset must be loaded before querying");

        let video_keys = self.meta.video_keys();
        let mut result: HashMap<String, pl::frame::DataFrame> = HashMap::new();

        for (key, q_idx) in query_indices {
            if video_keys.contains(key) {
                continue;
            }

            if dataset.column(key).is_err() {
                println!("Column {key} does not exist");
                continue;
            }

            let relative_indices = self.collect_relatives_indices(q_idx);

            let indices = pl::datatypes::UInt32Chunked::from_vec(
                "indices".into(),
                relative_indices.iter().map(|&x| x as u32).collect(),
            );
            let column_df = dataset.select([key.as_str()])?;
            let gathered = column_df.take(&indices)?;

            result.insert(key.clone(), gathered);
        }

        Ok(result)
    }

    fn query_videos(
        &self,
        query_timestamps: &HashMap<String, Vec<f64>>,
        ep_idx: usize,
    ) -> PolarsResult<HashMap<String, VideoFrames>> {
        let mut video_frames: HashMap<String, VideoFrames> = HashMap::new();

        for (vid_key, query_ts) in query_timestamps {
            let from_ts_col = format!("videos/{vid_key}/from_timestamp");

            let ep = self
                .meta
                .get_episode(ep_idx)
                .ok_or_else(|| {
                    PolarsError::ComputeError(format!("Could not find episode {ep_idx}").into())
                })?
                .select([col(&from_ts_col)])
                .collect()?;

            if ep.height() == 0 {
                return Err(PolarsError::ComputeError(
                    format!("Could not find episode metadata for episode_index {ep_idx}").into(),
                ));
            }

            let from_timestamp = ep.column(&from_ts_col)?.get(0)?.try_extract::<f64>()?;

            let shifted_query_ts: Vec<f64> =
                query_ts.iter().map(|ts| from_timestamp + ts).collect();

            // Get video file path for ep_idx and vid_key
            let video_rel_path = self.meta.get_video_file_path(ep_idx, vid_key).expect(
                format!(
                    "Could not get video file path for episode {} and video key {}",
                    ep_idx, vid_key
                )
                .as_str(),
            );
            let video_path = self.meta.root.join(video_rel_path);

            // TODO: decode_video_frames
            let frames = decode_video_frames(
                &video_path,
                &shifted_query_ts,
                self.tolerance_s,
                self.video_backend,
            ).map_err(|err| {
                PolarsError::ComputeError(
                    format!(
                        "Could not decode video frames for episode {ep_idx}, key {vid_key}, path {}: {err}",
                        video_path.display()
                    )
                    .into(),
                )
            })?;
            video_frames.insert(vid_key.clone(), frames);
        }

        Ok(video_frames)
    }

    pub fn get_item(&self, idx: usize) -> PolarsResult<DatasetItem> {
        let dataset = self
            .hf_dataset
            .as_ref()
            .ok_or_else(|| PolarsError::ComputeError("hf_dataset is not loaded".into()))?;

        if idx >= self.len() {
            return Err(PolarsError::OutOfBounds(
                format!(
                    "Index  {} is out of bounds for dataset with len {}",
                    idx,
                    self.len()
                )
                .into(),
            ));
        }

        // Convert the row into a HashMap
        let mut item: HashMap<String, DatasetItemValue> = HashMap::new();
        for column in dataset.get_columns() {
            item.insert(
                column.name().to_string(),
                DatasetItemValue::Polars(column.get(idx)?.into_static()),
            );
        }

        let ep_idx = dataset
            .column("episode_index")?
            .get(idx)?
            .try_extract::<u32>()? as usize;

        let abs_idx = dataset.column("index")?.get(idx)?.try_extract::<u32>()? as usize;

        let mut query_indices: Option<QueryIndices> = None;

        if self.delta_indices.is_some() {
            let (indices, padding) = self.get_query_indices(abs_idx, ep_idx)?;
            let query_result = self.query_hf_dataset(&indices)?;

            for (key, mask) in padding {
                item.insert(key, DatasetItemValue::BoolVec(mask));
            }

            for (key, value) in query_result {
                item.insert(key, DatasetItemValue::DataFrame(value));
            }

            query_indices = Some(indices);
        }

        if !self.meta.video_keys().is_empty() {
            let current_ts = dataset
                .column("timestamp")?
                .get(idx)?
                .try_extract::<f64>()?;

            let query_timestamps = self.get_query_timestamps(current_ts, query_indices.as_ref())?;

            let video_frames = self.query_videos(&query_timestamps, ep_idx)?;
            for (key, frames) in video_frames {
                item.insert(key, DatasetItemValue::VideoFrames(frames));
            }
        }

        // Extract task
        let task_idx = dataset
            .column("task_index")?
            .get(idx)?
            .try_extract::<u32>()? as usize;

        let task = self
            .meta
            .tasks
            .column("task")?
            .get(task_idx)?
            .str_value()
            .into_owned();

        item.insert("task".to_string(), DatasetItemValue::String(task));

        // Add subtask if available.
        if self.meta.info.features.contains_key("subtask_index") {
            if let Some(subtasks) = &self.meta.subtasks {
                let subtask_idx = dataset
                    .column("subtask_index")?
                    .get(idx)?
                    .try_extract::<u32>()? as usize;

                let subtask = subtasks
                    .column("subtask")?
                    .get(subtask_idx)?
                    .str_value()
                    .into_owned();

                item.insert("subtask".to_string(), DatasetItemValue::String(subtask));
            }
        }

        Ok(item)
    }
}

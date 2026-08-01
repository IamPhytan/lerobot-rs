use crate::datasets::utils;
use crate::datasets::{dataset_reader::DatasetReader, utils::FileError};
use crate::types::{DatasetItem, DeltaTimestamps};
use polars as pl;
use polars::error::PolarsResult;
use polars::lazy::prelude::LazyFrame;
use polars::prelude::{IntoLazy, col, lit};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct LeRobotDatasetMetadata {
    pub repo_id: String,
    pub root: PathBuf,
    revision: String,
    pub info: utils::DatasetInfo,
    pub tasks: polars::frame::DataFrame,
    pub subtasks: Option<polars::frame::DataFrame>,
    pub episodes: polars::frame::DataFrame,
    pub stats: utils::DatasetStats,
}

impl LeRobotDatasetMetadata {
    pub fn new(repo_id: &str, root: PathBuf, revision: &str) -> Self {
        // Load metadata
        let meta_dir = root.join("meta");

        // Info
        let info =
            utils::load_info(meta_dir.join("info.json")).expect("Error while reading dataset info");

        // Tasks
        let tasks = utils::load_tasks(meta_dir.join("tasks.parquet"))
            .expect("Error while reading dataset tasks");

        // Subtasks
        let subtasks = utils::load_subtasks(meta_dir.join("subtasks.parquet"));

        // Episodes
        let episodes = utils::load_episodes(meta_dir.join("episodes"))
            .expect("Error while reading dataset episodes");

        // Stats
        let stats = utils::load_stats(meta_dir.join("stats.json"))
            .expect("Error while reading dataset stats");

        Self {
            repo_id: repo_id.to_string(),
            root,
            revision: revision.to_string(),
            info,
            tasks,
            subtasks,
            episodes,
            stats,
        }
    }

    fn url_root(&self) -> String {
        format!("hf://datasets/{}", &self.repo_id.as_str())
    }

    fn _version(&self) -> &str {
        &self.info.codebase_version
    }

    pub fn get_episode(&self, ep_index: usize) -> Option<LazyFrame> {
        if ep_index >= self.episodes.height() {
            // TODO: add an error
            return None;
        }
        let ep: LazyFrame = self
            .episodes
            .clone()
            .lazy()
            .filter(col("episode_index").eq(ep_index as u32));

        Some(ep)
    }

    fn get_chunk_index(&self, ep: LazyFrame) -> Option<u32> {
        let chunk_index: u32 = ep
            .clone()
            .select([col("data/chunk_index")])
            .collect()
            .ok()?
            .column("data/chunk_index")
            .map_err(FileError::from)
            .ok()?
            .u32()
            .ok()?
            .get(0)?;

        Some(chunk_index)
    }

    fn get_file_index(&self, ep: LazyFrame) -> Option<u32> {
        let file_index = ep
            .clone()
            .select([col("data/file_index")])
            .collect()
            .ok()?
            .column("data/file_index")
            .map_err(FileError::from)
            .ok()?
            .u32()
            .ok()?
            .get(0)?;

        Some(file_index)
    }

    pub fn get_data_file_path(&self, ep_index: usize) -> Option<PathBuf> {
        let ep = self.get_episode(ep_index)?;
        let chunk_index = self.get_chunk_index(ep.clone())?;
        let file_index = self.get_file_index(ep.clone())?;

        let formatted_data_path = self
            .data_path()
            .replace("{chunk_index:03d}", format!("{chunk_index:03}").as_str())
            .replace("{file_index:03d}", format!("{file_index:03}").as_str());

        Some(formatted_data_path.into())
    }

    pub fn get_video_file_path(&self, ep_index: usize, vid_key: &str) -> Option<PathBuf> {
        let ep = self.get_episode(ep_index)?;
        let chunk_index = self.get_chunk_index(ep.clone())?;
        let file_index = self.get_file_index(ep.clone())?;

        let formatted_video_path = self
            .video_path()?
            .replace("{chunk_index:03d}", format!("{chunk_index:03}").as_str())
            .replace("{file_index:03d}", format!("{file_index:03}").as_str())
            .replace("{video_key}", vid_key);

        Some(formatted_video_path.into())
    }

    pub fn data_path(&self) -> &str {
        &self.info.data_path
    }

    fn video_path(&self) -> Option<&str> {
        if self.info.video_path.is_empty() {
            None
        } else {
            Some(self.info.video_path.as_str())
        }
    }

    fn robot_type(&self) -> Option<&str> {
        Some(&self.info.robot_type.as_str())
    }

    pub fn fps(&self) -> f32 {
        self.info.fps
    }

    fn features(&self) -> Option<HashMap<String, utils::DatasetFeature>> {
        Some(self.info.features.clone())
    }

    fn get_keys_by_filter(&self, filter_values: Vec<&str>) -> Vec<String> {
        self.features()
            .unwrap_or(HashMap::new())
            .iter()
            .filter_map(|(key, ft)| {
                if filter_values.contains(&ft.dtype.as_str()) {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    fn image_keys(&self) -> Vec<String> {
        self.get_keys_by_filter(vec!["image"])
    }

    pub fn video_keys(&self) -> Vec<String> {
        self.get_keys_by_filter(vec!["video"])
    }

    pub fn camera_keys(&self) -> Vec<String> {
        self.get_keys_by_filter(vec!["video", "image"])
    }

    fn names(&self) -> HashMap<String, Option<utils::DatasetFeatureNames>> {
        self.features()
            .unwrap_or(HashMap::new())
            .iter()
            .map(|(key, ft)| (key.clone(), ft.names.clone()))
            .collect()
    }

    fn shapes(&self) -> HashMap<String, Vec<usize>> {
        self.features()
            .unwrap_or(HashMap::new())
            .iter()
            .map(|(key, ft)| (key.clone(), ft.shape.clone()))
            .collect()
    }

    fn total_episodes(&self) -> usize {
        self.info.total_episodes
    }

    fn total_frames(&self) -> usize {
        self.info.total_frames
    }

    fn total_tasks(&self) -> usize {
        self.info.total_tasks
    }

    fn chunks_size(&self) -> usize {
        self.info.chunks_size
    }

    fn data_files_size_in_mb(&self) -> u32 {
        self.info.data_files_size_in_mb
    }

    fn video_files_size_in_mb(&self) -> u32 {
        self.info.video_files_size_in_mb
    }

    pub fn get_task_index(&self, task: &str) -> Option<usize> {
        let filtered_tasks = self
            .tasks
            .clone()
            .lazy()
            // .map(
            //     |f| {
            //         let _ = f.column("task").iter().map(|&val| println!("{val:?}"));
            //         Ok(f)
            //     },
            //     OptFlags::all(),
            //     None,
            //     None,
            // )
            .filter(col("task").eq(lit(task)))
            .select([col("task_index")])
            .limit(1)
            .collect()
            .expect(format!("Problem finding task: {}", task).as_str());

        if filtered_tasks.height() == 0 {
            return None;
        }

        filtered_tasks
            .column("task_index")
            .expect("No column 'task_index'")
            .u64()
            .expect("Task index cannot be converted to a u64")
            .get(0)
            .map(|x| x as usize)
    }
}

#[derive(Debug)]
pub struct LeRobotDataset {
    pub repo_id: String,
    pub meta: LeRobotDatasetMetadata,
    pub reader: DatasetReader,
    episodes: Option<Vec<usize>>,
}

impl LeRobotDataset {
    pub fn new(
        repo_id: &str,
        root: Option<PathBuf>,
        episodes: Option<Vec<usize>>,
        delta_timestamps: Option<DeltaTimestamps>,
        tolerance_s: Option<f64>,
        revision: Option<&str>,
    ) -> Self {
        let dataset_root = root.unwrap_or_else(|| crate::lerobot_home().join(repo_id));
        let revision = revision.unwrap_or("main");
        let meta = LeRobotDatasetMetadata::new(repo_id, dataset_root, revision);

        let tolerance_s = tolerance_s.unwrap_or(1e-4);

        // Dataset Reader
        let mut reader = DatasetReader::new(
            meta.clone(),
            episodes.clone(),
            tolerance_s,
            None,
            delta_timestamps,
        );
        reader.try_load();

        Self {
            repo_id: repo_id.to_string(),
            meta,
            reader,
            episodes,
        }
    }

    pub fn root(&self) -> &Path {
        &self.meta.root.as_path()
    }

    /// Number of frames in selected episodes
    fn fps(&self) -> f32 {
        self.meta.info.fps
    }

    fn num_frames(&self) -> usize {
        self.meta.total_frames()
    }

    fn num_episodes(&self) -> usize {
        self.meta.total_episodes()
    }

    fn features(&self) -> Option<HashMap<String, utils::DatasetFeature>> {
        self.meta.features()
    }

    fn hf_dataset(&mut self) -> &pl::frame::DataFrame {
        if self.reader.hf_dataset == None {
            self.reader.try_load();
        }
        self.reader
            .hf_dataset
            .as_ref()
            .expect("hf_dataset not loaded")
    }

    pub fn len(&self) -> usize {
        self.reader.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reader.is_empty()
    }

    pub fn get_item(&self, idx: usize) -> PolarsResult<DatasetItem> {
        self.reader.get_item(idx)
    }

    pub fn iter(&self) -> LeRobotDatasetIter<'_> {
        LeRobotDatasetIter {
            dataset: self,
            idx: 0,
            len: self.len(),
        }
    }
}

pub struct LeRobotDatasetIter<'a> {
    dataset: &'a LeRobotDataset,
    idx: usize,
    len: usize,
}

impl<'a> Iterator for LeRobotDatasetIter<'a> {
    type Item = PolarsResult<DatasetItem>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            return None;
        }

        let idx = self.idx;
        self.idx += 1;

        Some(self.dataset.get_item(idx))
    }
}

impl<'a> IntoIterator for &'a LeRobotDataset {
    type Item = PolarsResult<DatasetItem>;
    type IntoIter = LeRobotDatasetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

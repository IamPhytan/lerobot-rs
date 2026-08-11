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

/// Metadata container for a LeRobot dataset.
///
/// Similar to `LeRobotDatasetMetadata`` in [LeRobot](https://github.com/huggingface/lerobot).
///
/// Manages the ``info.json``, ``stats.json``, ``tasks.parquet``, and ``episodes/`` parquet files that describe a dataset's structure, content, and statistics.
#[derive(Debug, Clone)]
pub struct LeRobotDatasetMetadata {
    /// Repository identifier (e.g. ``'lerobot/aloha_sim'``).
    pub repo_id: String,
    /// Local directory for the dataset.When omitted, existing local datasets are looked up under ``$HF_LEROBOT_HOME/{repo_id}``.
    pub root: PathBuf,
    /// Git revision (branch, tag, or commit hash). Defaults to the current codebase version.
    pub revision: String,
    /// Dataset information imported from `meta/info.json`.
    ///
    /// See [`utils::DatasetInfo`].
    pub info: utils::DatasetInfo,
    /// Dataset tasks imported from `meta/tasks.parquet`.
    pub tasks: polars::frame::DataFrame,
    /// Dataset subtasks imported from `meta/subtasks.parquet`, if present.
    pub subtasks: Option<polars::frame::DataFrame>,
    /// Dataset episode metadata info imported from `meta/episodes/`.
    pub episodes: polars::frame::DataFrame,
    /// Dataset statistics used for normalization imported from `meta/stats.json`.
    ///
    /// See [`utils::DatasetStats`].
    pub stats: utils::DatasetStats,
}

impl LeRobotDatasetMetadata {
    /// Load metadata for an existing LeRobot dataset.
    ///
    /// Metadata files are loaded from the `meta/` directory under `root`, including dataset information, tasks, subtasks, episodes, and statistics.
    ///
    /// # Arguments
    ///
    /// * `repo_id`: Repository identifier (e.g. ``'lerobot/aloha_sim'``).
    /// * `root` - Local path to the dataset root directory.
    /// * `revision` - Git revision associated with the dataset (branch, tag, or commit hash). Optional, defaults to ``'main'``
    ///
    /// # Panics
    ///
    /// Panics if any required metadata file cannot be read or parsed:
    ///
    /// * `meta/info.json`
    /// * `meta/tasks.parquet`
    /// * `meta/episodes/`
    /// * `meta/stats.json`
    ///
    /// The `meta/subtasks.parquet` file is optional.
    pub fn new(repo_id: &str, root: PathBuf, revision: Option<&str>) -> Self {
        let revision = revision.unwrap_or("main").to_string();

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
            revision,
            info,
            tasks,
            subtasks,
            episodes,
            stats,
        }
    }

    /// Hugging Face Hub URL root for this dataset.
    pub fn url_root(&self) -> String {
        format!("hf://datasets/{}", &self.repo_id.as_str())
    }

    /// Codebase version used to create this dataset.
    fn _version(&self) -> &str {
        &self.info.codebase_version
    }

    /// Get metadata corresponding to episode `ep_index`
    ///
    /// # Arguments
    ///
    /// * `ep_index` - Index of the episode to retrieve.
    ///
    /// # Returns
    ///
    /// Returns a [`LazyFrame`] containing the metadata for the requested episode,
    /// or `None` if `ep_index` is out of bounds.
    pub fn get_episode(&self, ep_index: usize) -> Option<LazyFrame> {
        if ep_index >= self.episodes.height() {
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
        ep.select([col("data/chunk_index")])
            .collect()
            .ok()?
            .column("data/chunk_index")
            .map_err(FileError::from)
            .ok()?
            .u32()
            .ok()?
            .get(0)
    }

    fn get_file_index(&self, ep: LazyFrame) -> Option<u32> {
        ep.select([col("data/file_index")])
            .collect()
            .ok()?
            .column("data/file_index")
            .map_err(FileError::from)
            .ok()?
            .u32()
            .ok()?
            .get(0)
    }

    /// Return the relative parquet file path for the given episode index `ep_index`.
    ///
    /// # Arguments
    ///
    /// * `ep_index` - Zero-based episode index of the episode to retrieve.
    ///
    /// # Returns
    ///
    /// Path to the parquet file containing this episode's data,
    /// or `None` if the episode, chunk index, or file index cannot be found.
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

    /// Return the relative video file path for the given episode `ep_index` and video key `vid_key`.
    ///
    /// # Arguments
    ///
    /// * `ep_index` - Zero-based episode index of the episode to retrieve.
    /// * `vid_key` - Feature key identifying the video stream (e.g. ``'observation.images.laptop'``).
    ///
    /// # Returns
    ///
    /// Path to the video file containing this episode's frames,
    /// or `None` if the episode, chunk index, file index, or video path cannot be found.
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

    /// Formattable string for the parquet files.
    pub fn data_path(&self) -> &str {
        &self.info.data_path
    }

    /// Formattable string for the video files.
    pub fn video_path(&self) -> Option<&str> {
        if self.info.video_path.is_empty() {
            None
        } else {
            Some(self.info.video_path.as_str())
        }
    }

    /// Robot type used in recording this dataset.
    pub fn robot_type(&self) -> Option<&str> {
        Some(&self.info.robot_type.as_str())
    }

    /// Frames per second used during data collection.
    pub fn fps(&self) -> f32 {
        self.info.fps
    }

    /// All features contained in the dataset.
    pub fn features(&self) -> Option<HashMap<String, utils::DatasetFeature>> {
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

    /// Keys to access visual modalities stored as images.
    pub fn image_keys(&self) -> Vec<String> {
        self.get_keys_by_filter(vec!["image"])
    }

    /// Keys to access visual modalities stored as videos.
    pub fn video_keys(&self) -> Vec<String> {
        self.get_keys_by_filter(vec!["video"])
    }

    /// Keys to access visual modalities (regardless of their storage method).
    pub fn camera_keys(&self) -> Vec<String> {
        self.get_keys_by_filter(vec!["video", "image"])
    }

    /// Names of the various dimensions of vector modalities.
    pub fn names(&self) -> HashMap<String, Option<utils::DatasetFeatureNames>> {
        self.features()
            .unwrap_or(HashMap::new())
            .iter()
            .map(|(key, ft)| (key.clone(), ft.names.clone()))
            .collect()
    }

    /// Shapes for the different features.
    pub fn shapes(&self) -> HashMap<String, Vec<usize>> {
        self.features()
            .unwrap_or(HashMap::new())
            .iter()
            .map(|(key, ft)| (key.clone(), ft.shape.clone()))
            .collect()
    }

    /// Total number of episodes available.
    pub fn total_episodes(&self) -> usize {
        self.info.total_episodes
    }

    /// Total number of frames saved in this dataset.
    pub fn total_frames(&self) -> usize {
        self.info.total_frames
    }

    /// Total number of different tasks performed in this dataset.
    pub fn total_tasks(&self) -> usize {
        self.info.total_tasks
    }

    /// Max number of files per chunk.
    pub fn chunks_size(&self) -> usize {
        self.info.chunks_size
    }

    /// Max size of data file in mega bytes.
    pub fn data_files_size_in_mb(&self) -> u32 {
        self.info.data_files_size_in_mb
    }

    /// Max size of video file in mega bytes.
    pub fn video_files_size_in_mb(&self) -> u32 {
        self.info.video_files_size_in_mb
    }

    /// Given a `task` in natural language, returns its task_index if the task already exists in the dataset, otherwise return `None`.
    ///
    /// # Arguments
    ///
    /// * `task` - Natural-language description of the task.
    ///
    /// # Returns
    ///
    /// Returns the task index if a matching task exists, or `None` otherwise.
    /// # Panics
    ///
    /// Panics if the task table cannot be queried, if the `task_index` column is missing, or if the column cannot be interpreted as `u64`.
    pub fn get_task_index(&self, task: &str) -> Option<usize> {
        let filtered_tasks = self
            .tasks
            .clone()
            .lazy()
            .filter(col("task").eq(lit(task)))
            .select([col("task_index")])
            .limit(1)
            .collect()
            .unwrap_or_else(|_| panic!("Problem finding task: {task}"));

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

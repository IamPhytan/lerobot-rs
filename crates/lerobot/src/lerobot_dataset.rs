use crate::datasets::utils;
use polars::prelude::{IntoLazy, OptFlags, col, lit};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

pub struct LeRobotDatasetMetadata {
    repo_id: String,
    root: PathBuf,
    revision: String,
    pub info: utils::DatasetInfo,
    pub stats: utils::DatasetStats,
    pub tasks: polars::frame::DataFrame,
    pub episodes: polars::frame::DataFrame,
}

impl LeRobotDatasetMetadata {
    pub fn new(repo_id: &str, root: PathBuf, revision: &str) -> Self {
        // Load metadata
        let meta_dir = root.join("meta");

        // Info
        let info =
            utils::load_info(meta_dir.join("info.json")).expect("Error while reading dataset info");

        // Stats
        let stats = utils::load_stats(meta_dir.join("stats.json"))
            .expect("Error while reading dataset stats");

        // Tasks
        let tasks = utils::load_tasks(meta_dir.join("tasks.parquet"))
            .expect("Error while reading dataset tasks");

        // Episodes
        let episodes = utils::load_episodes(meta_dir.join("episodes"))
            .expect("Error while reading dataset episodes");

        Self {
            repo_id: repo_id.to_string(),
            root,
            revision: revision.to_string(),
            info,
            stats,
            tasks,
            episodes,
        }
    }

    fn url_root(&self) -> String {
        format!("hf://datasets/{}", &self.repo_id.clone())
    }

    fn _version(&self) -> &str {
        &self.info.codebase_version
    }

    fn data_path(&self) -> &str {
        &self.info.data_path
    }

    fn video_path(&self) -> Option<&str> {
        Some(&self.info.video_path.as_str())
    }

    fn robot_type(&self) -> Option<&str> {
        Some(&self.info.robot_type.as_str())
    }

    fn fps(&self) -> Option<f32> {
        Some(self.info.fps)
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
            .map(
                |f| {
                    let hello = f.column("task").iter().map(|&val| println!("{val:?}"));
                    Ok(f)
                },
                OptFlags::all(),
                None,
                None,
            )
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

struct Episode {}

pub struct LeRobotDataset {
    pub repo_id: String,
    pub meta: LeRobotDatasetMetadata,
    episodes: Vec<Episode>,
}

impl LeRobotDataset {
    pub fn new(
        repo_id: &str,
        root: Option<PathBuf>,
        episodes: Option<Vec<usize>>,
        delta_timestamps: Option<HashMap<&str, Vec<f64>>>,
        tolerance_s: Option<f64>,
        revision: Option<&str>,
    ) -> Self {
        let dataset_root = root.unwrap_or_else(|| crate::lerobot_home().join(repo_id));

        let meta = LeRobotDatasetMetadata::new(repo_id, dataset_root, "main");

        // Placeholder for loading episodes
        let episodes = Vec::new();

        Self {
            repo_id: repo_id.to_string(),
            meta,
            episodes,
        }
    }

    pub fn root(&self) -> &Path {
        &self.meta.root.as_path()
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }
}

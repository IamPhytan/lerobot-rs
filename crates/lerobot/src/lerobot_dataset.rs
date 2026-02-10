use crate::datasets::utils;
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

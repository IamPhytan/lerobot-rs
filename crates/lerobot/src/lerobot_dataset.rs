use crate::datasets::utils;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

pub struct LeRobotDatasetMetadata {
    repo_id: String,
    root: PathBuf,
    revision: String,
}

impl LeRobotDatasetMetadata {
    pub fn new(repo_id: &str, root: PathBuf, revision: &str) -> Self {
        let meta_info_path = root.join("meta/info.json");
        let info = utils::load_info(meta_info_path).expect("Error while reading info file");

        println!("{:?}", info);

        Self {
            repo_id: repo_id.to_string(),
            root,
            revision: revision.to_string(),
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

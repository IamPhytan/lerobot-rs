use std::path::{Path, PathBuf};

pub struct LeRobotDatasetMetadata {
    repo_id: String,
    root: PathBuf,
    revision: String,
}

struct Episode {}

pub struct LeRobotDataset {
    pub repo_id: String,
    pub meta: LeRobotDatasetMetadata,
    episodes: Vec<Episode>,
}

impl LeRobotDataset {
    pub fn new(repo_id: &str, root: Option<PathBuf>) -> Self {
        let dataset_root = root.unwrap_or_else(|| crate::lerobot_home().join(repo_id));

        let meta = LeRobotDatasetMetadata {
            repo_id: repo_id.to_string(),
            revision: String::from("main"),
            root: dataset_root,
        };

        // Placeholder for loading episodes
        let episodes = Vec::new();

        LeRobotDataset {
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

mod datasets;
pub mod lerobot_dataset;
use std::env;
use std::path::PathBuf;

pub use lerobot_dataset::LeRobotDataset;

pub fn default_path() -> PathBuf {
    env::home_dir()
        .expect("Could not find home directory")
        .join(".cache")
        .join("huggingface")
        .join("lerobot")
}

pub fn lerobot_home() -> PathBuf {
    env::var("LEROBOT_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_path().clone())
}

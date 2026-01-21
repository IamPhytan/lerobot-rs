pub mod lerobot_dataset;
use std::env;
use std::path::PathBuf;

pub fn default_path() -> PathBuf {
    env::home_dir()
        .expect("Could not find home directory")
        .join(".cache")
        .join("huggingface")
}

pub fn lerobot_home() -> PathBuf {
    env::var("LEROBOT_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_path().clone())
}

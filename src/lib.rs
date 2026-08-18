mod datasets;
pub mod lerobot_dataset;
use std::env;
use std::path::PathBuf;
mod types;
pub use datasets::dataset_reader::DatasetItemValue;
pub use datasets::video_utils::VideoFrames;
pub use types::DatasetItem;

pub use lerobot_dataset::LeRobotDataset;

/// Returns the default local directory used to store LeRobot datasets.
///
/// The default path is:
///
/// ```text
/// ~/.cache/huggingface/lerobot
/// ```
///
/// # Panics
///
/// Panics if the user's home directory cannot be determined.
pub fn default_path() -> PathBuf {
    env::home_dir()
        .expect("Could not find home directory")
        .join(".cache")
        .join("huggingface")
        .join("lerobot")
}

/// Returns the LeRobot home directory.
///
/// If the `LEROBOT_HOME` environment variable is set, its value is used.
/// Otherwise, [`default_path`] is returned.
///
/// # Panics
///
/// Panics if `LEROBOT_HOME` is not set and the user's home directory cannot
/// be determined.
pub fn lerobot_home() -> PathBuf {
    env::var("LEROBOT_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_path())
}

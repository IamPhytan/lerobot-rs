//! # lerobot
//!
//! Rust utilities for loading and reading [LeRobot] datasets.
//!
//! This crate provides [`LeRobotDataset`] for accessing LeRobot datasets, their metadata, episodes, and associated video data.
//!
//! By default, datasets are expected under `~/.cache/huggingface/lerobot`, unless the `LEROBOT_HOME` environment variable is set.
//!
//! ## Examples
//!
//! ```no_run
//! use lerobot::LeRobotDataset;
//!
//! let dataset = LeRobotDataset::new(
//!     "lerobot/pusht",
//!     None,
//!     None,
//!     None,
//!     None,
//!     None,
//! );
//!
//! println!(
//!     "Loaded dataset {} with {} items",
//!     dataset.repo_id,
//!     dataset.len()
//! );
//! ```
//!
//! [LeRobot]: https://github.com/huggingface/lerobot

#![warn(missing_docs)]
use std::env;
use std::path::PathBuf;

mod datasets;
mod lerobot_dataset;
mod types;

pub use datasets::dataset_reader::DatasetItemValue;
pub use datasets::video_utils::VideoFrames;
pub use lerobot_dataset::LeRobotDataset;
pub use types::DatasetItem;

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

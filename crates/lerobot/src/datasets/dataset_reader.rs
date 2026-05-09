use std::path::PathBuf;

use crate::datasets::utils::FileError;
use crate::datasets::utils::PathGlob;
use crate::lerobot_dataset::LeRobotDatasetMetadata;

#[derive(Debug)]
pub struct DatasetReader {
    meta: LeRobotDatasetMetadata,
    hf_dataset: Option<u32>,
}

impl DatasetReader {
    pub fn new(meta: LeRobotDatasetMetadata) -> Self {
        Self {
            meta,
            hf_dataset: None,
        }
    }

    pub fn try_load(&mut self, episodes: Option<Vec<i16>>) {
        self.hf_dataset = match self.load_hf_dataset(episodes) {
            Ok(value) => Some(value),
            _ => None,
        }
    }

    pub fn load_hf_dataset(&self, episodes: Option<Vec<i16>>) -> Result<u32, FileError> {
        let data_dir = self.meta.root.join("data");
        println!("Reading data dir: {:?}", data_dir);

        let files = data_dir
            .glob("**/*.parquet")
            .map(|p| p.expect("Error reading file {p}"))
            .collect::<Vec<PathBuf>>();

        println!("{:?}", files);

        Ok(5)
    }
}

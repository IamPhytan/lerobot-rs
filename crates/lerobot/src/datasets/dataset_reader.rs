use std::path::PathBuf;

use crate::datasets::utils::FileError;
use crate::datasets::utils::PathGlob;
use crate::lerobot_dataset::LeRobotDatasetMetadata;
use polars as pl;
use polars::lazy::prelude::LazyFrame;
use polars::prelude::{DataType, PlPath, UnionArgs, col, concat};

#[derive(Debug)]
pub struct DatasetReader {
    pub meta: LeRobotDatasetMetadata,
    pub hf_dataset: Option<pl::frame::DataFrame>,
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

    pub fn load_hf_dataset(
        &self,
        episodes: Option<Vec<i16>>,
    ) -> Result<pl::frame::DataFrame, FileError> {
        let data_dir = self.meta.root.join("data");
        println!("Reading data dir: {:?}", data_dir);

        let files = data_dir
            .glob("**/*.parquet")
            .map(|p| p.expect("Error reading file {p}"))
            .collect::<Vec<PathBuf>>();

        let data = files
            .iter()
            .map(|fpath| {
                LazyFrame::scan_parquet(
                    PlPath::new(fpath.to_str().expect("Polars Path error")),
                    Default::default(),
                )
                .map_err(FileError::from)
                .expect(format!("Error while scanning parquet file {:?}", fpath).as_str())
            })
            .collect::<Vec<LazyFrame>>();

        let all_data = concat(data, UnionArgs::default())?.collect()?;

        Ok(all_data)
    }
}

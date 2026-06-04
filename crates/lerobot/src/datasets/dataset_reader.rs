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
    pub episodes: Option<Vec<usize>>,
}

impl DatasetReader {
    pub fn new(meta: LeRobotDatasetMetadata, episodes: Option<Vec<usize>>) -> Self {
        Self {
            meta,
            hf_dataset: None,
            episodes,
        }
    }

    pub fn try_load(&mut self) {
        self.hf_dataset = match self.load_hf_dataset() {
            Ok(value) => Some(value),
            _ => None,
        }
    }

    pub fn load_hf_dataset(&self) -> Result<pl::frame::DataFrame, FileError> {
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

    pub fn get_episodes_file_paths(&self) -> Vec<PathBuf> {
        let episodes = self
            .episodes
            .clone()
            .unwrap_or((0..self.meta.info.total_episodes).collect::<Vec<usize>>());
        let mut fpaths = episodes
            .iter()
            .filter_map(|&ep_idx| self.meta.get_data_file_path(ep_idx))
            .collect::<Vec<PathBuf>>();

        let video_fpaths = self
            .meta
            .video_keys()
            .iter()
            .map(|vid_key| {
                episodes
                    .iter()
                    .filter_map(|&ep_idx| self.meta.get_video_file_path(ep_idx, vid_key))
                    .collect::<Vec<PathBuf>>()
            })
            .flatten()
            .collect::<Vec<PathBuf>>();

        fpaths.extend(video_fpaths);

        return fpaths;
    }
}

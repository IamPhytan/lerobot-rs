use std::collections::HashSet;
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
        };

        let complete = self.check_cache_episodes_sufficient();
        println!("COMPLETE {complete}");
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
                .with_column(col("episode_index").cast(DataType::UInt32))
            })
            .collect::<Vec<LazyFrame>>();

        let all_data = concat(data, UnionArgs::default())?.collect()?;

        Ok(all_data)
    }

    fn check_cache_episodes_sufficient(&self) -> bool {
        if (self.hf_dataset == None) && (self.len() == 0) {
            return false;
        }

        let available_episodes = self
            .hf_dataset
            .clone()
            .expect("Could not find hf_dataset")
            .column("episode_index")
            .expect("No column 'episode_index")
            .unique()
            .expect("Issue getting unique values")
            .u32()
            .expect("Episode indices cannot be converted to u32")
            .iter()
            .filter_map(|x| match x {
                Some(v) => Some(v as usize),
                None => None,
            })
            .collect::<Vec<usize>>();

        let requested_episodes: HashSet<usize> = match &self.episodes {
            Some(episodes) => HashSet::from_iter(episodes.clone()),
            None => HashSet::from_iter((0..self.meta.info.total_episodes).into_iter()),
        };

        if !requested_episodes.is_subset(&HashSet::from_iter(available_episodes)) {
            return false;
        }

        for ep_index in requested_episodes {
            for vid_key in self.meta.video_keys() {
                let video_path = self.meta.root.join(
                    self.meta
                        .get_video_file_path(ep_index, vid_key.as_str())
                        .expect("Could not get video file path"),
                );
                if !video_path.exists() {
                    panic!("Missing video file {}", video_path.display());
                    return false;
                }
            }
        }

        return true;
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

    pub fn len(&self) -> usize {
        self.hf_dataset.iter().len()
    }
}

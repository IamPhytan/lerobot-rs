use glob;
use polars::lazy::prelude::LazyFrame;
use polars::prelude::{DataType, PlPath, UnionArgs, col, concat};
use polars::{self as pl, io::SerReader};
use serde::Deserialize;
use serde_json;
use serde_json::Value;
use std::{collections::HashMap, error, fmt, fs::File, io, path::Path, path::PathBuf};

pub trait PathGlob {
    fn glob(&self, pattern: &str) -> glob::Paths;
}

impl PathGlob for Path {
    fn glob(&self, pattern: &str) -> glob::Paths {
        let path_pattern = self.join(pattern);
        glob::glob(path_pattern.to_str().expect("Invalid UTF-8 in path"))
            .expect("Invalid glob pattern")
    }
}

#[derive(Debug)]
pub enum FileError {
    Io(io::Error),
    Json(serde_json::Error),
    Polars(pl::error::PolarsError),
    Glob(glob::GlobError),
    PathEncoding,
}

impl From<serde_json::Error> for FileError {
    fn from(err: serde_json::Error) -> Self {
        FileError::Json(err)
    }
}

impl From<io::Error> for FileError {
    fn from(err: io::Error) -> Self {
        FileError::Io(err)
    }
}

impl From<pl::error::PolarsError> for FileError {
    fn from(err: pl::error::PolarsError) -> Self {
        FileError::Polars(err)
    }
}

impl From<glob::GlobError> for FileError {
    fn from(err: glob::GlobError) -> Self {
        FileError::Glob(err)
    }
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileError::Io(e) => write!(f, "I/O error: {}", e),
            FileError::Json(e) => write!(f, "JSON error: {}", e),
            FileError::Polars(e) => write!(f, "Polars error: {}", e),
            FileError::Glob(e) => write!(f, "Glob error: {}", e),
            FileError::PathEncoding => write!(f, "Path is not valid UTF-8"),
        }
    }
}

impl error::Error for FileError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            FileError::Io(e) => Some(e),
            FileError::Json(e) => Some(e),
            FileError::Polars(e) => Some(e),
            FileError::Glob(e) => Some(e),
            FileError::PathEncoding => None,
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum DatasetFeatureNames {
    Map(HashMap<String, Vec<String>>),
    Vec(Vec<String>),
}

#[derive(Deserialize, Debug, Clone)]
struct VideoInfo {
    #[serde(rename = "video.fps")]
    video_fps: f32,
    #[serde(rename = "video.codec")]
    video_codec: String,
    #[serde(rename = "video.pix_fmt")]
    video_pix_fmt: String,
    #[serde(rename = "video.is_depth_map")]
    video_is_depth_map: bool,
    has_audio: bool,
}

#[derive(Deserialize, Debug, Clone)]
pub struct DatasetFeature {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub names: Option<DatasetFeatureNames>,
    pub fps: Option<f32>,
    pub video_info: Option<VideoInfo>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct DatasetInfo {
    pub codebase_version: String,
    pub robot_type: String,
    pub total_episodes: usize,
    pub total_frames: usize,
    pub total_tasks: usize,
    pub chunks_size: usize,
    pub data_files_size_in_mb: u32,
    pub video_files_size_in_mb: u32,
    pub fps: f32,
    pub splits: HashMap<String, String>,
    pub data_path: String,
    pub video_path: String,
    pub features: HashMap<String, DatasetFeature>,
}

pub fn load_info<P: AsRef<Path>>(path: P) -> Result<DatasetInfo, FileError> {
    println!("Reading from file: {:?}", path.as_ref());
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    Ok(serde_json::from_reader(reader)?)
}

pub fn load_tasks<P: AsRef<Path>>(path: P) -> Result<pl::frame::DataFrame, FileError> {
    println!("Reading from file: {:?}", path.as_ref());

    let fpath = path.as_ref().to_str().ok_or(FileError::PathEncoding)?;

    let df = LazyFrame::scan_parquet(PlPath::new(fpath), Default::default())?
        .select([
            col("task_index").cast(DataType::UInt64),
            col("__index_level_0__").alias("task"),
        ])
        .collect()?;

    Ok(df)
}

pub fn load_subtasks<P: AsRef<Path>>(path: P) -> Option<pl::frame::DataFrame> {
    println!("Reading from file: {:?}", path.as_ref());

    let fpath = path.as_ref().to_str()?;

    let df = LazyFrame::scan_parquet(PlPath::new(fpath), Default::default())
        .ok()?
        // .select([
        //     col("task_index").cast(DataType::UInt64),
        //     col("__index_level_0__").alias("task"),
        // ])
        .collect()
        .ok()?;

    println!("{:?}", df);

    Some(df)
}

pub fn load_episodes<P: AsRef<Path>>(path: P) -> Result<pl::frame::DataFrame, FileError> {
    println!("Reading from file: {:?}", path.as_ref());

    let files = path
        .as_ref()
        .glob("**/*.parquet")
        .map(|p| p.expect("Error reading file {p}"))
        .collect::<Vec<PathBuf>>();

    let episodes = files
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

    let all_episodes = concat(episodes, UnionArgs::default())?.collect()?;

    Ok(all_episodes)
}

#[derive(Deserialize, Debug, Clone)]
pub struct DatasetFeatureStats {
    min: Vec<Value>,
    max: Vec<Value>,
    mean: Vec<Value>,
    std: Vec<Value>,
    count: Vec<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct DatasetStats {
    #[serde(flatten)]
    pub features: HashMap<String, DatasetFeatureStats>,
}

pub fn load_stats<P: AsRef<Path>>(path: P) -> Result<DatasetStats, FileError> {
    println!("Reading from file: {:?}", path.as_ref());
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    Ok(serde_json::from_reader(reader)?)
}

use serde::Deserialize;
use serde_json;
use std::{collections::HashMap, error, fmt, fs::File, io, path::Path};

#[derive(Debug)]
pub enum FileError {
    Io(io::Error),
    Json(serde_json::Error),
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

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileError::Io(e) => write!(f, "I/O error: {}", e),
            FileError::Json(e) => write!(f, "JSON error: {}", e),
        }
    }
}

impl error::Error for FileError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            FileError::Io(e) => Some(e),
            FileError::Json(e) => Some(e),
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum DatasetFeatureNames {
    Map(HashMap<String, Vec<String>>),
    Vec(Vec<String>),
}

#[derive(Deserialize, Debug)]
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

#[derive(Deserialize, Debug)]
struct DatasetFeature {
    dtype: String,
    shape: Vec<usize>,
    names: Option<DatasetFeatureNames>,
    fps: Option<f32>,
    video_info: Option<VideoInfo>,
}

#[derive(Deserialize, Debug)]
pub struct DatasetInfo {
    codebase_version: String,
    robot_type: String,
    total_episodes: usize,
    total_frames: usize,
    total_tasks: usize,
    chunks_size: usize,
    data_files_size_in_mb: u32,
    video_files_size_in_mb: u32,
    fps: u32,
    splits: HashMap<String, String>,
    data_path: String,
    video_path: String,
    features: HashMap<String, DatasetFeature>,
}

pub fn load_info<P: AsRef<Path>>(path: P) -> Result<DatasetInfo, FileError> {
    println!("Reading from file: {:?}", path.as_ref());
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    Ok(serde_json::from_reader(reader)?)
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum DatasetFeatureStatValue {
    F64(f64),
    Bool(bool),
}

#[derive(Deserialize, Debug)]
struct DatasetFeatureStats {
    min: Vec<DatasetFeatureStatValue>,
    max: Vec<DatasetFeatureStatValue>,
    mean: Vec<f64>,
    std: Vec<f64>,
    count: Vec<usize>,
}

#[derive(Deserialize, Debug)]
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

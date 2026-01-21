use serde::Deserialize;
use serde_json;
use std::{
    collections::HashMap,
    error, fmt,
    fs::File,
    io::{self, BufReader},
    path::Path,
};

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
pub struct DatasetInfo {
    codebase_version: String,
    robot_type: String,
    total_episodes: usize,
    total_frames: usize,
    total_tasks: usize,
    chunks_size: usize,
    fps: u32,
    splits: HashMap<String, String>,
    data_path: String,
    video_path: String,
}

pub fn load_info<P: AsRef<Path>>(path: P) -> Result<DatasetInfo, FileError> {
    println!("Reading from file: {:?}", path.as_ref());
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    Ok(serde_json::from_reader(reader)?)
}

use crate::datasets::dataset_reader::DatasetItemValue;
use polars::error::PolarsResult;
use std::collections::HashMap;

// Query Indices and associated Padding Masks
pub type QueryIndices = HashMap<String, Vec<usize>>;
pub type PaddingMask = HashMap<String, Vec<bool>>;

// Delta Timestamps and Indices
pub type DeltaTimestamps = HashMap<String, Vec<f64>>;
pub type DeltaIndices = HashMap<String, Vec<isize>>;

// Dataset Item
pub type DatasetItem = HashMap<String, DatasetItemValue>;

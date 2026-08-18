use crate::datasets::dataset_reader::DatasetItemValue;
use std::collections::HashMap;

/// Absolute dataset indices to query, grouped by feature key.
///
/// Indices are computed by applying each configured [`DeltaIndices`] offset
/// to the current absolute dataset index. Indices that fall outside the
/// current episode are clamped to the nearest valid episode index.
pub type QueryIndices = HashMap<String, Vec<usize>>;

/// Padding masks associated with queried feature indices.
///
/// Each key has the form `"{feature_key}_is_pad"`. A value is `true` when
/// the corresponding requested index falls outside the current episode and
/// therefore had to be clamped to an episode boundary.
pub type PaddingMask = HashMap<String, Vec<bool>>;

/// Timestamp offsets grouped by feature key.
///
/// Each value represents a temporal offset, in seconds, relative to the
/// current dataset item.
pub type DeltaTimestamps = HashMap<String, Vec<f64>>;

/// Relative item indices grouped by feature key.
///
/// Each value represents an index offset relative to the current dataset item.
pub type DeltaIndices = HashMap<String, Vec<isize>>;

/// A dataset item represented as a mapping from feature names to values.
pub type DatasetItem = HashMap<String, DatasetItemValue>;

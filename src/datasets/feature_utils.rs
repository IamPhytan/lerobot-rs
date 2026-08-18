use std::collections::HashMap;

use crate::types::{DeltaIndices, DeltaTimestamps};

pub fn check_delta_timestamps(
    delta_timestamps: &DeltaTimestamps,
    fps: f32,
    tolerance_s: f64,
) -> Result<(), String> {
    let mut outside_tolerance: DeltaTimestamps = HashMap::new();

    for (key, delta_ts) in delta_timestamps {
        let mut bad_values = Vec::new();

        for &ts in delta_ts {
            let frames = ts * fps as f64;
            let error_s = (frames - frames.round()).abs() / fps as f64;

            if error_s > tolerance_s {
                bad_values.push(ts);
            }
        }

        if !bad_values.is_empty() {
            outside_tolerance.insert(key.clone(), bad_values);
        }
    }

    if !outside_tolerance.is_empty() {
        return Err(format!(
            "The following delta_timestamps are outside the tolerance range. \
             Please make sure they are multiples of 1/{fps} +/- {tolerance_s}: \
             {outside_tolerance:?}"
        ));
    }

    Ok(())
}

/// Convert delta timestamps in seconds to delta indices in frames.
///
/// Each timestamp offset is multiplied by `fps` and rounded to the nearest
/// frame index.
///
/// # Arguments
///
/// * `delta_timestamps` - Timestamp offsets, in seconds, grouped by feature key.
/// * `fps` - Dataset frame rate, in frames per second.
///
/// # Returns
///
/// Returns the corresponding relative frame indices, grouped by feature key.
pub fn get_delta_indices(delta_timestamps: &DeltaTimestamps, fps: f32) -> DeltaIndices {
    delta_timestamps
        .iter()
        .map(|(key, delta_ts)| {
            let indices = delta_ts
                .iter()
                .map(|d| (d * fps as f64).round() as isize)
                .collect();
            (key.clone(), indices)
        })
        .collect()
}

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

#[cfg(test)]
mod tests {
    use super::{check_delta_timestamps, get_delta_indices};
    use crate::types::DeltaTimestamps;
    use std::collections::HashMap;

    // Inspired by https://github.com/huggingface/lerobot/blob/main/tests/datasets/test_delta_timestamps.py

    const FEATURE_KEYS: [&str; 2] = ["action", "state"];

    fn valid_delta_timestamps(fps: u32, start: isize, end: isize) -> DeltaTimestamps {
        FEATURE_KEYS
            .iter()
            .map(|key| {
                let values = (start..end).map(|i| i as f64 / fps as f64).collect();
                ((*key).to_string(), values)
            })
            .collect()
    }

    #[test]
    fn check_delta_timestamps_valid() {
        let fps = 30.0;
        let tolerance_s = 1e-4;
        let delta_timestamps = valid_delta_timestamps(fps as u32, -10, 10);

        assert!(check_delta_timestamps(&delta_timestamps, fps, tolerance_s).is_ok());
    }

    #[test]
    fn check_delta_timestamps_slightly_off() {
        let fps = 30.0;
        let tolerance_s = 1e-4;
        let mut delta_timestamps = valid_delta_timestamps(fps as u32, -10, 10);

        for timestamps in delta_timestamps.values_mut() {
            timestamps[3] += tolerance_s * 0.9;
            let len = timestamps.len();
            timestamps[len - 3] += tolerance_s * 0.9;
        }

        assert!(check_delta_timestamps(&delta_timestamps, fps, tolerance_s).is_ok());
    }

    #[test]
    fn check_delta_timestamps_invalid() {
        let fps = 30.0;
        let tolerance_s = 1e-4;
        let mut delta_timestamps = valid_delta_timestamps(fps as u32, -10, 10);

        for timestamps in delta_timestamps.values_mut() {
            timestamps[3] += tolerance_s * 1.1;
        }

        assert!(check_delta_timestamps(&delta_timestamps, fps, tolerance_s).is_err());
    }

    #[test]
    fn check_delta_timestamps_empty() {
        let delta_timestamps = HashMap::new();

        assert!(check_delta_timestamps(&delta_timestamps, 30.0, 1e-4).is_ok());
    }

    #[test]
    fn delta_indices() {
        let fps = 50;
        let delta_timestamps = valid_delta_timestamps(fps, -100, 100);

        let expected = FEATURE_KEYS
            .iter()
            .map(|key| ((*key).to_string(), (-100..100).collect::<Vec<isize>>()))
            .collect::<HashMap<_, _>>();

        assert_eq!(expected, get_delta_indices(&delta_timestamps, fps as f32));
    }
}

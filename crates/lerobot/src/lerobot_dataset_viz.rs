use lerobot::LeRobotDataset;
pub fn visualize_dataset(dataset: &LeRobotDataset) {
    for (idx, item) in dataset.iter().enumerate() {
        let item = match item {
            Ok(item) => item,
            Err(err) => {
                eprintln!("Failed to load item {idx}: {err}");
                continue;
            }
        };
    }
    ()
}

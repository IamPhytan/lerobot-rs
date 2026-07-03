use lerobot::LeRobotDataset;
use tqdm::Iter;
pub fn visualize_dataset(dataset: &LeRobotDataset) {
    for (idx, item) in dataset.iter().enumerate().tqdm().total(Some(dataset.len())) {
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

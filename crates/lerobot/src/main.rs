use lerobot;

fn main() {
    println!(
        "LeRobot home directory: {:?}",
        lerobot::lerobot_home().as_path()
    );

    let dataset = lerobot::LeRobotDataset::new("lerobot/pusht", None, None, None, None, None);
    println!(
        "Loaded dataset of size {} with repo ID: {} and root {}",
        dataset.len(),
        dataset.repo_id,
        dataset.root().display()
    );

    println!("Dataset metadata: {:?}", dataset.meta.info);
    println!("Dataset statistics: {:?}", dataset.meta.stats);
    println!("Dataset tasks: {}", dataset.meta.tasks);
    // println!("Dataset episodes: {:?}", dataset.meta.episodes);
}

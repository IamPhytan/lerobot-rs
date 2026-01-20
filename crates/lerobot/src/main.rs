use lerobot;

fn main() {
    println!(
        "LeRobot home directory: {:?}",
        lerobot::lerobot_home().as_path()
    );

    let dataset = lerobot::lerobot_dataset::LeRobotDataset::new("user/my_dataset", None);
}

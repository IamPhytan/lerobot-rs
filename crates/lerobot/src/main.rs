use lerobot;

mod lerobot_cli;
use clap::Parser;
use lerobot_cli::LeRobotCli;

fn main() {
    let cli = LeRobotCli::parse();

    match cli.command {
        lerobot_cli::Command::Open { repo } => {
            let dataset =
                lerobot::LeRobotDataset::new(repo.repo_id.as_str(), None, None, None, None, None);

            println!(
                "Loaded dataset of size {} with repo ID: {} and root {}",
                dataset.len(),
                dataset.repo_id,
                dataset.root().display()
            );

            println!("Dataset metadata: {:?}", dataset.meta.info);
            println!("Dataset statistics: {:?}", dataset.meta.stats);
            println!("Dataset tasks: {}", dataset.meta.tasks);
            println!("Dataset subtasks: {:?}", dataset.meta.subtasks);
            println!("Dataset episodes: {:?}", dataset.meta.episodes);

            let task_index = dataset.meta.get_task_index("put the white mug on the left plate and put the yellow and white mug on the right plate");

            println!("{task_index:?}");
        }
        #[cfg(feature = "viz")]
        lerobot_cli::Command::DatasetViz {
            repo,
            episode_index,
            batch_size,
            num_workers,
            mode,
        } => {
            let dataset =
                lerobot::LeRobotDataset::new(repo.repo_id.as_str(), None, None, None, None, None);
            println!("{:?}", dataset);
        }
    }
}

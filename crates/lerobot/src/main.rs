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
            // println!("Dataset episodes: {:?}", dataset.meta.episodes);
        }
    }
}

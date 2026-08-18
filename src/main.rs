//! Command-line interface for the `lerobot` crate.

mod lerobot_cli;

use clap::Parser;
use lerobot_cli::LeRobotCli;

#[cfg(feature = "viz")]
mod lerobot_dataset_viz;

fn main() {
    let cli = LeRobotCli::parse();

    match cli.command {
        lerobot_cli::Command::Open { repo } => {
            let dataset = lerobot::LeRobotDataset::new(
                repo.repo_id.as_str(),
                repo.root,
                None,
                None,
                None,
                None,
            );

            println!(
                "Loaded dataset of size {} with repo ID: {} and root {}",
                dataset.len(),
                dataset.repo_id,
                dataset.root().display()
            );

            println!("\n========================= METADATA =========================\n");

            println!("Dataset metadata: {:?}", dataset.meta.info);
            println!("Dataset statistics: {:?}", dataset.meta.stats);
            println!("Dataset tasks: {}", dataset.meta.tasks);
            println!("Dataset subtasks: {:?}", dataset.meta.subtasks);
            println!("Dataset episodes: {:?}", dataset.meta.episodes);
        }
        #[cfg(feature = "viz")]
        lerobot_cli::Command::DatasetViz {
            repo,
            episode_index,
            mode,
            tolerance_s,
        } => {
            let dataset = lerobot::LeRobotDataset::new(
                repo.repo_id.as_str(),
                repo.root,
                Some(vec![episode_index as usize]),
                None,
                Some(tolerance_s),
                None,
            );

            let _ = lerobot_dataset_viz::visualize_dataset(&dataset, episode_index as usize, mode);
        }
    }
}

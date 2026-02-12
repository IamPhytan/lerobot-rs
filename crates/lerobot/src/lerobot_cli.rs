use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
pub struct RepoArgs {
    /// Dataset Repository identifier
    #[arg(long)]
    pub repo_id: String,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Open a dataset
    Open {
        #[command(flatten)]
        repo: RepoArgs,
    },

    /// Visualize a dataset
    #[cfg(feature = "dataset-viz")]
    DatasetViz {
        #[command(flatten)]
        repo: RepoArgs,

        /// Episode to visualize
        #[arg(long, value_parser = clap::value_parser!(u64).range(0..))]
        episode_index: u64,

        /// Batch size loaded by DataLoader.
        #[arg(long, default_value_t = 32)]
        batch_size: u32,

        ///Number of processes of Dataloader for loading the data.
        #[arg(long, default_value_t = 4)]
        num_workers: u32,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct LeRobotCli {
    #[command(subcommand)]
    pub command: Command,
}

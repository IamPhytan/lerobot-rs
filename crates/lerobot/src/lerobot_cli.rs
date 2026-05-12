use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
pub struct RepoArgs {
    /// Dataset Repository identifier
    #[arg(long)]
    pub repo_id: String,
}

#[cfg(feature = "viz")]
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DatasetVizMode {
    Local,
    Distant,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Open a dataset
    Open {
        #[command(flatten)]
        repo: RepoArgs,
    },

    /// Visualize a dataset
    #[cfg(feature = "viz")]
    DatasetViz {
        #[command(flatten)]
        repo: RepoArgs,

        /// Episode to visualize
        #[arg(long, value_parser = clap::value_parser!(u64).range(0..))]
        episode_index: u64,

        /// Batch size loaded by DataLoader.
        #[arg(long, default_value_t = 32)]
        batch_size: u32,

        /// Number of processes of Dataloader for loading the data.
        #[arg(long, default_value_t = 4)]
        num_workers: u32,

        /// Mode of viewing between 'local' or 'distant'.
        /// 'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally.
        /// 'distant' creates a server on the distant machine where the data is stored.
        /// Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine.
        #[arg(long, default_value = "local")]
        mode: DatasetVizMode,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct LeRobotCli {
    #[command(subcommand)]
    pub command: Command,
}

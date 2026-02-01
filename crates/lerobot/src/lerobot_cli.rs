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
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct LeRobotCli {
    #[command(subcommand)]
    pub command: Command,
}

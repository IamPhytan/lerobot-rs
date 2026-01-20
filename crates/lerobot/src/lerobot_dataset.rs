struct LeRobotDataset {
    repo_id: String,
    root: String,
    meta: LeRobotDatasetMetadata,
    episodes: Vec<Episode>,
}

struct LeRobotDatasetMetadata {
    repo_id: String,
    revision: String,
}

struct Episode {}

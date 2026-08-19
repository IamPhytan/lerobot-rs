# lerobot

[![On crates.io](https://img.shields.io/crates/v/lerobot.svg)](https://crates.io/crates/lerobot)
[![On docs.rs](https://docs.rs/lerobot/badge.svg)](https://docs.rs/lerobot)

> Rust utilities for loading, inspecting, and visualizing [LeRobot](https://github.com/huggingface/lerobot) datasets.

`lerobot` provides Rust utilities for working with locally stored LeRobot datasets, including their metadata, episode data, features, and associated video frames.

By default, datasets are expected under:

```text
~/.cache/huggingface/lerobot
```

This location can be overridden with the `LEROBOT_HOME` environment variable.

## Features

This crate provides:

* Loading of LeRobot dataset metadata:
    * dataset information and statistics
    * tasks and subtasks
    * episode metadata
    * feature definitions
* Reading episode data stored as Parquet files.
* Loading selected episodes.
* Random access to and iteration over dataset items.
* Access to image and video features.
* A command-line interface for quickly inspecting datasets.
* Optional dataset visualization with Rerun, similar to `lerobot-dataset-viz`.

---

## Usage

To use this library, simply add `lerobot` to your `Cargo.toml` file:

```toml
[dependencies]
lerobot = "0.1.0"
```

Datasets are loaded from `$LEROBOT_HOME/<repo_id>` or, when `LEROBOT_HOME` is not set, `~/.cache/huggingface/lerobot/<repo_id>`.

## Examples

### Loading a dataset

```rust
use lerobot::LeRobotDataset;

fn main() {
    let dataset = LeRobotDataset::new(
        "lerobot/pusht",
        None, // root
        None, // episodes
        None, // delta timestamps
        None, // timestamp tolerance
        None, // revision
    );

    println!("Dataset: {}", dataset.repo_id);
    println!("Root: {}", dataset.root().display());
    println!("Frames: {}", dataset.len());
    println!("Episodes: {}", dataset.num_episodes());
    println!("FPS: {}", dataset.fps());
}
```

A subset of episodes can be selected when loading a dataset:

```rust
use lerobot::LeRobotDataset;

fn main() {
    let dataset = LeRobotDataset::new(
        "lerobot/pusht",
        None,
        Some(vec![0, 1, 2]),
        None,
        None,
        None,
    );

    println!("Loaded {} frames", dataset.len());
}
```

### Reading dataset items

Dataset items can be accessed individually or iterated over:

```rust
use lerobot::LeRobotDataset;

fn main() {
    let dataset = LeRobotDataset::new(
        "lerobot/pusht",
        None,
        Some(vec![0]),
        None,
        None,
        None,
    );

    let first = dataset
        .get_item(0)
        .expect("could not read dataset item");

    for (feature, value) in first {
        println!("{feature}: {value:?}");
    }
}
```

### Iterating over a dataset

`LeRobotDataset` implements iteration by reference:

```rust
use lerobot::LeRobotDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = LeRobotDataset::new(
        "lerobot/pusht",
        None,
        None,
        None,
        None,
        None,
    );

    for item in &dataset {
        let item = item?;
        println!("{item:?}");
    }

    Ok(())
}
```

## Command-line Interface

You can install `lerobot-rs` with:

```sh
cargo install lerobot
```

Inspect a locally available dataset with:

```sh
lerobot-rs open --repo-id lerobot/pusht
```

A custom dataset root can also be specified:

```sh
lerobot-rs open \
    --repo-id lerobot/pusht \
    --root /path/to/dataset
```

See all available options with:

```sh
lerobot-rs open --help
```

### Visualization

Dataset visualization is available through the optional `viz` feature:

```sh
cargo install lerobot --features viz
```

Visualize an episode with Rerun:

```sh
lerobot-rs dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

For remote visualization:

```sh
lerobot-rs dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant
```

See all visualization options with:

```sh
lerobot-rs dataset-viz --help
```

## LeRobot

This crate is built around the [`LeRobotDataset v3.0`](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) format developed and used by the [LeRobot](https://github.com/huggingface/lerobot) project.
Its dataset loading structure intentionally follows the upstream LeRobot implementation closely, both for compatibility and as a way to better understand how LeRobot datasets are organized.
For information about creating datasets, dataset conventions, supported robots, and the broader LeRobot ecosystem, refer to [the upstream project](https://github.com/huggingface/lerobot).

## Roadmap

Some features are still planned. Contributions are welcome:

- [ ] Add tensor and array representations
    - [ ] [Candle tensors](https://docs.rs/candle-core/latest/candle_core/struct.Tensor.html) for model training.
    - [ ] [`ndarray`](https://docs.rs/ndarray/latest/ndarray) arrays.
- [ ] Pull datasets directly from the Hugging Face Hub, using [`hf-hub`](https://docs.rs/hf-hub/latest/hf_hub).
- [ ] Improve video decoding and querying performance for large dataset episodes.
- [ ] Add Foxglove visualization support, similar to [LeRobot `v0.6.0`](https://github.com/huggingface/lerobot/releases/tag/v0.6.0)


## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

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

To use the latest published release, add `lerobot` to your `Cargo.toml` file:

```toml
[dependencies]
lerobot = "0.5.1"
```

> **Builder availability:**
> The `LeRobotDataset::builder()` API is currently only available on this repository's `main` branch and is not included in the published `0.5.1` release.
> Until its publication, you can instead rely on a Git dependency:
>
> ```toml
> [dependencies]
> lerobot = { git = "https://github.com/IamPhytan/lerobot-rs", branch = "main" }
> ```

Datasets must already be stored locally.
By default, they are loaded from `$LEROBOT_HOME/<repo_id>` or, when `LEROBOT_HOME` is not set, `~/.cache/huggingface/lerobot/<repo_id>`.

## Examples

### Loading a dataset

Use `LeRobotDataset::builder()` to configure a dataset, and call `build()`
when you are ready to load it:

```rust
use lerobot::LeRobotDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = LeRobotDataset::builder("lerobot/pusht")
        .build()?;

    println!("Dataset: {}", dataset.repo_id);
    println!("Root: {}", dataset.root().display());
    println!("Frames: {}", dataset.len());
    println!("Episodes: {}", dataset.num_episodes());
    println!("FPS: {}", dataset.fps());

    Ok(())
}
```

### Builder configuration

Select episodes or override the local dataset settings by chaining `with_*` methods:

```rust
use lerobot::LeRobotDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = LeRobotDataset::builder("lerobot/pusht")
        .with_root("/path/to/dataset")
        .with_episodes(vec![0, 1, 2])
        .with_tolerance_s(1e-3)
        .with_revision("main")
        .build()?;

    println!("Loaded {} frames", dataset.len());
    Ok(())
}
```

Available options:

| Method | Description | Default |
| --- | --- | --- |
| `with_root(path)` | Use a custom local dataset directory. | `$LEROBOT_HOME/<repo_id>` or the default cache path |
| `with_episodes(indices)` | Read only the selected episode indices. | All episodes |
| `with_delta_timestamps(offsets)` | Configure timestamp offsets for temporally related data. | No offsets |
| `with_tolerance_s(seconds)` | Set the timestamp-matching tolerance in seconds. | `1e-4` |
| `with_revision(revision)` | Set the metadata revision identifier. | `"main"` |

`build()` returns a `Result<LeRobotDataset, FileError>` for metadata-loading errors.

### Reading dataset items

Dataset items can be accessed individually with `get_item()`:

```rust
use lerobot::LeRobotDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = LeRobotDataset::builder("lerobot/pusht")
        .with_episodes(vec![0])
        .build()?;

    let first = dataset.get_item(0)?;

    for (feature, value) in first {
        println!("{feature}: {value:?}");
    }

    Ok(())
}
```

### Iterating over a dataset

`LeRobotDataset` implements iteration by reference:

```rust
use lerobot::LeRobotDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = LeRobotDataset::builder("lerobot/pusht")
        .build()?;

    for item in &dataset {
        let item = item?;
        println!("{item:?}");
    }

    Ok(())
}
```

### Using the legacy constructor

You can also use `LeRobotDataset::new()` to instantiate a LeRobotDataset variable.
Unlike the builder, it requires to specify all optional arguments:

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

    println!("Loaded {} frames", dataset.len());
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

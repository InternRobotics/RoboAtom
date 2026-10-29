# RoboAtom: Large-scale Atomic-Action Supervision for Robot Learning

![RoboAtom overview](https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/figures/roboatom-overview.png)

<p align="center">
  <a href="https://internrobotics.github.io/RoboAtom/">🌐 Project page</a> ·
  <a href="https://github.com/InternRobotics/RoboAtom">💻 Code</a> ·
  <a href="https://arxiv.org/abs/2610.40207">📄 Paper</a> ·
  <a href="https://huggingface.co/datasets/InternRobotics/RoboAtom">🤗 Data &amp; benchmark</a> ·
  <a href="https://huggingface.co/InternRobotics/RoboAtom">🤖 Models</a>
</p>

<p align="center">
  <a href="#installation">🚀 Installation</a> ·
  <a href="#visual-annotation">🎬 Visual annotation</a> ·
  <a href="#physextend">🧩 PhysExtend</a> ·
  <a href="#data">🗂️ Data</a> ·
  <a href="#quick-start-benchmark">🦾 Inference</a> ·
  <a href="#evaluation">📊 Evaluation</a>
</p>

## Overview

RoboAtom brings time-aligned atomic-action supervision to robot learning. Each action combines a shared skill label, an object-specific description, and temporal boundaries. The paper describes RoboAtom-Data with **1.45 million trajectories**, **6.88 million skill-labeled segments**, and **16,882.1 hours**; available dataset subsets are documented in their release cards.

| Component | What you can do | Entry point |
| --- | --- | --- |
| **RoboAtom-Data** | Read atomic-action annotations for 11 datasets, individually or as one JSONL | [Data card](https://huggingface.co/datasets/InternRobotics/RoboAtom#roboatom-data) |
| **RoboAtom-Bench** | Evaluate on 500 trajectories, 4,018 GT segments, and 721 videos | [Evaluation](#evaluation) |
| **Visual annotation** | Inspect synchronized camera views, edit action segments, save annotations, and collaborate with reviewers | [`roboatom annotate`](#visual-annotation) |
| **PhysExtend** | Propagate a known action sequence using same-task references and robot state/action signals | [`roboatom extend`](#physextend) |
| **RoboAtom-Anno** | Predict atomic actions from videos with the 9B or 27B model | [Inference quick start](#quick-start-benchmark) |

## Installation

Run the following from the code repository root. Python **3.10+** is required; model inference needs **Python 3.11+** and CUDA-enabled PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[annotation,physextend]'
```

Choose additional dependencies for your workflow:

| Extra | Use |
| --- | --- |
| `annotation` | Browser annotation UI and offline model review |
| `physextend` | Physical-signal-based annotation propagation |
| `inference` | RoboAtom-Anno service and video preprocessing |

For inference, also run `python -m pip install -e '.[inference]'`. The inference extra pins `transformers==5.8.0`; install a compatible PyTorch/CUDA combination for your system.

## Visual annotation

The package includes the **SubtaskAnnotator-based browser UI**, its frontend, and collaboration tools in [`src/roboatom/annotation/`](https://github.com/InternRobotics/RoboAtom/tree/main/src/roboatom/annotation).

Start with the bundled two-view example:

```bash
roboatom annotate \
  --input examples/annotation.jsonl \
  --output outputs/annotations.jsonl \
  --media-root examples \
  --config configs/annotation.ini \
  --host 127.0.0.1 --port 3001
```

Open `http://127.0.0.1:3001`. Inspect the views and timeline, adjust frame boundaries, skill labels, and action descriptions, then save to `outputs/annotations.jsonl`. The demo uses the same short clip for both views to illustrate the input format; replace them with your own camera recordings.

The UI supports single-user editing, assigned annotator ranges, a reviewer role, and overlays of offline model review suggestions. The [annotation guide](https://github.com/InternRobotics/RoboAtom/blob/main/docs/annotation.md) covers input formats, saving and resuming, model review, and collaboration. Configuration: [`configs/annotation.ini`](https://github.com/InternRobotics/RoboAtom/blob/main/configs/annotation.ini); examples: [`examples/`](https://github.com/InternRobotics/RoboAtom/tree/main/examples).

## PhysExtend

`roboatom extend` transfers boundaries from one or more annotated trajectories of the same task to an unannotated target with the same ordered skill sequence. It extracts model-free physical features, decodes each reference candidate, then applies sequence, gripper, event, DTW, and dataset/embodiment routing steps. Short actions are flagged in diagnostics rather than deleted.

Inputs are JSONL reference and target manifests plus numeric sensor parquet files. `signals_path` may be a local path or a remote URI supported by the configured storage backend. References include ordered `segments`; targets omit them. Both identify `episode_id`, `task_id`, `num_frames`, `fps`, and `signals_path`; `dataset` and `embodiment` are recommended for routing. Outputs are prediction JSONL and a diagnostics JSONL file with each evaluated step. Public intervals use zero-based, closed-inclusive `start_frame` and `end_frame`.

Run on the bundled example:

```bash
roboatom extend \
  --references examples/references.jsonl \
  --input examples/trajectories.jsonl \
  --data-root examples \
  --output outputs/phys_extend_predictions.jsonl \
  --route multi
```

Use `--route single` for one reference trajectory. The CLI requires every reference to cover its full episode. References for a target must agree on their ordered skill sequence and per-step subtask descriptions; conflicting references cause an explicit error instead of silently selecting an execution variant. After reviewing distinct variants, set the same `task_variant` string on each target and its matching references to select the intended group. A target without `task_variant` cannot use a task group containing variant-marked references. PhysExtend requires readable state/action signals and does not infer an unknown skill sequence from video alone. The command uses the bundled final recipe. It writes its feature, schema, and candidate caches to `<output_stem>_physextend_work/` next to the requested output; rerunning the same output rebuilds that workspace.

### Storage configuration

Local manifests work without deployment-specific storage aliases. For remote sources, configure your own rclone/Petrel credentials outside the repository. Copy [`configs/storage.example.json`](https://github.com/InternRobotics/RoboAtom/blob/main/configs/storage.example.json), set dataset aliases, and export `ROBOATOM_STORAGE_CONFIG` to that JSON file before starting the command. Relative local roots resolve from the configuration file directory.

## Data

Download all RoboAtom-Data annotations as one JSONL, or a single dataset from `RoboAtom-Data/by_dataset/`:

```bash
hf download InternRobotics/RoboAtom --repo-type dataset \
  --include 'RoboAtom-Data/roboatom_data.jsonl' --local-dir ./data
```

See the [data card](https://huggingface.co/datasets/InternRobotics/RoboAtom#roboatom-data) for the per-dataset files and record fields.

## Quick start: benchmark

Download the benchmark and either model from their respective Hugging Face repositories. Choose any download locations; these examples use `./data` and `./model`:

```bash
hf download InternRobotics/RoboAtom --repo-type dataset \
  --include 'RoboAtom-Bench/**' --local-dir ./data
hf download InternRobotics/RoboAtom --repo-type model \
  --include 'RoboAtom-Anno-27B/**' --local-dir ./model
```

After installing the inference dependencies, validate the benchmark and start the service. Replace `/path/to/data` and `/path/to/model` with your download locations:

```bash
roboatom validate \
  --benchmark-root /path/to/data/RoboAtom-Bench \
  --media-root /path/to/data/RoboAtom-Bench

roboatom serve \
  --model /path/to/model/RoboAtom-Anno-27B \
  --media-root /path/to/data/RoboAtom-Bench
```

The service listens on `http://127.0.0.1:3000`. In another terminal:

```bash
roboatom infer \
  --input /path/to/data/RoboAtom-Bench/benchmark.jsonl \
  --output outputs/roboatom_predictions.jsonl \
  --base-url http://127.0.0.1:3000
```

To run the smaller model, download `RoboAtom-Anno-9B/**` instead and point `--model` to its directory. Frames are sampled uniformly across the video at a target rate of 2 FPS, including both endpoints and capped at 448 frames. Both models use BF16 and deterministic generation.

## Evaluation

Replace `/path/to/data` with your download location and `/path/to/RoboAtom` with your code checkout:

```bash
roboatom evaluate \
  --gt /path/to/data/RoboAtom-Bench/gt.jsonl \
  --benchmark-input /path/to/data/RoboAtom-Bench/benchmark.jsonl \
  --prediction outputs/roboatom_predictions.jsonl \
  --protocol /path/to/RoboAtom/configs/roboatom_bench.yaml \
  --output-dir outputs/roboatom_evaluation
```

The scorer requires complete prediction coverage and writes `metrics.json`, `summary.tsv`, and `episodes.jsonl`. The benchmark protocol reports temporal Micro F1 at 0.3/0.5/0.7, Skill F1@0.7, mIoU, and Edit. See the [benchmark card](https://huggingface.co/datasets/InternRobotics/RoboAtom) for input fields and frame conventions.

## Release layout

| Repository | Contents |
| --- | --- |
| [GitHub](https://github.com/InternRobotics/RoboAtom) | Code, annotation UI, configs, examples, tests, and the project page (`homepage/`) |
| [HF dataset](https://huggingface.co/datasets/InternRobotics/RoboAtom) | `RoboAtom-Data/` (per-dataset and combined JSONL), `RoboAtom-Bench/` (inputs, GT, 721 videos), `project_media/` (website figures, posters, logos, videos) |
| [HF models](https://huggingface.co/InternRobotics/RoboAtom) | `RoboAtom-Anno-9B/`, `RoboAtom-Anno-27B/` |

## Command reference

| Command | Purpose |
| --- | --- |
| `roboatom annotate` | Launch visual annotation or model-assisted review; prepare/merge collaboration outputs |
| `roboatom review` | Generate offline model review suggestions |
| `roboatom extend` | Run PhysExtend boundary propagation |
| `roboatom validate` | Validate benchmark identities, annotations, and media |
| `roboatom serve` | Serve RoboAtom-Anno |
| `roboatom infer` | Write predictions from the inference service |
| `roboatom evaluate` | Score saved predictions |

Use `roboatom <command> --help` for arguments. CLI paths resolve from the current directory; configuration file references resolve from the configuration directory.

## Tests

```bash
python -m pip install pytest
pytest -q
```

The bundled examples support local annotation and PhysExtend checks. Full model inference needs merged weights and a suitable GPU. The project website source is in [`homepage/`](https://github.com/InternRobotics/RoboAtom/tree/main/homepage); figures and videos are hosted in [project_media](https://huggingface.co/datasets/InternRobotics/RoboAtom/tree/main/project_media). See its [deployment guide](https://github.com/InternRobotics/RoboAtom/blob/main/homepage/docs/deployment.md).

## License and citation

Code is released under the [MIT license](https://github.com/InternRobotics/RoboAtom/blob/main/LICENSE). See [`THIRD_PARTY_NOTICES.md`](https://github.com/InternRobotics/RoboAtom/blob/main/THIRD_PARTY_NOTICES.md) for component attribution, and [`CITATION.cff`](https://github.com/InternRobotics/RoboAtom/blob/main/CITATION.cff) for citation information. RoboAtom-Anno weights are Apache-2.0, following their Qwen base models; benchmark videos retain their source-specific terms.

If you use RoboAtom in your research, please cite:

```bibtex
@misc{wei2026roboatomlargescaleatomicactionsupervision,
      title={RoboAtom: Large-scale Atomic-Action Supervision for Robot Learning},
      author={Xueyuan Wei and Xuanxu Wang and Xudong Li and Yuping He and Wanshui Gan and Jiaqi Peng and Wenqi Guo and Jianjun Zhou and Junting Dong and Chunhua Shen and Haoyu Guo and Mulin Yu and Tao Lu},
      year={2026},
      eprint={2610.40207},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.40207},
}
```

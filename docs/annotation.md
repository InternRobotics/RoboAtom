# Visual annotation guide

[Back to RoboAtom](https://github.com/InternRobotics/RoboAtom/blob/main/README.md#visual-annotation)

The SubtaskAnnotator-based UI combines video playback, editable action segments, human review, and optional offline model review overlays. Run all commands from the code repository root.

## Install and launch

```bash
python -m pip install -e '.[annotation]'
roboatom annotate \
  --input examples/annotation.jsonl \
  --output outputs/annotations.jsonl \
  --media-root examples \
  --config configs/annotation.ini \
  --host 127.0.0.1 --port 3001
```

Open `http://127.0.0.1:3001`. Choose a trajectory, inspect the camera views and timeline, edit the task and action segments, and save. `--media-root` resolves relative video paths; `--output` selects the saved annotation file. Parent output directories are created when saving.

## Input format

The bundled [`examples/annotation.jsonl`](https://github.com/InternRobotics/RoboAtom/blob/main/examples/annotation.jsonl) illustrates the native JSONL format:

```json
{
  "episode_index": "target",
  "frames": 150,
  "fps": 30,
  "tasks": ["Move the block."],
  "views": {"front": "demo.mp4", "wrist": "demo.mp4"},
  "view_order": ["front", "wrist"],
  "action_config": [
    {"seg_id": 0, "start_frame": 0, "end_frame": 74, "skill": "Pick", "action_text": "Pick up the block."},
    {"seg_id": 1, "start_frame": 75, "end_frame": 149, "skill": "Place", "action_text": "Place the block."}
  ]
}
```

Use the original video's frame count and FPS. Frame indices are zero-based with both endpoints included: frames 0–74 contain 75 frames. The example repeats one clip as two views; real multiview inputs should reference synchronized recordings. A single-view record may use `path` instead of `views`.

The editor represents segments internally as half-open intervals: the same 75-frame segment is shown as `[0, 75)`, and the last segment ends at the frame count. Contiguous segments share a boundary. On save, the UI converts boundaries back to the detected input convention.

The UI also accepts the package's public format (`episode_id`, `num_frames`, `segments`) and preserves that format when saving. Skill choices come from [`configs/annotation.ini`](https://github.com/InternRobotics/RoboAtom/blob/main/configs/annotation.ini).

## Edit, save, and resume

- Use the timeline and segment editor to adjust boundaries, skills, and descriptions.
- Save to the file selected by `--output`. To resume a single-user session, use the saved file as `--input`; choose the same file as `--output` to continue updating it.
- The output has companion `.audit.jsonl` and `.status.json` files for recorded changes and progress.
- Keep the original input alongside your saved output. Discard controls record the disposition of a trajectory.

```bash
roboatom annotate \
  --input outputs/annotations.jsonl \
  --output outputs/annotations.jsonl \
  --media-root examples \
  --config configs/annotation.ini \
  --host 127.0.0.1 --port 3001
```

## Model review overlay

To inspect the bundled synthetic review suggestion:

```bash
roboatom annotate \
  --input examples/annotation.jsonl \
  --output outputs/reviewed_annotations.jsonl \
  --media-root examples \
  --config configs/annotation.ini \
  --mode model-review \
  --review examples/review_suggestions.jsonl \
  --host 127.0.0.1 --port 3001
```

The overlay presents verdicts, issues, and suggested fixes for manual review. To generate suggestions for your own data, copy [`configs/review.yaml`](https://github.com/InternRobotics/RoboAtom/blob/main/configs/review.yaml), set its `base_url` and `model`, and supply the service key through `OPENAI_API_KEY` or `NEWAPI_API_KEY`:

```bash
roboatom review \
  --input examples/annotation.jsonl \
  --output outputs/model_review.jsonl \
  --media-root examples \
  --config configs/review.yaml
```

The review command sends sampled video evidence and text to the configured model service. The bundled review fixture can be inspected without making model requests.

## Multiple annotators and a reviewer

[`examples/collaboration.json`](https://github.com/InternRobotics/RoboAtom/blob/main/examples/collaboration.json) assigns the demo trajectory to an annotator and includes a reviewer. Manifest paths resolve relative to the manifest file. For your dataset, set `master_input_json`, `merged_output_json`, and each annotator's `start_index`, `count`, and `output_json`. Ranges refer to the input trajectory list and start at zero.

Prepare the assigned seed files before the first launch:

```bash
roboatom annotate \
  --manifest examples/collaboration.json \
  --media-root examples \
  --config configs/annotation.ini \
  --prepare
```

Then launch the shared UI:

```bash
roboatom annotate \
  --manifest examples/collaboration.json \
  --media-root examples \
  --config configs/annotation.ini \
  --host 127.0.0.1 --port 3001
```

Select an annotator or the reviewer in the browser. Each annotator has a separate output; the reviewer can inspect the combined results. After saving, merge using the same manifest:

```bash
roboatom annotate \
  --manifest examples/collaboration.json \
  --media-root examples \
  --config configs/annotation.ini \
  --merge
```

The merged file is written to `merged_output_json`. Relaunch with the same manifest to resume each annotator's saved output.

## Checks

```bash
pytest -q tests/annotation tests/test_release_interfaces.py
```

These tests cover annotation persistence, public-format round trips, review scores, collaboration, discard handling, audit records, and media streaming behavior.

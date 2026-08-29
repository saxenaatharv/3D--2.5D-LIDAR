# Context: Adaptive Variable Resolution 2.5D Lidar Mapping

DRDO / IDEX submission (Problem Statement 26053). Builds a foveated
2.5D occupancy/semantic map from lidar point clouds — fine grid cells
near the sensor, coarser cells farther out, like human vision. Uses a
pretrained RandLA-Net (Open3D-ML) backbone to segment points, then
projects them into that variable-resolution grid.

## What it actually does

1. A pretrained SemanticKITTI model (RandLA-Net, optionally KPFCNN)
   classifies each lidar point into one of 19 SemanticKITTI classes.
2. Those 19 classes get collapsed into 3 super-classes: terrain,
   static obstacle, dynamic object (`class_mapping.py`).
3. Points get inserted into `VariableResolutionGrid` (`grid_engine.py`)
   — a sparse dict keyed by `(ring_id, ix, iy)`, where each concentric
   ring around the sensor has its own cell size (e.g. 5cm near, 50cm
   at 100m). Empty space costs nothing since it's a dict, not a dense
   array.
4. Metrics get computed: FPS, active cell count, memory saved vs. an
   equivalent uniform high-res grid, and mIoU if ground truth is
   available.

## Two ways to run it

- **Web app** (`app.py` + `variant.html`): FastAPI backend, upload
  frames, pick pretrained/fine-tuned checkpoint, watch it run, get
  CSV/JSON out. This is the recommended entry point per the README.
- **CLI** (`run_inference.py`, `finetune.py`): OpenCV desktop
  dashboard, good for training and batch/offline runs over SSH.

Both paths run the same core logic — grid engine, class mapping,
metrics — just different entry points.

## File map

| File | Role |
|---|---|
| `grid_engine.py` | The core data structure — sparse ring-based variable-resolution grid |
| `class_mapping.py` | 19 SemanticKITTI classes → 3 super-classes (terrain/static/dynamic) |
| `backends.py` | Swappable model backend: real Open3D-ML model, or a heuristic fallback if no checkpoint/deps are available |
| `run_inference.py` | CLI inference script — loads a checkpoint, runs it over a folder of frames, renders + logs |
| `finetune.py` | Fine-tunes the pretrained backbone on new labelled data; never overwrites the pretrained config, always writes a fresh output config |
| `dataset_utils.py` | Converts raw point clouds + labels into SemanticKITTI folder layout, with a 90/10 train/test split |
| `visualize_dashboard.py` | Live OpenCV rendering for the CLI path |
| `metrics.py` | IoU tracking + memory-saving calculation |
| `app.py` | FastAPI backend for the web UI |
| `variant.html` | Web frontend |
| `generate_test_frame.py` | Makes a synthetic frame for smoke-testing without real data |
| `config_pretrained.yaml` / `config_finetuned.yaml` | Point at their respective checkpoints, plus grid/training/viz settings |

## Notable design choices worth knowing

- **Two checkpoints ship in the repo**: `randlanet_semantickitti_...pth`
  (official Open3D-ML pretrained weights) and `finetuned_ckpt_00040.pth`
  (already fine-tuned). Both configs point at one each, so pretrained
  vs. fine-tuned runs never clobber each other's outputs.
- **`finetune.py` is careful about not mutating the input config** — it
  reads the starting checkpoint from `--config` but always writes
  results to a separate `--output_config`, so `config_pretrained.yaml`
  stays a stable reference no matter how many times you fine-tune.
- **The pretrained checkpoint uses 3 input channels (xyz only)**, not 4
  (xyz + intensity). Since Open3D-ML's SemanticKITTI loader always
  returns intensity as a feature, `finetune.py` monkey-patches
  `get_data()` on the dataset split class to strip it back off, so
  fine-tuning stays architecturally compatible with the pretrained
  weights.
- **`backends.py`'s heuristic fallback is explicit about not being a
  real model** — it's a height threshold + a deterministic hash split,
  used only when Open3D-ML/torch aren't installed or the checkpoint
  can't load. It never gets silently mistaken for the real thing:
  every prediction reports which backend produced it, and a failed
  Open3D-ML load is never cached (so fixing the checkpoint path later
  actually takes effect on the next call).
- **KPFCNN is present in code but marked under construction** on the
  web frontend — reachable only via a direct API call.
- Grid insertion is per-frame by default (`grid.reset()` before each
  frame in `run_inference.py`) — you'd remove that line if you wanted
  a persistent accumulated map instead of a per-frame snapshot.

## Setup gotchas

- Needs PyTorch 2.2.2 + Open3D-ML 0.17+, version-matched — the README
  has exact pip commands for both GPU (CUDA 11.8) and CPU-only.
- Web app extras (`fastapi`, `uvicorn`, `python-multipart`, `pydantic`)
  aren't in `requirements.txt` and need a separate install.
- `checkpoints/*.pth` will silently be a tiny Git LFS pointer file
  instead of real weights if LFS isn't pulled — check file size before
  assuming a checkpoint load failure is something else.

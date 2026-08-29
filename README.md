# Adaptive Variable Resolution 2.5D Lidar Mapping

A foveated 2.5D occupancy/semantic map for lidar point clouds — like human vision, it keeps fine resolution close to the sensor and lets it get coarser with distance, so you're not wasting memory on empty space far away. Built for DRDO/IDEX (Problem Statement 26053).

It uses a pretrained **RandLA-Net** (via Open3D-ML) to segment every point into one of 19 SemanticKITTI classes, collapses those into 3 super-classes (terrain / static obstacle / dynamic object), and drops them into a sparse ring-based grid where each ring around the sensor has its own cell size.

## How it works

1. A pretrained model classifies each point (19 SemanticKITTI classes).
2. Those get mapped down to 3 classes: **terrain**, **static obstacle**, **dynamic object**.
3. Points land in a `VariableResolutionGrid` — a plain dict keyed by `(ring_id, x_index, y_index)`, so empty cells cost nothing. Rings closer to the sensor use smaller cells (e.g. 5cm), rings farther out use bigger ones (up to 50cm at 100m).
4. It logs FPS, how many cells are actually in use, how much memory that saved vs. a uniform high-res grid, and mIoU if you have ground truth.

There are two ways to run this — a web app and a CLI. Both share the same core logic (grid engine, class mapping, metrics), just different entry points.

## Files

| File | What it does |
|---|---|
| `grid_engine.py` | The sparse, ring-based variable-resolution grid — the core data structure. |
| `class_mapping.py` | Maps the 19 SemanticKITTI classes down to terrain/static/dynamic, plus color codes for rendering. |
| `backends.py` | Swaps between the real Open3D-ML model and a heuristic fallback (height threshold + hash split) when no checkpoint/deps are available — always reports which one actually ran. |
| `run_inference.py` | CLI script: loads a checkpoint, runs it over a folder of lidar frames, renders and logs results. Doesn't train anything. |
| `finetune.py` | Fine-tunes the pretrained backbone on your own labelled data. Starts from the pretrained weights, never trains from scratch. Never touches the config you pass it — always writes a new output config. |
| `dataset_utils.py` | Converts raw point clouds + labels into the folder layout Open3D-ML expects, with a reproducible 90/10 train/test split. |
| `visualize_dashboard.py` | Live OpenCV rendering used by the CLI path. |
| `metrics.py` | IoU tracking and the memory-saving calculation. |
| `app.py` | FastAPI backend for the web UI. |
| `variant.html` | The web frontend — a static page, just open it in a browser. |
| `generate_test_frame.py` | Generates a synthetic point cloud so you can smoke-test the pipeline without real data. |
| `config_pretrained.yaml` | Points at the pretrained checkpoint. |
| `config_finetuned.yaml` | Points at the fine-tuned checkpoint. |
| `randlanet_semantickitti_202201071330utc.pth` | The pretrained RandLA-Net weights (official Open3D-ML model zoo). |
| `finetuned_ckpt_00040.pth` | An already fine-tuned checkpoint, ready to use. |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# PyTorch, version-matched to Open3D-ML 0.17+
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118   # GPU (CUDA 11.8)
# pip install torch==2.2.2 torchvision==0.17.2                                                     # CPU only

# only needed if you're running the web app:
pip install fastapi "uvicorn[standard]" python-multipart pydantic
```

**One thing to fix before you run anything:** both config files expect checkpoints under a `checkpoints/` folder (`checkpoint_path: checkpoints/...pth`), but the two `.pth` files ship at the repo root. Either move them or update the path:

```bash
mkdir -p checkpoints
mv randlanet_semantickitti_202201071330utc.pth finetuned_ckpt_00040.pth checkpoints/
```

If you're on CPU only, also change `device: cuda` to `device: cpu` in both config files.

## Running it

### Option A — Web app (recommended)

Start the API:

```bash
uvicorn app:app --reload --port 8000
```

Then just open `variant.html` in a browser (double-click it, or `python -m http.server` and browse to it — no build step needed). From there:

1. Upload a lidar frame (`.bin`, KITTI-style float32 N×4, or `.npy`, N×3/N×4).
2. Pick pretrained or fine-tuned, and CPU or GPU.
3. Click **Check backend** — it should say `open3d-ml` (if it says `heuristic-fallback`, your checkpoint path is wrong — double check step above).
4. Run it, watch the colored grid render, download the CSV/JSON.

If the API isn't on `localhost:8000`, there's an "API base" field in the page to point it elsewhere.

### Option B — CLI

Generate a test frame first if you don't have real data yet:

```bash
python generate_test_frame.py
```

Then run inference:

```bash
python run_inference.py --config config_pretrained.yaml
# or
python run_inference.py --config config_finetuned.yaml
```

This opens a live OpenCV window, writes per-frame images to `outputs/<pretrained|finetuned>/outputimage/`, and a `run_log.csv` with FPS/cell-count/memory-saving/mIoU per frame. Drop more `.bin`/`.npy` frames into `data/inference_frames/` (path set in the config) to process a batch. Press `q` to quit early.

## Fine-tuning on your own data

```bash
python finetune.py --config config_pretrained.yaml \
  --raw_pc_dir path/to/your/point_clouds \
  --raw_label_dir path/to/your/labels \
  --output_config config_finetuned.yaml
```

This converts your data into SemanticKITTI layout (90/10 train/test split by default, `--train_ratio` to change it), fine-tunes starting from the pretrained checkpoint, and writes a **new** config file pointing at the result — `config_pretrained.yaml` is never modified, so it always stays a stable reference.

If your data's already in SemanticKITTI layout (`sequences/00` = train, `sequences/01` = test), skip `--raw_pc_dir`/`--raw_label_dir` and just point `paths.finetune_data_dir` in the config at it.

## Good to know

- The pretrained checkpoint only takes xyz (3 channels), not xyz+intensity — `finetune.py` strips the intensity channel automatically so fine-tuning stays compatible.
- `run_inference.py` resets the grid every frame by default (a per-frame snapshot, not an accumulated map). Remove the `grid.reset()` call in the loop if you want points to persist across frames.
- KPFCNN is in the code but not wired up on the web frontend yet — it's reachable only through a direct API call.
- The `HeuristicBackend` in `backends.py` is not a real model — it's just a fallback used when Open3D-ML/torch aren't available or the checkpoint fails to load, and every prediction says which backend actually produced it.

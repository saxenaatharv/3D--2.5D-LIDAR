# Adaptive Variable Resolution 2.5D Lidar Mapping
DRDO / IDEX — Problem Statement 26053

A foveated, human-vision-inspired 2.5D occupancy/semantic map built on top
of a pretrained RandLA-Net (Open3D-ML) backbone. Space around the ego
sensor is split into concentric rings, each with its own cell size —
fine resolution near the sensor, coarser far away — so memory scales with
what actually matters instead of a fixed uniform grid.

Two ways to use it:
- **CLI pipeline** (`run_inference.py`, `finetune.py`) — desktop OpenCV dashboard.
- **Web app** (`app.py` + `variant.html`) — browser UI, upload frames, run,
  view results, download CSV/JSON.

---

## 1. Project structure

```
SIH/
├── config_pretrained.yaml     <- points at the ORIGINAL Open3D-ML checkpoint
├── config_finetuned.yaml      <- points at YOUR fine-tuned checkpoint (auto-written by finetune.py)
├── class_mapping.py           <- SemanticKITTI 19 classes -> terrain/static/dynamic
├── grid_engine.py             <- Variable Resolution 2.5D Grid Engine (foveated sparse grid)
├── visualize_dashboard.py     <- live OpenCV dashboard renderer (CLI path)
├── metrics.py                 <- IoU + memory-saving report
├── run_inference.py           <- SCRIPT 1: pretrained/fine-tuned inference, no training
├── dataset_utils.py           <- converts raw point clouds + labels into SemanticKITTI layout,
│                                  with a 90/10 train/test SPLIT (see Section 5)
├── finetune.py                <- SCRIPT 2: fine-tunes the pretrained checkpoint
├── generate_test_frame.py     <- synthetic smoke-test frame generator
├── requirements.txt
├── app.py                     <- FastAPI backend for the web UI
├── backends.py                <- pluggable model backend (real Open3D-ML model, or
│                                  a transparent heuristic fallback if no checkpoint loads)
├── variant.html                <- the web frontend (talks to app.py)
├── lidar25d/                   <- Python virtual environment (created below)
├── checkpoints/                 <- .pth files go here (pretrained + fine-tuned)
├── data/
│   ├── inference_frames/        <- lidar frames to run inference on
│   ├── finetune_raw/             <- your raw point clouds + labels, before conversion
│   └── finetune_dataset/         <- auto-populated by dataset_utils.py (sequences/00 = train, 01 = test)
└── outputs/
    ├── pretrained/               <- run_log.csv + dashboard PNGs from config_pretrained.yaml runs
    └── finetuned/                <- same, from config_finetuned.yaml runs
```

---

## 2. Setup (one-time)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip curl unzip

cd ~/Desktop/SIH
python3 -m venv lidar25d
source lidar25d/bin/activate
pip install --upgrade pip

# PyTorch, version-matched to Open3D-ML (2.2.2)
# GPU (NVIDIA + CUDA 11.8):
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
# CPU only, use this line INSTEAD:
# pip install torch==2.2.2 torchvision==0.17.2

pip install numpy pyyaml opencv-python matplotlib "open3d>=0.17.0" tensorboard "numpy<2"
pip install fastapi "uvicorn[standard]" python-multipart pydantic   # only needed for the web app

mkdir -p checkpoints data/inference_frames data/finetune_raw outputs/pretrained outputs/finetuned
```

If you're on CPU only, set `model.device: "cpu"` in both `config_pretrained.yaml` and
`config_finetuned.yaml`.

---

## 3. Get the pretrained checkpoint

```bash
curl -L -o checkpoints/randlanet_semantickitti_202201071330utc.pth \
  https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth
```

This is a real Open3D-ML model-zoo checkpoint trained end-to-end on
SemanticKITTI — `run_inference.py` / `finetune.py` load it via
`pipeline.load_ckpt(...)`, never starting from random weights.

`config_pretrained.yaml` already points `model.checkpoint_path` at this file.

---

## 4. Run INFERENCE (pretrained or fine-tuned) — CLI path

Drop `.bin` (KITTI-style float32 N×4) or `.npy` (N×3/N×4) frames into
`data/inference_frames/`, or generate a synthetic smoke-test frame:

```bash
python generate_test_frame.py
```

Then run either model — each writes to its own output folder so results
never overwrite each other:

```bash
# pretrained weights
python run_inference.py --config config_pretrained.yaml

# your fine-tuned weights
python run_inference.py --config config_finetuned.yaml
```

Each run:
- loads the model checkpoint from that config's `model.checkpoint_path`
- segments every frame, remaps to terrain / static-obstacle / dynamic-object
- builds the variable-resolution grid and shows a live OpenCV dashboard
- writes `outputs/<pretrained|finetuned>/run_log.csv` (fps, active cells, memory saving, mIoU)
- saves a PNG of each frame's dashboard to `outputs/<pretrained|finetuned>/outputimage/`

Press **`q`** in the dashboard window to stop early.

```bash
cat outputs/pretrained/run_log.csv
cat outputs/finetuned/run_log.csv
```

---

## 5. FINE-TUNE on your own data (90/10 train/test split)

`finetune.py` always starts from `config_pretrained.yaml`'s checkpoint and
**never overwrites that file**. It writes a brand-new `config_finetuned.yaml`
pointing at the new weights, so `config_pretrained.yaml` stays a permanent,
untouched reference no matter how many times you re-train.

Your raw data (point clouds + a matching `.npy` label array per frame,
values 0=terrain/1=static/2=dynamic or your own SemanticKITTI ids) gets
split **90% train / 10% held-out test**, shuffled with a fixed seed —
into two separate SemanticKITTI-style sequences (`sequences/00` = train,
`sequences/01` = test), so training and evaluation never see the same
frames:

```bash
python finetune.py \
  --config config_pretrained.yaml \
  --output_config config_finetuned.yaml \
  --raw_pc_dir data/finetune_raw/pointclouds \
  --raw_label_dir data/finetune_raw/labels \
  --train_ratio 0.9
```

Watch for this line early in the console output to confirm the split
actually happened:

```
[dataset_utils] Split N frames -> X train (sequence 00) / Y test (sequence 01), train_ratio=0.9, seed=42
```

Verify on disk:

```bash
ls data/finetune_dataset/sequences/00/velodyne/ | wc -l   # ~90%
ls data/finetune_dataset/sequences/01/velodyne/ | wc -l   # ~10%
```

If your data is already in SemanticKITTI layout with `sequences/00` and
`sequences/01` present, skip `--raw_pc_dir`/`--raw_label_dir` entirely and
just run:

```bash
python finetune.py --config config_pretrained.yaml --output_config config_finetuned.yaml
```

Epochs / batch size / learning rate / frozen layers / split ratio are all
editable under `training:` in `config_pretrained.yaml`.

When training finishes, `config_finetuned.yaml` is written/overwritten
automatically with the new checkpoint path — re-run Section 4's second
command to try it out.

---

## 6. Run the WEB APP

The web app (`app.py` FastAPI backend + `variant.html` frontend) wraps the
same pipeline behind a browser UI: upload frames, pick a checkpoint, run,
watch progress, view the color-coded map, download `run_log.csv` /
`results.json`.

```bash
cd ~/Desktop/SIH
source lidar25d/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Leave that running, then open `variant.html` directly in a browser
(double-click it, or `xdg-open variant.html`).

**Confirm the server is up** (optional, second terminal):
```bash
curl http://localhost:8000/api/health
```

### Using a real trained model instead of the fallback

The web app has a **"Pretrained checkpoint path"** text box. If you leave
it empty, it silently uses `HeuristicBackend` — a transparent placeholder
(height threshold, not a trained model) meant only to prove the wiring
works when no model is loaded. It is **not** your fine-tuned model.

To use your real weights:
1. Type the checkpoint path into that box (relative to the folder you ran
   `uvicorn` from), e.g.:
   ```
   checkpoints/finetuned_ckpt_00040.pth
   ```
   or
   ```
   checkpoints/randlanet_semantickitti_202201071330utc.pth
   ```
2. Click **"Check backend"** and confirm the badge says `backend: open3d-ml`
   (not `heuristic-fallback`).
3. Then run as normal — dynamic/static/terrain colors will now come from
   the real model.

If it still falls back after entering a correct path, check that
`open3d.ml.torch` actually imports in the same venv running `uvicorn`:
```bash
python -c "import open3d.ml.torch as ml3d; print('ok')"
```

### API reference (for anyone extending the frontend)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/frames` | upload a `.bin`/`.npy` frame |
| GET | `/api/frames` | list uploaded frames |
| DELETE | `/api/frames/{frame_id}` | drop a frame |
| GET | `/api/backend/status` | which classifier is active (real model vs heuristic) |
| POST | `/api/runs` | start a run over a set of frame ids + config |
| GET | `/api/runs/{run_id}/status` | poll progress |
| GET | `/api/runs/{run_id}/results` | per-frame metrics + grid cells |
| GET | `/api/runs/{run_id}/log.csv` | run as CSV, same columns as `run_log.csv` |

Frame/run data lives in memory only — fine for a demo, not for
multi-user production (swap `FRAMES`/`RUNS` in `app.py` for Redis/a DB
before deploying beyond a single session).

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on run | venv not activated, or a package not installed | `source lidar25d/bin/activate`, then `pip install -r requirements.txt` |
| `FileNotFoundError: Checkpoint not found` | `model.checkpoint_path` in the config doesn't exist on disk | `ls checkpoints/`, fix the path in the config, or re-download (Section 3) |
| Web app shows mostly terrain / wrong colors | checkpoint path box left empty → using the heuristic fallback | see Section 6, "Using a real trained model" |
| `config_pretrained.yaml` no longer points at the pretrained checkpoint | you're on an old copy of `finetune.py` that overwrote the input config in place | use the version in this repo — it always writes a separate `--output_config` and never touches the input config |
| Fine-tuning trained on 100% of frames, no held-out test set | `data/finetune_dataset/sequences/` had no `01/` folder — `--raw_pc_dir`/`--raw_label_dir` weren't passed, or an old `dataset_utils.py` was used | `rm -rf data/finetune_dataset cache logs` then re-run Section 5's command with `--raw_pc_dir`/`--raw_label_dir` set |
| No lidar frames found | `data/inference_frames/` is empty | drop `.bin`/`.npy` frames in, or `python generate_test_frame.py` for a smoke test |

---

## 8. Credits

Pretrained checkpoints from the official Open3D-ML model zoo
(`isl-org/Open3D-ML`), trained on SemanticKITTI.

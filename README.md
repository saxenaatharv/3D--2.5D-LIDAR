# Adaptive Variable Resolution 2.5D Lidar Mapping
DRDO / IDEX — Problem Statement 26053

A foveated, human-vision-inspired 2.5D occupancy/semantic map built on top
of a pretrained RandLA-Net (Open3D-ML) backbone. Space around the ego
sensor is split into concentric rings, each with its own cell size — fine
resolution near the sensor, coarser far away — so memory scales with what
actually matters instead of a fixed uniform grid.

There are **two ways to run this project**:

| | Recommended | What it gives you |
|---|---|---|
| **A. Web App** (`app.py` + `variant.html`) | ✅ **Yes — start here** | Browser UI: upload frames, pick a checkpoint (pretrained or fine-tuned), run, watch live progress, view the color-coded foveated grid, download CSV/JSON. No OpenCV window, no desktop needed — works over a network too. |
| **B. CLI pipeline** (`run_inference.py`, `finetune.py`) | Use for training / offline batch runs | Desktop OpenCV dashboard, scriptable, best for fine-tuning and for processing a whole folder of frames unattended. |

This README covers **both**, start to finish: environment setup, getting the
checkpoints (pretrained + fine-tuned — including the ones already shipped
in this repo), getting sample lidar data (2 ready-to-use sample frames, plus
full commands to pull real data from SemanticKITTI and nuScenes), and
running each path.

---

## Table of contents

1. [Project structure](#1-project-structure)
2. [Setup (one-time, needed for both paths)](#2-setup-one-time-needed-for-both-paths)
3. [Checkpoints — pretrained and fine-tuned](#3-checkpoints--pretrained-and-fine-tuned)
4. [Sample data](#4-sample-data)
   - [4.1 Two ready-made sample frames](#41-two-ready-made-sample-frames)
   - [4.2 Full datasets — SemanticKITTI](#42-full-datasets--semantickitti)
   - [4.3 Full datasets — nuScenes](#43-full-datasets--nuscenes)
5. [PATH A — Run the WEB APP (recommended)](#5-path-a--run-the-web-app-recommended)
6. [PATH B — Run the CLI pipeline (local, no web)](#6-path-b--run-the-cli-pipeline-local-no-web)
7. [Fine-tuning on your own data](#7-fine-tuning-on-your-own-data-9010-traintest-split)
8. [Troubleshooting](#8-troubleshooting)
9. [Credits](#9-credits)

---

## 1. Project structure

```
SIH/
├── config_pretrained.yaml     <- points at the ORIGINAL Open3D-ML checkpoint
├── config_finetuned.yaml      <- points at the fine-tuned checkpoint
├── class_mapping.py           <- SemanticKITTI 19 classes -> terrain/static/dynamic
├── grid_engine.py             <- Variable Resolution 2.5D Grid Engine (foveated sparse grid)
├── visualize_dashboard.py     <- live OpenCV dashboard renderer (CLI path)
├── metrics.py                 <- IoU + memory-saving report
├── run_inference.py           <- SCRIPT 1: pretrained/fine-tuned inference, no training (CLI path)
├── dataset_utils.py           <- converts raw point clouds + labels into SemanticKITTI layout,
│                                  with a 90/10 train/test SPLIT (see Section 7)
├── finetune.py                <- SCRIPT 2: fine-tunes the pretrained checkpoint (CLI path)
├── generate_test_frame.py     <- synthetic smoke-test frame generator
├── requirements.txt
├── app.py                     <- FastAPI backend for the web UI (PATH A)
├── backends.py                <- pluggable model backend (real Open3D-ML model, or a
│                                  transparent heuristic fallback if no checkpoint loads)
├── variant.html                <- the web frontend — open this in a browser (PATH A)
├── lidar25d/                   <- Python virtual environment (created below)
├── checkpoints/                 <- .pth files live here — PRETRAINED + FINE-TUNED, side by side
│   ├── randlanet_semantickitti_202201071330utc.pth   <- pretrained (Section 3.1)
│   └── finetuned_ckpt_00040.pth                       <- fine-tuned (Section 3.2)
├── data/
│   ├── inference_frames/        <- lidar frames to run inference on (Section 4)
│   ├── finetune_raw/             <- your raw point clouds + labels, before conversion
│   └── finetune_dataset/         <- auto-populated by dataset_utils.py (sequences/00 = train, 01 = test)
└── outputs/
    ├── pretrained/               <- run_log.csv + dashboard PNGs from config_pretrained.yaml runs
    └── finetuned/                <- same, from config_finetuned.yaml runs
```

---

## 2. Setup (one-time, needed for both paths)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip curl unzip git

cd ~/Desktop/SIH        # or wherever you cloned this repo
python3 -m venv lidar25d
source lidar25d/bin/activate
pip install --upgrade pip

# PyTorch, version-matched to Open3D-ML (2.2.2)
# GPU (NVIDIA + CUDA 11.8):
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
# CPU only, use this line INSTEAD:
# pip install torch==2.2.2 torchvision==0.17.2

pip install numpy pyyaml opencv-python matplotlib "open3d>=0.17.0" tensorboard "numpy<2"
pip install fastapi "uvicorn[standard]" python-multipart pydantic   # needed for the WEB APP (Path A)

mkdir -p checkpoints data/inference_frames data/finetune_raw outputs/pretrained outputs/finetuned
```

Or simply:

```bash
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart pydantic   # web app extras aren't in requirements.txt
```

**CPU-only machines:** set `model.device: "cpu"` in both `config_pretrained.yaml`
and `config_finetuned.yaml`, and pick **CPU** in the web UI's device toggle
(Section 5) or pass `device: cpu` when calling the API.

**Verify the install** before moving on:
```bash
python -c "import torch, open3d, cv2, yaml; print('core deps ok')"
python -c "import open3d.ml.torch as ml3d; print('open3d-ml ok')"
python -c "import fastapi, uvicorn; print('web app deps ok')"
```

---

## 3. Checkpoints — pretrained and fine-tuned

Both configs simply point at a `.pth` file under `checkpoints/`:

```bash
grep checkpoint_path config_pretrained.yaml config_finetuned.yaml
# config_pretrained.yaml:  checkpoint_path: checkpoints/randlanet_semantickitti_202201071330utc.pth
# config_finetuned.yaml:   checkpoint_path: checkpoints/finetuned_ckpt_00040.pth
```

### 3.1 Pretrained checkpoint

If this repo already ships `checkpoints/randlanet_semantickitti_202201071330utc.pth`
(e.g. committed to git or pulled via Git LFS), you're done — skip straight
to Section 4. Confirm it's there and non-empty:

```bash
ls -lh checkpoints/randlanet_semantickitti_202201071330utc.pth
```

If it's missing, download the official Open3D-ML model-zoo checkpoint
(a real end-to-end SemanticKITTI-trained checkpoint — `run_inference.py`
and `finetune.py` load it via `pipeline.load_ckpt(...)`, never starting
from random weights):

```bash
curl -L -o checkpoints/randlanet_semantickitti_202201071330utc.pth \
  https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth
```

### 3.2 Fine-tuned checkpoint

If a fine-tuned checkpoint has already been trained and committed to this
repo (for example `checkpoints/finetuned_ckpt_00040.pth`, or pulled via
Git LFS / `git pull`), you can use it immediately — no training needed.
Confirm it's there:

```bash
ls -lh checkpoints/finetuned_ckpt_00040.pth
cat config_finetuned.yaml | grep checkpoint_path
```

If `config_finetuned.yaml`'s `checkpoint_path` points at a *different*
filename than what's actually in `checkpoints/` (e.g. someone fine-tuned
again later and the repo has `finetuned_ckpt_00080.pth` instead), either:
- edit `config_finetuned.yaml`'s `checkpoint_path` to match the file that's
  actually present, **or**
- use that exact filename directly in the web app's checkpoint box
  (Section 5.2) / the CLI's `--config` (Section 6).

If **no** fine-tuned checkpoint exists yet in the repo, you can train one
yourself — see [Section 7](#7-fine-tuning-on-your-own-data-9010-traintest-split).
Until then, just use the pretrained checkpoint from 3.1.

### 3.3 If checkpoints are stored with Git LFS

Large `.pth` files are often tracked with Git LFS instead of committed
directly. If `checkpoints/*.pth` looks tiny (a few hundred bytes, a text
pointer file) instead of tens/hundreds of MB, pull the real binary:

```bash
git lfs install
git lfs pull
ls -lh checkpoints/
```

---

## 4. Sample data

You need at least one lidar frame (`.bin`, KITTI-style float32 N×4
`x,y,z,intensity`, or `.npy`, N×3/N×4) to run either path.

### 4.1 Two ready-made sample frames

If this repo ships sample frames (e.g. under `data/samples/` or attached to
a release), they're the fastest way to try the pipeline with no external
downloads:

```bash
ls data/samples/
# e.g. sample_frame_01.bin  sample_frame_02.bin
```

- **Web app:** drag-and-drop (or use the file picker on) both files in
  **Step 01 — Upload** — see Section 5.3. No conversion needed.
- **CLI:** copy them into the folder `run_inference.py` reads from:
  ```bash
  cp data/samples/sample_frame_01.bin data/inference_frames/
  cp data/samples/sample_frame_02.bin data/inference_frames/
  ```

If you don't have real sample frames yet, generate a synthetic smoke-test
frame instead (proves the wiring works end-to-end, not meant to give
meaningful semantics):

```bash
python generate_test_frame.py
# writes data/inference_frames/000000.bin
```

You can also add labels: to enable live IoU when you do have ground truth,
attach a matching `.label` (or `.npy`) folder — see `paths.gt_label_dir` in
the YAML configs, or the "Attach .label folder" control in the web UI.

### 4.2 Full datasets — SemanticKITTI

This is the dataset the pretrained checkpoint was trained on, and the
format `dataset_utils.py` converts everything into. Useful if you want more
real frames than the 2 samples, or want to properly evaluate mIoU.

```bash
mkdir -p data/semantickitti && cd data/semantickitti

# 1) Velodyne point clouds (~80 GB, all 22 sequences) — or grab just sequence 00
curl -L -o data_odometry_velodyne.zip \
  http://www.semantic-kitti.org/assets/data_odometry_velodyne.zip

# 2) SemanticKITTI label files (~179 MB)
curl -L -o data_odometry_labels.zip \
  http://www.semantic-kitti.org/assets/data_odometry_labels.zip

unzip -q data_odometry_velodyne.zip
unzip -q data_odometry_labels.zip
cd ../..
```

> SemanticKITTI's asset links occasionally move — if `curl` 404s, get the
> current mirror links from https://www.semantic-kitti.org/dataset.html
> (Downloads section) and substitute them above; some mirrors require
> filling out a short registration form first.

This unpacks into the standard `sequences/00 … 21/velodyne/*.bin` +
`sequences/00 … 10/labels/*.label` layout, which is exactly what
`config_pretrained.yaml` / `config_finetuned.yaml`'s
`paths.finetune_data_dir` and `dataset_utils.py`'s output layout expect —
point `--raw_pc_dir` / `--raw_label_dir` (Section 7) at a sequence's
`velodyne/` and `labels/` folders, or use the sequence folders directly for
evaluation. Individual `.bin` frames from `velodyne/` can also be copied
straight into `data/inference_frames/` for a normal inference run (Section 5/6)
since they're already in the exact float32 N×4 format this project expects.

### 4.3 Full datasets — nuScenes

nuScenes is a second real-world autonomous-driving lidar dataset, useful for
testing generalization beyond SemanticKITTI. Its raw `.pcd.bin` sweeps are
lidar-only point clouds you can also feed into this pipeline (they carry
`x,y,z,intensity` plus a ring index — trim to the first 4 columns to match
this project's N×4 format).

```bash
mkdir -p data/nuscenes && cd data/nuscenes

# nuScenes requires a free account at https://www.nuscenes.org/sign-up
# Once logged in, copy the dataset's signed download link and pass it here,
# or use the nuScenes devkit CLI. Example for the small, no-signup "mini" split:
pip install nuscenes-devkit

python - <<'PY'
from nuscenes.utils.data_io import load_bin_file  # sanity import only
print("nuscenes-devkit installed OK — now download nuScenes-mini:")
print("  https://www.nuscenes.org/download  ->  'Full dataset (v1.0)' -> 'Mini'")
PY

# After downloading and extracting v1.0-mini (~4 GB) into data/nuscenes/,
# raw lidar sweeps live under:
#   data/nuscenes/samples/LIDAR_TOP/*.pcd.bin
ls data/nuscenes/samples/LIDAR_TOP/ | head
cd ../..
```

Convert a nuScenes `.pcd.bin` sweep (x,y,z,intensity,ring — N×5 float32) to
this project's plain N×4 `.bin` format before using it:

```bash
python - <<'PY'
import numpy as np
src = "data/nuscenes/samples/LIDAR_TOP/<pick_a_file>.pcd.bin"
pts = np.fromfile(src, dtype=np.float32).reshape(-1, 5)[:, :4]   # drop ring column
pts.tofile("data/inference_frames/nuscenes_frame_000.bin")
print("wrote", pts.shape[0], "points")
PY
```

Then run inference/upload it exactly like any other `.bin` frame (Section 5/6).

---

## 5. PATH A — Run the WEB APP (recommended)

The web app (`app.py` FastAPI backend + `variant.html` frontend) wraps the
same pipeline behind a browser UI: upload frames, **pick which checkpoint to
run** (pretrained or fine-tuned, or any other `.pth`), run, watch progress,
view the color-coded foveated map, download `run_log.csv` / `results.json`.
This is the recommended way to use the project day-to-day — no OpenCV
window, works headless/remote, and lets you compare checkpoints side by
side without re-running scripts.

Unlike the CLI path, the web app does **not** read `config_pretrained.yaml` /
`config_finetuned.yaml` directly — those two files are the CLI's config
switch (Section 6). In the web UI, the equivalent switch is a single
**checkpoint path text box**: whatever `.pth` path you type there is what
the run uses, so you flip between "pretrained" and "fine-tuned" by changing
that one field (or via the `checkpoint_path` field of the `/api/runs`
request, if calling the API directly).

### 5.1 Start the server

```bash
cd ~/Desktop/SIH
source lidar25d/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Leave that running, then open `variant.html` directly in a browser
(double-click it, or `xdg-open variant.html`). It talks to the API at
`http://localhost:8000` — no build step, no separate frontend server, no
node/npm needed. `--host 0.0.0.0` also lets you reach it from another
machine on the same network at `http://<this-machine's-ip>:8000`.

**Confirm the server is up** (optional, second terminal):
```bash
curl http://localhost:8000/api/health
```

### 5.2 Choosing which checkpoint the web app uses

In **Step 02 — Configure the run**, under *"Pretrained checkpoint path"*:

| Want to run... | Type into the checkpoint path box | Then click |
|---|---|---|
| **Pretrained** (official Open3D-ML SemanticKITTI weights) | `checkpoints/randlanet_semantickitti_202201071330utc.pth` | **Check backend** |
| **Fine-tuned** (the checkpoint shipped in this repo, or the one you trained in Section 7) | `checkpoints/finetuned_ckpt_00040.pth` (match whatever `config_finetuned.yaml` / `checkpoints/` actually has — Section 3.2) | **Check backend** |
| **Heuristic fallback** (no model, wiring/smoke-test only) | leave the box empty | — |
| **Any other checkpoint** | any `.pth` path resolvable from the folder `uvicorn` was launched in | **Check backend** |

Tip: pull the exact filename straight from the YAML so there's no typo risk:
```bash
grep checkpoint_path config_pretrained.yaml config_finetuned.yaml
```
Copy the value after `checkpoint_path:` straight into the web UI's text box.

After typing a path, click **Check backend** and confirm the status badge
turns green and reads `backend: open3d-ml` — that means your real weights
loaded successfully. If it stays amber / says `heuristic-fallback`, the run
will use the transparent placeholder classifier (height threshold, not a
trained model), not your checkpoint — see Troubleshooting (Section 8).

You can switch checkpoints as many times as you like within the same
session: change the text box, click **Check backend** again, then **Run**
(Step 03) — each run remembers whichever checkpoint was active at the time
it was started, so a pretrained run and a fine-tuned run can sit side by
side in your results history without conflicting.

### 5.3 Also switchable in the same Configure step

- **Model** — RandLA-Net (enabled) or KPConv/KPFCNN (placeholder, under
  construction).
- **Device** — GPU · CUDA or CPU, matching `model.device` in the YAML
  configs. Pick CPU here if you set `device: "cpu"` in Section 2.
- **Ring layout** — near/mid/far/edge radii and cell sizes, editable
  directly in the ring table; mirrors the `rings:` block in the YAML
  configs.
- **Advanced → Fine-tune this model on your own labelled data** — exposes
  the same epochs / batch size / learning rate / frozen-layers knobs as
  `finetune.py --config config_pretrained.yaml`, so you can kick off a new
  fine-tuning run without leaving the browser; the resulting checkpoint can
  be dropped straight back into the checkpoint path box above.

### 5.4 Full walkthrough

1. **Step 01 — Upload frames.** Drag in `.bin`/`.npy` files — e.g. the 2
   sample frames from Section 4.1, frames pulled from SemanticKITTI/nuScenes
   (Section 4.2/4.3), or your own DRDO lidar captures. Each upload returns a
   `frame_id` and point count shown in the UI.
2. **Step 02 — Configure.** Set the checkpoint path (5.2), model, device,
   and ring layout, then click **Check backend** and confirm it's green.
3. **Step 03 — Run.** Watch live progress: current/total frames, fps,
   active cells, memory-saving %, running log tail.
4. **Step 04 — Results.** View the color-coded foveated grid per frame
   (terrain / static / dynamic), download `run_log.csv` (same columns as
   the CLI's `outputs/<...>/run_log.csv`) and `results.json` for
   downstream analysis.
5. Repeat with a different checkpoint path to compare pretrained vs.
   fine-tuned side by side — nothing gets overwritten, each run keeps its
   own `run_id`.

### 5.5 API reference (for scripting / calling checkpoints directly)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/frames` | upload a `.bin`/`.npy` frame |
| GET | `/api/frames` | list uploaded frames |
| DELETE | `/api/frames/{frame_id}` | drop a frame |
| GET | `/api/backend/status?checkpoint_path=...&device=...` | which classifier is active for a given checkpoint (real model vs heuristic) |
| POST | `/api/runs` | start a run — body includes `checkpoint_path`, `device`, `model`, `rings`, `frame_ids` |
| GET | `/api/runs/{run_id}/status` | poll progress |
| GET | `/api/runs/{run_id}/results` | per-frame metrics + grid cells |
| GET | `/api/runs/{run_id}/log.csv` | run as CSV, same columns as `run_log.csv` |

Example — upload a sample frame and kick off a fine-tuned run purely via
the API, no browser:

```bash
# 1) upload
FRAME_ID=$(curl -s -F "file=@data/samples/sample_frame_01.bin" \
  http://localhost:8000/api/frames | python -c "import sys,json;print(json.load(sys.stdin)['frame_id'])")

# 2) run against the fine-tuned checkpoint
curl -X POST http://localhost:8000/api/runs \
  -H "Content-Type: application/json" \
  -d "{
        \"frame_ids\": [\"$FRAME_ID\"],
        \"checkpoint_path\": \"checkpoints/finetuned_ckpt_00040.pth\",
        \"device\": \"cpu\",
        \"model\": \"RandLANet\"
      }"
```
Swap `checkpoint_path` for `checkpoints/randlanet_semantickitti_202201071330utc.pth`
to run the pretrained weights instead — every other field can stay the same.

Frame/run data lives in memory only — fine for a demo, not for
multi-user production (swap `FRAMES`/`RUNS` in `app.py` for Redis/a DB
before deploying beyond a single session).

---

## 6. PATH B — Run the CLI pipeline (local, no web)

Use this path if you want a desktop OpenCV dashboard, are doing batch/offline
processing of many frames unattended, or are training (Section 7) — the web
app's in-browser fine-tuning panel calls the same code, but the CLI is more
convenient for long training runs you want to leave running in a terminal /
over SSH.

### 6.1 Get frames into place

Copy your sample/real frames into `data/inference_frames/` (Section 4), or
generate a synthetic smoke-test frame:

```bash
python generate_test_frame.py
```

### 6.2 Run inference — pretrained or fine-tuned

Each config points at its own checkpoint (Section 3) and writes to its own
output folder, so results never overwrite each other:

```bash
# pretrained weights
python run_inference.py --config config_pretrained.yaml

# fine-tuned weights (shipped in the repo, or trained yourself — Section 7)
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

To run against a checkpoint that isn't wired into either YAML yet (e.g. a
newly-trained one with a different filename), either edit
`config_finetuned.yaml`'s `checkpoint_path`, or copy the YAML and point the
copy at the new file:

```bash
cp config_finetuned.yaml config_finetuned_v2.yaml
sed -i 's|checkpoint_path:.*|checkpoint_path: checkpoints/finetuned_ckpt_00080.pth|' config_finetuned_v2.yaml
python run_inference.py --config config_finetuned_v2.yaml
```

---

## 7. Fine-tuning on your own data (90/10 train/test split)

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

You can point `--raw_pc_dir`/`--raw_label_dir` at your own DRDO captures,
or at a SemanticKITTI sequence's `velodyne/`/`labels/` folders from
Section 4.2 to fine-tune/re-evaluate on that data.

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
automatically with the new checkpoint path — re-run Section 6.2's second
command (CLI), or type the new path into the web app's checkpoint box
(Section 5.2), to try it out. Commit the new `.pth` under `checkpoints/`
(via Git LFS if it's large) if you want it available the next time someone
sets up the repo per Section 3.2.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on run | venv not activated, or a package not installed | `source lidar25d/bin/activate`, then `pip install -r requirements.txt` (and the web-app extras in Section 2 if using Path A) |
| `FileNotFoundError: Checkpoint not found` | `model.checkpoint_path` in the config doesn't exist on disk | `ls checkpoints/`, fix the path in the config, or get it per Section 3 |
| Web app shows mostly terrain / wrong colors, or badge says `heuristic-fallback` | checkpoint path box left empty, or the path doesn't resolve on the server → using the heuristic fallback | retype the exact path from `grep checkpoint_path config_*.yaml` into the box (Section 5.2), click **Check backend**, confirm badge says `open3d-ml` |
| Web app still uses the old checkpoint after fine-tuning | you fine-tuned but never changed the checkpoint path box, or `uvicorn` was launched from a different working directory than the path assumes | check the *new* path written to `config_finetuned.yaml` (`grep checkpoint_path config_finetuned.yaml`) and paste that exact string into the box |
| `checkpoints/*.pth` is a few hundred bytes, not tens of MB | it's a Git LFS pointer file, not the real binary | `git lfs install && git lfs pull` (Section 3.3) |
| `config_pretrained.yaml` no longer points at the pretrained checkpoint | you're on an old copy of `finetune.py` that overwrote the input config in place | use the version in this repo — it always writes a separate `--output_config` and never touches the input config |
| Fine-tuning trained on 100% of frames, no held-out test set | `data/finetune_dataset/sequences/` had no `01/` folder — `--raw_pc_dir`/`--raw_label_dir` weren't passed, or an old `dataset_utils.py` was used | `rm -rf data/finetune_dataset cache logs` then re-run Section 7's command with `--raw_pc_dir`/`--raw_label_dir` set |
| No lidar frames found | `data/inference_frames/` is empty | drop `.bin`/`.npy` frames in (Section 4), or `python generate_test_frame.py` for a smoke test |
| SemanticKITTI download link 404s | dataset host reorganized/moved assets | get current links from https://www.semantic-kitti.org/dataset.html |
| nuScenes download fails / no link | requires a free account | sign up at https://www.nuscenes.org/sign-up, then use https://www.nuscenes.org/download |
| nuScenes frame shape errors (`N x 5` vs expected `N x 4`) | raw nuScenes `.pcd.bin` sweeps include a ring-index column this project doesn't expect | drop the 5th column before use — see the conversion snippet in Section 4.3 |
| `import open3d.ml.torch` fails inside the venv running `uvicorn` | wrong venv, or torch/open3d version mismatch | `source lidar25d/bin/activate` before `uvicorn app:app ...`, and confirm `python -c "import open3d.ml.torch as ml3d; print('ok')"` succeeds in that same shell |

---

## 9. Credits

Pretrained checkpoints from the official Open3D-ML model zoo
(`isl-org/Open3D-ML`), trained on SemanticKITTI.
SemanticKITTI dataset: www.semantic-kitti.org.
nuScenes dataset: www.nuscenes.org (Motional).

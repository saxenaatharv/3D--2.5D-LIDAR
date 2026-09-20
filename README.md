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

---

## 9. Accuracy evaluation module (VaRLA vs sparse uniform 5 cm)

`eval_accuracy.py` is a **separate, additive module**. It does not import
from, modify, or change the outputs of `run_inference.py` or `finetune.py`.
It reads ground-truth `.label` files and the same `.bin` point clouds, and
writes everything to `results/` -- nothing under `outputs/` is touched.

### What it compares

Both grids are built from the **same points and the same z-clip**, per scan:

- **VaRLA** -- the configurable ring schedule (default: 0-10 m @5cm,
  10-30 m @20cm, 30-60 m @35cm, 60-100 m @50cm), same as `config*.yaml`'s
  `grid.rings`.
- **Sparse uniform 5 cm** -- a single ring covering the full 0-100 m range
  at a fixed 5cm cell size, built with the exact same
  `VariableResolutionGrid` code path (so any difference in the numbers
  below comes from cell SIZE, not from two different implementations).

### Metrics

| Code | Metric | Definition |
|---|---|---|
| E1 | Obstacle recall | Fraction of GT obstacle (static+dynamic) points whose cell's MAJORITY class is an obstacle group. Also reported: static-only, dynamic-only, a looser "any-vote" recall, and `false_obstacle_rate` (terrain points whose cell is wrongly obstacle-majority). |
| E2 | Height error (cm) | For every occupied uniform-5cm cell, `|uniform.z_max - varla.z_max|` at the VaRLA cell physically containing that uniform cell's centre. Mean / median / p95, overall and obstacle-cells-only. |
| E3 | Range-band breakdown | E1 + E2 recomputed per horizontal-distance band (default matches the ring boundaries). Worst band = lowest VaRLA obstacle recall. |
| E4 | Point spacing vs range | Median nearest-neighbour spacing (KD-tree) per band, to sanity-check the far-range cell sizes against actual sensor resolution. |
| E5 (bonus) | Ring-schedule ablation | Obstacle recall vs mean active cells for 5 alternative ring schedules. `--ablation` |
| E6 (bonus) | Size + build time | `pickle.dumps()` byte size and `insert_points()` wall-clock time, labelled "entry/size/time" (never "RAM", since RAM was not profiled). CPU model printed alongside. `--benchmark` |

### How to run

Standalone (recommended -- keeps eval fully decoupled from the model run):

```bash
python eval_accuracy.py \
  --scans_dir data/finetune_dataset/sequences/00/velodyne \
  --labels_dir data/finetune_dataset/sequences/00/labels \
  --out_dir results
```

Add `--ablation` for E5 and `--benchmark` for E6:

```bash
python eval_accuracy.py \
  --scans_dir data/finetune_dataset/sequences/00/velodyne \
  --labels_dir data/finetune_dataset/sequences/00/labels \
  --ablation --benchmark
```

Override the ring schedule without touching any config file (format
`r_min:r_max:cell_cm`, comma-separated):

```bash
python eval_accuracy.py --rings "0:10:5,10:30:20,30:60:35,60:100:50" ...
```

Or via `run_inference.py`'s opt-in flag (runs the normal pipeline first,
unchanged, then hands off to `eval_accuracy.run_eval()` using that
config's ring schedule):

```bash
python run_inference.py --config config_pretrained.yaml --eval_accuracy --eval_ablation --eval_benchmark
```

### Outputs (all under `results/`)

- `accuracy_per_scan.csv` -- one row per scan, every E1/E2 number
- `accuracy_per_band.csv` -- E1/E3/E4 aggregated per range band
- `accuracy_summary.json` -- mean +/- std over all scans, `worst_band_m`,
  `worst_band_recall`, `worst_scan`, full per-band table
- `recall_by_band.png` -- grouped bar, VaRLA vs uniform recall by band
  (white background, large fonts, slide-ready)
- `schedule_tradeoff.png` -- E5 scatter (only with `--ablation`)
- `ablation_summary.json` -- E5 raw numbers (only with `--ablation`)

The console also prints the five headline numbers for a slide at the end
of every run:

```
[eval_accuracy] ===== SLIDE NUMBERS =====
  1. VaRLA obstacle recall      : ...
  2. Uniform 5cm obstacle recall: ...
  3. Mean height error (cm)     : ...
  4. Worst range band           : ... m
  5. Worst band VaRLA recall    : ...
==========================================
```

### Honesty notes

- The console always prints which sequence and exactly which frame stems
  were used (`[eval_accuracy] Using N scans, frame indices: ...`) -- if
  all scans come from one sequence, that is stated plainly, not hidden.
- `accuracy_summary.json`'s `worst_scan` / `worst_scan_varla_recall` are
  always the single worst-performing scan, not cherry-picked.
- `false_obstacle_rate` exists specifically so a grid cannot "cheat" its
  recall number by making every cell obstacle-majority.
- E6 numbers are explicitly labelled "entry / size / time" -- this module
  never claims to measure RAM, FPS, or on-device latency.

### Unit tests

```bash
python -m unittest tests/test_eval_accuracy.py -v
```

Covers, on a tiny hand-built synthetic scene with cell indices computed
directly from the grid's own floor-division formula (not approximated):
recall stays within `[0, 1]`, the uniform 5cm grid retains at least as
much obstacle detail as VaRLA's coarser far-range cells in a scene
designed to isolate that effect, and a point sitting exactly on a ring
boundary is assigned to exactly one ring, never zero or two.

### New dependency

E4 (point spacing) uses `scipy.spatial.cKDTree`:

```bash
pip install scipy
```

If scipy isn't installed, E4 is skipped with a printed warning -- E1/E2/E3/E5/E6
still run normally.

---

## 10. Performance profiling (compute + memory)

`perf_profile.py` adds real inference/RAM/VRAM numbers to replace "unprofiled"
on the deck. It is a **separate, additive module** -- `run_inference.py`'s
existing columns and their VALUES never change, whether `--profile` is on
or off. Profiling only APPENDS new columns to the same `run_log.csv` you
already use for the 50-scan replay.

### Run it

```bash
pip install psutil   # new dependency for CPU RAM tracking

python run_inference.py --config config_pretrained.yaml --profile
```

This automatically:
1. Prints + saves model size (`outputs/pretrained/model_size.json`)
2. Runs 3 warm-up passes (change with `--warmup_passes N`) on the first
   frame, discarded, before the timed loop opens the CSV
3. Times every stage per scan: frame load, RandLA-Net inference, semantic
   remap, sparse grid build, dashboard render (CUDA-synced if `model.device: cuda`)
4. Tracks peak GPU memory (`torch.cuda.max_memory_allocated()`, reset per
   scan) and peak CPU RSS (`psutil`, sampled once per scan)
5. Appends `frame_load_ms, inference_ms, remap_ms, grid_build_ms,
   render_ms, total_latency_ms, fps_measured, peak_gpu_mem_mb,
   peak_cpu_ram_mb, timestamp` to `run_log.csv` -- existing columns
   unchanged
6. Prints + saves a slide-ready mean +/- std markdown table
   (`outputs/pretrained/perf_summary.md`)

Re-summarize any already-profiled CSV later without re-running inference:

```bash
python perf_profile.py --summarize outputs/pretrained/run_log.csv
```

### Assumptions flagged for review

`perf_profile.py`'s module docstring lists every judgment call made while
wiring this up (please read before citing the numbers on a slide):

1. The pre-existing `fps` column is untouched; a separate `fps_measured`
   column (from `1000/total_latency_ms`, perf_counter-based) is added --
   the two will be close but are not defined identically.
2. Model size is a per-run constant, written once to `model_size.json`,
   not duplicated onto every CSV row.
3. "Frame load" only wraps `load_points()` -- no GPU sync needed there.
4. "Dashboard render" times `dash.render()` + `dash.save_image()` together
   as one stage, matching your 5-stage list.
5. CPU RSS is sampled once per scan (after all stages), not before/after each.
6. GPU peak memory is reset once at the start of each scan and read once
   at the end -- it is the peak across ALL that scan's stages combined,
   not a separate peak per stage (resetting before every stage would
   itself perturb the very thing being measured).
7. Warm-up passes reuse the first real frame's point cloud, not a
   synthetic one.

If any of these don't match how you want the numbers framed, say which one
and it's a small change.

### Jetson / edge device profiling (optional, run alongside)

If you have a Jetson (or similar) and want real power/thermal/utilization
numbers to fold into the same report, `tegrastats` runs independently of
this script and logs on its own schedule -- correlate the two using the
`timestamp` column this module now writes to every CSV row:

```bash
# Terminal 1: start tegrastats logging BEFORE the profiled run
sudo tegrastats --interval 500 --logfile tegrastats_log.txt &
TEGRA_PID=$!

# Terminal 1 (same shell): run the profiled inference
python run_inference.py --config config_pretrained.yaml --profile

# stop tegrastats once done
kill $TEGRA_PID
```

`tegrastats_log.txt` has one line per 500ms with its own timestamp, GPU/CPU
utilization %, power draw (mW), and temperatures. To fold it into the same
report:

```bash
# find the wall-clock start/end of your profiled run from the CSV's
# first and last `timestamp` column, then grep that window out of the
# tegrastats log:
head -2 outputs/pretrained/run_log.csv | tail -1 | cut -d, -f10   # first timestamp
tail -1 outputs/pretrained/run_log.csv | cut -d, -f10             # last timestamp
grep -A 100000 "<first HH:MM:SS>" tegrastats_log.txt | grep -B 100000 "<last HH:MM:SS>" > tegrastats_window.txt
```

Then average the `POM_5V_IN` (or equivalent power rail) and `GPU@` /
`CPU@` temperature fields across `tegrastats_window.txt` the same way
`perf_profile.summarize_csv` averages the CSV columns, and add those as
extra rows on the same slide table. I have not run this myself (no Jetson
available here) -- flagging that the exact power-rail field NAME varies by
Jetson model (Orin vs Xavier vs Nano use different labels in tegrastats'
output), so check one raw line of your `tegrastats_log.txt` first and
adjust the field name accordingly.

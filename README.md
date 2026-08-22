# Adaptive Variable Resolution 2.5D Lidar Mapping — Setup & Run Guide
DRDO / IDEX — Problem Statement 26053

All commands below are typed into a **Linux (Ubuntu) terminal**, assuming
your project folder is `~/Desktop/SIH`. The core inference pipeline only
needs these six files in that folder:

```
class_mapping.py
config.yaml
grid_engine.py
metrics.py
run_inference.py
visualize_dashboard.py
```

(`finetune.py`, `dataset_utils.py`, and `generate_test_frame.py` are only
needed if you plan to fine-tune on your own data or generate a synthetic
smoke-test frame — see Section 6.)

---

## 0. System packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip curl unzip
```

---

## 1. Go to your project folder

```bash
cd ~/Desktop/SIH
ls
```

---

## 2. Create and activate a virtual environment

```bash
python3 -m venv lidar25d
source lidar25d/bin/activate
```

---

## 3. Install PyTorch (version-matched to Open3D-ML: 2.2.2)

```bash
pip install --upgrade pip

# GPU build (NVIDIA GPU + CUDA 11.8 driver) -- fastest, recommended
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# If you do NOT have an NVIDIA GPU, use this line INSTEAD of the one above:
# pip install torch==2.2.2 torchvision==0.17.2
```

If you installed the CPU build of torch, open `config.yaml` and set
`model.device: "cpu"`.

---

## 4. Install remaining dependencies

```bash
pip install numpy pyyaml opencv-python matplotlib "open3d>=0.17.0" tensorboard "numpy<2"
```

---

## 5. Create required folders

```bash
mkdir -p checkpoints data/inference_frames outputs
```

---

## 6. Get the pretrained checkpoint (real weights, trained on SemanticKITTI — not random init)

```bash
curl -L -o checkpoints/randlanet_semantickitti_202201071330utc.pth https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth
```

(Optional, if you want to try KPConv instead of RandLA-Net):

```bash
curl -L -o checkpoints/kpconv_semantickitti_202009090354utc.pth https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth
```

If you switch to KPConv, edit `config.yaml`:
```yaml
model:
  name: "KPFCNN"
  checkpoint_path: "checkpoints/kpconv_semantickitti_202009090354utc.pth"
```

Both checkpoints come straight from the official Open3D-ML model zoo
(`isl-org/Open3D-ML`) and are trained end-to-end on SemanticKITTI — real
outdoor driving Lidar with road/building/vehicle/pedestrian classes, i.e.
the same kind of data this project targets. `run_inference.py` (and
`finetune.py`, if you use it) calls `pipeline.load_ckpt(...)`, so the
network starts from these learned weights, never from random
initialization.

---

## 7. Add Lidar frames to run inference on

You have two options:

**Option A — download a small real SemanticKITTI subset (recommended for a first run):**

```bash
mkdir -p data/finetune_raw
curl -L -o data/SemanticKittiTiny.zip https://pl-flash-data.s3.amazonaws.com/SemanticKittiTiny.zip
unzip -o data/SemanticKittiTiny.zip -d data/finetune_raw

rm -f data/inference_frames/*.bin
cp data/finetune_raw/SemanticKittiTiny/train/00/scans/*.bin data/inference_frames/
ls data/inference_frames/ | wc -l
```

**Option B — use your own frames.** Drop `.bin` frames (KITTI-style:
float32, N×4 = x,y,z,intensity) or `.npy` point clouds (N×3 or N×4) into:

```
data/inference_frames/
```

(A synthetic smoke-test frame can also be generated with
`python generate_test_frame.py`, if that script is present in your
folder — see Section 9.)

---

## 8. Run INFERENCE with the pretrained model

```bash
cd ~/Desktop/SIH
source lidar25d/bin/activate
python run_inference.py --config config.yaml
```

This will:
- load the pretrained RandLA-Net/KPConv weights (no training happens here)
- run semantic segmentation on every frame in `data/inference_frames`
- remap predictions to terrain / static-obstacle / dynamic-object
- build the variable-resolution 2.5D grid (5cm cells out to 10m, widening to
  50cm cells out to 100m)
- open a live OpenCV dashboard window showing the color-coded map, FPS, and
  memory-saving percentage vs. an equivalent uniform 5cm grid
- write `outputs/run_log.csv` with per-frame FPS / active cell count /
  memory saving / mIoU (mIoU only populates if you set `paths.gt_label_dir`
  in `config.yaml` to a folder of matching `.label` files)
- save a PNG of the dashboard for every frame to `outputs/outputimage/`

Press **`q`** in the dashboard window to stop early.

**Check the output:**

```bash
cat outputs/run_log.csv
ls outputs/outputimage/
```

---

## 9. (Optional) FINE-TUNE the pretrained model on your own data

This step needs three extra files that aren't part of the minimal
inference set: `finetune.py`, `dataset_utils.py`, and `requirements.txt`.
Copy them into `~/Desktop/SIH` first.

If your own data is just raw point clouds + a per-point label array
(`.npy`, values 0=terrain / 1=static / 2=dynamic, or your own SemanticKITTI-style ids):

```bash
python finetune.py --config config.yaml --raw_pc_dir path/to/your/pointclouds --raw_label_dir path/to/your/labels
```

If your data is already organized in SemanticKITTI's native layout
(`sequences/00/velodyne/*.bin`, `sequences/00/labels/*.label`), just point
`paths.finetune_data_dir` in `config.yaml` at it and run:

```bash
python finetune.py --config config.yaml
```

This script:
- loads the **same pretrained checkpoint** from `checkpoints/` as a starting
  point (`pipeline.load_ckpt(...)`) — fine-tuning, not training from scratch
- trains for `training.epochs` epochs (edit `config.yaml` to change
  epochs / batch size / learning rate / how many early layers to freeze)
- saves the adapted checkpoint under `./logs/`

To use the fine-tuned weights afterwards, copy the new `.pth` file into
`checkpoints/` and update `model.checkpoint_path` in `config.yaml`, then
re-run Section 8.

---

## 10. Project structure

```
SIH/
├── config.yaml              <- single source of truth: model, grid rings, paths, training
├── class_mapping.py         <- SemanticKITTI 19 classes -> terrain/static/dynamic
├── grid_engine.py           <- Variable Resolution 2.5D Grid Engine (foveated sparse grid)
├── visualize_dashboard.py   <- live OpenCV dashboard renderer
├── metrics.py                <- IoU + memory-saving report
├── run_inference.py          <- SCRIPT 1: pretrained-model inference, no training
├── dataset_utils.py          <- (optional) converts your raw data into SemanticKITTI layout
├── finetune.py                <- (optional) SCRIPT 2: fine-tunes the pretrained checkpoint
├── generate_test_frame.py     <- (optional) synthetic smoke-test frame generator
├── requirements.txt           <- (optional, only needed for finetune.py's own dependency pin)
├── lidar25d/                  <- Python virtual environment (created in Section 2)
├── checkpoints/                <- pretrained .pth files go here
├── data/
│   ├── inference_frames/       <- put lidar frames here for Section 8
│   ├── finetune_raw/            <- downloaded/extracted SemanticKittiTiny subset (Section 7)
│   └── finetune_dataset/        <- auto-populated by dataset_utils.py, if used
└── outputs/                     <- run_log.csv and exported dashboard PNGs land here
    └── outputimage/
```

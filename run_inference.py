"""
run_inference.py
------------------
INFERENCE-ONLY script. Loads a PRETRAINED semantic segmentation backbone
(RandLA-Net or KPConv, trained on SemanticKITTI, downloaded from the
official Open3D-ML model zoo) and runs it on a folder of raw Lidar frames.

For each frame:
    1. forward pass through the pretrained network -> per-point class id
    2. remap predictions to {terrain, static, dynamic} super-classes
    3. insert into the Variable Resolution 2.5D Grid Engine (foveated:
       5cm cells near the sensor, widening to 50cm cells at 100m)
    4. render live on the OpenCV dashboard
    5. log FPS / active-cell count / memory saving / (optional) mIoU

No weights are updated by this script. Use finetune.py to adapt the
backbone to your own labelled data.
"""

import os
import time
import argparse
import glob
import yaml
import numpy as np

import open3d.ml as _ml3d
import open3d.ml.torch as ml3d   # switch to open3d.ml.tf if config.yaml model.backend == "tf"

from class_mapping import to_super_class
from grid_engine import VariableResolutionGrid
from visualize_dashboard import Dashboard
from metrics import RunningIoU, memory_saving_report


def load_points(path):
    if path.endswith(".bin"):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.shape[1] == 3:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), np.float32)], axis=1)
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported point cloud file: {path}")


def build_pipeline(cfg):
    model_name = cfg["model"]["name"]
    ckpt = cfg["model"]["checkpoint_path"]
    device = cfg["model"]["device"]

    if model_name == "RandLANet":
        model = ml3d.models.RandLANet(num_classes=cfg["model"]["num_classes"])
    elif model_name == "KPFCNN":
        model = ml3d.models.KPFCNN(num_classes=cfg["model"]["num_classes"])
    else:
        raise ValueError(f"Unknown model '{model_name}' -- use RandLANet or KPFCNN")

    pipeline = ml3d.pipelines.SemanticSegmentation(model, device=device)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt}. Download the pretrained weights first "
            "(see README.md 'Download the pretrained checkpoint' step)."
        )
    pipeline.load_ckpt(ckpt_path=ckpt)   # <-- loads PRETRAINED weights, NOT random init
    pipeline.model.eval()
    return pipeline


def run(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline = build_pipeline(cfg)

    grid_cfg = cfg["grid"]
    grid = VariableResolutionGrid(
        rings_cfg=grid_cfg["rings"],
        ego_origin=tuple(grid_cfg["ego_origin"]),
        z_clip=tuple(grid_cfg["z_clip"]),
    )
    dash = Dashboard(
        grid,
        image_size=tuple(cfg["visualization"]["image_size"]),
        window_name=cfg["visualization"]["window_name"],
    )

    frames = sorted(glob.glob(os.path.join(cfg["paths"]["inference_input_dir"], "*")))
    if not frames:
        raise FileNotFoundError(
            f"No lidar frames found in {cfg['paths']['inference_input_dir']}. "
            "Drop .bin (KITTI-style N x 4 float32) or .npy point cloud files there first."
        )

    gt_dir = cfg["paths"].get("gt_label_dir", "")
    iou_meter = RunningIoU(num_classes=3) if gt_dir else None

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    image_out_dir = os.path.join(cfg["paths"]["output_dir"], "outputimage")
    os.makedirs(image_out_dir, exist_ok=True)
    log_path = os.path.join(cfg["paths"]["output_dir"], "run_log.csv")
    log_f = open(log_path, "w")
    log_f.write("frame,fps,active_cells,memory_saving_percent,mean_iou\n")

    for fi, frame_path in enumerate(frames):
        t0 = time.time()
        pts = load_points(frame_path)
        xyz, intensity = pts[:, :3], pts[:, 3]

        data = {"point": xyz, "feat": None, "label": np.zeros(len(xyz), dtype=np.int32)}
        results = pipeline.run_inference(data)   # forward pass through the PRETRAINED network
        pred_labels = results["predict_labels"]  # (N,) SemanticKITTI class ids

        super_labels = to_super_class(pred_labels)

        grid.reset()   # remove this line to accumulate a persistent map across frames instead
        grid.insert_points(xyz, super_labels)

        mean_iou_txt = "n/a"
        if iou_meter is not None:
            gt_path = os.path.join(gt_dir, os.path.splitext(os.path.basename(frame_path))[0] + ".label")
            if os.path.exists(gt_path):
                gt_raw = np.fromfile(gt_path, dtype=np.uint32) & 0xFFFF
                gt_super = to_super_class(gt_raw)
                iou_meter.update(np.clip(super_labels, 0, 2), np.clip(gt_super, 0, 2))
                mean_iou_txt = f"{iou_meter.mean_iou():.3f}"

        mem_report = memory_saving_report(grid)
        key = dash.render(extra_text=[
            f"Model: {cfg['model']['name']} (pretrained on {cfg['model']['dataset_for_pretrain']})",
            f"Frame: {fi + 1}/{len(frames)}",
            f"Memory saving vs uniform 5cm grid: {mem_report['memory_saving_percent']}%",
            f"mIoU (terrain/static/dynamic): {mean_iou_txt}",
        ])

        fps = 1.0 / max(time.time() - t0, 1e-6)
        log_f.write(f"{fi},{fps:.2f},{mem_report['sparse_cells_used']},"
                    f"{mem_report['memory_saving_percent']},{mean_iou_txt}\n")

        image_path = os.path.join(image_out_dir, f"frame_{fi:06d}.png")
        dash.save_image(image_path)
        print(f"[run_inference] Saved dashboard image to {image_path}")

        if key == ord("q"):
            break

    log_f.close()
    dash.close()
    print(f"[run_inference] Done. Per-frame log written to {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    run(args.config)

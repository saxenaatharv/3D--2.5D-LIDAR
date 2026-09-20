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

--eval_accuracy is an OPT-IN flag (default off) that, after the normal run
completes, hands off to eval_accuracy.py's ground-truth accuracy-evaluation
module. It changes nothing about the default run_inference.py behaviour or
its existing outputs -- eval_accuracy is a completely separate module that
is only imported/called if this flag is passed.

--profile is a SEPARATE opt-in flag (default off) that wraps each stage
(frame load, RandLA-Net inference, semantic remap, sparse grid build,
dashboard render) in perf_profile.StageTimer, tracks peak GPU/CPU memory,
runs a warm-up phase before the timed loop, and appends new columns to
run_log.csv. The EXISTING columns (frame, fps, active_cells,
memory_saving_percent, mean_iou, overall_accuracy, terrain_accuracy,
static_accuracy, dynamic_accuracy) and their values are completely
unchanged whether --profile is on or off -- profiling only APPENDS columns.
See perf_profile.py's module docstring for the specific assumptions this
makes (flagged there for your review).
"""

import os
import time
import json
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
import perf_profile


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


def run(cfg_path, profile=False, warmup_passes=3):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline = build_pipeline(cfg)
    device = cfg["model"]["device"]

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

    if profile:
        # Model size report -- constant for the whole run, written once as
        # a sidecar JSON rather than duplicated onto every CSV row (see
        # perf_profile.py's module docstring, assumption #2).
        size_report = perf_profile.model_size_report(pipeline.model)
        size_path = os.path.join(cfg["paths"]["output_dir"], "model_size.json")
        with open(size_path, "w") as sf:
            json.dump(size_report, sf, indent=2)
        print(f"[run_inference] Model size: {size_report['num_params']:,} parameters, "
              f"{size_report['size_mb']:.2f} MB -> {size_path}")

        # Warm-up phase: throwaway passes on the first real frame, run
        # BEFORE the timed loop opens the CSV, so one-time CUDA kernel
        # compilation/allocation never lands in a logged row.
        warmup_pts = load_points(frames[0])
        perf_profile.run_warmup(pipeline, warmup_pts[:, :3], n_passes=warmup_passes, device=device)

    log_f = open(log_path, "w")
    header = ("frame,fps,active_cells,memory_saving_percent,mean_iou,"
               "overall_accuracy,terrain_accuracy,static_accuracy,dynamic_accuracy")
    if profile:
        header += "," + ",".join(perf_profile.PERF_COLUMNS)
    log_f.write(header + "\n")

    for fi, frame_path in enumerate(frames):
        t0 = time.time()

        if profile:
            perf_profile.reset_gpu_peak(device)
            stage_ms = {}
            with perf_profile.StageTimer(device=device) as t_load:
                pts = load_points(frame_path)
                xyz, intensity = pts[:, :3], pts[:, 3]
            stage_ms["frame_load"] = t_load.elapsed_ms

            data = {"point": xyz, "feat": None, "label": np.zeros(len(xyz), dtype=np.int32)}
            with perf_profile.StageTimer(device=device) as t_inf:
                results = pipeline.run_inference(data)
            stage_ms["inference"] = t_inf.elapsed_ms
            pred_labels = results["predict_labels"]

            with perf_profile.StageTimer(device=device) as t_remap:
                super_labels = to_super_class(pred_labels)
            stage_ms["remap"] = t_remap.elapsed_ms

            with perf_profile.StageTimer(device=device) as t_grid:
                grid.reset()
                grid.insert_points(xyz, super_labels)
            stage_ms["grid_build"] = t_grid.elapsed_ms
        else:
            pts = load_points(frame_path)
            xyz, intensity = pts[:, :3], pts[:, 3]

            data = {"point": xyz, "feat": None, "label": np.zeros(len(xyz), dtype=np.int32)}
            results = pipeline.run_inference(data)   # forward pass through the PRETRAINED network
            pred_labels = results["predict_labels"]  # (N,) SemanticKITTI class ids

            super_labels = to_super_class(pred_labels)

            grid.reset()   # remove this line to accumulate a persistent map across frames instead
            grid.insert_points(xyz, super_labels)

        mean_iou_txt = "n/a"
        overall_acc_txt = "n/a"
        terrain_acc_txt = static_acc_txt = dynamic_acc_txt = "n/a"
        overall_acc = terrain_acc = static_acc = dynamic_acc = None
        if iou_meter is not None:
            gt_path = os.path.join(gt_dir, os.path.splitext(os.path.basename(frame_path))[0] + ".label")
            if os.path.exists(gt_path):
                gt_raw = np.fromfile(gt_path, dtype=np.uint32) & 0xFFFF
                gt_super = to_super_class(gt_raw)
                iou_meter.update(np.clip(super_labels, 0, 2), np.clip(gt_super, 0, 2))
                mean_iou_txt = f"{iou_meter.mean_iou():.3f}"

                overall_acc = iou_meter.overall_accuracy()
                terrain_acc, static_acc, dynamic_acc = iou_meter.per_class_accuracy()
                overall_acc_txt = f"{overall_acc:.3f}"
                terrain_acc_txt = f"{terrain_acc:.3f}" if not np.isnan(terrain_acc) else "n/a"
                static_acc_txt = f"{static_acc:.3f}" if not np.isnan(static_acc) else "n/a"
                dynamic_acc_txt = f"{dynamic_acc:.3f}" if not np.isnan(dynamic_acc) else "n/a"

        mem_report = memory_saving_report(grid)

        if profile:
            with perf_profile.StageTimer(device=device) as t_render:
                key = dash.render(extra_text=[
                    f"Model: {cfg['model']['name']} (pretrained on {cfg['model']['dataset_for_pretrain']})",
                    f"Frame: {fi + 1}/{len(frames)}",
                    f"Memory saving vs uniform 5cm grid: {mem_report['memory_saving_percent']}%",
                    f"mIoU (terrain/static/dynamic): {mean_iou_txt}",
                    f"Overall accuracy: {overall_acc_txt}",
                    f"Per-class acc  T:{terrain_acc_txt}  S:{static_acc_txt}  D:{dynamic_acc_txt}",
                ])
                image_path = os.path.join(image_out_dir, f"frame_{fi:06d}.png")
                dash.save_image(image_path)
            stage_ms["render"] = t_render.elapsed_ms

            peak_gpu_mb = perf_profile.get_gpu_peak_mb(device)
            peak_cpu_mb = perf_profile.get_cpu_rss_mb()
            perf_fields = perf_profile.perf_row_to_csv_fields(stage_ms, peak_gpu_mb, peak_cpu_mb)
        else:
            key = dash.render(extra_text=[
                f"Model: {cfg['model']['name']} (pretrained on {cfg['model']['dataset_for_pretrain']})",
                f"Frame: {fi + 1}/{len(frames)}",
                f"Memory saving vs uniform 5cm grid: {mem_report['memory_saving_percent']}%",
                f"mIoU (terrain/static/dynamic): {mean_iou_txt}",
                f"Overall accuracy: {overall_acc_txt}",
                f"Per-class acc  T:{terrain_acc_txt}  S:{static_acc_txt}  D:{dynamic_acc_txt}",
            ])

        fps = 1.0 / max(time.time() - t0, 1e-6)
        log_line = (f"{fi},{fps:.2f},{mem_report['sparse_cells_used']},"
                    f"{mem_report['memory_saving_percent']},{mean_iou_txt},"
                    f"{overall_acc_txt},{terrain_acc_txt},{static_acc_txt},{dynamic_acc_txt}")
        if profile:
            log_line += "," + ",".join(str(perf_fields[c]) for c in perf_profile.PERF_COLUMNS)
        log_f.write(log_line + "\n")

        if not profile:
            image_path = os.path.join(image_out_dir, f"frame_{fi:06d}.png")
            dash.save_image(image_path)
        print(f"[run_inference] Saved dashboard image to {image_path}")

        if key == ord("q"):
            break

    log_f.close()
    dash.close()
    print(f"[run_inference] Done. Per-frame log written to {log_path}")

    if profile:
        model_size_path = os.path.join(cfg["paths"]["output_dir"], "model_size.json")
        print("\n[run_inference] ===== PERFORMANCE SUMMARY =====")
        perf_profile.summarize_csv(log_path, model_size_path=model_size_path)

    if iou_meter is not None:
        summary = iou_meter.summary_dict()
        report_path = os.path.join(cfg["paths"]["output_dir"], "accuracy_report.json")
        with open(report_path, "w") as rf:
            json.dump(summary, rf, indent=2)
        print(f"[run_inference] Accuracy over {len(frames)} frames vs DRDO ground truth:")
        print(f"  Overall accuracy : {summary['overall_accuracy']}")
        print(f"  Mean IoU         : {summary['mean_iou']}")
        print(f"  Terrain accuracy : {summary['terrain_accuracy']}")
        print(f"  Static accuracy  : {summary['static_accuracy']}")
        print(f"  Dynamic accuracy : {summary['dynamic_accuracy']}")
        print(f"  Full report      : {report_path}")
    else:
        print("[run_inference] No gt_label_dir set in config -> no accuracy computed. "
              "Set paths.gt_label_dir to your DRDO .label folder to get accuracy numbers.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--eval_accuracy", action="store_true",
                     help="OPT-IN: after the normal run, also run eval_accuracy.py's "
                          "VaRLA-vs-uniform-5cm ground-truth accuracy evaluation. Does not "
                          "change anything about the run above -- purely additive.")
    ap.add_argument("--eval_scans_dir", default=None,
                     help="Only used with --eval_accuracy. Defaults to "
                          "data/finetune_dataset/sequences/00/velodyne")
    ap.add_argument("--eval_labels_dir", default=None,
                     help="Only used with --eval_accuracy. Defaults to "
                          "data/finetune_dataset/sequences/00/labels")
    ap.add_argument("--eval_ablation", action="store_true",
                     help="Only used with --eval_accuracy. Also run the E5 ring-schedule ablation.")
    ap.add_argument("--eval_benchmark", action="store_true",
                     help="Only used with --eval_accuracy. Also run the E6 size/time benchmark.")
    ap.add_argument("--profile", action="store_true",
                     help="OPT-IN: wrap each pipeline stage in timers, track peak GPU/CPU memory, "
                          "run a warm-up phase, and append new columns to run_log.csv. Does not "
                          "change any existing column's value -- purely additive. See "
                          "perf_profile.py's docstring for the assumptions this makes.")
    ap.add_argument("--warmup_passes", type=int, default=3,
                     help="Only used with --profile. Throwaway inference passes before the timed "
                          "loop starts (default 3).")
    args = ap.parse_args()
    run(args.config, profile=args.profile, warmup_passes=args.warmup_passes)

    if args.eval_accuracy:
        # Imported here, not at module top, so a machine without eval_accuracy's
        # optional deps (scipy) can still run plain inference with this flag unset.
        from eval_accuracy import run_eval, DEFAULT_RINGS

        with open(args.config, "r") as f:
            _cfg = yaml.safe_load(f)
        rings_cfg = _cfg.get("grid", {}).get("rings", DEFAULT_RINGS)

        scans_dir = args.eval_scans_dir or "data/finetune_dataset/sequences/00/velodyne"
        labels_dir = args.eval_labels_dir or "data/finetune_dataset/sequences/00/labels"

        print("\n[run_inference] --eval_accuracy set: handing off to eval_accuracy.run_eval() ...")
        run_eval(
            scans_dir, labels_dir, rings_cfg=rings_cfg,
            out_dir="results", ablation=args.eval_ablation, benchmark=args.eval_benchmark,
        )
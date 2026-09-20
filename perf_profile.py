"""
perf_profile.py
------------------
Performance-profiling utilities for the VaRLA pipeline (frame load ->
RandLA-Net inference -> semantic remap -> sparse ring grid build ->
dashboard render). Purely additive: this module does not change any
existing metric run_inference.py already computes (fps, active_cells,
memory_saving_percent, mean_iou, accuracy columns) -- it only supplies the
timing/memory helpers that run_inference.py calls, and appends NEW columns
to the same run_log.csv.

ASSUMPTIONS / THINGS FLAGGED FOR THE USER (please double-check on your box):
  1. run_inference.py's existing "fps" column (1 / wall-clock over the
     whole per-frame loop body via time.time()) is left completely
     untouched. This module adds a SEPARATE "fps_measured" column, derived
     from perf_counter()-based total_latency_ms as you specified
     (1000/total_ms). The two numbers should be close but are not
     guaranteed identical, since the old one used time.time() over a
     slightly different span (it also includes the CSV log write). I did
     not merge them into one column so nothing you already measured
     changes.
  2. Model size (param count + MB) is a CONSTANT for the whole run, not a
     per-scan quantity, so I did NOT add it as a repeated per-row CSV
     column. It's written once to a sidecar
     outputs/<run>/model_size.json and included in the printed summary
     table. If you specifically want it duplicated onto every CSV row
     instead, tell me and I'll add it as a constant column.
  3. "Frame load" stage timing wraps only the load_points() call. GPU
     synchronize is skipped for this stage since file I/O + numpy never
     touches CUDA.
  4. Dashboard render timing wraps BOTH dash.render() and dash.save_image()
     together as one "render" stage (your spec listed "map/dashboard
     render" as one stage, not two) -- flagging in case you wanted them
     split.
  5. CPU RSS is sampled ONCE per scan (after all stages complete), per
     your "once per scan is fine" fallback -- not before/after each stage,
     to keep the per-frame loop overhead low.
  6. Peak GPU memory is reset once at the START of each scan (before frame
     load) and read once at the END of each scan (after render) --
     i.e. it captures the peak across ALL stages for that scan combined,
     not a separate peak per stage. torch.cuda's peak-memory API is
     global per reset, so getting a true per-stage GPU peak would require
     a reset before every single stage, which would itself perturb the
     numbers being measured; I judged whole-scan peak to be the more
     honest number to report. Flagging this trade-off explicitly.
  7. Warm-up passes reuse the FIRST real frame's point cloud (frames[0]),
     run through pipeline.run_inference() the same way as the timed loop,
     but are not logged to CSV and do not affect any stat. This matches
     "throwaway inference passes" in your request. If you'd rather warm up
     on a synthetic/dummy cloud instead of a real frame, say so.
"""

import csv
import json
import os
import time

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


# --------------------------------------------------------------- timing --
class StageTimer:
    """
    Context manager for timing one pipeline stage.

        with StageTimer(device="cuda") as t:
            do_the_stage()
        elapsed_ms = t.elapsed_ms

    If device starts with "cuda" and torch/CUDA are available, calls
    torch.cuda.synchronize() right before stopping the clock, so async
    GPU kernels are actually finished before we measure -- otherwise the
    timer would return before the GPU work completes and under-report
    the stage's true cost.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self._is_cuda = (
            isinstance(device, str) and device.startswith("cuda")
            and torch is not None and torch.cuda.is_available()
        )
        self.elapsed_ms = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_cuda:
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        return False  # don't suppress exceptions


# ------------------------------------------------------------ GPU memory --
def cuda_available(device="cpu"):
    return (
        isinstance(device, str) and device.startswith("cuda")
        and torch is not None and torch.cuda.is_available()
    )


def reset_gpu_peak(device="cpu"):
    """Call at the START of each scan, before any stage runs."""
    if cuda_available(device):
        torch.cuda.reset_peak_memory_stats()


def get_gpu_peak_mb(device="cpu"):
    """Call at the END of each scan. Returns None if not on CUDA (not 0.0,
    so it's never confused with 'measured zero usage')."""
    if not cuda_available(device):
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ------------------------------------------------------------ CPU memory --
def get_cpu_rss_mb():
    """Process resident set size in MB. Returns None if psutil isn't installed."""
    if psutil is None:
        return None
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)


# ------------------------------------------------------------ model size --
def model_size_report(model):
    """
    sum(p.numel() for p in model.parameters()) and the corresponding byte
    size (each parameter's actual dtype size, not assumed float32), as
    specified. Returns a plain dict, JSON-serialisable.
    """
    num_params = 0
    size_bytes = 0
    for p in model.parameters():
        n = p.numel()
        num_params += n
        size_bytes += n * p.element_size()
    return {
        "num_params": int(num_params),
        "size_mb": round(size_bytes / (1024 ** 2), 3),
    }


# ---------------------------------------------------------------- warmup --
def run_warmup(pipeline, xyz, n_passes=3, device="cpu"):
    """
    Runs n_passes throwaway forward passes through pipeline.run_inference()
    using the SAME call shape as the real per-frame loop (data dict with
    feat=None, matching the xyz-only checkpoint), so one-time CUDA kernel
    compilation / cuDNN autotune / first-alloc costs land here instead of
    polluting scan 0's timed numbers. Results are discarded. Prints a short
    per-pass timing line so you can see the warm-up curve flattening out.
    """
    print(f"[perf_profile] Running {n_passes} warm-up pass(es) before the timed loop...")
    for i in range(n_passes):
        data = {"point": xyz, "feat": None, "label": np.zeros(len(xyz), dtype=np.int32)}
        with StageTimer(device=device) as t:
            pipeline.run_inference(data)
        print(f"[perf_profile]   warm-up pass {i + 1}/{n_passes}: {t.elapsed_ms:.1f} ms")


# ----------------------------------------------------------------- CSV --
PERF_COLUMNS = [
    "timestamp",
    "frame_load_ms",
    "inference_ms",
    "remap_ms",
    "grid_build_ms",
    "render_ms",
    "total_latency_ms",
    "fps_measured",
    "peak_gpu_mem_mb",
    "peak_cpu_ram_mb",
]


def perf_row_to_csv_fields(stage_ms: dict, peak_gpu_mb, peak_cpu_mb):
    """
    stage_ms: dict with keys frame_load, inference, remap, grid_build, render
    (all in ms). Returns an OrderedDict-friendly dict of the new columns,
    ready to be merged into the existing row dict before writing.
    total_latency_ms is the SUM of the five stage timings (not a separate
    wall-clock measurement), so it is internally consistent with the
    per-stage breakdown by construction.
    """
    total_ms = sum(stage_ms.values())
    fps_measured = 1000.0 / total_ms if total_ms > 0 else float("nan")
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frame_load_ms": round(stage_ms["frame_load"], 3),
        "inference_ms": round(stage_ms["inference"], 3),
        "remap_ms": round(stage_ms["remap"], 3),
        "grid_build_ms": round(stage_ms["grid_build"], 3),
        "render_ms": round(stage_ms["render"], 3),
        "total_latency_ms": round(total_ms, 3),
        "fps_measured": round(fps_measured, 3),
        "peak_gpu_mem_mb": round(peak_gpu_mb, 2) if peak_gpu_mb is not None else "n/a",
        "peak_cpu_ram_mb": round(peak_cpu_mb, 2) if peak_cpu_mb is not None else "n/a",
    }


# ------------------------------------------------------------- summary --
def _mean_std(values):
    vals = [v for v in values if v is not None and v != "n/a" and not (isinstance(v, float) and np.isnan(v))]
    vals = [float(v) for v in vals]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def summarize_csv(csv_path, model_size_path=None):
    """
    Reads a run_log.csv that has the PERF_COLUMNS appended (as written by
    run_inference.py with profiling enabled) and returns/prints a
    slide-ready markdown table of mean +/- std across all rows for: total
    latency, each stage, peak VRAM, peak RAM, FPS, plus model size if the
    sidecar JSON is found.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")
    missing = [c for c in PERF_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"{csv_path} is missing profiling columns {missing} -- "
            f"this CSV was written WITHOUT --profile. Re-run with profiling enabled."
        )

    n = len(rows)
    metrics = {}
    for col in ["frame_load_ms", "inference_ms", "remap_ms", "grid_build_ms",
                "render_ms", "total_latency_ms", "fps_measured",
                "peak_gpu_mem_mb", "peak_cpu_ram_mb"]:
        mean, std = _mean_std([r[col] for r in rows])
        metrics[col] = (mean, std)

    model_size = None
    if model_size_path and os.path.exists(model_size_path):
        with open(model_size_path) as f:
            model_size = json.load(f)

    lines = []
    lines.append(f"# Performance summary ({n} scans)\n")
    lines.append("| Metric | Mean | Std |")
    lines.append("|---|---|---|")
    label_map = {
        "frame_load_ms": "Frame load (ms)",
        "inference_ms": "RandLA-Net inference (ms)",
        "remap_ms": "Semantic remap (ms)",
        "grid_build_ms": "Sparse grid build (ms)",
        "render_ms": "Dashboard render (ms)",
        "total_latency_ms": "**Total latency (ms)**",
        "fps_measured": "**FPS (measured)**",
        "peak_gpu_mem_mb": "Peak GPU memory (MB)",
        "peak_cpu_ram_mb": "Peak CPU RAM (MB)",
    }
    for col, (mean, std) in metrics.items():
        if np.isnan(mean):
            lines.append(f"| {label_map[col]} | n/a | n/a |")
        else:
            lines.append(f"| {label_map[col]} | {mean:.2f} | {std:.2f} |")

    if model_size:
        lines.append("")
        lines.append(f"**Model size**: {model_size['num_params']:,} parameters, "
                      f"{model_size['size_mb']:.2f} MB")

    table = "\n".join(lines)
    print(table)
    return table


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Summarize a profiled run_log.csv into a slide-ready table.")
    ap.add_argument("--summarize", required=True, help="Path to run_log.csv (must have --profile columns)")
    ap.add_argument("--model_size_json", default=None,
                     help="Path to model_size.json sidecar (default: same dir as the CSV)")
    args = ap.parse_args()

    model_size_path = args.model_size_json
    if model_size_path is None:
        candidate = os.path.join(os.path.dirname(args.summarize), "model_size.json")
        if os.path.exists(candidate):
            model_size_path = candidate

    summary_md = summarize_csv(args.summarize, model_size_path=model_size_path)
    out_path = os.path.join(os.path.dirname(args.summarize), "perf_summary.md")
    with open(out_path, "w") as f:
        f.write(summary_md + "\n")
    print(f"\n[perf_profile] Wrote {out_path}")


if __name__ == "__main__":
    main()

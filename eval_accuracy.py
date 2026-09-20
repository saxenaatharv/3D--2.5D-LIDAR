"""
eval_accuracy.py
------------------
Accuracy-evaluation module for VaRLA (variable-resolution ring grid) vs a
uniform sparse 5 cm grid, both built from the SAME points and the SAME
z-clip. This module is purely additive: it does not import from, modify,
or change the behaviour of run_inference.py, finetune.py, or any existing
output file. It only reads ground-truth labels + point clouds and writes
new files under results/.

Run standalone:
    python eval_accuracy.py --scans_dir <velodyne dir> --labels_dir <labels dir>

Or via the existing pipeline's opt-in CLI flag (does not run by default):
    python run_inference.py --config config_pretrained.yaml --eval_accuracy

-----------------------------------------------------------------------
DEFINITIONS
-----------------------------------------------------------------------
Grids compared (built from the SAME xyz points + SAME z_clip, per scan):
    VaRLA   -- VariableResolutionGrid with the configurable ring schedule
               (default: 0-10 m @5cm, 10-30 m @20cm, 30-60 m @35cm,
               60-100 m @50cm).
    Uniform -- VariableResolutionGrid with a single "ring" covering the
               full 0-100 m range at a fixed 5 cm cell size. This reuses
               the exact same insert/lookup code path as VaRLA (same
               floor-division cell-key formula), so any difference in the
               metrics below comes only from cell SIZE, not from two
               different implementations.

E1 Obstacle recall
    For every GT point labelled static or dynamic obstacle, the point is
    "kept" if the grid cell containing it has an obstacle class (static or
    dynamic) as its MAJORITY vote. Reported for VaRLA and Uniform, and
    separately for static-only and dynamic-only obstacle points.
    "any-vote" recall is a looser secondary number: kept if the cell has
    ANY obstacle vote at all (majority or not) -- this exists specifically
    so a coarse, terrain-majority cell that still contains a few obstacle
    votes doesn't get silently counted as a total loss under the stricter
    majority metric, and vice versa a coarse cell cannot inflate the
    strict number either.
    false_obstacle_rate = fraction of TERRAIN gt points whose cell
    majority is (wrongly) an obstacle class -- this is the metric that
    would catch a grid "cheating" by making every cell obstacle-majority.

E2 Height error (cm)
    For every OCCUPIED uniform-5cm cell, error = |uniform_cell.z_max -
    varla_cell.z_max| where varla_cell is whichever VaRLA cell physically
    contains that uniform cell's centre. Reported as mean / median / p95,
    overall and restricted to cells whose GT majority is an obstacle class.

E3 Range-band breakdown
    E1 and E2 recomputed within each horizontal-distance band (default
    0-10, 10-30, 30-60, 60-100 m, matching the VaRLA ring boundaries).
    The worst band is whichever has the lowest VaRLA obstacle recall.

E4 Point spacing vs range
    Median nearest-neighbour spacing (metres, horizontal x,y only) of the
    raw point cloud within each band, via a KD-tree. This is a property of
    the SENSOR DATA, not of either grid -- included to sanity-check that
    the far-range cell sizes (35/50 cm) are not wildly finer or coarser
    than what the sensor can actually resolve out there.

E5 Ring-schedule ablation (bonus, --ablation)
    Re-runs E1's VaRLA obstacle recall + mean active cells for several
    alternative ring schedules, to show the recall/memory trade-off as a
    scatter plot.

E6 Serialized size + build time (bonus, --benchmark)
    Size in bytes of the grid's cell dict via pickle.dumps (an honest,
    measured number -- NOT a RAM estimate), and wall-clock time for
    insert_points() in milliseconds. CPU model is printed alongside so
    the timing number has necessary context. Labelled "entry / size /
    time", never "RAM", since RAM was not profiled.
-----------------------------------------------------------------------
"""

import argparse
import csv
import glob
import json
import os
import pickle
import platform
import time

import numpy as np
import yaml

from class_mapping import to_super_class
from grid_engine import VariableResolutionGrid

DEFAULT_RINGS = [
    {"name": "near", "r_min": 0.0, "r_max": 10.0, "cell_size": 0.05},
    {"name": "mid", "r_min": 10.0, "r_max": 30.0, "cell_size": 0.20},
    {"name": "far", "r_min": 30.0, "r_max": 60.0, "cell_size": 0.35},
    {"name": "edge", "r_min": 60.0, "r_max": 100.0, "cell_size": 0.50},
]

ABLATION_SCHEDULES = {
    "5-10-20-30": [
        {"name": "near", "r_min": 0.0, "r_max": 10.0, "cell_size": 0.05},
        {"name": "mid", "r_min": 10.0, "r_max": 30.0, "cell_size": 0.10},
        {"name": "far", "r_min": 30.0, "r_max": 60.0, "cell_size": 0.20},
        {"name": "edge", "r_min": 60.0, "r_max": 100.0, "cell_size": 0.30},
    ],
    "5-20-35-50 (default)": DEFAULT_RINGS,
    "10-20-40-60": [
        {"name": "near", "r_min": 0.0, "r_max": 10.0, "cell_size": 0.10},
        {"name": "mid", "r_min": 10.0, "r_max": 30.0, "cell_size": 0.20},
        {"name": "far", "r_min": 30.0, "r_max": 60.0, "cell_size": 0.40},
        {"name": "edge", "r_min": 60.0, "r_max": 100.0, "cell_size": 0.60},
    ],
    "5-15-25-40": [
        {"name": "near", "r_min": 0.0, "r_max": 10.0, "cell_size": 0.05},
        {"name": "mid", "r_min": 10.0, "r_max": 30.0, "cell_size": 0.15},
        {"name": "far", "r_min": 30.0, "r_max": 60.0, "cell_size": 0.25},
        {"name": "edge", "r_min": 60.0, "r_max": 100.0, "cell_size": 0.40},
    ],
    "5-30-50-70": [
        {"name": "near", "r_min": 0.0, "r_max": 10.0, "cell_size": 0.05},
        {"name": "mid", "r_min": 10.0, "r_max": 30.0, "cell_size": 0.30},
        {"name": "far", "r_min": 30.0, "r_max": 60.0, "cell_size": 0.50},
        {"name": "edge", "r_min": 60.0, "r_max": 100.0, "cell_size": 0.70},
    ],
}

OBSTACLE_CLASSES = (1, 2)  # static, dynamic
TERRAIN_CLASS = 0


# --------------------------------------------------------------------- I/O
def uniform_rings(cell_size=0.05, r_max=100.0):
    return [{"name": "uniform", "r_min": 0.0, "r_max": r_max, "cell_size": cell_size}]


def load_points_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_gt_labels(path):
    """SemanticKITTI .label: uint32 per point, lower 16 bits = semantic id."""
    raw = np.fromfile(path, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.int64)


def discover_scans(scans_dir, labels_dir, limit=None):
    """
    Finds every .bin in scans_dir with a matching .label (same stem) in
    labels_dir. Returns a sorted list of (stem, bin_path, label_path).
    Does NOT assume any fixed count -- caller prints however many it finds.
    """
    bin_paths = sorted(glob.glob(os.path.join(scans_dir, "*.bin")))
    pairs = []
    for bp in bin_paths:
        stem = os.path.splitext(os.path.basename(bp))[0]
        lp = os.path.join(labels_dir, stem + ".label")
        if os.path.exists(lp):
            pairs.append((stem, bp, lp))
        else:
            print(f"[eval_accuracy] WARNING: no label file for {stem}, skipping.")
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


# ------------------------------------------------------------ core metrics
def _valid_mask_and_r(xyz, ego_origin, z_clip, max_r):
    x = xyz[:, 0] - ego_origin[0]
    y = xyz[:, 1] - ego_origin[1]
    z = xyz[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    valid = (z >= z_clip[0]) & (z <= z_clip[1]) & (r < max_r)
    return valid, r


def obstacle_recall(grid, xyz, gt_super, mask, obstacle_classes=OBSTACLE_CLASSES):
    """
    Strict (majority) and any-vote recall for whichever points `mask`
    selects (e.g. all obstacle points, or obstacle points within one band).
    Returns (strict_recall, any_vote_recall, n_points).
    """
    sel = np.where(mask)[0]
    total = len(sel)
    if total == 0:
        return float("nan"), float("nan"), 0
    kept = 0
    any_vote = 0
    for i in sel:
        cell = grid.cell_at(float(xyz[i, 0]), float(xyz[i, 1]))
        if cell is None:
            continue
        if cell.dominant_class in obstacle_classes:
            kept += 1
        if any(cell.class_votes[c] > 0 for c in obstacle_classes):
            any_vote += 1
    return kept / total, any_vote / total, total


def false_obstacle_rate(grid, xyz, gt_super, mask, obstacle_classes=OBSTACLE_CLASSES):
    """Fraction of TERRAIN points (selected by `mask`) whose cell majority
    is wrongly an obstacle class."""
    sel = np.where(mask)[0]
    total = len(sel)
    if total == 0:
        return float("nan"), 0
    false_n = 0
    for i in sel:
        cell = grid.cell_at(float(xyz[i, 0]), float(xyz[i, 1]))
        if cell is None:
            continue
        if cell.dominant_class in obstacle_classes:
            false_n += 1
    return false_n / total, total


def height_errors_cm(varla, uniform, obstacle_only_classes=OBSTACLE_CLASSES):
    """
    For every occupied uniform-5cm cell, |uniform.z_max - varla.z_max| at
    the VaRLA cell physically containing that uniform cell's centre.
    Returns (all_errors_cm, obstacle_only_errors_cm) as plain lists.
    """
    all_errs, obstacle_errs = [], []
    for (ring_id, ix, iy), ucell in uniform.cells.items():
        x0, y0, x1, y1 = uniform.cell_world_rect(ring_id, ix, iy)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        vcell = varla.cell_at(cx, cy)
        if vcell is None:
            continue
        err_cm = abs(ucell.z_max - vcell.z_max) * 100.0
        all_errs.append(err_cm)
        if ucell.dominant_class in obstacle_only_classes:
            obstacle_errs.append(err_cm)
    return all_errs, obstacle_errs


def band_edges_from_rings(rings_cfg):
    return [(r["r_min"], r["r_max"]) for r in sorted(rings_cfg, key=lambda r: r["r_min"])]


def median_nn_spacing_by_band(xyz, ego_origin, bands, valid_mask, r_values):
    """Median nearest-neighbour spacing (m, horizontal) per band, via KD-tree."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("[eval_accuracy] scipy not installed -- E4 point-spacing skipped. "
              "Install with: pip install scipy")
        return {band: float("nan") for band in bands}

    xy = xyz[valid_mask][:, :2]
    r = r_values[valid_mask]
    out = {}
    for lo, hi in bands:
        band_mask = (r >= lo) & (r < hi)
        pts = xy[band_mask]
        if len(pts) < 2:
            out[(lo, hi)] = float("nan")
            continue
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        out[(lo, hi)] = float(np.median(dists[:, 1]))
    return out


# --------------------------------------------------------------- per-scan
def evaluate_scan(xyz, gt_labels_raw, rings_cfg, ego_origin=(0.0, 0.0), z_clip=(-3.0, 3.0),
                   benchmark=False):
    """
    Runs VaRLA + Uniform grids on one scan's GT-labelled points and returns
    a flat dict of every metric this module reports for that scan.
    """
    gt_super = to_super_class(gt_labels_raw)
    max_r = max(r["r_max"] for r in rings_cfg)

    varla = VariableResolutionGrid(rings_cfg, ego_origin=ego_origin, z_clip=z_clip)
    uniform = VariableResolutionGrid(uniform_rings(0.05, max_r), ego_origin=ego_origin, z_clip=z_clip)

    t0 = time.perf_counter()
    varla.insert_points(xyz, gt_super)
    t_varla_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    uniform.insert_points(xyz, gt_super)
    t_uniform_ms = (time.perf_counter() - t0) * 1000.0

    valid, r = _valid_mask_and_r(xyz, ego_origin, z_clip, max_r)
    labels_full = np.full(xyz.shape[0], -1, dtype=np.int64)
    labels_full[valid] = gt_super[valid]

    obstacle_mask = valid & np.isin(labels_full, OBSTACLE_CLASSES)
    static_mask = valid & (labels_full == 1)
    dynamic_mask = valid & (labels_full == 2)
    terrain_mask = valid & (labels_full == TERRAIN_CLASS)

    result = {}

    for gname, grid in (("varla", varla), ("uniform", uniform)):
        strict, any_vote, n = obstacle_recall(grid, xyz, labels_full, obstacle_mask)
        result[f"{gname}_recall"] = strict
        result[f"{gname}_recall_any_vote"] = any_vote
        s_strict, s_any, s_n = obstacle_recall(grid, xyz, labels_full, static_mask)
        d_strict, d_any, d_n = obstacle_recall(grid, xyz, labels_full, dynamic_mask)
        result[f"{gname}_recall_static"] = s_strict
        result[f"{gname}_recall_dynamic"] = d_strict
        fo_rate, fo_n = false_obstacle_rate(grid, xyz, labels_full, terrain_mask)
        result[f"{gname}_false_obstacle_rate"] = fo_rate
        result[f"{gname}_active_cells"] = grid.sparse_cell_count()

    all_errs, obs_errs = height_errors_cm(varla, uniform)
    result["mean_height_error_cm"] = float(np.mean(all_errs)) if all_errs else float("nan")
    result["median_height_error_cm"] = float(np.median(all_errs)) if all_errs else float("nan")
    result["p95_height_error_cm"] = float(np.percentile(all_errs, 95)) if all_errs else float("nan")
    result["mean_height_error_cm_obstacle_only"] = float(np.mean(obs_errs)) if obs_errs else float("nan")

    # E3: per-band breakdown
    bands = band_edges_from_rings(rings_cfg)
    band_rows = []
    for lo, hi in bands:
        band_sel = valid & (r >= lo) & (r < hi)
        band_obstacle = band_sel & np.isin(labels_full, OBSTACLE_CLASSES)
        row = {"band_lo": lo, "band_hi": hi}
        for gname, grid in (("varla", varla), ("uniform", uniform)):
            strict, any_vote, n = obstacle_recall(grid, xyz, labels_full, band_obstacle)
            row[f"{gname}_recall"] = strict
            row[f"{gname}_n_obstacle_points"] = n
        band_rows.append(row)

    result["_band_rows"] = band_rows
    result["_n_points_total"] = int(xyz.shape[0])
    result["_n_points_valid"] = int(valid.sum())
    result["_n_obstacle_points"] = int(obstacle_mask.sum())
    result["_n_terrain_points"] = int(terrain_mask.sum())

    if benchmark:
        result["_varla_build_ms"] = t_varla_ms
        result["_uniform_build_ms"] = t_uniform_ms
        result["_varla_size_bytes"] = len(pickle.dumps(varla.cells))
        result["_uniform_size_bytes"] = len(pickle.dumps(uniform.cells))

    return result, varla, uniform, valid, r, labels_full


# -------------------------------------------------------------- plotting
def _plot_recall_by_band(band_summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{int(b['band_lo'])}-{int(b['band_hi'])} m" for b in band_summary]
    varla_vals = [b["varla_recall_mean"] for b in band_summary]
    uniform_vals = [b["uniform_recall_mean"] for b in band_summary]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(x - width / 2, varla_vals, width, label="VaRLA", color="#2f6fed")
    ax.bar(x + width / 2, uniform_vals, width, label="Sparse uniform 5 cm", color="#e0a100")

    ax.set_ylabel("Obstacle recall", fontsize=16)
    ax.set_xlabel("Range band (horizontal distance from sensor)", fontsize=16)
    ax.set_title("Obstacle recall by range band: VaRLA vs Sparse uniform 5 cm", fontsize=17, pad=45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=13, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.14), ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def _plot_ablation_tradeoff(ablation_rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.set_facecolor("white")
    for row in ablation_rows:
        ax.scatter(row["mean_active_cells"], row["mean_recall"], s=140, color="#2f6fed")
        ax.annotate(row["schedule_name"], (row["mean_active_cells"], row["mean_recall"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=12)

    ax.set_xlabel("Mean active cells per scan", fontsize=16)
    ax.set_ylabel("Mean obstacle recall", fontsize=16)
    ax.set_title("Ring-schedule trade-off: recall vs active cells", fontsize=18)
    ax.tick_params(labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------- driver
def run_eval(scans_dir, labels_dir, rings_cfg=None, out_dir="results", limit=None,
             ablation=False, benchmark=False, sequence_name=None,
             ego_origin=(0.0, 0.0), z_clip=(-3.0, 3.0)):
    rings_cfg = rings_cfg or DEFAULT_RINGS
    os.makedirs(out_dir, exist_ok=True)

    pairs = discover_scans(scans_dir, labels_dir, limit=limit)
    if not pairs:
        raise FileNotFoundError(
            f"No .bin/.label pairs found under scans_dir={scans_dir}, labels_dir={labels_dir}"
        )

    seq_label = sequence_name or os.path.basename(os.path.dirname(scans_dir.rstrip("/"))) or "unknown"
    frame_indices = [stem for stem, _, _ in pairs]
    print(f"[eval_accuracy] Sequence: {seq_label}")
    print(f"[eval_accuracy] Using {len(pairs)} scans, frame indices: "
          f"{frame_indices[0]}..{frame_indices[-1]} "
          f"({', '.join(frame_indices)})")
    print(f"[eval_accuracy] Ring schedule: "
          + ", ".join(f"{r['name']} {r['r_min']}-{r['r_max']}m@{int(r['cell_size']*100)}cm" for r in rings_cfg))

    bands = band_edges_from_rings(rings_cfg)
    per_scan_rows = []
    band_accum = {band: {"varla_recall": [], "uniform_recall": []} for band in bands}
    spacing_accum = {band: [] for band in bands}

    for stem, bin_path, label_path in pairs:
        pts = load_points_bin(bin_path)
        xyz = pts[:, :3]
        gt_raw = load_gt_labels(label_path)
        if gt_raw.shape[0] != xyz.shape[0]:
            print(f"[eval_accuracy] WARNING: {stem} has {xyz.shape[0]} points but "
                  f"{gt_raw.shape[0]} labels -- skipping this scan.")
            continue

        result, varla, uniform, valid, r, labels_full = evaluate_scan(
            xyz, gt_raw, rings_cfg, ego_origin=ego_origin, z_clip=z_clip, benchmark=benchmark
        )

        spacing = median_nn_spacing_by_band(xyz, ego_origin, bands, valid, r)
        for band in bands:
            spacing_accum[band].append(spacing[band])

        for band_row in result["_band_rows"]:
            band_key = (band_row["band_lo"], band_row["band_hi"])
            band_accum[band_key]["varla_recall"].append(band_row["varla_recall"])
            band_accum[band_key]["uniform_recall"].append(band_row["uniform_recall"])

        row = {"scan": stem}
        row.update({k: v for k, v in result.items() if not k.startswith("_")})
        row["n_points_total"] = result["_n_points_total"]
        row["n_points_valid"] = result["_n_points_valid"]
        row["n_obstacle_points"] = result["_n_obstacle_points"]
        row["n_terrain_points"] = result["_n_terrain_points"]
        if benchmark:
            row["varla_build_ms"] = result["_varla_build_ms"]
            row["uniform_build_ms"] = result["_uniform_build_ms"]
            row["varla_size_bytes"] = result["_varla_size_bytes"]
            row["uniform_size_bytes"] = result["_uniform_size_bytes"]
        per_scan_rows.append(row)

    if not per_scan_rows:
        raise RuntimeError("No scans were successfully evaluated (all skipped due to point/label mismatch).")

    # ---- write per-scan CSV ----
    csv_path = os.path.join(out_dir, "accuracy_per_scan.csv")
    fieldnames = list(per_scan_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_scan_rows:
            w.writerow(row)
    print(f"[eval_accuracy] Wrote {csv_path}")

    # ---- band summary (mean over scans) ----
    band_summary = []
    for (lo, hi) in bands:
        vr = [v for v in band_accum[(lo, hi)]["varla_recall"] if not np.isnan(v)]
        ur = [v for v in band_accum[(lo, hi)]["uniform_recall"] if not np.isnan(v)]
        sp = [v for v in spacing_accum[(lo, hi)] if not np.isnan(v)]
        band_summary.append({
            "band_lo": lo, "band_hi": hi,
            "varla_recall_mean": float(np.mean(vr)) if vr else float("nan"),
            "varla_recall_std": float(np.std(vr)) if vr else float("nan"),
            "uniform_recall_mean": float(np.mean(ur)) if ur else float("nan"),
            "uniform_recall_std": float(np.std(ur)) if ur else float("nan"),
            "median_point_spacing_m": float(np.median(sp)) if sp else float("nan"),
        })

    valid_bands = [b for b in band_summary if not np.isnan(b["varla_recall_mean"])]
    worst_band = min(valid_bands, key=lambda b: b["varla_recall_mean"]) if valid_bands else None

    def _mean(key):
        vals = [row[key] for row in per_scan_rows if not np.isnan(row.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(key):
        vals = [row[key] for row in per_scan_rows if not np.isnan(row.get(key, float("nan")))]
        return float(np.std(vals)) if vals else float("nan")

    summary = {
        "sequence": seq_label,
        "n_scans": len(per_scan_rows),
        "frame_indices": frame_indices,
        "ring_schedule": rings_cfg,
        "varla_recall": round(_mean("varla_recall"), 4),
        "varla_recall_std": round(_std("varla_recall"), 4),
        "uniform_recall": round(_mean("uniform_recall"), 4),
        "uniform_recall_std": round(_std("uniform_recall"), 4),
        "varla_recall_static": round(_mean("varla_recall_static"), 4),
        "uniform_recall_static": round(_mean("uniform_recall_static"), 4),
        "varla_recall_dynamic": round(_mean("varla_recall_dynamic"), 4),
        "uniform_recall_dynamic": round(_mean("uniform_recall_dynamic"), 4),
        "varla_false_obstacle_rate": round(_mean("varla_false_obstacle_rate"), 4),
        "uniform_false_obstacle_rate": round(_mean("uniform_false_obstacle_rate"), 4),
        "mean_height_error_cm": round(_mean("mean_height_error_cm"), 3),
        "median_height_error_cm": round(_mean("median_height_error_cm"), 3),
        "p95_height_error_cm": round(_mean("p95_height_error_cm"), 3),
        "mean_height_error_cm_obstacle_only": round(_mean("mean_height_error_cm_obstacle_only"), 3),
        "varla_active_cells_mean": round(_mean("varla_active_cells"), 1),
        "uniform_active_cells_mean": round(_mean("uniform_active_cells"), 1),
        "worst_band_m": f"{int(worst_band['band_lo'])}-{int(worst_band['band_hi'])}" if worst_band else None,
        "worst_band_recall": round(worst_band["varla_recall_mean"], 4) if worst_band else None,
        "per_band": band_summary,
    }

    if benchmark:
        summary["varla_build_ms_mean"] = round(_mean("varla_build_ms"), 3)
        summary["uniform_build_ms_mean"] = round(_mean("uniform_build_ms"), 3)
        summary["varla_size_bytes_mean"] = round(_mean("varla_size_bytes"), 1)
        summary["uniform_size_bytes_mean"] = round(_mean("uniform_size_bytes"), 1)
        summary["cpu_model"] = platform.processor() or platform.uname().processor or "unknown"

    # worst scan, honestly reported
    worst_scan = min(per_scan_rows, key=lambda r: r["varla_recall"] if not np.isnan(r["varla_recall"]) else 1.0)
    summary["worst_scan"] = worst_scan["scan"]
    summary["worst_scan_varla_recall"] = round(worst_scan["varla_recall"], 4)

    json_path = os.path.join(out_dir, "accuracy_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval_accuracy] Wrote {json_path}")

    # ---- band CSV (separate file, keeps per-scan CSV columns sane) ----
    band_csv_path = os.path.join(out_dir, "accuracy_per_band.csv")
    with open(band_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(band_summary[0].keys()))
        w.writeheader()
        for row in band_summary:
            w.writerow(row)
    print(f"[eval_accuracy] Wrote {band_csv_path}")

    # ---- PNG 1: recall by band ----
    png1_path = os.path.join(out_dir, "recall_by_band.png")
    _plot_recall_by_band(band_summary, png1_path)
    print(f"[eval_accuracy] Wrote {png1_path}")

    # ---- E5 ablation (optional) ----
    if ablation:
        print("[eval_accuracy] Running E5 ring-schedule ablation...")
        ablation_rows = []
        for sched_name, sched_rings in ABLATION_SCHEDULES.items():
            recalls, cell_counts = [], []
            for stem, bin_path, label_path in pairs:
                pts = load_points_bin(bin_path)
                xyz = pts[:, :3]
                gt_raw = load_gt_labels(label_path)
                if gt_raw.shape[0] != xyz.shape[0]:
                    continue
                gt_super = to_super_class(gt_raw)
                max_r = max(r["r_max"] for r in sched_rings)
                grid = VariableResolutionGrid(sched_rings, ego_origin=ego_origin, z_clip=z_clip)
                grid.insert_points(xyz, gt_super)
                valid, r = _valid_mask_and_r(xyz, ego_origin, z_clip, max_r)
                labels_full = np.full(xyz.shape[0], -1, dtype=np.int64)
                labels_full[valid] = gt_super[valid]
                obstacle_mask = valid & np.isin(labels_full, OBSTACLE_CLASSES)
                strict, _, _ = obstacle_recall(grid, xyz, labels_full, obstacle_mask)
                if not np.isnan(strict):
                    recalls.append(strict)
                cell_counts.append(grid.sparse_cell_count())
            ablation_rows.append({
                "schedule_name": sched_name,
                "mean_recall": float(np.mean(recalls)) if recalls else float("nan"),
                "mean_active_cells": float(np.mean(cell_counts)) if cell_counts else float("nan"),
            })
            print(f"  {sched_name}: mean_recall={ablation_rows[-1]['mean_recall']:.4f}, "
                  f"mean_active_cells={ablation_rows[-1]['mean_active_cells']:.0f}")

        png2_path = os.path.join(out_dir, "schedule_tradeoff.png")
        _plot_ablation_tradeoff(ablation_rows, png2_path)
        print(f"[eval_accuracy] Wrote {png2_path}")

        ablation_json_path = os.path.join(out_dir, "ablation_summary.json")
        with open(ablation_json_path, "w") as f:
            json.dump(ablation_rows, f, indent=2)
        print(f"[eval_accuracy] Wrote {ablation_json_path}")
    else:
        print("[eval_accuracy] Skipping E5 ablation (pass --ablation to run it).")

    if not benchmark:
        print("[eval_accuracy] Skipping E6 benchmark (pass --benchmark to run it).")

    # ---- the five headline numbers for the slide ----
    print("\n[eval_accuracy] ===== SLIDE NUMBERS =====")
    print(f"  1. VaRLA obstacle recall      : {summary['varla_recall']}")
    print(f"  2. Uniform 5cm obstacle recall: {summary['uniform_recall']}")
    print(f"  3. Mean height error (cm)     : {summary['mean_height_error_cm']}")
    print(f"  4. Worst range band           : {summary['worst_band_m']} m")
    print(f"  5. Worst band VaRLA recall    : {summary['worst_band_recall']}")
    print("==========================================\n")

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans_dir", default="data/finetune_dataset/sequences/00/velodyne")
    ap.add_argument("--labels_dir", default="data/finetune_dataset/sequences/00/labels")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of scans (default: all found)")
    ap.add_argument("--rings", default=None,
                     help="Override ring schedule, e.g. '0:10:5,10:30:20,30:60:35,60:100:50' "
                          "(r_min:r_max:cell_cm per ring, comma-separated). Default matches config.yaml.")
    ap.add_argument("--sequence_name", default=None)
    ap.add_argument("--ablation", action="store_true", help="Run E5 ring-schedule ablation + scatter plot")
    ap.add_argument("--benchmark", action="store_true", help="Run E6 size/time measurement")
    args = ap.parse_args()

    rings_cfg = DEFAULT_RINGS
    if args.rings:
        rings_cfg = []
        for i, part in enumerate(args.rings.split(",")):
            r_min, r_max, cell_cm = part.split(":")
            rings_cfg.append({
                "name": f"ring{i}", "r_min": float(r_min), "r_max": float(r_max),
                "cell_size": float(cell_cm) / 100.0,
            })

    run_eval(args.scans_dir, args.labels_dir, rings_cfg=rings_cfg, out_dir=args.out_dir,
              limit=args.limit, ablation=args.ablation, benchmark=args.benchmark,
              sequence_name=args.sequence_name)


if __name__ == "__main__":
    main()

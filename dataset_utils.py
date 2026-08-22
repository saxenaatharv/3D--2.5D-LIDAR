"""
dataset_utils.py
------------------
Helpers to bring an arbitrary point cloud + label dataset into the folder
layout Open3D-ML's SemanticKITTI dataset loader expects, so the same
pretrained-model fine-tuning pipeline works no matter where your data
originally came from.

Expected OUTPUT layout (created by convert_folder):
    <finetune_data_dir>/
        sequences/
            00/
                velodyne/   *.bin    float32, N x 4: x, y, z, intensity
                labels/     *.label  uint32 per point, SemanticKITTI-style id

INPUT expectations:
    raw_pc_dir     -> one point cloud file per frame (.bin / .npy / .pcd / .ply)
    raw_label_dir  -> one .npy file per frame with the SAME stem as the point
                       cloud file, containing an integer label per point.
                       These can be your own class ids (see RAW_TO_KITTI).

If your raw labels are already 0=terrain / 1=static / 2=dynamic, leave
RAW_TO_KITTI as-is: it maps each super-class to one representative
SemanticKITTI id so the pretrained decoder head still gets a sensible
supervision target during fine-tuning. If your raw labels are something
else entirely, edit RAW_TO_KITTI to point at the nearest SemanticKITTI
class (see class_mapping.SEMANTIC_KITTI_NAMES for the full id table).
"""

import os
import glob
import numpy as np

RAW_TO_KITTI = {0: 9, 1: 13, 2: 1}   # terrain->road(9), static->building(13), dynamic->car(1)


def convert_folder(raw_pc_dir, raw_label_dir, out_dir, seq="00", raw_label_map=None):
    raw_label_map = raw_label_map or RAW_TO_KITTI
    velo_out = os.path.join(out_dir, "sequences", seq, "velodyne")
    label_out = os.path.join(out_dir, "sequences", seq, "labels")
    os.makedirs(velo_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    pc_files = sorted(glob.glob(os.path.join(raw_pc_dir, "*")))
    if not pc_files:
        raise FileNotFoundError(f"No point cloud files found in {raw_pc_dir}")

    for i, pc_path in enumerate(pc_files):
        stem = os.path.splitext(os.path.basename(pc_path))[0]
        label_path = os.path.join(raw_label_dir, stem + ".npy")

        pts = _load_points_any(pc_path)
        if pts.shape[1] == 3:
            intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
            pts = np.concatenate([pts, intensity], axis=1)
        pts.astype(np.float32).tofile(os.path.join(velo_out, f"{i:06d}.bin"))

        if os.path.exists(label_path):
            raw_labels = np.load(label_path).astype(np.int64)
            kitti_labels = np.vectorize(lambda v: raw_label_map.get(int(v), 0))(raw_labels)
            kitti_labels.astype(np.uint32).tofile(os.path.join(label_out, f"{i:06d}.label"))
        else:
            print(f"[dataset_utils] WARNING: no label file for {stem}, skipping label export.")

    print(f"[dataset_utils] Wrote {len(pc_files)} frames to {out_dir}")


def _load_points_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    if ext == ".npy":
        return np.load(path)[:, :3]
    if ext in (".pcd", ".ply"):
        import open3d as o3d
        pc = o3d.io.read_point_cloud(path)
        return np.asarray(pc.points)
    raise ValueError(f"Unsupported point cloud format: {ext}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Convert a custom lidar dataset into SemanticKITTI layout.")
    ap.add_argument("--raw_pc_dir", required=True)
    ap.add_argument("--raw_label_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    convert_folder(args.raw_pc_dir, args.raw_label_dir, args.out_dir)

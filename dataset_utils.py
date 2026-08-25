"""
dataset_utils.py
------------------
Helpers to bring an arbitrary point cloud + label dataset into the folder
layout Open3D-ML's SemanticKITTI dataset loader expects, so the same
pretrained-model fine-tuning pipeline works no matter where your data
originally came from.

Expected OUTPUT layout (created by convert_folder / convert_folder_split):
    <finetune_data_dir>/
        sequences/
            00/
                velodyne/   *.bin    float32, N x 4: x, y, z, intensity
                labels/     *.label  uint32 per point, SemanticKITTI-style id
            01/             (only created by convert_folder_split)
                velodyne/   ...
                labels/     ...

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
    """
    Original behaviour: dumps every frame into a single sequence ("00").
    Kept for backwards compatibility -- prefer convert_folder_split() below
    when you actually want a held-out train/test split.
    """
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


def convert_folder_split(raw_pc_dir, raw_label_dir, out_dir, train_ratio=0.9, seed=42, raw_label_map=None):
    """
    Like convert_folder, but produces a REAL held-out split by writing two
    SemanticKITTI-style sequences (Open3D-ML's SemanticKITTI loader splits
    by sequence folder, not by percentage, so this is the correct way to
    get a genuine train/test split rather than training and testing on the
    same frames):

        sequences/00/  <- train_ratio fraction of frames (e.g. 90%) -> TRAIN
        sequences/01/  <- remaining frames                (e.g. 10%) -> TEST

    Frames are shuffled with a fixed seed before splitting, so the split is
    reproducible across runs instead of just taking "first N frames".
    """
    raw_label_map = raw_label_map or RAW_TO_KITTI
    pc_files = sorted(glob.glob(os.path.join(raw_pc_dir, "*")))
    if not pc_files:
        raise FileNotFoundError(f"No point cloud files found in {raw_pc_dir}")

    rng = np.random.default_rng(seed)
    order = np.arange(len(pc_files))
    rng.shuffle(order)
    n_train = int(round(len(order) * train_ratio))
    train_indices = sorted(order[:n_train].tolist())
    test_indices = sorted(order[n_train:].tolist())

    def _write(indices, seq):
        velo_out = os.path.join(out_dir, "sequences", seq, "velodyne")
        label_out = os.path.join(out_dir, "sequences", seq, "labels")
        os.makedirs(velo_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)

        for out_i, orig_i in enumerate(indices):
            pc_path = pc_files[orig_i]
            stem = os.path.splitext(os.path.basename(pc_path))[0]
            label_path = os.path.join(raw_label_dir, stem + ".npy")

            pts = _load_points_any(pc_path)
            if pts.shape[1] == 3:
                intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
                pts = np.concatenate([pts, intensity], axis=1)
            pts.astype(np.float32).tofile(os.path.join(velo_out, f"{out_i:06d}.bin"))

            if os.path.exists(label_path):
                raw_labels = np.load(label_path).astype(np.int64)
                kitti_labels = np.vectorize(lambda v: raw_label_map.get(int(v), 0))(raw_labels)
                kitti_labels.astype(np.uint32).tofile(os.path.join(label_out, f"{out_i:06d}.label"))
            else:
                print(f"[dataset_utils] WARNING: no label file for {stem}, skipping label export.")

        return len(indices)

    n_tr = _write(train_indices, "00")
    n_te = _write(test_indices, "01")
    print(f"[dataset_utils] Split {len(pc_files)} frames -> {n_tr} train (sequence 00) / "
          f"{n_te} test (sequence 01), train_ratio={train_ratio}, seed={seed}")


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
    ap.add_argument("--train_ratio", type=float, default=None,
                     help="If set, uses convert_folder_split (90/10-style split) instead of dumping "
                          "everything into one sequence. e.g. --train_ratio 0.9")
    args = ap.parse_args()
    if args.train_ratio is not None:
        convert_folder_split(args.raw_pc_dir, args.raw_label_dir, args.out_dir, train_ratio=args.train_ratio)
    else:
        convert_folder(args.raw_pc_dir, args.raw_label_dir, args.out_dir)

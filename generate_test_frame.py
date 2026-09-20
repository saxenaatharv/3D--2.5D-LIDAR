"""
generate_test_frame.py
------------------------
Creates a synthetic, KITTI-style .bin point cloud purely so you can
smoke-test the pipeline (checkpoint loads, model runs, grid engine builds,
dashboard renders) BEFORE you plug in real DRDO lidar data.

This is not meant to give meaningful semantic predictions -- it is only
meant to prove the wiring works end to end.
"""

import os
import numpy as np

OUT_DIR = "data/inference_frames"
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(0)

# Flat "ground" ring out to 40m
n_ground = 20000
r = rng.uniform(1, 40, n_ground)
theta = rng.uniform(0, 2 * np.pi, n_ground)
gx, gy = r * np.cos(theta), r * np.sin(theta)
gz = rng.normal(-1.6, 0.02, n_ground)  # sensor ~1.6m above ground

# A few vertical "obstacle" clusters (poles/walls)
n_obs = 4000
ox = rng.uniform(-15, 15, n_obs)
oy = rng.uniform(5, 30, n_obs)
oz = rng.uniform(-1.5, 1.5, n_obs)

x = np.concatenate([gx, ox])
y = np.concatenate([gy, oy])
z = np.concatenate([gz, oz])
intensity = rng.uniform(0, 1, x.shape[0]).astype(np.float32)

pts = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
out_path = os.path.join(OUT_DIR, "000000.bin")
pts.tofile(out_path)

print(f"Wrote {pts.shape[0]} synthetic points to {out_path}")
print("You can now run: python run_inference.py --config config.yaml")

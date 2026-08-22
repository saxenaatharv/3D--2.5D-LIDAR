"""
grid_engine.py
---------------
Variable Resolution 2.5D Grid Engine.

Implements a foveated (human-vision-like) elevation + semantic map. Space
around the ego sensor is split into concentric radial "rings"; each ring
owns its own cell size, so resolution degrades smoothly with distance
instead of relying on one fixed-size matrix (which would either waste
memory near the sensor or under-sample far away).

Data structure: a Python dict keyed by (ring_id, ix, iy) -> CellStats.
This sparse hash-grid means empty space costs ~0 memory, and every ring
is addressed with its own local integer coordinate system, so there is
no aliasing / alignment error between rings of different resolution --
each ring's cells simply exist at whatever radius produced them.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class CellStats:
    z_min: float = 1e9
    z_max: float = -1e9
    z_sum: float = 0.0
    count: int = 0
    class_votes: list = field(default_factory=lambda: [0, 0, 0])  # terrain, static, dynamic

    def update(self, z, super_class):
        self.z_min = min(self.z_min, z)
        self.z_max = max(self.z_max, z)
        self.z_sum += z
        self.count += 1
        if super_class in (0, 1, 2):
            self.class_votes[super_class] += 1

    @property
    def z_mean(self):
        return self.z_sum / self.count if self.count else 0.0

    @property
    def dominant_class(self):
        if self.count == 0:
            return -1
        return int(np.argmax(self.class_votes))


class VariableResolutionGrid:
    def __init__(self, rings_cfg, ego_origin=(0.0, 0.0), z_clip=(-3.0, 3.0)):
        """
        rings_cfg: list of dicts with keys r_min, r_max, cell_size
        """
        self.rings = sorted(rings_cfg, key=lambda r: r["r_min"])
        self.ego_x, self.ego_y = ego_origin
        self.z_min_clip, self.z_max_clip = z_clip
        self.cells = {}   # (ring_id, ix, iy) -> CellStats
        self._max_r = self.rings[-1]["r_max"]

    def reset(self):
        self.cells.clear()

    def _find_ring(self, r):
        for idx, ring in enumerate(self.rings):
            if ring["r_min"] <= r < ring["r_max"]:
                return idx, ring
        return None, None

    def insert_points(self, xyz, super_class_labels):
        """
        xyz: (N,3) float array in the ego/sensor frame
        super_class_labels: (N,) int array, values in {-1,0,1,2}
        """
        xyz = np.asarray(xyz)
        x = xyz[:, 0] - self.ego_x
        y = xyz[:, 1] - self.ego_y
        z = xyz[:, 2]

        valid = (z >= self.z_min_clip) & (z <= self.z_max_clip)
        r = np.sqrt(x ** 2 + y ** 2)
        valid &= r < self._max_r
        x, y, z, r = x[valid], y[valid], z[valid], r[valid]
        labels = np.asarray(super_class_labels)[valid]

        for xi, yi, zi, ri, li in zip(x, y, z, r, labels):
            ring_id, ring = self._find_ring(ri)
            if ring is None:
                continue
            cs = ring["cell_size"]
            ix = int(np.floor(xi / cs))
            iy = int(np.floor(yi / cs))
            key = (ring_id, ix, iy)
            cell = self.cells.get(key)
            if cell is None:
                cell = CellStats()
                self.cells[key] = cell
            cell.update(float(zi), int(li))

    def cell_world_rect(self, ring_id, ix, iy):
        """Return (x0, y0, x1, y1) world-frame corners of a cell, for rendering."""
        cs = self.rings[ring_id]["cell_size"]
        x0 = self.ego_x + ix * cs
        y0 = self.ego_y + iy * cs
        return x0, y0, x0 + cs, y0 + cs

    # ---- memory accounting, consumed by metrics.py ----
    def sparse_cell_count(self):
        return len(self.cells)

    def equivalent_uniform_cell_count(self, uniform_cell_size=None):
        """
        Cells a single fixed-resolution grid covering the same max radius
        would need, at the finest cell size actually used. Demonstrates the
        memory saving of the foveated approach.
        """
        cs = uniform_cell_size or self.rings[0]["cell_size"]
        span = 2 * self._max_r
        side = int(np.ceil(span / cs))
        return side * side

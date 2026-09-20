"""
tests/test_eval_accuracy.py
------------------------------
Unit tests for eval_accuracy.py on a tiny, hand-built synthetic scene.
No real dataset required -- run with:

    python -m unittest tests/test_eval_accuracy.py -v

or

    python -m pytest tests/test_eval_accuracy.py -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid_engine import VariableResolutionGrid
from eval_accuracy import (
    DEFAULT_RINGS,
    uniform_rings,
    obstacle_recall,
    evaluate_scan,
    band_edges_from_rings,
)


class TestRingBoundary(unittest.TestCase):
    """A point sitting exactly on a ring boundary must be assigned to
    exactly one ring -- never zero, never two."""

    def test_boundary_point_assigned_to_exactly_one_ring(self):
        grid = VariableResolutionGrid(DEFAULT_RINGS, ego_origin=(0.0, 0.0), z_clip=(-3.0, 3.0))
        # r_min <= r < r_max means the boundary belongs to the ring where it
        # is the LOWER edge (r_min), not the ring above where it would be
        # r_max (exclusive). This is exactly what _find_ring enforces.
        for boundary_r in (10.0, 30.0, 60.0):
            key = grid.locate(boundary_r, 0.0)
            self.assertIsNotNone(key, f"boundary point at r={boundary_r} was not assigned to any ring")
            ring_id = key[0]
            ring = grid.rings[ring_id]
            self.assertEqual(ring["r_min"], boundary_r,
                              f"r={boundary_r} should land in the ring where it is r_min, "
                              f"got ring '{ring['name']}' ({ring['r_min']}-{ring['r_max']})")

    def test_point_at_max_radius_is_unassigned(self):
        """r == max_r is outside every ring (r_max is exclusive on the last ring too)."""
        grid = VariableResolutionGrid(DEFAULT_RINGS, ego_origin=(0.0, 0.0), z_clip=(-3.0, 3.0))
        key = grid.locate(100.0, 0.0)
        self.assertIsNone(key)


class TestSyntheticSceneRecall(unittest.TestCase):
    """
    Tiny hand-built scene: a handful of terrain points plus a small cluster
    of obstacle points placed FAR from the sensor (inside the 60-100m ring,
    where VaRLA uses 50cm cells vs uniform's 5cm). The obstacle cluster is
    deliberately split across two adjacent uniform-5cm cells but falls
    inside ONE coarse 50cm VaRLA cell that is also flooded with many more
    terrain points -- so VaRLA's majority vote for that cell tips to
    terrain (losing the obstacle), while the uniform grid's finer cells
    keep the obstacle points in cells the terrain doesn't dominate.
    """

    def setUp(self):
        # Deterministic construction, computed directly from the grid's own
        # cell-index formula (floor(coord / cell_size)) so the scene's
        # behaviour is guaranteed rather than hoped-for:
        #
        # VaRLA "edge" ring (60-100m) uses 50cm cells; uniform grid uses 5cm
        # cells everywhere. We place 6 terrain points and 2 obstacle points
        # that all land in the SAME single VaRLA cell (ix=150, iy=0 at
        # cs=0.5) -- so VaRLA's majority vote there is terrain (6 > 2),
        # losing the obstacle. Under the uniform 5cm grid the two obstacle
        # points fall into their OWN cell (ix=1504, iy=4 at cs=0.05) that no
        # terrain point shares, so that cell's majority is obstacle and both
        # points are kept.
        terrain_x = [75.0, 75.05, 75.10, 75.30, 75.35, 75.40]
        terrain_y = [0.05] * 6
        obstacle_x = [75.21, 75.22]
        obstacle_y = [0.21, 0.22]

        x = np.array(terrain_x + obstacle_x, dtype=np.float32)
        y = np.array(terrain_y + obstacle_y, dtype=np.float32)
        z = np.array([-1.6] * 6 + [0.5, 0.5], dtype=np.float32)
        self.xyz = np.stack([x, y, z], axis=1)

        self.gt_super = np.array([0] * 6 + [1, 1], dtype=np.int64)  # 0=terrain, 1=static obstacle

    def test_recall_is_bounded(self):
        result, varla, uniform, valid, r, labels_full = evaluate_scan(
            self.xyz, self._raw_kitti_labels(), DEFAULT_RINGS
        )
        for key in ("varla_recall", "uniform_recall", "varla_recall_static", "uniform_recall_static"):
            val = result[key]
            self.assertTrue(0.0 <= val <= 1.0, f"{key}={val} out of [0,1] bounds")

    def test_uniform_recall_at_least_varla_recall_in_far_band(self):
        """The whole point of this synthetic scene: at 50cm VaRLA cells,
        a coarse cell dominated by terrain should swallow the obstacle
        vote, while uniform 5cm cells keep it separate -- so uniform
        recall must be >= VaRLA recall here."""
        result, varla, uniform, valid, r, labels_full = evaluate_scan(
            self.xyz, self._raw_kitti_labels(), DEFAULT_RINGS
        )
        self.assertGreaterEqual(
            result["uniform_recall"], result["varla_recall"],
            "uniform 5cm grid should retain at least as much obstacle detail "
            "as the coarser VaRLA far-range cells in this synthetic scene"
        )

    def _raw_kitti_labels(self):
        # map our super-class scheme back to one representative SemanticKITTI
        # id per class, since evaluate_scan expects raw KITTI ids and calls
        # to_super_class() itself (mirrors dataset_utils.RAW_TO_KITTI).
        raw = np.where(self.gt_super == 1, 13, 9)  # 13=building(static), 9=road(terrain)
        return raw.astype(np.int64)


class TestBandEdges(unittest.TestCase):
    def test_band_edges_match_ring_schedule(self):
        bands = band_edges_from_rings(DEFAULT_RINGS)
        self.assertEqual(bands, [(0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 100.0)])


if __name__ == "__main__":
    unittest.main()

"""
visualize_dashboard.py
------------------------
Real-time dashboard renderer for the Adaptive 2.5D Lidar Map. Draws every
occupied cell of a VariableResolutionGrid as a rectangle, color-coded by
dominant class and shaded by elevation, with an FPS / active-cell overlay
and the concentric ring boundaries so the foveation is visible at a glance.
Pure OpenCV -- runs in a normal desktop window on Windows, no extra
display server required.
"""

import time
import numpy as np
import cv2

from class_mapping import SUPER_CLASS_COLOR_BGR


class Dashboard:
    def __init__(self, grid, image_size=(900, 900), window_name="Adaptive 2.5D Lidar Map"):
        self.grid = grid
        self.w, self.h = image_size
        self.window_name = window_name
        self.px_per_m = self.w / (2 * grid._max_r)
        self._t_prev = time.time()
        self._fps = 0.0
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _world_to_px(self, x, y):
        px = int(self.w / 2 + x * self.px_per_m)
        py = int(self.h / 2 - y * self.px_per_m)
        return px, py

    def _tick_fps(self):
        now = time.time()
        dt = now - self._t_prev
        self._t_prev = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        self._fps = 0.9 * self._fps + 0.1 * inst_fps if self._fps else inst_fps
        return self._fps

    def render(self, extra_text=None, wait_ms=1):
        img = np.full((self.h, self.w, 3), 20, dtype=np.uint8)

        for (ring_id, ix, iy), cell in self.grid.cells.items():
            x0, y0, x1, y1 = self.grid.cell_world_rect(ring_id, ix, iy)
            p0 = self._world_to_px(x0, y1)
            p1 = self._world_to_px(x1, y0)
            base_color = np.array(SUPER_CLASS_COLOR_BGR[cell.dominant_class], dtype=np.float32)
            elev_norm = np.clip(
                (cell.z_mean - self.grid.z_min_clip) / (self.grid.z_max_clip - self.grid.z_min_clip + 1e-6),
                0, 1,
            )
            shade = 0.5 + 0.5 * elev_norm
            color = tuple(int(c) for c in np.clip(base_color * shade, 0, 255))
            cv2.rectangle(img, p0, p1, color, thickness=-1)

        # ego marker
        cx, cy = self._world_to_px(0, 0)
        cv2.circle(img, (cx, cy), 6, (255, 255, 0), -1)

        # ring boundaries, so the foveation is visually obvious
        for ring in self.grid.rings:
            radius_px = int(ring["r_max"] * self.px_per_m)
            cv2.circle(img, (cx, cy), radius_px, (90, 90, 90), 1)

        fps = self._tick_fps()
        cv2.putText(img, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.putText(img, f"Active cells: {self.grid.sparse_cell_count()}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if extra_text:
            for i, line in enumerate(extra_text):
                cv2.putText(img, line, (15, 90 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (200, 200, 200), 1)

        # legend: terrain / static obstacle / dynamic object colour key
        legend_items = [
            ("Terrain", SUPER_CLASS_COLOR_BGR[0]),
            ("Static Obstacle", SUPER_CLASS_COLOR_BGR[1]),
            ("Dynamic Object", SUPER_CLASS_COLOR_BGR[2]),
        ]
        legend_y0 = self.h - 20 * len(legend_items) - 15
        for i, (label, color) in enumerate(legend_items):
            ly = legend_y0 + i * 22
            cv2.rectangle(img, (15, ly), (35, ly + 16), color, thickness=-1)
            cv2.putText(img, label, (45, ly + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (230, 230, 230), 1)

        self._last_img = img  # keep a copy so save_image() works without re-rendering

        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(wait_ms) & 0xFF
        return key  # caller checks for 'q' to quit

    def save_image(self, path):
        """Save the most recently rendered frame to disk as a PNG (or any cv2-supported ext)."""
        if getattr(self, "_last_img", None) is None:
            raise RuntimeError("No frame has been rendered yet -- call render() before save_image().")
        cv2.imwrite(path, self._last_img)

    def close(self):
        cv2.destroyWindow(self.window_name)

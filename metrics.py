"""
metrics.py
-----------
Performance metrics for the pipeline: per-class IoU (when ground-truth
labels are available), and the memory saving of the variable resolution
grid versus an equivalent uniform high-resolution grid. FPS is tracked
live inside visualize_dashboard.Dashboard and logged by run_inference.py.
"""

import numpy as np


class RunningIoU:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, gt):
        """pred, gt: 1D int arrays with values in [0, num_classes-1]; negatives are ignored"""
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        mask = (gt >= 0) & (pred >= 0)
        pred, gt = pred[mask], gt[mask]
        for p, g in zip(pred, gt):
            self.confusion[g, p] += 1

    def per_class_iou(self):
        ious = []
        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp
            denom = tp + fp + fn
            ious.append(tp / denom if denom > 0 else float("nan"))
        return ious

    def mean_iou(self):
        ious = [v for v in self.per_class_iou() if not np.isnan(v)]
        return float(np.mean(ious)) if ious else float("nan")

    def per_class_accuracy(self):
        """Recall per class: of all GT points that truly belong to class c,
        what fraction did the model correctly label as c? This is the
        'how well is it identifying dynamic/static/terrain' number."""
        accs = []
        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            total_gt = self.confusion[c, :].sum()
            accs.append(tp / total_gt if total_gt > 0 else float("nan"))
        return accs

    def overall_accuracy(self):
        """Fraction of ALL points (any class) correctly classified."""
        tp_total = np.trace(self.confusion)
        total = self.confusion.sum()
        return float(tp_total / total) if total > 0 else float("nan")

    def summary_dict(self, class_names=("terrain", "static", "dynamic")):
        accs = self.per_class_accuracy()
        ious = self.per_class_iou()
        return {
            "overall_accuracy": round(self.overall_accuracy(), 4),
            "mean_iou": round(self.mean_iou(), 4),
            **{f"{class_names[c]}_accuracy": round(accs[c], 4) if not np.isnan(accs[c]) else None
               for c in range(self.num_classes)},
            **{f"{class_names[c]}_iou": round(ious[c], 4) if not np.isnan(ious[c]) else None
               for c in range(self.num_classes)},
            "confusion_matrix": self.confusion.tolist(),
        }


def memory_saving_report(grid):
    sparse = grid.sparse_cell_count()
    uniform = grid.equivalent_uniform_cell_count()
    saving_pct = 100.0 * (1 - sparse / uniform) if uniform else 0.0
    return {
        "sparse_cells_used": sparse,
        "equivalent_uniform_cells": uniform,
        "memory_saving_percent": round(saving_pct, 2),
    }

"""
backends.py
------------
Pluggable classification backends behind one interface:

    backend.predict(xyz) -> (labels, source_name)

HeuristicBackend is always available -- a transparent placeholder
(height threshold + a deterministic pseudo-random split) that stands
in for a real segmentation model. It never claims to be anything else;
`source_name` on every prediction says exactly which backend produced it.

Open3DMLBackend wires to the SAME code path as run_inference.py
(open3d.ml RandLA-Net / KPConv, loaded from a pretrained SemanticKITTI
checkpoint via pipeline.load_ckpt). It only activates if open3d-ml/torch
are installed AND the checkpoint file exists on disk -- this sandbox
does not have either, so it will not be the active backend here, but
the code is real and matches build_pipeline() in run_inference.py.
"""

import os
import numpy as np

from class_mapping import to_super_class


class HeuristicBackend:
    """Placeholder classifier -- NOT a trained model. Height threshold for
    terrain, deterministic hash-based split for the rest."""
    name = "heuristic-fallback"

    def __init__(self, device="cpu", note=None):
        self.device = device
        self._note = note or (
            "Height-based heuristic placeholder -- no learned weights. "
            "Swap in a real checkpoint to activate Open3DMLBackend."
        )

    def predict(self, xyz):
        xyz = np.asarray(xyz)
        z = xyz[:, 2]
        idx = np.arange(xyz.shape[0])
        h = np.abs(np.sin(idx * 12.9898) * 43758.5453) % 1.0
        labels = np.where(z < -1.0, 0, np.where(h < 0.78, 1, 2)).astype(np.int8)
        return labels, self.name

    def status(self):
        return {
            "backend": self.name,
            "is_pretrained_model": False,
            "note": self._note,
        }


class Open3DMLBackend:
    """Same forward-pass path as run_inference.py's build_pipeline()."""
    name = "open3d-ml"

    def __init__(self, model_name, checkpoint_path, device="cpu"):
        import open3d.ml as _ml3d          # noqa: F401
        import open3d.ml.torch as ml3d

        if model_name == "RandLANet":
            model = ml3d.models.RandLANet(num_classes=19)
        elif model_name == "KPFCNN":
            model = ml3d.models.KPFCNN(num_classes=19)
        else:
            raise ValueError(f"Unknown model '{model_name}'")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        self.pipeline = ml3d.pipelines.SemanticSegmentation(model, device=device)
        self.pipeline.load_ckpt(ckpt_path=checkpoint_path)
        self.pipeline.model.eval()
        self.model_name = model_name
        self.device = device

    def predict(self, xyz):
        data = {"point": xyz, "feat": None, "label": np.zeros(len(xyz), dtype=np.int32)}
        results = self.pipeline.run_inference(data)
        kitti_labels = results["predict_labels"]
        return to_super_class(kitti_labels), self.name

    def status(self):
        return {
            "backend": self.name,
            "is_pretrained_model": True,
            "model": self.model_name,
            "device": self.device,
        }


_cache = {}

# KPConv (KPFCNN) is under construction on the frontend -- the model card is
# disabled there so this should only ever be reached via a direct API call.
# Report that plainly instead of attempting to build/load it.
UNDER_CONSTRUCTION_MODELS = {"KPFCNN"}


def get_backend(model_name="RandLANet", device="cpu", checkpoint_path=None):
    """Returns a real Open3D-ML backend if it can actually be constructed;
    otherwise falls back to the heuristic and says so.

    Only a SUCCESSFUL Open3DMLBackend construction is cached. A failed
    attempt (missing torch/open3d, bad checkpoint path, checkpoint not
    downloaded yet) must NOT be cached under this key -- otherwise fixing
    the checkpoint path or installing the dependency later would still
    silently return the old cached heuristic forever, since the key
    (model_name, device, checkpoint_path) never changes. HeuristicBackend
    itself is trivial to construct, so there's no cost to not caching it.
    """
    if model_name in UNDER_CONSTRUCTION_MODELS:
        return HeuristicBackend(
            device,
            note=f"'{model_name}' is under construction and not available yet -- "
                 f"using heuristic-fallback classifier instead. Use 'RandLANet'.",
        )

    key = (model_name, device, checkpoint_path)
    if key in _cache:
        return _cache[key]

    if checkpoint_path:
        try:
            backend = Open3DMLBackend(model_name, checkpoint_path, device)
            _cache[key] = backend
            return backend
        except Exception:
            pass  # do NOT cache the failure -- retry next call in case the
                  # checkpoint / dependency shows up later

    return HeuristicBackend(device)

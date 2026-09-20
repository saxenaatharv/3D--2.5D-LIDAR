"""
class_mapping.py
------------------
Maps the pretrained model's native SemanticKITTI 19-class output onto the
3 super-categories required by this project:

    0 -> DRIVABLE TERRAIN   (road, parking, sidewalk, other-ground, terrain)
    1 -> STATIC OBSTACLE    (building, fence, vegetation, trunk, pole, traffic-sign)
    2 -> DYNAMIC OBJECT     (car, bicycle, motorcycle, truck, other-vehicle,
                             person, bicyclist, motorcyclist)
   -1 -> UNLABELED / IGNORE

Label ids follow the standard SemanticKITTI 19-class training scheme used
by the Open3D-ML RandLA-Net / KPConv model zoo checkpoints.
"""

import numpy as np

SEMANTIC_KITTI_NAMES = {
    0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
    5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
    9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
    13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
    17: "terrain", 18: "pole", 19: "traffic-sign",
}

DRIVABLE_TERRAIN = {9, 10, 11, 12, 17}
STATIC_OBSTACLE  = {13, 14, 15, 16, 18, 19}
DYNAMIC_OBJECT   = {1, 2, 3, 4, 5, 6, 7, 8}
UNLABELED        = {0}

SUPER_CLASS_ID = {"unlabeled": -1, "terrain": 0, "static": 1, "dynamic": 2}
SUPER_CLASS_NAMES = {-1: "unlabeled", 0: "terrain", 1: "static", 2: "dynamic"}

# OpenCV uses BGR ordering
SUPER_CLASS_COLOR_BGR = {
    -1: (60, 60, 60),      # unlabeled -> dark grey
     0: (255, 80, 0),      # terrain        -> blue
     1: (0, 230, 255),     # static obstacle -> yellow
     2: (0, 0, 255),       # dynamic object -> red
}


def to_super_class(label_ids):
    """
    Vectorised remap of raw SemanticKITTI predictions/labels (numpy int array)
    into the 3-class scheme consumed by the grid engine.
    """
    label_ids = np.asarray(label_ids)
    out = np.full(label_ids.shape, SUPER_CLASS_ID["unlabeled"], dtype=np.int8)
    out[np.isin(label_ids, list(DRIVABLE_TERRAIN))] = SUPER_CLASS_ID["terrain"]
    out[np.isin(label_ids, list(STATIC_OBSTACLE))]  = SUPER_CLASS_ID["static"]
    out[np.isin(label_ids, list(DYNAMIC_OBJECT))]   = SUPER_CLASS_ID["dynamic"]
    return out

"""
finetune.py
------------
Fine-tunes the SAME pretrained backbone used in run_inference.py on a new
labelled dataset of your own. Training starts from the pretrained
SemanticKITTI checkpoint -- it is NOT trained from random weights.

Your raw data can live anywhere; pass --raw_pc_dir / --raw_label_dir and
this script converts it into the folder layout Open3D-ML's SemanticKITTI
dataset loader expects (see dataset_utils.py), then runs the standard
supervised fine-tuning loop and checkpoints the adapted model.

If you already have data in SemanticKITTI layout, skip --raw_pc_dir /
--raw_label_dir and just point paths.finetune_data_dir in config.yaml at
it directly.
"""

import argparse
import yaml

import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.torch.datasets import SemanticKITTI

from dataset_utils import convert_folder


def build_model(cfg):
    model_name = cfg["model"]["name"]
    if model_name == "RandLANet":
        return ml3d.models.RandLANet(num_classes=cfg["model"]["num_classes"])
    if model_name == "KPFCNN":
        return ml3d.models.KPFCNN(num_classes=cfg["model"]["num_classes"])
    raise ValueError(f"Unknown model '{model_name}' -- use RandLANet or KPFCNN")


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.raw_pc_dir and args.raw_label_dir:
        convert_folder(args.raw_pc_dir, args.raw_label_dir, cfg["paths"]["finetune_data_dir"])

    dataset = SemanticKITTI(
        dataset_path=cfg["paths"]["finetune_data_dir"],
        cache_dir="./cache",
        training_split=["00"],
        validation_split=["00"],
        test_split=["00"],
    )

    model = build_model(cfg)
    pipeline = ml3d.pipelines.SemanticSegmentation(
        model=model,
        dataset=dataset,
        device=cfg["model"]["device"],
        max_epoch=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        optimizer={"lr": cfg["training"]["learning_rate"]},
        save_ckpt_freq=cfg["training"]["save_every"],
    )

    # Start from the PRETRAINED SemanticKITTI checkpoint, not random weights.
    pipeline.load_ckpt(ckpt_path=cfg["model"]["checkpoint_path"])

    freeze_n = cfg["training"].get("freeze_backbone_layers", 0)
    if freeze_n > 0:
        for i, (name, param) in enumerate(pipeline.model.named_parameters()):
            if i < freeze_n:
                param.requires_grad = False
        print(f"[finetune] Froze the first {freeze_n} parameter tensors; the rest fine-tune normally.")

    pipeline.run_train()
    print("[finetune] Fine-tuning complete. The adapted checkpoint was saved under "
          "./logs/ (see the console output above for the exact filename Open3D-ML used). "
          "Copy that .pth file into checkpoints/ and point config.yaml's "
          "model.checkpoint_path at it to use the fine-tuned weights in run_inference.py.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--raw_pc_dir", default=None, help="Optional: folder of your own point clouds")
    ap.add_argument("--raw_label_dir", default=None, help="Optional: matching folder of your own .npy labels")
    args = ap.parse_args()
    main(args)

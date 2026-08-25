"""
finetune.py
------------
Fine-tunes the SAME pretrained backbone used in run_inference.py on a new
labelled dataset of your own. Training starts from the pretrained
SemanticKITTI checkpoint -- it is NOT trained from random weights.

IMPORTANT: this script NEVER modifies the --config file you pass in. It
reads the starting checkpoint from --config (this should always be your
pretrained config, e.g. config_pretrained.yaml) and writes a SEPARATE
config file (--output_config, default config_finetuned.yaml) pointing at
the new fine-tuned checkpoint. This means config_pretrained.yaml always
keeps pointing at the pretrained weights, no matter how many times you
fine-tune -- you can always run inference against either one.

Your raw data can live anywhere; pass --raw_pc_dir / --raw_label_dir and
this script splits it 90/10 (train/test, configurable via
training.train_split in the config, or --train_ratio) into the folder
layout Open3D-ML's SemanticKITTI dataset loader expects (see
dataset_utils.convert_folder_split), then runs the standard supervised
fine-tuning loop and checkpoints the adapted model.

If you already have data in SemanticKITTI layout with sequences 00
(train) and 01 (test), skip --raw_pc_dir / --raw_label_dir and just point
paths.finetune_data_dir in the config at it directly.
"""

import argparse
import glob
import os
import shutil
import copy
import yaml

import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.torch.datasets import SemanticKITTI

from dataset_utils import convert_folder_split


def build_model(cfg):
    model_name = cfg["model"]["name"]
    # The pretrained checkpoint was trained with in_channels=3 (xyz only) --
    # confirmed by its fc0.weight shape [8, 3]. Read from config so it stays
    # explicit and easy to change if you ever fine-tune from a different
    # checkpoint that does use intensity.
    in_channels = cfg["model"].get("in_channels", 3)
    if model_name == "RandLANet":
        return ml3d.models.RandLANet(num_classes=cfg["model"]["num_classes"],
                                      in_channels=in_channels)
    if model_name == "KPFCNN":
        return ml3d.models.KPFCNN(num_classes=cfg["model"]["num_classes"],
                                   in_channels=in_channels)
    raise ValueError(f"Unknown model '{model_name}' -- use RandLANet or KPFCNN")


def _drop_intensity_feature(dataset):
    """
    Open3D-ML's SemanticKITTI dataset split always returns
    data['feat'] = intensity (shape N,1) -- this is hard-coded in the
    library's own semantickitti.py and can't be turned off via config.
    That would make every training sample arrive as 4 channels (xyz +
    intensity), but our pretrained checkpoint was trained on 3 channels
    (xyz only -- see build_model above). So we monkey-patch get_data() on
    the split class to strip the intensity channel back off, keeping the
    fine-tuned model architecture compatible with the pretrained weights
    we're starting from.
    """
    probe_split = dataset.get_split("training")
    split_cls = type(probe_split)
    if getattr(split_cls, "_intensity_dropped", False):
        return  # already patched (e.g. if this is called more than once)

    orig_get_data = split_cls.get_data

    def get_data_xyz_only(self, idx):
        data = orig_get_data(self, idx)
        data["feat"] = None
        return data

    split_cls.get_data = get_data_xyz_only
    split_cls._intensity_dropped = True


def _export_finetuned_checkpoint(cfg, output_config_path):
    """
    pipeline.run_train() saves the adapted weights under ./logs/.../checkpoint/
    -- it does NOT touch checkpoints/ or any config file. This finds the
    checkpoint Open3D-ML just wrote (the most recently modified .pth under
    ./logs/), copies it into checkpoints/, and writes a BRAND NEW config
    file at output_config_path with model.checkpoint_path updated to point
    at the copy. The config you originally passed via --config (your
    pretrained config) is left completely untouched.
    """
    candidates = glob.glob(os.path.join("logs", "**", "checkpoint", "*.pth"), recursive=True)
    if not candidates:
        print("[finetune] WARNING: fine-tuning finished but no checkpoint file was found "
              "under ./logs/ -- no output config was written. Check the training output "
              "above for where Open3D-ML actually saved its checkpoint.")
        return

    latest = max(candidates, key=os.path.getmtime)

    os.makedirs("checkpoints", exist_ok=True)
    dest_path = os.path.join("checkpoints", f"finetuned_{os.path.basename(latest)}")
    shutil.copy2(latest, dest_path)

    out_cfg = copy.deepcopy(cfg)
    out_cfg["model"]["checkpoint_path"] = dest_path
    out_cfg["paths"]["output_dir"] = out_cfg["paths"].get("output_dir", "outputs").rstrip("/") \
        if "finetuned" in out_cfg["paths"].get("output_dir", "") else "outputs/finetuned"

    with open(output_config_path, "w") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    print(f"[finetune] Fine-tuned checkpoint copied to {dest_path}")
    print(f"[finetune] Wrote NEW config file: {output_config_path} (checkpoint_path -> {dest_path})")
    print(f"[finetune] Your original config was NOT modified -- it still points at the pretrained weights.")
    print(f"[finetune] Run inference with either:")
    print(f"[finetune]   python run_inference.py --config <your_pretrained_config>.yaml")
    print(f"[finetune]   python run_inference.py --config {output_config_path}")


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_ratio = args.train_ratio if args.train_ratio is not None else cfg["training"].get("train_split", 0.9)

    if args.raw_pc_dir and args.raw_label_dir:
        convert_folder_split(args.raw_pc_dir, args.raw_label_dir,
                              cfg["paths"]["finetune_data_dir"], train_ratio=train_ratio)

    # sequence "00" = train (train_ratio, e.g. 90%), sequence "01" = held-out test (e.g. 10%).
    # validation uses the same held-out split as test since we only have two sequences.
    dataset = SemanticKITTI(
        dataset_path=cfg["paths"]["finetune_data_dir"],
        cache_dir="./cache",
        training_split=["00"],
        validation_split=["01"],
        test_split=["01"],
    )
    _drop_intensity_feature(dataset)

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
    # --config should be your PRETRAINED config (e.g. config_pretrained.yaml)
    # so this always starts from the right weights.
    pipeline.load_ckpt(ckpt_path=cfg["model"]["checkpoint_path"])

    freeze_n = cfg["training"].get("freeze_backbone_layers", 0)
    if freeze_n > 0:
        for i, (name, param) in enumerate(pipeline.model.named_parameters()):
            if i < freeze_n:
                param.requires_grad = False
        print(f"[finetune] Froze the first {freeze_n} parameter tensors; the rest fine-tune normally.")

    print(f"[finetune] Starting from checkpoint: {cfg['model']['checkpoint_path']}")
    print(f"[finetune] Train/test split: {train_ratio:.0%} / {1 - train_ratio:.0%} "
          f"(sequence 00 / sequence 01)")

    pipeline.run_train()
    _export_finetuned_checkpoint(cfg, args.output_config)
    print("[finetune] Fine-tuning complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_pretrained.yaml",
                     help="Config to start fine-tuning FROM. This file is never modified.")
    ap.add_argument("--output_config", default="config_finetuned.yaml",
                     help="NEW config file to write, pointing at the fine-tuned checkpoint.")
    ap.add_argument("--raw_pc_dir", default=None, help="Optional: folder of your own point clouds")
    ap.add_argument("--raw_label_dir", default=None, help="Optional: matching folder of your own .npy labels")
    ap.add_argument("--train_ratio", type=float, default=None,
                     help="Train fraction, e.g. 0.9 for 90/10. Defaults to training.train_split in config.")
    args = ap.parse_args()
    main(args)

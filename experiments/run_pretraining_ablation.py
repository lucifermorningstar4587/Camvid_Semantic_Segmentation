import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet101, ResNet101_Weights
from transformers import SegformerForSemanticSegmentation
import pandas as pd

from data.dataset import CamVidDataset
from losses.ce_loss import CrossEntropyLossWrapper
from utils.logger import FormattedLogger
from utils.trainer import fit_model
from experiments.train_semantic_transfer import (
    DeepLabV3PlusResNet101,
    build_semantic_transferred_deeplab,
    build_semantic_transferred_segformer,
    get_512_train_transforms,
    get_512_val_transforms
)


# =========================================================================
# ImageNet-1K Pretrained Model Builders
# =========================================================================
def build_imagenet_deeplab(device: str = "cuda") -> nn.Module:
    """Builds DeepLabV3+ ResNet-101 initialized with official ImageNet-1K backbone weights."""
    model = DeepLabV3PlusResNet101(num_classes=11)
    resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    model.backbone.conv1.load_state_dict(resnet.conv1.state_dict())
    model.backbone.bn1.load_state_dict(resnet.bn1.state_dict())
    model.backbone.layer1.load_state_dict(resnet.layer1.state_dict())
    model.backbone.layer2.load_state_dict(resnet.layer2.state_dict())
    model.backbone.layer3.load_state_dict(resnet.layer3.state_dict())
    model.backbone.layer4.load_state_dict(resnet.layer4.state_dict())
    print("[DeepLabV3+] Initialized with official ImageNet-1K ResNet-101 backbone.")
    return model.to(device)


def build_imagenet_segformer(device: str = "cuda") -> nn.Module:
    """Builds SegFormer-B2 initialized with official ImageNet-1K MiT-B2 backbone."""
    class SegFormerWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b2",
                num_labels=11,
                ignore_mismatched_sizes=True
            )
        def forward(self, x):
            orig_h, orig_w = x.shape[2], x.shape[3]
            out = self.inner(pixel_values=x).logits
            return F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    print("[SegFormer-B2] Initialized with official ImageNet-1K MiT-B2 backbone (nvidia/mit-b2).")
    return SegFormerWrapper().to(device)


# =========================================================================
# Main Pretraining Ablation Pipeline
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="ImageNet vs Cityscapes Pretraining Ablation (Unified High-Performance Recipe)")
    parser.add_argument("--data_root", type=str, default="./data/CamVid")
    parser.add_argument("--log_dir", type=str, default="./outputs/new_logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/new_checkpoints")
    parser.add_argument("--pretraining", type=str, default="all",
                        choices=["all", "imagenet", "cityscapes"],
                        help="Select pretraining domain: 'imagenet', 'cityscapes', or 'all'")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "deeplabv3plus", "segformer_b2"],
                        help="Select model: 'deeplabv3plus', 'segformer_b2', or 'all'")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("[PRETRAINING ABLATION: IMAGENET VS CITYSCAPES TRANSFER]")
    print(f"   Domain: {args.pretraining.upper()} | Model(s): {args.model.upper()} | Epochs: {args.epochs}")
    print(f"   Input Size: 512x512 | Device: {device}")
    print(f"   Logs Directory:        {args.log_dir}")
    print(f"   Checkpoints Directory: {args.checkpoint_dir}")
    print("=" * 80)

    # 512x512 Datasets and Loaders
    train_dataset = CamVidDataset(args.data_root, split="train", transforms=get_512_train_transforms())
    val_dataset = CamVidDataset(args.data_root, split="val", transforms=get_512_val_transforms())

    use_pin_memory = (device == "cuda")
    num_workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    pretrain_list = ["imagenet", "cityscapes"] if args.pretraining == "all" else [args.pretraining.lower()]
    models_list = ["deeplabv3plus", "segformer_b2"] if args.model == "all" else [args.model.lower()]

    configs = [{"model": m, "pretraining": p} for p in pretrain_list for m in models_list]

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.log_dir) / "02_pretraining_ablation_summary.csv"

    existing_records = []
    if summary_path.exists():
        try:
            existing_records = pd.read_csv(summary_path).to_dict("records")
        except Exception:
            existing_records = []

    new_results = []

    for cfg in configs:
        exp_name = f"02_pretrain_{cfg['model']}_{cfg['pretraining']}"
        logger = FormattedLogger(exp_name, log_dir=args.log_dir)
        logger.log_message("=" * 70)
        logger.log_message(f"Starting Pretraining Ablation: {cfg['model'].upper()} with {cfg['pretraining'].upper()} weights ({args.epochs} epochs)")
        logger.log_message("=" * 70)

        # Build model with exact specified domain weights
        if cfg["model"] == "deeplabv3plus":
            if cfg["pretraining"] == "cityscapes":
                model = build_semantic_transferred_deeplab(device=device)
            else:
                model = build_imagenet_deeplab(device=device)

            # DeepLabV3+ proven optimizer: SGD with polynomial decay
            optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=1.0)
            logger.log_message("[Optimizer] DeepLabV3+ SGD: lr=0.005, momentum=0.9, PolynomialLR(power=1.0)")

        else:  # segformer_b2
            if cfg["pretraining"] == "cityscapes":
                model = build_semantic_transferred_segformer(device=device)
            else:
                model = build_imagenet_segformer(device=device)

            # SegFormer-B2 proven optimizer: AdamW with cosine annealing
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            logger.log_message("[Optimizer] SegFormer-B2 AdamW: lr=0.0001, weight_decay=0.01, CosineAnnealingLR")

        criterion = CrossEntropyLossWrapper()

        ckpt_path = f"{args.checkpoint_dir}/{exp_name}_best.pth"
        fit_res = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            logger=logger,
            device=device,
            save_path=ckpt_path
        )

        record = {
            "model": cfg["model"],
            "pretraining": cfg["pretraining"],
            "best_val_miou": round(float(fit_res["best_val_miou"]), 4),
            "best_val_dice": round(float(fit_res["best_metrics"].get("mean_dice_f1", 0.0)), 4),
            "best_val_acc": round(float(fit_res["best_metrics"].get("pixel_accuracy", 0.0)), 4),
            "best_val_biou": round(float(fit_res["best_metrics"].get("boundary_iou", 0.0)), 4),
            "checkpoint": ckpt_path
        }
        new_results.append(record)
        logger.save_summary(record)

        # Merge incrementally by (model, pretraining) key to guarantee no duplicates
        current_map = {(r.get("model"), r.get("pretraining")): r for r in existing_records}
        for r in new_results:
            current_map[(r.get("model"), r.get("pretraining"))] = r
        df_all = pd.DataFrame(list(current_map.values()))
        df_all.to_csv(summary_path, index=False)

        # Also mirror to pretraining_ablation_summary.csv in the same directory
        alt_summary = Path(args.log_dir) / "pretraining_ablation_summary.csv"
        df_all.to_csv(alt_summary, index=False)
        logger.log_message(f"Result for {cfg['model']} ({cfg['pretraining']}) saved to {summary_path}")

    df_results = pd.read_csv(summary_path)
    print("\n" + "=" * 80)
    print("=== CURRENT PRETRAINING ABLATION SUMMARY TABLE ===")
    print(df_results.to_string(index=False))
    print(f"Summary saved to {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

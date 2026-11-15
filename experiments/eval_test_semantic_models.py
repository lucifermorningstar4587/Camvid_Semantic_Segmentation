"""
Zero-Leakage Pure Test Set Evaluation for Semantic Transferred Models
====================================================================
Evaluates DeepLabV3+ and SegFormer-B2 on the official CamVid test set (233 images).
Guarantees:
- Strictly model.eval() and torch.no_grad().
- All parameter gradients disabled (requires_grad=False).
- No model parameter updates, no data leakage, pure inference.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from data.dataset import CamVidDataset, CAMVID_CLASSES
from experiments.train_semantic_transfer import (
    DeepLabV3PlusResNet101,
    get_512_val_transforms
)
from utils.metrics import SegmentationMetricEvaluator


class SegFormerWrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        out = self.inner(pixel_values=x).logits
        return F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)


def load_deeplab_eval(checkpoint_path: str, device: str) -> nn.Module:
    model = DeepLabV3PlusResNet101(num_classes=11)
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_segformer_eval(checkpoint_path: str, device: str) -> nn.Module:
    config = SegformerConfig.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_labels=11
    )
    inner = SegformerForSemanticSegmentation(config)
    model = SegFormerWrapper(inner)
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def evaluate_model_on_test(model: nn.Module, model_name: str, test_loader: DataLoader, device: str):
    print("=" * 80)
    print(f"EVALUATING {model_name.upper()} ON TEST SET (Zero-Leakage, Frozen Weights)")
    print("=" * 80)

    evaluator = SegmentationMetricEvaluator(num_classes=11, ignore_index=255, compute_boundary=True)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            evaluator.update(preds, targets_np)

            if batch_idx % 15 == 0 or batch_idx == len(test_loader):
                print(f"  [{batch_idx:02d}/{len(test_loader)}] test batches processed...")

    metrics = evaluator.compute_all_metrics()

    print("\n" + "-" * 80)
    print(f"{model_name.upper()} TEST RESULTS:")
    print(f"   Test mIoU:              {metrics['mIoU'] * 100:.2f}%")
    print(f"   Pixel Accuracy:         {metrics['pixel_accuracy'] * 100:.2f}%")
    print(f"   Mean Dice / F1:         {metrics['mean_dice_f1'] * 100:.2f}%")
    print(f"   Boundary IoU:           {metrics['boundary_iou'] * 100:.2f}%")
    print("-" * 80)
    print("   Per-Class IoU Breakdown:")
    per_class = metrics.get("per_class_iou", {})
    for idx, c_name in enumerate(CAMVID_CLASSES):
        iou_val = per_class.get(idx, 0.0) if isinstance(per_class, dict) else per_class[idx]
        print(f"     - Class {idx:02d} ({c_name:<11}): {iou_val * 100:6.2f}%")
    print("=" * 80 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Zero-Leakage Test Evaluation")
    parser.add_argument("--model", type=str, default="both", choices=["both", "deeplabv3plus", "segformer_b2"])
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/new_checkpoints")
    parser.add_argument("--seed", type=str, default="42", help="Seed to evaluate ('42', '123', '456', or 'all')")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "./data/CamVid"
    ckpt_dir = Path(args.checkpoint_dir)
    seeds = [42, 123, 456] if args.seed.lower() == "all" else [int(args.seed)]

    # Test transforms: exactly the same deterministic resizing & normalization as validation (no data augmentation)
    test_dataset = CamVidDataset(data_root, split="test", transforms=get_512_val_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    print(f"Loaded test split: {len(test_dataset)} images.")

    models_to_eval = ["deeplabv3plus", "segformer_b2"] if args.model == "both" else [args.model]
    all_seed_results = {m: [] for m in models_to_eval}

    for s in seeds:
        print(f"\n=======================================================")
        print(f"EVALUATING ON TEST SET - SEED {s}")
        print(f"=======================================================")
        for m in models_to_eval:
            # Check for seeded checkpoint first, then fallback to canonical
            ckpt_path = ckpt_dir / f"04_semantic_{m}_seed{s}_best.pth"
            if not ckpt_path.exists():
                ckpt_path = ckpt_dir / f"04_semantic_{m}_best.pth"
            if not ckpt_path.exists():
                # Check legacy checkpoints folder
                ckpt_path = Path(f"./outputs/checkpoints/04_semantic_{m}_best.pth")

            if not ckpt_path.exists():
                print(f"[WARNING] Checkpoint not found for {m} (Seed {s}). Skipping.")
                continue

            print(f"--> Loading {m} checkpoint: {ckpt_path}")
            if "segformer" in m:
                model = load_segformer_eval(str(ckpt_path), device)
            else:
                model = load_deeplab_eval(str(ckpt_path), device)

            m_res = evaluate_model_on_test(model, f"{m} (Seed {s})", test_loader, device)
            m_res["seed"] = s
            all_seed_results[m].append(m_res)

    # Print summary across seeds
    import numpy as np
    print("\n" + "=" * 80)
    print("FINAL COMPREHENSIVE TEST SET BENCHMARK")
    print("=" * 80)

    for m in models_to_eval:
        res_list = all_seed_results[m]
        if not res_list:
            continue
        print(f"\n--- {m.upper()} ({len(res_list)} Seed(s) Evaluated) ---")
        mious = [r["mIoU"] * 100 for r in res_list]
        accs = [r["pixel_accuracy"] * 100 for r in res_list]
        dices = [r["mean_dice_f1"] * 100 for r in res_list]
        bious = [r["boundary_iou"] * 100 for r in res_list]

        if len(res_list) > 1:
            print(f"  Test mIoU:      {np.mean(mious):.2f}% ± {np.std(mious):.2f}% (Seeds: {[round(x, 2) for x in mious]})")
            print(f"  Pixel Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
            print(f"  Mean Dice / F1: {np.mean(dices):.2f}% ± {np.std(dices):.2f}%")
            print(f"  Boundary IoU:   {np.mean(bious):.2f}% ± {np.std(bious):.2f}%")
        else:
            print(f"  Test mIoU:      {mious[0]:.2f}%")
            print(f"  Pixel Accuracy: {accs[0]:.2f}%")
            print(f"  Mean Dice / F1: {dices[0]:.2f}%")
            print(f"  Boundary IoU:   {bious[0]:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()

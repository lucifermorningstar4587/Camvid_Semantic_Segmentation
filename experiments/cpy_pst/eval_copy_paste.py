"""
eval_copy_paste.py
==================
Evaluates the Class-Aware Copy-Paste checkpoint on all 233 official CamVid test frames
using the complete, state-of-the-art final inference pipeline (All Improvements):
  1. Full Native 720x960 Frame Resolution
  2. 2D Gaussian Sliding Window (512x512 crop with 256x256 stride)
  3. 3-Scale Multi-Scale Pyramid [0.85, 1.0, 1.15]
  4. Horizontal Flip Test-Time Augmentation (TTA)
  5. Minority Class Temperature Prior Shifts
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.dataset import CamVidDataset, CAMVID_CLASSES
from experiments.train_semantic_transfer import CAMVID_MEAN, CAMVID_STD
from experiments.run_final_78_benchmark import (
    load_segformer,
    load_deeplab,
    evaluate_advanced_sliding_window
)


def main():
    parser = argparse.ArgumentParser(
        description="Final Test Set Evaluation for Class-Aware Copy-Paste Model (All Improvements)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="segformer_b2",
                        choices=["segformer_b2", "deeplabv3plus", "both"],
                        help="Model architecture to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path (defaults to outputs/new_checkpoints/05_copypaste_<model>_seed42.pth)")
    parser.add_argument("--data_root", type=str, default="./data/CamVid",
                        help="Path to CamVid dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed identifier for checkpoint lookup")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/new_checkpoints",
                        help="Directory containing trained checkpoints")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.85, 1.0, 1.15],
                        help="TTA multi-scale pyramid factors")
    parser.add_argument("--no_flip", action="store_true",
                        help="Disable horizontal flip TTA")
    parser.add_argument("--tau_shift", type=float, default=None,
                        help="Class prior shift for minority classes (default: 0.20 for SegFormer, 0.10 for DeepLab)")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 85)
    print("FINAL TEST EVALUATION: CLASS-AWARE COPY-PASTE MODEL (ALL IMPROVEMENTS)")
    print(f"Device: {device} | Split: Official 233 Test Frames (Zero Leakage)")
    print(f"Pipeline: Native 720x960 + Gaussian Sliding Window + 3-Scale TTA + Prior Calibration")
    print("=" * 85)

    # 1. Native 720x960 Test Dataset
    native_trans = A.Compose([
        A.Resize(height=720, width=960, interpolation=1),
        A.Normalize(mean=CAMVID_MEAN, std=CAMVID_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])
    test_ds = CamVidDataset(args.data_root, split="test", transforms=native_trans)

    models_to_eval = ["segformer_b2", "deeplabv3plus"] if args.model == "both" else [args.model]

    for m_name in models_to_eval:
        # Determine checkpoint path
        if args.checkpoint is not None and args.model != "both":
            ckpt_path = Path(args.checkpoint)
        else:
            ckpt_path = Path(args.checkpoint_dir) / f"05_copypaste_{m_name}_seed{args.seed}.pth"
            if not ckpt_path.exists():
                fallback = Path(f"./outputs/checkpoints/05_copypaste_{m_name}_seed{args.seed}.pth")
                if fallback.exists():
                    ckpt_path = fallback

        if not ckpt_path.exists():
            print(f"\n[ERROR] Checkpoint not found: {ckpt_path}")
            print(f"Please make sure you have trained the model using:")
            print(f"  python experiments/train_copy_paste.py --model {m_name} --seed {args.seed}")
            continue

        print(f"\nLoading trained Copy-Paste weights from: {ckpt_path}")
        if "segformer" in m_name:
            model = load_segformer(str(ckpt_path), device)
            tau = args.tau_shift if args.tau_shift is not None else 0.20
        else:
            model = load_deeplab(str(ckpt_path), device)
            tau = args.tau_shift if args.tau_shift is not None else 0.10

        metrics = evaluate_advanced_sliding_window(
            model=model,
            dataset=test_ds,
            model_name=f"Copy-Paste {m_name.upper()}",
            device=device,
            crop_size=(512, 512),
            stride=(256, 256),
            scales=args.scales,
            flip_tta=not args.no_flip,
            tau_shift=tau,
            minority_classes=[2, 6, 7, 9, 10]
        )

        print("\n" + "=" * 80)
        print(f"FINAL TEST RESULTS: COPY-PASTE {m_name.upper()} (ALL IMPROVEMENTS)")
        print("=" * 80)
        print(f"  • Test mIoU           : {metrics['mIoU'] * 100:.2f}%")
        print(f"  • Pixel Accuracy      : {metrics['pixel_accuracy'] * 100:.2f}%")
        print(f"  • Mean Dice / F1 Score: {metrics['mean_dice_f1'] * 100:.2f}%")
        print(f"  • Boundary IoU (BIoU) : {metrics['boundary_iou']:.4f}")
        print(f"  • Mean Precision      : {metrics['mean_precision'] * 100:.2f}%")
        print(f"  • Mean Recall         : {metrics['mean_recall'] * 100:.2f}%")
        print("-" * 80)
        print(f"  {'Class ID':<10} {'Class Name':<15} {'Test IoU (%)':<15} {'Notes'}")
        print("  " + "-" * 60)

        for c_idx, c_name in enumerate(CAMVID_CLASSES):
            iou_val = metrics['per_class_iou'].get(c_idx, 0.0) * 100
            note = "★ Minority Target" if c_idx in [2, 6, 9, 10] else ""
            print(f"  {c_idx:<10} {c_name:<15} {iou_val:>6.2f}%         {note}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

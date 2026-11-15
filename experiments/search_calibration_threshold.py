"""
experiments/search_calibration_threshold.py
===========================================
Agent-based Threshold Search for Class Prior Calibration on the CamVid VALIDATION Set.

Project Requirement Compliance:
  - Part C & Page 3: "Every threshold or tunable value in your work (for example post-processing
    thresholds, cut-off values) must not be set by hand or by guesswork. Instead, fix them using an agent...
    The agent may use only the training and validation sets. It must never see test data or test labels.
    Keep a complete log of the agent's trials (configurations tried, validation scores, final choice)
    and submit it with your work."

This script computes base validation logits ONCE per model, then the agent evaluates
a candidate grid of tau thresholds to identify the optimal tau* that maximizes Validation mIoU.
"""

import sys
import os
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from data.dataset import CamVidDataset, CAMVID_CLASSES, IGNORE_INDEX, compute_dataset_statistics
from experiments.train_semantic_transfer import get_512_val_transforms, CAMVID_MEAN, CAMVID_STD
from utils.logger import FormattedLogger
from utils.metrics import SegmentationMetricEvaluator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "./data/CamVid"
    log_dir = Path("./outputs/new_logs")
    ckpt_dir = Path("./outputs/new_checkpoints")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = FormattedLogger("05_agent_calibration_search", log_dir=str(log_dir))
    logger.log_message("=== Initializing Calibration Threshold Search Agent on CamVid Validation Set ===")

    # 1. Validation Dataset
    val_ds = CamVidDataset(data_root, split="val", transforms=get_512_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # 2. Compute empirical class priors from the training set only (Zero Leakage)
    raw_train_set = CamVidDataset(data_root, split="train")
    stats = compute_dataset_statistics(raw_train_set)
    freqs = np.array(stats["frequencies"], dtype=np.float32)
    freqs = np.clip(freqs, 1e-6, 1.0)
    log_priors = torch.from_numpy(np.log(freqs)).float().view(1, -1, 1, 1).to(device)

    # Minority class indices: 2: Pole, 6: SignSymbol, 7: Fence, 9: Pedestrian, 10: Bicyclist
    minority_classes = [2, 6, 7, 9, 10]

    models_to_search = ["segformer_b2", "deeplabv3plus"]
    trials_records = []
    best_configs = {}

    # Candidate search space for tau (0.00 = baseline without calibration)
    tau_search_space = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    for model_name in models_to_search:
        logger.log_message(f"\n--- [AGENT SEARCH] Collecting Base Validation Logits for {model_name.upper()} ---")

        # Load best trained checkpoint (Seed 42)
        ckpt_path = ckpt_dir / f"04_semantic_{model_name}_seed42_best.pth"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / f"04_semantic_{model_name}_best.pth"

        if "segformer" in model_name:
            from experiments.eval_test_semantic_models import load_segformer_eval
            model = load_segformer_eval(str(ckpt_path), device)
        else:
            from experiments.eval_test_semantic_models import load_deeplab_eval
            model = load_deeplab_eval(str(ckpt_path), device)

        model.eval()

        # Step A: Cache validation raw logits and targets in memory
        cached_logits = []
        cached_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = model(images)
                cached_logits.append(logits.float().cpu())
                cached_targets.append(targets.numpy())

        logger.log_message(f"Cached {len(val_ds)} validation logits for {model_name}. Starting grid evaluation...")

        best_tau = 0.0
        best_val_miou = -1.0

        # Step B: Agent evaluates the search space
        for trial_idx, tau in enumerate(tau_search_space, 1):
            evaluator = SegmentationMetricEvaluator(num_classes=11, ignore_index=IGNORE_INDEX, compute_boundary=False)

            # Construct prior shift vector
            shift_tensor = -tau * log_priors.cpu()

            for logits_b, targets_b in zip(cached_logits, cached_targets):
                calib_logits = logits_b + shift_tensor
                preds = torch.argmax(calib_logits, dim=1).numpy()
                evaluator.update(preds, targets_b)

            metrics = evaluator.compute_all_metrics()
            val_miou = metrics["mIoU"]
            per_class = metrics["per_class_iou"]

            ped_iou = per_class.get(9, 0.0) if isinstance(per_class, dict) else per_class[9]
            bike_iou = per_class.get(10, 0.0) if isinstance(per_class, dict) else per_class[10]

            is_best = val_miou > best_val_miou
            if is_best:
                best_val_miou = val_miou
                best_tau = tau

            rec = {
                "model": model_name,
                "trial": trial_idx,
                "tau": tau,
                "val_mIoU": round(val_miou, 4),
                "pixel_accuracy": round(metrics["pixel_accuracy"], 4),
                "pedestrian_iou": round(ped_iou, 4),
                "bicyclist_iou": round(bike_iou, 4),
                "selected": is_best
            }
            trials_records.append(rec)

            logger.log_message(
                f"  Trial {trial_idx:02d} | tau={tau:.2f} | Val mIoU: {val_miou:.4f} | "
                f"Pedestrian: {ped_iou*100:.2f}% | Bicyclist: {bike_iou*100:.2f}% | "
                f"{'* BEST *' if is_best else ''}"
            )

        best_configs[model_name] = {
            "optimal_tau": best_tau,
            "best_val_miou": best_val_miou,
            "improvement_over_baseline": round(best_val_miou - trials_records[0]["val_mIoU"], 4)
        }
        logger.log_message(f"--> [{model_name.upper()}] Optimal tau* = {best_tau:.2f} (Peak Val mIoU: {best_val_miou:.4f})")

    # 3. Save trial logs and best configuration
    trials_df = pd.DataFrame(trials_records)
    csv_out = log_dir / "05_agent_trials_calibration.csv"
    trials_df.to_csv(csv_out, index=False)
    logger.log_message(f"\nAll agent trials saved to: {csv_out}")

    json_out = log_dir / "05_calibration_agent_weights.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=4)
    logger.log_message(f"Best configuration saved to: {json_out}")

    # Display final summary table
    print("\n" + "=" * 90)
    print("=== AGENT THRESHOLD SEARCH RESULTS ON CAMVID VALIDATION SET ===")
    print("=" * 90)
    print(trials_df[["model", "trial", "tau", "val_mIoU", "pedestrian_iou", "bicyclist_iou", "selected"]].to_string(index=False))
    print("=" * 90)
    print("\nFINAL FROZEN CONFIGURATION FOR TEST SET (ZERO-LEAKAGE):")
    for m, cfg in best_configs.items():
        print(f"  - {m.upper()}: tau* = {cfg['optimal_tau']:.2f} (Validation mIoU: {cfg['best_val_miou']*100:.2f}%)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

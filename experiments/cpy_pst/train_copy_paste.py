import sys
import os
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any

from data.dataset import CamVidDataset
from experiments.train_semantic_transfer import (
    build_semantic_transferred_deeplab,
    build_semantic_transferred_segformer,
    CAMVID_TO_CITYSCAPES,
    get_512_train_transforms,
    get_512_val_transforms,
)
from losses.composite_loss import CompositeLossCEJML
from improvements.copy_paste import ClassAwareCopyPaste
from utils.logger import FormattedLogger
from utils.trainer import fit_model
from utils.metrics import SegmentationMetricEvaluator
import pandas as pd


CAMVID_CLASSES = [
    "Sky", "Building", "Pole", "Road", "Pavement",
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_split(
    model: nn.Module,
    dataset: CamVidDataset,
    device: str,
    batch_size: int = 4,
    split_name: str = "Test"
) -> Dict[str, Any]:
    """Evaluates the model on a dataset split and returns full per-class metrics."""
    model.eval()
    evaluator = SegmentationMetricEvaluator(num_classes=11)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
        pin_memory=(device == "cuda")
    )

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            evaluator.update(preds, targets_np)

    metrics = evaluator.compute_all_metrics()
    return metrics


def print_evaluation_summary(metrics: Dict[str, Any], split_name: str, logger: FormattedLogger):
    """Prints a structured summary of overall and per-class segmentation metrics."""
    logger.log_message(f"\n{'='*75}")
    logger.log_message(f"=== {split_name.upper()} SET EVALUATION RESULTS ===")
    logger.log_message(f"{'='*75}")
    logger.log_message(f"  • Mean IoU (mIoU)     : {metrics['mIoU'] * 100:.2f}%")
    logger.log_message(f"  • Pixel Accuracy      : {metrics['pixel_accuracy'] * 100:.2f}%")
    logger.log_message(f"  • Mean Dice / F1 Score: {metrics['mean_dice_f1'] * 100:.2f}%")
    logger.log_message(f"  • Boundary IoU (BIoU) : {metrics['boundary_iou']:.4f}")
    logger.log_message(f"  • Mean Precision      : {metrics['mean_precision'] * 100:.2f}%")
    logger.log_message(f"  • Mean Recall         : {metrics['mean_recall'] * 100:.2f}%")
    logger.log_message(f"{'-'*75}")
    logger.log_message(f"  {'Class ID':<10} {'Class Name':<15} {'IoU (%)':<12} {'Dice / F1 (%)':<15}")
    logger.log_message(f"  {'-'*55}")

    per_class_iou = metrics.get("per_class_iou", {})
    per_class_dice = metrics.get("per_class_dice", {})

    for idx, cname in enumerate(CAMVID_CLASSES):
        iou_val = per_class_iou.get(cname, per_class_iou.get(idx, 0.0)) * 100
        dice_val = per_class_dice.get(cname, per_class_dice.get(idx, 0.0)) * 100
        marker = " [MINORITY TARGET]" if idx in [2, 6, 9, 10] else ""
        logger.log_message(f"  {idx:<10} {cname:<15} {iou_val:>6.2f}%      {dice_val:>6.2f}%{marker}")

    logger.log_message(f"{'='*75}\n")


def train_single_model(
    model_name: str,
    args: argparse.Namespace,
    device: str,
    logger: FormattedLogger
) -> Dict[str, Any]:
    """Trains a single architecture using Cityscapes 1:1 Semantic Transfer + Class-Aware Copy-Paste."""
    set_seed(args.seed)
    logger.log_message(f"\n{'#'*80}")
    logger.log_message(f"# STARTING COPY-PASTE EXPERIMENT: {model_name.upper()} (Seed: {args.seed})")
    logger.log_message(f"# Mode: Cityscapes 1:1 Semantic Transfer Initialization (Backbone + 19->11 Classifier)")
    logger.log_message(f"{'#'*80}")

    # 1. Setup Data & Copy-Paste Augmenter
    logger.log_message("[STEP 1/5] Initializing Class-Aware Copy-Paste Bank from CamVid Train Set...")
    raw_train_ds = CamVidDataset(args.data_root, split="train", transforms=None)
    
    cp_augmenter = ClassAwareCopyPaste(
        target_classes=args.target_classes,
        p=args.copy_paste_p,
        max_paste_objects=args.max_paste_objects
    )
    cp_augmenter.populate_bank(raw_train_ds, max_items=args.max_bank_items)

    # 2. Build Transformed Datasets
    logger.log_message("[STEP 2/5] Creating Augmented Train, Val, and Test DataLoaders...")
    train_ds = CamVidDataset(
        args.data_root,
        split="train",
        transforms=get_512_train_transforms(),
        copy_paste_augmenter=cp_augmenter
    )
    val_ds = CamVidDataset(args.data_root, split="val", transforms=get_512_val_transforms())
    test_ds = CamVidDataset(args.data_root, split="test", transforms=get_512_val_transforms())

    num_workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )

    # 3. Model Architecture Instantiation via 1:1 Cityscapes Semantic Transfer
    logger.log_message(f"[STEP 3/5] Instantiating {model_name} with Cityscapes 1:1 Semantic Transfer...")
    if "segformer" in model_name:
        model = build_semantic_transferred_segformer(device=device)
        weight_decay = 1e-2
        lr = args.lr if args.lr != 5e-3 else 1e-4
    elif "deeplab" in model_name:
        model = build_semantic_transferred_deeplab(device=device)
        weight_decay = 1e-4
        lr = args.lr
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CompositeLossCEJML(lambda_ce=args.lambda_ce, lambda_jml=args.lambda_jml)

    # Output paths
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    save_ckpt = Path(args.checkpoint_dir) / f"05_copypaste_{model_name}_seed{args.seed}.pth"

    # 4. Training Loop
    logger.log_message(f"[STEP 4/5] Training for {args.epochs} epochs with Composite Loss "
                       f"({args.lambda_ce:.2f} CE + {args.lambda_jml:.2f} JML)...")
    training_res = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        logger=logger,
        device=device,
        save_path=str(save_ckpt),
        use_amp=args.use_amp
    )

    # 5. Final Evaluation on Selected Splits
    logger.log_message("[STEP 5/5] Loading Best Checkpoint for Final Performance Evaluation...")
    if save_ckpt.exists():
        ckpt_data = torch.load(save_ckpt, map_location=device)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        model.load_state_dict(state_dict)

    results = {
        "model": model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "pretraining": "cityscapes_1to1_semantic_transfer",
        "copy_paste_p": args.copy_paste_p,
        "best_val_mIoU": training_res.get("best_miou", 0.0),
    }

    if args.eval_split in ["val", "both"]:
        val_metrics = evaluate_split(model, val_ds, device=device, batch_size=args.batch_size, split_name="Validation")
        print_evaluation_summary(val_metrics, "Validation Split", logger)
        results["val_mIoU"] = val_metrics["mIoU"]
        results["val_acc"] = val_metrics["pixel_accuracy"]
        results["val_dice"] = val_metrics["mean_dice_f1"]
        results["val_biou"] = val_metrics["boundary_iou"]

    if args.eval_split in ["test", "both"]:
        test_metrics = evaluate_split(model, test_ds, device=device, batch_size=args.batch_size, split_name="Test (Official 233)")
        print_evaluation_summary(test_metrics, "Official Test Split", logger)
        results["test_mIoU"] = test_metrics["mIoU"]
        results["test_acc"] = test_metrics["pixel_accuracy"]
        results["test_dice"] = test_metrics["mean_dice_f1"]
        results["test_biou"] = test_metrics["boundary_iou"]

        for cid in args.target_classes:
            cname = CAMVID_CLASSES[cid]
            iou = test_metrics.get("per_class_iou", {}).get(cname, test_metrics.get("per_class_iou", {}).get(cid, 0.0))
            results[f"test_iou_{cname.lower()}"] = iou

    # Persist summary record
    summary_csv = Path(args.log_dir) / "05_copy_paste_experiments_summary.csv"
    df_new = pd.DataFrame([results])
    if summary_csv.exists():
        df_old = pd.read_csv(summary_csv)
        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["model", "seed"], keep="last")
    else:
        df_all = df_new
    df_all.to_csv(summary_csv, index=False)
    logger.log_message(f"Saved experiment metrics to: {summary_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Class-Aware Copy-Paste Data Augmentation (Cityscapes 1:1 Semantic Transfer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="segformer_b2",
                        choices=["segformer_b2", "deeplabv3plus", "all"],
                        help="Model architecture to train with Copy-Paste augmentation")
    parser.add_argument("--data_root", type=str, default="./data/CamVid",
                        help="Root directory of CamVid dataset")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Total training epochs (25 epochs matches 2,300 unified budget)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training and evaluation batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate for AdamW optimizer")
    parser.add_argument("--lambda_ce", type=float, default=0.65,
                        help="Weight for Cross-Entropy loss in composite objective")
    parser.add_argument("--lambda_jml", type=float, default=1.10,
                        help="Weight for Jaccard Multi-Loss in composite objective")
    parser.add_argument("--copy_paste_p", type=float, default=0.5,
                        help="Probability of applying Copy-Paste augmentation per sample")
    parser.add_argument("--max_paste_objects", type=int, default=3,
                        help="Maximum number of donor instances pasted into recipient image")
    parser.add_argument("--max_bank_items", type=int, default=200,
                        help="Maximum training frames scanned to populate donor object bank")
    parser.add_argument("--target_classes", type=int, nargs="+", default=[2, 6, 9, 10],
                        help="Target minority class indices (2=Pole, 6=SignSymbol, 9=Pedestrian, 10=Bicyclist)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible dataset sampling and weight initialization")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/new_checkpoints",
                        help="Directory to save best checkpoint weights")
    parser.add_argument("--log_dir", type=str, default="./outputs/new_logs",
                        help="Directory for logs and summary CSV records")
    parser.add_argument("--eval_split", type=str, default="both", choices=["val", "test", "both"],
                        help="Dataset splits to evaluate post-training")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable Automatic Mixed Precision (AMP)")

    args = parser.parse_args()
    args.use_amp = not args.no_amp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = FormattedLogger("05_copypaste_experiment", log_dir=args.log_dir)

    logger.log_message(f"Copy-Paste Augmentation Experiment Runner Initialized")
    logger.log_message(f"Initialization: Cityscapes 1:1 Semantic Transfer (Both Backbone + Classifier Head)")
    logger.log_message(f"Device: {device} | Model: {args.model.upper()} | Copy-Paste Probability: {args.copy_paste_p}")
    logger.log_message(f"Target Minority Classes: {[CAMVID_CLASSES[c] for c in args.target_classes]}")

    models_to_run = ["segformer_b2", "deeplabv3plus"] if args.model == "all" else [args.model]

    all_results = []
    for m_name in models_to_run:
        if args.model == "all":
            if m_name == "deeplabv3plus":
                args.lambda_ce = 0.75
                args.lambda_jml = 0.95
            else:
                args.lambda_ce = 0.65
                args.lambda_jml = 1.10

        res = train_single_model(m_name, args, device, logger)
        all_results.append(res)

    logger.log_message("\n=== ALL COPY-PASTE EXPERIMENTS COMPLETED SUCCESSFULLY ===")
    for r in all_results:
        logger.log_message(f"  • {r['model'].upper()} (Seed {r['seed']}): Val mIoU = {r.get('val_mIoU', 0.0)*100:.2f}% | Test mIoU = {r.get('test_mIoU', 0.0)*100:.2f}%")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from data.dataset import CamVidDataset
from losses.ce_loss import CrossEntropyLossWrapper
from losses.lovasz_loss import LovaszSoftmaxLoss
from losses.dice_loss import MultiClassDiceLoss
from losses.focal_loss import MultiClassFocalLoss
from losses.jml_loss import JaccardMetricLoss
from utils.logger import FormattedLogger
from utils.trainer import fit_model
from utils.metrics import SegmentationMetricEvaluator
from experiments.eval_test_semantic_models import load_deeplab_eval, load_segformer_eval
from experiments.train_semantic_transfer import (
    build_semantic_transferred_deeplab,
    build_semantic_transferred_segformer,
    get_512_train_transforms,
    get_512_val_transforms
)


def get_loss_criterion(loss_name: str) -> torch.nn.Module:
    name = loss_name.lower()
    if name == "ce":
        return CrossEntropyLossWrapper()
    elif name == "lovasz":
        return LovaszSoftmaxLoss()
    elif name == "dice":
        return MultiClassDiceLoss()
    elif name == "focal":
        return MultiClassFocalLoss()
    elif name == "jml":
        return JaccardMetricLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from: 'ce', 'lovasz', 'dice', 'focal', 'jml'.")


def main():
    parser = argparse.ArgumentParser(description="Loss Functions Ablation (Unified High-Performance Recipe)")
    parser.add_argument("--data_root", type=str, default="./data/CamVid")
    parser.add_argument("--log_dir", type=str, default="./outputs/new_logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/new_checkpoints")
    parser.add_argument("--loss", type=str, default="all",
                        choices=["all", "ce", "lovasz", "dice", "focal", "jml"],
                        help="Select a specific loss function to train, or 'all'")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "deeplabv3plus", "segformer_b2"],
                        help="Select a specific model to train, or 'all'")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_only", action="store_true", help="Evaluate existing trained checkpoints without training")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split for evaluation (default: test)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("[5 LOSS FUNCTIONS ABLATION STUDY: UNIFIED HIGH-PERFORMANCE RECIPE]")
    print(f"   Loss: {args.loss.upper()} | Model(s): {args.model.upper()} | Epochs: {args.epochs}")
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

    losses_to_run = ["ce", "lovasz", "dice", "focal", "jml"] if args.loss == "all" else [args.loss.lower()]
    models_to_run = ["deeplabv3plus", "segformer_b2"] if args.model == "all" else [args.model.lower()]

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ---------------- EVALUATION ONLY MODE (TEST / VAL ABLATION TABLE) ----------------
    if args.eval_only:
        eval_split = args.split
        logger = FormattedLogger(f"03_losses_ablation_{eval_split}", log_dir=args.log_dir)
        logger.log_message(f"=== Evaluating Loss Functions Ablation on CamVid [{eval_split.upper()}] Partition ===")

        eval_dataset = CamVidDataset(args.data_root, split=eval_split, transforms=get_512_val_transforms())
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
        print(f"Loaded {eval_split} split: {len(eval_dataset)} images.")

        eval_losses = losses_to_run.copy()
        if "composite" not in eval_losses and args.loss == "all":
            eval_losses.append("composite")

        eval_records = []
        for model_name in models_to_run:
            print("\n" + "=" * 80)
            print(f"EVALUATING MODEL: {model_name.upper()} ON {eval_split.upper()}")
            print("=" * 80)

            for loss_name in eval_losses:
                if loss_name == "composite":
                    ckpt_p = Path(args.checkpoint_dir) / f"04_semantic_{model_name}_seed42_best.pth"
                    if not ckpt_p.exists():
                        ckpt_p = Path(args.checkpoint_dir) / f"04_semantic_{model_name}_best.pth"
                else:
                    ckpt_p = Path(args.checkpoint_dir) / f"03_loss_{model_name}_{loss_name}_best.pth"

                if not ckpt_p.exists():
                    print(f"[WARNING] Checkpoint not found: {ckpt_p}. Skipping.")
                    continue

                print(f"--> [{model_name.upper()}] Evaluating Loss: {loss_name.upper():<10} | Checkpoint: {ckpt_p.name}")
                if "segformer" in model_name:
                    model = load_segformer_eval(str(ckpt_p), device)
                else:
                    model = load_deeplab_eval(str(ckpt_p), device)

                evaluator = SegmentationMetricEvaluator(num_classes=11, compute_boundary=True)
                with torch.no_grad():
                    for images, targets in eval_loader:
                        images = images.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                            outputs = model(images)
                        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                        evaluator.update(preds, targets.numpy())

                m = evaluator.compute_all_metrics()
                rec = {
                    "model": model_name,
                    "loss": loss_name,
                    "split": eval_split,
                    f"{eval_split}_mIoU": round(m["mIoU"], 4),
                    "pixel_accuracy": round(m["pixel_accuracy"], 4),
                    "dice_f1": round(m["mean_dice_f1"], 4),
                    "precision": round(m["mean_precision"], 4),
                    "recall": round(m["mean_recall"], 4),
                    "boundary_iou": round(m["boundary_iou"], 4),
                    "checkpoint": str(ckpt_p)
                }
                eval_records.append(rec)
                logger.log_message(
                    f"  [{model_name.upper()}] Loss: {loss_name.upper():<10} | "
                    f"{eval_split.capitalize()} mIoU: {m['mIoU']*100:.2f}% | Pixel Acc: {m['pixel_accuracy']*100:.2f}% | "
                    f"Dice/F1: {m['mean_dice_f1']*100:.2f}% | Boundary IoU: {m['boundary_iou']:.4f}"
                )

        eval_summary_path = Path(args.log_dir) / f"03_losses_ablation_{eval_split}_summary.csv"
        df_eval = pd.DataFrame(eval_records)
        df_eval.to_csv(eval_summary_path, index=False)
        logger.log_message(f"Results saved to: {eval_summary_path}")

        print("\n" + "=" * 90)
        print(f"=== LOSS FUNCTIONS ABLATION ON [{eval_split.upper()}] SET ({len(eval_dataset)} IMAGES) ===")
        print("=" * 90)
        disp_df = df_eval.copy()
        disp_df[f"{eval_split}_mIoU"] = disp_df[f"{eval_split}_mIoU"].apply(lambda v: f"{v*100:.2f}%")
        disp_df["pixel_accuracy"] = disp_df["pixel_accuracy"].apply(lambda v: f"{v*100:.2f}%")
        disp_df["dice_f1"] = disp_df["dice_f1"].apply(lambda v: f"{v*100:.2f}%")
        print(disp_df[["model", "loss", f"{eval_split}_mIoU", "pixel_accuracy", "dice_f1", "boundary_iou"]].to_string(index=False))
        print("=" * 90)
        return

    # ---------------- STANDARD TRAINING MODE ----------------
    summary_path = Path(args.log_dir) / "03_losses_ablation_summary.csv"

    existing_records = []
    if summary_path.exists():
        try:
            existing_records = pd.read_csv(summary_path).to_dict("records")
        except Exception:
            existing_records = []

    new_results = []

    for model_name in models_to_run:
        for loss_name in losses_to_run:
            exp_name = f"03_loss_{model_name}_{loss_name}"
            logger = FormattedLogger(exp_name, log_dir=args.log_dir)
            logger.log_message("=" * 70)
            logger.log_message(f"Starting Loss Ablation: {model_name.upper()} with {loss_name.upper()} Loss ({args.epochs} epochs)")
            logger.log_message("=" * 70)

            # Build model with 100% Cityscapes transferred representations
            if model_name == "deeplabv3plus":
                model = build_semantic_transferred_deeplab(device=device)
                optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=1.0)
                logger.log_message("[Optimizer] DeepLabV3+ SGD: lr=0.005, momentum=0.9, PolynomialLR(power=1.0)")
            else:
                model = build_semantic_transferred_segformer(device=device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
                logger.log_message("[Optimizer] SegFormer-B2 AdamW: lr=0.0001, weight_decay=0.01, CosineAnnealingLR")

            criterion = get_loss_criterion(loss_name)

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
                "model": model_name,
                "loss": loss_name,
                "best_val_miou": round(float(fit_res["best_val_miou"]), 4),
                "best_val_dice": round(float(fit_res["best_metrics"].get("mean_dice_f1", 0.0)), 4),
                "best_val_acc": round(float(fit_res["best_metrics"].get("pixel_accuracy", 0.0)), 4),
                "best_val_biou": round(float(fit_res["best_metrics"].get("boundary_iou", 0.0)), 4),
                "checkpoint": ckpt_path
            }
            new_results.append(record)
            logger.save_summary(record)

            # Update master CSV incrementally by (model, loss) key
            current_map = {(r.get("model"), r.get("loss")): r for r in existing_records}
            for r in new_results:
                current_map[(r.get("model"), r.get("loss"))] = r
            df_all = pd.DataFrame(list(current_map.values()))
            df_all.to_csv(summary_path, index=False)
            logger.log_message(f"Result for {model_name} with {loss_name} saved to {summary_path}")

    # Display updated summary
    df_losses = pd.read_csv(summary_path)
    print("\n" + "=" * 80)
    print("=== CURRENT LOSSES ABLATION SUMMARY TABLE ===")
    print(df_losses.to_string(index=False))
    print(f"Summary saved to {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

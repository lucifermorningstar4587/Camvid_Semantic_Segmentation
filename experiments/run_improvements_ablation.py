import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from data.dataset import CamVidDataset, compute_dataset_statistics
from experiments.train_semantic_transfer import get_512_train_transforms, get_512_val_transforms, CAMVID_MEAN, CAMVID_STD
from experiments.run_final_78_benchmark import evaluate_advanced_sliding_window
from models.model_factory import build_model
from losses.composite_loss import CompositeLossCEJML
from improvements.copy_paste import ClassAwareCopyPaste
from improvements.sliding_window import sliding_window_inference
from improvements.tta import multi_scale_tta_inference
from improvements.calibration import ClassPriorCalibrator
from utils.logger import FormattedLogger
from utils.trainer import fit_model
from utils.metrics import SegmentationMetricEvaluator
import pandas as pd

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_with_improvements(
    model: torch.nn.Module,
    val_dataset: CamVidDataset,
    device: str,
    use_sliding_window: bool = False,
    use_tta: bool = False,
    calibrator: ClassPriorCalibrator = None
) -> dict:
    model.eval()
    evaluator = SegmentationMetricEvaluator(num_classes=11)

    for i in range(len(val_dataset)):
        img_tensor, mask_tensor = val_dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        target_np = mask_tensor.numpy()

        with torch.no_grad():
            if use_sliding_window:
                logits = sliding_window_inference(model, img_tensor, patch_size=(512, 512), stride=(256, 256), device=device)
            elif use_tta:
                probs = multi_scale_tta_inference(model, img_tensor, scales=[0.85, 1.0, 1.15], flip=True, device=device)
                logits = torch.log(probs.clamp(min=1e-7))
            else:
                logits = model(img_tensor)

            if calibrator is not None:
                preds = calibrator.calibrate_and_predict(logits).cpu().numpy()[0]
            else:
                preds = torch.argmax(logits, dim=1).cpu().numpy()[0]

        evaluator.update(preds, target_np)

    return evaluator.compute_all_metrics()


def update_and_persist_record(csv_path: str, new_records: list, logger: FormattedLogger):
    """Safely merge new step records into master CSV, avoiding duplicates."""
    df_existing = pd.DataFrame()
    if Path(csv_path).exists():
        try:
            df_existing = pd.read_csv(csv_path)
        except Exception:
            df_existing = pd.DataFrame()

    df_new = pd.DataFrame(new_records)
    if not df_existing.empty:
        # Remove any existing rows matching (model, seed, type, stage) of new records
        keys = ["model", "seed", "type", "stage"]
        combined = pd.concat([df_existing, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=keys, keep="last")
    else:
        combined = df_new

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    logger.log_message(f"Results saved and merged into: {csv_path}")


def log_step_metric(logger: FormattedLogger, stage_type: str, stage_name: str, metrics: dict):
    logger.log_message(
        f"  --> [{stage_type.upper()}] {stage_name} | "
        f"mIoU: {metrics['mIoU']:.4f} | Dice/F1: {metrics['mean_dice_f1']:.4f} | "
        f"Precision: {metrics['mean_precision']:.4f} | Recall: {metrics['mean_recall']:.4f} | "
        f"Pixel Acc: {metrics['pixel_accuracy']:.4f} | Boundary IoU: {metrics['boundary_iou']:.4f}"
    )


def print_summary_table(csv_path: str, logger: FormattedLogger):
    """Calculates and displays Mean ± Std table across all 3 seeds for each stage."""
    if not Path(csv_path).exists():
        logger.log_message("No records found yet to summarize.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Filter to Progressive Cumulative ladder (removes redundant duplicate rows)
    if "type" in df.columns:
        df = df[df["type"] == "Cumulative"]

    agg = df.groupby(["model", "stage"]).agg({
        "mIoU": ["mean", "std"],
        "dice_f1": ["mean", "std"],
        "pixel_accuracy": ["mean", "std"],
        "boundary_iou": ["mean", "std"]
    }).reset_index()

    agg.columns = ["model", "stage", "mIoU_mean", "mIoU_std", "dice_mean", "dice_std", "acc_mean", "acc_std", "b_iou_mean", "b_iou_std"]
    agg["mIoU (Mean ± Std)"] = agg.apply(lambda r: f"{r['mIoU_mean']:.4f} ± {r['mIoU_std']:.4f}" if pd.notnull(r['mIoU_std']) else f"{r['mIoU_mean']:.4f}", axis=1)
    agg["Boundary IoU (Mean ± Std)"] = agg.apply(lambda r: f"{r['b_iou_mean']:.4f} ± {r['b_iou_std']:.4f}" if pd.notnull(r['b_iou_std']) else f"{r['b_iou_mean']:.4f}", axis=1)

    print("\n" + "=" * 100)
    print("=== CURRENT IMPROVEMENTS ABLATION PROGRESS (MEAN ± STD ACROSS SEEDS) ===")
    print(agg[["model", "stage", "mIoU (Mean ± Std)", "Boundary IoU (Mean ± Std)"]].to_string(index=False))
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Run Specific Improvement Stage or All Stages Across 3 Seeds")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "1", "2", "3", "4", "5", "6", "summary"],
                        help="Select improvement stage to execute: 1 (Baseline), 2 (Sliding Window), 3 (TTA), 4 (Calibration), 5 (Copy-Paste), 6 (Extended Convergence), or 'all'/'summary'")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to evaluate on")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "segformer_b2", "deeplabv3plus"])
    parser.add_argument("--seed", type=str, default="all",
                        choices=["all", "42", "123", "456"])
    parser.add_argument("--data_root", type=str, default="./data/CamVid")
    parser.add_argument("--log_dir", type=str, default="./outputs/new_logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/new_checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--extended_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_ce", type=float, default=0.5)
    parser.add_argument("--lambda_jml", type=float, default=0.5)
    args = parser.parse_args()

    master_csv = f"{args.log_dir}/05_improvements_ablation_{args.split}_steps.csv"
    logger = FormattedLogger(f"05_improvement_{args.split}_stage_{args.stage}", log_dir=args.log_dir)

    if args.stage == "summary":
        print_summary_table(master_csv, logger)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_to_run = ["segformer_b2", "deeplabv3plus"] if args.model == "all" else [args.model]
    seeds_to_run = [42, 123, 456] if args.seed == "all" else [int(args.seed)]
    stages_to_run = [1, 2, 3, 4, 5, 6] if args.stage == "all" else [int(args.stage)]

    # Load dataset calibration priors
    raw_train_set = CamVidDataset(args.data_root, split="train")
    stats = compute_dataset_statistics(raw_train_set)
    calibrator = ClassPriorCalibrator(class_frequencies=stats["frequencies"], temperature=1.0, tau=0.25)

    # Evaluation datasets
    native_trans = A.Compose([
        A.Resize(height=720, width=960, interpolation=1),
        A.Normalize(mean=CAMVID_MEAN, std=CAMVID_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])
    eval_ds = CamVidDataset(args.data_root, split=args.split, transforms=get_512_val_transforms())
    eval_ds_native = CamVidDataset(args.data_root, split=args.split, transforms=native_trans)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    # Training datasets (if Stage 1/5/6 training is explicitly requested)
    train_ds = CamVidDataset(args.data_root, split="train", transforms=get_512_train_transforms())
    num_workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))

    for model_name in models_to_run:
        for seed in seeds_to_run:
            set_seed(seed)
            exp_prefix = f"05_imp_{model_name}_seed{seed}"
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            stage1_ckpt = f"{args.checkpoint_dir}/{exp_prefix}_stage1.pth"
            cp_ckpt = f"{args.checkpoint_dir}/{exp_prefix}_cp.pth"
            final_ckpt = f"{args.checkpoint_dir}/{exp_prefix}_final_best.pth"

            # Helper to get base trained model for post-processing evaluation stages (2, 3, 4)
            def load_base_model():
                seed_ckpt = Path(f"{args.checkpoint_dir}/04_semantic_{model_name}_seed{seed}_best.pth")
                canonical_ckpt = Path(f"{args.checkpoint_dir}/04_semantic_{model_name}_best.pth")
                legacy_ckpt = Path(stage1_ckpt)

                target_ckpt = seed_ckpt if seed_ckpt.exists() else (canonical_ckpt if canonical_ckpt.exists() else legacy_ckpt)
                if not target_ckpt.exists():
                    logger.log_message(f"Notice: {target_ckpt} not found! Automatically training Stage 1 first...")
                    m_fresh = build_model(model_name, pretraining="cityscapes", num_classes=11).to(device)
                    train_stage_1(m_fresh)
                    return m_fresh

                if "segformer" in model_name:
                    from experiments.eval_test_semantic_models import load_segformer_eval
                    return load_segformer_eval(str(target_ckpt), device)
                else:
                    from experiments.eval_test_semantic_models import load_deeplab_eval
                    return load_deeplab_eval(str(target_ckpt), device)

            def train_stage_1(m):
                crit = CompositeLossCEJML(lambda_ce=args.lambda_ce, lambda_jml=args.lambda_jml)
                opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
                fit_model(m, train_loader, eval_loader, crit, opt, sched, args.epochs, logger, device, stage1_ckpt)

            new_step_records = []

            # ---------------- STAGE 1: BASELINE COMPOSITE LOSS ----------------
            if 1 in stages_to_run:
                logger.log_message(f"[{model_name.upper()} | Seed {seed}] Evaluating Stage 1: Baseline Composite Loss on {args.split.upper()}...")
                model_s1 = load_base_model()
                val_eval = SegmentationMetricEvaluator(num_classes=11, compute_boundary=True)
                with torch.no_grad():
                    for img, mask in eval_loader:
                        out = model_s1(img.to(device))
                        val_eval.update(torch.argmax(out, dim=1).cpu().numpy(), mask.numpy())
                m1 = val_eval.compute_all_metrics()
                log_step_metric(logger, "Progressive", f"1. Baseline Composite Loss ({args.split})", m1)
                rec = {"model": model_name, "seed": seed, "type": "Cumulative", "stage": "1. Baseline Composite Loss"}
                rec.update({k: m1[k] for k in ["mIoU", "pixel_accuracy", "boundary_iou"]})
                rec["dice_f1"] = m1["mean_dice_f1"]
                rec["precision"] = m1["mean_precision"]
                rec["recall"] = m1["mean_recall"]
                new_step_records.append(rec)

            # ---------------- STAGE 2: NATIVE SLIDING WINDOW ----------------
            if 2 in stages_to_run:
                logger.log_message(f"[{model_name.upper()} | Seed {seed}] Evaluating Improvement 2: Native 720x960 Sliding Window on {args.split.upper()}...")
                m = load_base_model()
                m2 = evaluate_advanced_sliding_window(
                    m, eval_ds_native, f"{model_name} (S2)", device=device,
                    crop_size=(512, 512), stride=(256, 256), scales=[1.0], flip_tta=False, tau_shift=0.0
                )
                log_step_metric(logger, "Progressive", "2. + Native Sliding Window", m2)
                rec = {"model": model_name, "seed": seed, "type": "Cumulative", "stage": "2. + Native Sliding Window"}
                rec.update({k: m2[k] for k in ["mIoU", "pixel_accuracy", "boundary_iou"]})
                rec["dice_f1"] = m2["mean_dice_f1"]
                rec["precision"] = m2["mean_precision"]
                rec["recall"] = m2["mean_recall"]
                new_step_records.append(rec)

            # ---------------- STAGE 3: MULTI-SCALE TTA PYRAMID ----------------
            if 3 in stages_to_run:
                logger.log_message(f"[{model_name.upper()} | Seed {seed}] Evaluating Improvement 3: Multi-Scale TTA Pyramid on {args.split.upper()}...")
                m = load_base_model()
                # Progressive Ladder: Native Sliding Window + Multi-Scale TTA
                m3_cum = evaluate_advanced_sliding_window(
                    m, eval_ds_native, f"{model_name} (S3)", device=device,
                    crop_size=(512, 512), stride=(256, 256), scales=[0.85, 1.0, 1.15], flip_tta=True, tau_shift=0.0
                )
                log_step_metric(logger, "Progressive", "3. + Multi-Scale TTA", m3_cum)

                rec_cum = {"model": model_name, "seed": seed, "type": "Cumulative", "stage": "3. + Multi-Scale TTA"}
                rec_cum.update({k: m3_cum[k] for k in ["mIoU", "pixel_accuracy", "boundary_iou"]})
                rec_cum["dice_f1"] = m3_cum["mean_dice_f1"]
                rec_cum["precision"] = m3_cum["mean_precision"]
                rec_cum["recall"] = m3_cum["mean_recall"]
                new_step_records.append(rec_cum)

            # ---------------- STAGE 4: CLASS PRIOR CALIBRATION / SHIFT ----------------
            if 4 in stages_to_run:
                logger.log_message(f"[{model_name.upper()} | Seed {seed}] Evaluating Improvement 4: Class Prior Calibration on {args.split.upper()}...")
                m = load_base_model()
                # Full Peak Pipeline: Native Sliding Window + Multi-Scale TTA + Prior Shift (matches run_final_78)
                tau_shift = 0.20 if "segformer" in model_name else 0.10
                m4_cum = evaluate_advanced_sliding_window(
                    m, eval_ds_native, f"{model_name} (S4 Full Pipeline)", device=device,
                    crop_size=(512, 512), stride=(256, 256), scales=[0.85, 1.0, 1.15], flip_tta=True, tau_shift=tau_shift
                )
                log_step_metric(logger, "Progressive", "4. + Class Prior Calibration (Full Peak Pipeline)", m4_cum)

                rec_cum = {"model": model_name, "seed": seed, "type": "Cumulative", "stage": "4. + Class Prior Calibration (Full Peak Pipeline)"}
                rec_cum.update({k: m4_cum[k] for k in ["mIoU", "pixel_accuracy", "boundary_iou"]})
                rec_cum["dice_f1"] = m4_cum["mean_dice_f1"]
                rec_cum["precision"] = m4_cum["mean_precision"]
                rec_cum["recall"] = m4_cum["mean_recall"]
                new_step_records.append(rec_cum)

            # ---------------- STAGE 5: CLASS-AWARE COPY-PASTE ----------------
            if 5 in stages_to_run:
                logger.log_message(f"[{model_name.upper()} | Seed {seed}] Executing Improvement 5: Class-Aware Copy-Paste Training ({args.epochs} eps)...")
                cp_aug = ClassAwareCopyPaste(target_classes=[2, 6, 9, 10], p=0.5)
                cp_aug.populate_bank(raw_train_set)
                train_ds_cp = CamVidDataset(args.data_root, split="train", transforms=get_512_train_transforms(), copy_paste_augmenter=cp_aug)
                train_loader_cp = DataLoader(train_ds_cp, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

                m_cp = build_model(model_name, pretraining="cityscapes", num_classes=11).to(device)
                crit = CompositeLossCEJML(lambda_ce=args.lambda_ce, lambda_jml=args.lambda_jml)
                opt = torch.optim.AdamW(m_cp.parameters(), lr=args.lr, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

                fit_cp = fit_model(m_cp, train_loader_cp, eval_loader, crit, opt, sched, args.epochs, logger, device, cp_ckpt)
                m5_ind = fit_cp["best_metrics"]
                m5_cum = evaluate_with_improvements(m_cp, eval_ds, device=device, use_tta=True, calibrator=calibrator)

                log_step_metric(logger, "Individual", "5. Standalone Copy-Paste", m5_ind)
                log_step_metric(logger, "Cumulative", "5. + Class-Aware Copy-Paste", m5_cum)

                rec_ind = {"model": model_name, "seed": seed, "type": "Individual", "stage": "5. Standalone Copy-Paste"}
                rec_ind.update({k: m5_ind[k] for k in ["mIoU", "pixel_accuracy", "boundary_iou"]})
                rec_ind["dice_f1"] = m5_ind["mean_dice_f1"]
                rec_ind["precision"] = m5_ind["mean_precision"]
                rec_ind["recall"] = m5_ind["mean_recall"]
                new_step_records.append(rec_ind)

                rec_cum = {"model": model_name, "seed": seed, "type": "Cumulative", "stage": "5. + Class-Aware Copy-Paste"}
                rec_cum.update({k: m5_cum[k] for k in ["mIoU", "pixel_accuracy", "boundary_iou"]})
                rec_cum["dice_f1"] = m5_cum["mean_dice_f1"]
                rec_cum["precision"] = m5_cum["mean_precision"]
                rec_cum["recall"] = m5_cum["mean_recall"]
                new_step_records.append(rec_cum)

            # ---------------- STAGE 6: EXTENDED CONVERGENCE ----------------
            if 6 in stages_to_run:
                logger.log_message(f"[{model_name.upper()} | Seed {seed}] Executing Improvement 6: Cosine Extended Convergence ({args.extended_epochs} eps)...")
                cp_aug = ClassAwareCopyPaste(target_classes=[2, 6, 9, 10], p=0.5)
                cp_aug.populate_bank(raw_train_set)
                train_ds_cp = CamVidDataset(args.data_root, split="train", transforms=get_512_train_transforms(), copy_paste_augmenter=cp_aug)
                train_loader_cp = DataLoader(train_ds_cp, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

                m_ext = build_model(model_name, pretraining="cityscapes", num_classes=11).to(device)
                crit = CompositeLossCEJML(lambda_ce=args.lambda_ce, lambda_jml=args.lambda_jml)
                opt = torch.optim.AdamW(m_ext.parameters(), lr=args.lr, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.extended_epochs, eta_min=1e-6)

                fit_model(m_ext, train_loader_cp, eval_loader, crit, opt, sched, args.extended_epochs, logger, device, final_ckpt)
                m6 = evaluate_with_improvements(m_ext, eval_ds_native, device=device, use_sliding_window=True, calibrator=calibrator)
                log_step_metric(logger, "Cumulative", "6. + Cosine Extended Convergence (Full Pipeline)", m6)

                rec_ext = {"model": model_name, "seed": seed, "type": "Cumulative", "stage": "6. + Cosine Extended Convergence (Full Pipeline)"}
                rec_ext.update({k: m6[k] for k in ["mIoU", "pixel_accuracy", "boundary_iou"]})
                rec_ext["dice_f1"] = m6["mean_dice_f1"]
                rec_ext["precision"] = m6["mean_precision"]
                rec_ext["recall"] = m6["mean_recall"]
                new_step_records.append(rec_ext)

            # Immediately persist this step's records
            if new_step_records:
                update_and_persist_record(master_csv, new_step_records, logger)

    print_summary_table(master_csv, logger)

if __name__ == "__main__":
    main()

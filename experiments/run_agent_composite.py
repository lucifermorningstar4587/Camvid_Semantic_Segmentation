import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from data.dataset import CamVidDataset, get_training_transforms, get_validation_transforms
from models.model_factory import build_model
from losses.composite_loss import CompositeLossCEJML
from agent.loss_agent import GoalDrivenLossSearchAgent
from utils.logger import FormattedLogger
from utils.trainer import fit_model, train_one_epoch, validate
from utils.metrics import SegmentationMetricEvaluator
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Agent-Driven Composite Loss Optimization and Training")
    parser.add_argument("--data_root", type=str, default="./data/CamVid")
    parser.add_argument("--log_dir", type=str, default="./outputs/logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/checkpoints")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "search", "train"],
                        help="Execution mode: 'search' (Bayesian agent search only), 'train' (model training only), or 'all'")
    parser.add_argument("--model", type=str, default="all", choices=["all", "deeplabv3plus", "segformer_b2"],
                        help="Model(s) to train in 'train' mode: 'deeplabv3plus', 'segformer_b2', or 'all'")
    parser.add_argument("--agent_trials", type=int, default=12,
                        help="Number of Optuna TPE search trials")
    parser.add_argument("--search_epochs", type=int, default=3,
                        help="Epochs per proxy trial during agent search")
    parser.add_argument("--search_deeplab", action="store_true",
                        help="Run a separate agent search for DeepLab‑V3+ (in addition to the default SegFormer search)")
    parser.add_argument("--full_epochs", type=int, default=25,
                        help="Full training epochs for composite models")
    parser.add_argument("--lambda_ce", type=float, default=None,
                        help="Manual weight for CE loss (if training directly without search)")
    parser.add_argument("--lambda_jml", type=float, default=None,
                        help="Manual weight for JML loss (if training directly without search)")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Agent Search & Composite Pipeline on device: {device} | Mode: {args.mode.upper()}")

    # Datasets
    train_dataset = CamVidDataset(args.data_root, split="train", transforms=get_training_transforms())
    val_dataset = CamVidDataset(args.data_root, split="val", transforms=get_validation_transforms())

    num_workers = 0 if os.name == "nt" else 2
    use_pin_memory = (device == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    weights_json_path = Path(args.log_dir) / "04_agent_best_weights.json"
    best_lambda_ce = None
    best_lambda_jml = None

    # =========================================================================
    # STEP 1: GOAL-DRIVEN AGENT SEARCH
    # =========================================================================
    if args.mode in ["search", "all"]:
        agent_logger = FormattedLogger("04_agent_optimization", log_dir=args.log_dir)
        agent_logger.log_message(f"Starting Bayesian Loss-Weight Search Agent ({args.agent_trials} trials, {args.search_epochs} epochs/trial)...")

        def evaluate_loss_weights(lambda_ce: float, lambda_jml: float, trial_idx: int) -> float:
            proxy_model = build_model("segformer_b2", pretraining="cityscapes", num_classes=11).to(device)
            composite_crit = CompositeLossCEJML(lambda_ce=lambda_ce, lambda_jml=lambda_jml)
            proxy_opt = torch.optim.AdamW(proxy_model.parameters(), lr=args.lr, weight_decay=1e-4)
            
            train_eval = SegmentationMetricEvaluator()
            val_eval = SegmentationMetricEvaluator()

            best_val = 0.0
            for ep in range(1, args.search_epochs + 1):
                train_one_epoch(proxy_model, train_loader, composite_crit, proxy_opt, device, train_eval)
                _, v_metrics = validate(proxy_model, val_loader, composite_crit, device, val_eval)
                if v_metrics["mIoU"] > best_val:
                    best_val = v_metrics["mIoU"]

            return best_val

        agent = GoalDrivenLossSearchAgent(
            study_name="ce_jml_loss_weights",
            n_trials=args.agent_trials,
            log_dir=args.log_dir,
            random_seed=42
        )

        agent_result = agent.run_agent(eval_fn=evaluate_loss_weights)
        best_lambda_ce = agent_result["best_lambda_ce"]
        best_lambda_jml = agent_result["best_lambda_jml"]

        # Persist discovered best weights to JSON for subsequent training
        weights_info = {
            "lambda_ce": float(best_lambda_ce),
            "lambda_jml": float(best_lambda_jml),
            "best_val_miou": float(agent_result["best_val_miou"]),
            "best_trial": int(agent_result["best_trial_num"]),
            "trials_csv": agent_result["trials_csv"]
        }
        # Save initial (SegFormer) weights
        with open(weights_json_path, "w", encoding="utf-8") as f:
            json.dump(weights_info, f, indent=4)

        # If user requested a separate search for DeepLab‑V3+, run it now and augment the JSON
        if args.search_deeplab:
            agent_logger.log_message("Starting separate Bayesian search for DeepLab‑V3+ loss weights...")
            def evaluate_deeplab_weights(lambda_ce: float, lambda_jml: float, trial_idx: int) -> float:
                proxy_model = build_model("deeplabv3plus", pretraining="cityscapes", num_classes=11).to(device)
                composite_crit = CompositeLossCEJML(lambda_ce=lambda_ce, lambda_jml=lambda_jml)
                proxy_opt = torch.optim.AdamW(proxy_model.parameters(), lr=args.lr, weight_decay=1e-4)
                train_eval = SegmentationMetricEvaluator()
                val_eval = SegmentationMetricEvaluator()
                best_val = 0.0
                for ep in range(1, args.search_epochs + 1):
                    train_one_epoch(proxy_model, train_loader, composite_crit, proxy_opt, device, train_eval)
                    _, v_metrics = validate(proxy_model, val_loader, composite_crit, device, val_eval)
                    if v_metrics["mIoU"] > best_val:
                        best_val = v_metrics["mIoU"]
                return best_val

            deeplab_agent = GoalDrivenLossSearchAgent(
                study_name="ce_jml_loss_weights_deeplab",
                n_trials=args.agent_trials,
                log_dir=args.log_dir,
                random_seed=42,
            )
            deeplab_result = deeplab_agent.run_agent(eval_fn=evaluate_deeplab_weights)
            # Update JSON with DeepLab‑specific weights
            weights_info.update({
                "deeplab_lambda_ce": float(deeplab_result["best_lambda_ce"]),
                "deeplab_lambda_jml": float(deeplab_result["best_lambda_jml"]),
                "deeplab_best_val_miou": float(deeplab_result["best_val_miou"]),
                "deeplab_best_trial": int(deeplab_result["best_trial_num"]),
                "deeplab_trials_csv": deeplab_result["trials_csv"]
            })
            with open(weights_json_path, "w", encoding="utf-8") as f:
                json.dump(weights_info, f, indent=4)
            agent_logger.log_message(f"DeepLab‑V3+ search completed: lambda_ce={deeplab_result['best_lambda_ce']}, lambda_jml={deeplab_result['best_lambda_jml']}")
            agent_logger.log_message(f"Combined weights saved to: {weights_json_path.resolve()}")

        agent_logger.log_message(f"Agent Search Completed! Optimal weights: lambda_ce={best_lambda_ce}, lambda_jml={best_lambda_jml} (Val mIoU: {agent_result['best_val_miou']:.4f})")
        agent_logger.log_message(f"Best weights configuration saved to: {weights_json_path.resolve()}")

        if args.mode == "search":
            print("\n" + "=" * 80)
            print(f"[AGENT SEARCH COMPLETED SUCCESSFULLY]")
            print(f"   Optimal Lambda CE:  {best_lambda_ce}")
            print(f"   Optimal Lambda JML: {best_lambda_jml}")
            print(f"   Config saved to: {weights_json_path}")
            print(f"   Now execute training mode whenever ready:")
            print(f"     !python -u experiments/run_agent_composite.py --mode train --full_epochs {args.full_epochs}")
            print("=" * 80)
            return

    # =========================================================================
    # STEP 2: FULL COMPOSITE LOSS MODEL TRAINING
    # =========================================================================
    if args.mode in ["train", "all"]:
        # Resolve weights: manual args > saved json > defaults
        if args.lambda_ce is not None and args.lambda_jml is not None:
            best_lambda_ce = args.lambda_ce
            best_lambda_jml = args.lambda_jml
            print(f"Using manually specified common weights: lambda_ce={best_lambda_ce}, lambda_jml={best_lambda_jml}")
        elif weights_json_path.exists():
            with open(weights_json_path, "r", encoding="utf-8") as f:
                saved_weights = json.load(f)
            # generic weights (fallback)
            best_lambda_ce = float(saved_weights.get("lambda_ce", 0.5))
            best_lambda_jml = float(saved_weights.get("lambda_jml", 0.5))
            # model‑specific weights if present
            deeplab_lambda_ce = saved_weights.get("deeplab_lambda_ce")
            deeplab_lambda_jml = saved_weights.get("deeplab_lambda_jml")
            segformer_lambda_ce = saved_weights.get("segformer_lambda_ce")
            segformer_lambda_jml = saved_weights.get("segformer_lambda_jml")
            print(f"Loaded weights from {weights_json_path}: common lambda_ce={best_lambda_ce}, lambda_jml={best_lambda_jml}")
            if deeplab_lambda_ce is not None or deeplab_lambda_jml is not None:
                print(f"  DeepLab‑V3+ overrides: lambda_ce={deeplab_lambda_ce}, lambda_jml={deeplab_lambda_jml}")
            if segformer_lambda_ce is not None or segformer_lambda_jml is not None:
                print(f"  SegFormer‑B2 overrides: lambda_ce={segformer_lambda_ce}, lambda_jml={segformer_lambda_jml}")
        else:
            best_lambda_ce = 0.5
            best_lambda_jml = 0.5
            deeplab_lambda_ce = deeplab_lambda_jml = segformer_lambda_ce = segformer_lambda_jml = None
            print(f"Notice: No agent weights file found. Using balanced defaults: lambda_ce=0.5, lambda_jml=0.5")

        models = ["deeplabv3plus", "segformer_b2"] if args.model == "all" else [args.model.lower()]
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        summary_path = Path(args.log_dir) / "04_agent_composite_summary.csv"
        existing_records = []
        if summary_path.exists():
            try:
                existing_records = pd.read_csv(summary_path).to_dict("records")
            except Exception:
                existing_records = []

        new_results = []

        for model_name in models:
            exp_name = f"04_composite_{model_name}"
            logger = FormattedLogger(exp_name, log_dir=args.log_dir)
            logger.log_message(f"===========================================================")
            logger.log_message(f"Training {model_name.upper()} with Agent-Optimized Composite Loss: {best_lambda_ce}*CE + {best_lambda_jml}*JML ({args.full_epochs} epochs)")
            logger.log_message(f"===========================================================")

            model = build_model(model_name, pretraining="cityscapes", num_classes=11)
            criterion = CompositeLossCEJML(lambda_ce=best_lambda_ce, lambda_jml=best_lambda_jml)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.full_epochs, eta_min=1e-6)

            ckpt_path = f"{args.checkpoint_dir}/{exp_name}_best.pth"
            fit_res = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=args.full_epochs,
                logger=logger,
                device=device,
                save_path=ckpt_path
            )

            record = {
                "model": model_name,
                "lambda_ce": best_lambda_ce,
                "lambda_jml": best_lambda_jml,
                "best_val_miou": round(float(fit_res["best_val_miou"]), 4),
                "best_val_dice": round(float(fit_res["best_metrics"].get("mean_dice_f1", 0.0)), 4),
                "best_val_acc": round(float(fit_res["best_metrics"].get("pixel_accuracy", 0.0)), 4),
                "best_val_biou": round(float(fit_res["best_metrics"].get("boundary_iou", 0.0)), 4),
                "checkpoint": ckpt_path
            }
            new_results.append(record)
            logger.save_summary(record)

            # Update summary csv incrementally
            updated_records = [r for r in existing_records if r.get("model") != model_name] + new_results
            pd.DataFrame(updated_records).to_csv(summary_path, index=False)

        print("\n" + "=" * 80)
        print("=== AGENT COMPOSITE LOSS TRAINING SUMMARY ===")
        print(pd.read_csv(summary_path).to_string(index=False))
        print(f"Summary saved to: {summary_path.resolve()}")
        print("=" * 80)

if __name__ == "__main__":
    main()

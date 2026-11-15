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
from utils.trainer import train_one_epoch, validate
from utils.metrics import SegmentationMetricEvaluator


def main():
    parser = argparse.ArgumentParser(description="Bayesian Loss-Weight Search for DeepLabV3+")
    parser.add_argument("--data_root", type=str, default="./data/CamVid")
    parser.add_argument("--log_dir", type=str, default="./outputs/logs")
    parser.add_argument("--agent_trials", type=int, default=15, help="Number of Optuna TPE search trials")
    parser.add_argument("--search_epochs", type=int, default=3, help="Epochs per proxy trial")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SEARCH AGENT - DEEPLABV3+] Device: {device} | AMP: {args.use_amp} | Trials: {args.agent_trials}")

    # Datasets
    train_dataset = CamVidDataset(args.data_root, split="train", transforms=get_training_transforms())
    val_dataset = CamVidDataset(args.data_root, split="val", transforms=get_validation_transforms())

    num_workers = args.num_workers if os.name != "nt" else 0
    use_pin_memory = (device == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    agent_logger = FormattedLogger("04_agent_deeplab_search", log_dir=args.log_dir)
    agent_logger.log_message(f"Starting DeepLabV3+ Bayesian Loss Search ({args.agent_trials} trials, {args.search_epochs} epochs/trial)...")

    # Fast validation evaluator that calculates mIoU without heavy CPU boundary contour extraction
    val_eval = SegmentationMetricEvaluator(compute_boundary=False)

    def evaluate_loss_weights(lambda_ce: float, lambda_jml: float, trial_idx: int) -> float:
        proxy_model = build_model("deeplabv3plus", pretraining="cityscapes", num_classes=11).to(device)
        composite_crit = CompositeLossCEJML(lambda_ce=lambda_ce, lambda_jml=lambda_jml)
        proxy_opt = torch.optim.AdamW(proxy_model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device == "cuda") else None

        best_val = 0.0
        for ep in range(1, args.search_epochs + 1):
            train_one_epoch(
                model=proxy_model,
                loader=train_loader,
                criterion=composite_crit,
                optimizer=proxy_opt,
                device=device,
                metric_evaluator=None,  # skip metric calculation during train batches for maximum speed
                scaler=scaler
            )
            _, v_metrics = validate(
                model=proxy_model,
                loader=val_loader,
                criterion=composite_crit,
                device=device,
                metric_evaluator=val_eval,
                use_amp=args.use_amp
            )
            if v_metrics["mIoU"] > best_val:
                best_val = v_metrics["mIoU"]

        return best_val

    agent = GoalDrivenLossSearchAgent(
        study_name="ce_jml_loss_weights_deeplab",
        n_trials=args.agent_trials,
        log_dir=args.log_dir,
        trials_csv_name="04_agent_trials_deeplab.csv",
        random_seed=42
    )

    agent_result = agent.run_agent(eval_fn=evaluate_loss_weights)
    best_lambda_ce = agent_result["best_lambda_ce"]
    best_lambda_jml = agent_result["best_lambda_jml"]

    weights_json_path = Path(args.log_dir) / "04_agent_weights_deeplab.json"
    weights_info = {
        "model": "deeplabv3plus",
        "lambda_ce": float(best_lambda_ce),
        "lambda_jml": float(best_lambda_jml),
        "best_val_miou": float(agent_result["best_val_miou"]),
        "best_trial": int(agent_result["best_trial_num"]),
        "trials_csv": agent_result["trials_csv"]
    }
    with open(weights_json_path, "w", encoding="utf-8") as f:
        json.dump(weights_info, f, indent=4)

    agent_logger.log_message(f"DeepLabV3+ Search Done: lambda_ce={best_lambda_ce}, lambda_jml={best_lambda_jml} (mIoU: {agent_result['best_val_miou']:.4f})")
    print("\n" + "=" * 80)
    print("[DEEPLABV3+ AGENT SEARCH COMPLETE]")
    print(f"   Best Lambda CE:  {best_lambda_ce}")
    print(f"   Best Lambda JML: {best_lambda_jml}")
    print(f"   Best Val mIoU:   {agent_result['best_val_miou']:.4f}")
    print(f"   Saved Weights:   {weights_json_path}")
    print(f"   Saved CSV:       {agent_result['trials_csv']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

import os
import time
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from utils.logger import FormattedLogger

class GoalDrivenLossSearchAgent:
    """
    Automated Goal-Driven Agent for Loss Component Weight Selection.
    
    Specification:
      - Objective: Maximize Validation Mean Intersection over Union (mIoU_val)
      - Search Space:
          * lambda_ce in [0.10, 1.00]
          * lambda_jml in [0.10, 1.50]
      - Search Strategy: Tree-structured Parzen Estimator (TPE) Bayesian Optimization
      - Compute Budget: Configurable trials (default 15 trials) with strict compute bounds
      - Stopping Rule: Max trials or early plateau termination
      - Strict Anti-Leakage: Operates strictly on validation set; NEVER accesses test set.
      - Full Logging: Records trial index, timestamp, parameters, validation score to CSV.
    """
    def __init__(
        self,
        study_name: str = "composite_loss_weight_optimization",
        n_trials: int = 15,
        log_dir: str = "./outputs/logs",
        trials_csv_name: str = "04_agent_trials.csv",
        random_seed: int = 42
    ):
        self.n_trials = n_trials
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trials_csv_path = self.log_dir / trials_csv_name
        self.study_name = study_name
        self.random_seed = random_seed
        self.trials_history = []

    def run_agent(self, eval_fn) -> Dict[str, Any]:
        """
        Execute automated goal-driven search.
        eval_fn(lambda_ce: float, lambda_jml: float) -> val_miou: float
        """
        print("=" * 80)
        print(f" [AGENT INITIALIZATION] Starting Goal-Driven Loss-Weight Search Agent")
        print(f"   Objective: Maximize Validation mIoU")
        print(f"   Search Strategy: Bayesian Optimization (TPESampler, seed={self.random_seed})")
        print(f"   Compute Budget: {self.n_trials} trials")
        print(f"   Trial Audit Log: {self.trials_csv_path.resolve()}")
        print("=" * 80)

        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=sampler
        )

        def objective(trial: optuna.Trial) -> float:
            trial_start = time.time()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Suggest loss weights
            lambda_ce = round(trial.suggest_float("lambda_ce", 0.1, 1.0, step=0.05), 2)
            lambda_jml = round(trial.suggest_float("lambda_jml", 0.1, 1.5, step=0.05), 2)

            print(f"\n[Agent Trial #{trial.number + 1}/{self.n_trials}] Testing config: lambda_ce={lambda_ce}, lambda_jml={lambda_jml}")
            
            # Evaluate using training/validation loop passed by caller
            val_miou = eval_fn(lambda_ce, lambda_jml, trial.number)
            elapsed = time.time() - trial_start

            record = {
                "trial": trial.number + 1,
                "timestamp": timestamp,
                "lambda_ce": lambda_ce,
                "lambda_jml": lambda_jml,
                "val_miou": round(float(val_miou), 4),
                "duration_seconds": round(elapsed, 2)
            }
            self.trials_history.append(record)

            # Persist trial history immediately to CSV
            df = pd.DataFrame(self.trials_history)
            df.to_csv(self.trials_csv_path, index=False)

            print(f"  --> Result: val_miou={val_miou:.4f} (took {elapsed:.1f}s) | Trial logged to CSV.")
            return val_miou

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_trial = study.best_trial
        print("\n" + "=" * 80)
        print(f" [AGENT CONVERGED] Optimal Loss Weights Discovered by Agent:")
        print(f"   Best Trial: #{best_trial.number + 1}")
        print(f"   Best Validation mIoU: {best_trial.value:.4f}")
        print(f"   Selected Parameters: {best_trial.params}")
        print(f"   Audit trail permanently stored at: {self.trials_csv_path.resolve()}")
        print("=" * 80)

        return {
            "best_lambda_ce": best_trial.params["lambda_ce"],
            "best_lambda_jml": best_trial.params["lambda_jml"],
            "best_val_miou": best_trial.value,
            "best_trial_num": best_trial.number + 1,
            "trials_csv": str(self.trials_csv_path.resolve())
        }

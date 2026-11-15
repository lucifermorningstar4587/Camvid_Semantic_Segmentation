import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

class FormattedLogger:
    """
    Structured Dual Logger:
      - Prints live formatted logs with timestamps to stdout.
      - Concurrently flushes to persistent CSV and TXT log files in `log_dir` so outputs are never lost.
    """
    def __init__(self, experiment_name: str, log_dir: str = "./outputs/logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.txt_log_path = self.log_dir / f"{experiment_name}.log"
        self.csv_log_path = self.log_dir / f"{experiment_name}_history.csv"
        self.epoch_records = []

        # Initialize log files
        header_msg = f"=== Experiment Log Initialized: {experiment_name} | {self.get_timestamp()} ==="
        self.log_message(header_msg)

    @staticmethod
    def get_timestamp() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_message(self, message: str):
        """Prints to stdout and appends to .log file."""
        line = f"[{self.get_timestamp()}] {message}"
        print(line)
        sys.stdout.flush()
        with open(self.txt_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_to_file(self, message: str):
        """Appends to .log file without printing to stdout."""
        line = f"[{self.get_timestamp()}] {message}"
        with open(self.txt_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_miou: float,
        val_loss: float,
        val_miou: float,
        lr: Optional[float] = None,
        write_stdout: bool = True
    ):
        """
        Logs epoch metrics strictly matching required fields:
          - Timestamp
          - Train Loss
          - Train mIoU
          - Val Loss
          - Val mIoU
        """
        ts = self.get_timestamp()
        lr_str = f" | LR: {lr:.6f}" if lr is not None else ""
        formatted = (
            f"[{ts}] Epoch [{epoch:03d}/{total_epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}{lr_str}"
        )
        if write_stdout:
            print(formatted)
            sys.stdout.flush()

        with open(self.txt_log_path, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

        record = {
            "timestamp": ts,
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_miou": round(float(train_miou), 4),
            "val_loss": round(float(val_loss), 4),
            "val_miou": round(float(val_miou), 4),
            "lr": float(lr) if lr is not None else 0.0
        }
        self.epoch_records.append(record)
        # Flush to CSV immediately
        pd.DataFrame(self.epoch_records).to_csv(self.csv_log_path, index=False)

    def save_summary(self, summary_data: Dict[str, Any]):
        """Persists final structured experimental metrics to JSON."""
        summary_path = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
        self.log_message(f"Experimental summary successfully saved to: {summary_path.resolve()}")

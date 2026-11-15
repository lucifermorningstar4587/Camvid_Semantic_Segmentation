import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.logger import FormattedLogger
from utils.metrics import SegmentationMetricEvaluator

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    metric_evaluator: Optional[SegmentationMetricEvaluator] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    **kwargs
) -> Tuple[float, float]:
    """Train for one epoch cleanly with optional AMP support."""
    model.train()
    if metric_evaluator is not None:
        metric_evaluator.reset()
    total_loss = 0.0
    total_batches = len(loader)
    use_amp = (scaler is not None) and (device == "cuda")

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        if metric_evaluator is not None:
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            metric_evaluator.update(preds, targets_np)

    train_miou = metric_evaluator.compute_all_metrics()["mIoU"] if metric_evaluator is not None else 0.0
    avg_loss = total_loss / max(1, total_batches)
    return avg_loss, train_miou


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    metric_evaluator: SegmentationMetricEvaluator,
    use_amp: bool = False,
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """Validate model cleanly without noisy progress bars."""
    model.eval()
    metric_evaluator.reset()
    total_loss = 0.0
    total_batches = len(loader)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            metric_evaluator.update(preds, targets_np)

    metrics = metric_evaluator.compute_all_metrics()
    avg_loss = total_loss / max(1, total_batches)
    return avg_loss, metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    logger: FormattedLogger,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    use_amp: bool = True
) -> Dict[str, Any]:
    """
    Standard training loop with clean, epoch-wise transition messages and a single
    structured summary box per completed epoch. Completely eliminates screen-flooding progress bars.
    """
    model = model.to(device)
    train_evaluator = SegmentationMetricEvaluator()
    val_evaluator = SegmentationMetricEvaluator()
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device == "cuda") else None

    best_val_miou = 0.0
    best_metrics = {}
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n>>> Epoch [{epoch:02d}/{epochs:02d}] Training in progress ({n_train_batches} batches on {device.upper()})...", flush=True)

        # 1. Train one epoch
        train_loss, train_miou = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            metric_evaluator=train_evaluator,
            scaler=scaler
        )

        print(f"--> Epoch [{epoch:02d}/{epochs:02d}] Validating ({n_val_batches} batches)...", flush=True)

        # 2. Validate
        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metric_evaluator=val_evaluator,
            use_amp=(use_amp and device == "cuda")
        )
        val_miou = val_metrics["mIoU"]
        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step()

        # 3. Checkpoint
        is_best = False
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_metrics = val_metrics
            is_best = True
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)

        elapsed = time.time() - epoch_start
        ckpt_tag = " [NEW BEST CHECKPOINT SAVED]" if is_best else ""

        # 4. Print clean single summary box per epoch
        summary_box = (
            f"+------------------------------------------------------------------------------------+\n"
            f"| Epoch [{epoch:02d}/{epochs:02d}] Summary | Duration: {elapsed:5.1f}s | Device: {device.upper():<6s}                        |\n"
            f"| Train: Loss = {train_loss:.4f} | mIoU = {train_miou:.4f}                                                    |\n"
            f"| Val:   Loss = {val_loss:.4f} | mIoU = {val_miou:.4f} (Best: {best_val_miou:.4f}){ckpt_tag:<32s}|\n"
            f"| Learning Rate: {current_lr:.2e}                                                                  |\n"
            f"+------------------------------------------------------------------------------------+"
        )
        print(summary_box, flush=True)

        # 5. Persist to file
        logger.log_epoch(
            epoch=epoch,
            total_epochs=epochs,
            train_loss=train_loss,
            train_miou=train_miou,
            val_loss=val_loss,
            val_miou=val_miou,
            lr=current_lr,
            write_stdout=False
        )

        if is_best:
            logger.log_to_file(f"--> [CHECKPOINT] New best model saved! (Val mIoU: {best_val_miou:.4f}) to {save_path}")

    return {
        "best_val_miou": best_val_miou,
        "best_metrics": best_metrics
    }

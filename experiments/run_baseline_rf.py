import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import numpy as np
import cv2
from tqdm import tqdm
from data.dataset import CamVidDataset, CAMVID_CLASSES
from models.baseline_rf import TraditionalRandomForestBaseline
from utils.metrics import SegmentationMetricEvaluator
from utils.logger import FormattedLogger

def main():
    parser = argparse.ArgumentParser(description="Run Traditional ML Random Forest Baseline on CamVid")
    parser.add_argument("--data_root", type=str, default="./data/CamVid")
    parser.add_argument("--log_dir", type=str, default="./outputs/logs")
    parser.add_argument("--max_samples", type=int, default=1500)
    args = parser.parse_args()

    logger = FormattedLogger("01_baseline_rf", log_dir=args.log_dir)
    logger.log_message("Starting Traditional Random Forest Baseline Training...")

    # Load Train and Test datasets
    train_dataset = CamVidDataset(args.data_root, split="train")
    test_dataset = CamVidDataset(args.data_root, split="test")

    rf_baseline = TraditionalRandomForestBaseline(n_estimators=50, max_depth=16, random_state=42)
    rf_baseline.fit_from_dataset(train_dataset, max_samples_per_image=args.max_samples)

    logger.log_message(f"Evaluating Random Forest on official Test set ({len(test_dataset)} images)...")
    evaluator = SegmentationMetricEvaluator(num_classes=11)

    for i in tqdm(range(len(test_dataset)), desc="Evaluating Test Set", file=sys.stdout):
        img_p = str(test_dataset.images[i])
        lbl_p = str(test_dataset.labels[i])
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(lbl_p, cv2.IMREAD_UNCHANGED)
        if lbl.ndim == 3:
            from data.dataset import rgb_to_mask
            lbl = rgb_to_mask(lbl)

        h, w = lbl.shape
        img = cv2.resize(img, (w, h))
        pred = rf_baseline.predict_image(img)
        evaluator.update(pred, lbl)

    metrics = evaluator.compute_all_metrics()
    logger.log_message("=== Random Forest Baseline Test Results ===")
    logger.log_message(f"Test mIoU: {metrics['mIoU']:.4f}")
    logger.log_message(f"Test Mean Dice/F1: {metrics['mean_dice_f1']:.4f}")
    logger.log_message(f"Test Mean Precision: {metrics['mean_precision']:.4f}")
    logger.log_message(f"Test Mean Recall: {metrics['mean_recall']:.4f}")
    logger.log_message(f"Test Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    logger.log_message(f"Test Boundary IoU: {metrics['boundary_iou']:.4f}")

    logger.save_summary(metrics)

if __name__ == "__main__":
    main()

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple

class SegmentationMetricEvaluator:
    """
    Computes all standard and fine-grained segmentation evaluation metrics:
      - mIoU (Mean Intersection over Union)
      - Mean Dice / F1 Score
      - Mean Precision
      - Mean Recall
      - Pixel Accuracy
      - Per-Class IoUs
      - Boundary IoU (Cheng et al., CVPR 2021)
      - Confusion Matrix
    """
    def __init__(self, num_classes: int = 11, ignore_index: int = 255, compute_boundary: bool = True):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.compute_boundary = compute_boundary
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.boundary_intersections = np.zeros(num_classes, dtype=np.float64)
        self.boundary_unions = np.zeros(num_classes, dtype=np.float64)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.boundary_intersections = np.zeros(self.num_classes, dtype=np.float64)
        self.boundary_unions = np.zeros(self.num_classes, dtype=np.float64)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        preds: (B, H, W) or (H, W) class indices
        targets: (B, H, W) or (H, W) ground truth with ignore_index
        """
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)

        valid_mask = (targets_flat != self.ignore_index) & (targets_flat < self.num_classes) & (preds_flat < self.num_classes)
        p = preds_flat[valid_mask]
        t = targets_flat[valid_mask]

        indices = self.num_classes * t + p
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

        if not self.compute_boundary:
            return

        # Compute Boundary IoU per 2D image
        if preds.ndim == 2:
            preds_batch = preds[None, ...]
            targets_batch = targets[None, ...]
        else:
            preds_batch = preds
            targets_batch = targets

        for b in range(preds_batch.shape[0]):
            self._update_boundary_iou(preds_batch[b], targets_batch[b])

    def _extract_boundary(self, mask: np.ndarray, dilation: int = 2) -> np.ndarray:
        """Extract boundary contour of a binary mask using morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation * 2 + 1, dilation * 2 + 1))
        mask_uint8 = (mask > 0).astype(np.uint8)
        dilated = cv2.dilate(mask_uint8, kernel)
        eroded = cv2.erode(mask_uint8, kernel)
        return (dilated - eroded) > 0

    def _update_boundary_iou(self, pred: np.ndarray, target: np.ndarray):
        valid = (target != self.ignore_index)
        for c in range(self.num_classes):
            gt_c = (target == c) & valid
            pr_c = (pred == c) & valid
            if gt_c.sum() == 0 and pr_c.sum() == 0:
                continue

            gt_b = self._extract_boundary(gt_c)
            pr_b = self._extract_boundary(pr_c)

            intersection = np.logical_and(gt_b, pr_b).sum()
            union = np.logical_or(gt_b, pr_b).sum()

            self.boundary_intersections[c] += intersection
            self.boundary_unions[c] += union

    def compute_all_metrics(self) -> Dict[str, any]:
        """Calculates final aggregated metrics across all evaluated samples."""
        cm = self.confusion_matrix
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Per-class IoU: TP / (TP + FP + FN)
        denominator = tp + fp + fn
        iou_per_class = np.divide(tp, denominator, out=np.zeros_like(tp), where=(denominator > 0))

        # Dice / F1: 2*TP / (2*TP + FP + FN)
        dice_den = 2 * tp + fp + fn
        dice_per_class = np.divide(2 * tp, dice_den, out=np.zeros_like(tp), where=(dice_den > 0))

        # Precision: TP / (TP + FP)
        prec_den = tp + fp
        precision_per_class = np.divide(tp, prec_den, out=np.zeros_like(tp), where=(prec_den > 0))

        # Recall: TP / (TP + FN)
        rec_den = tp + fn
        recall_per_class = np.divide(tp, rec_den, out=np.zeros_like(tp), where=(rec_den > 0))

        # Pixel Accuracy: sum(TP) / total_valid_pixels
        total_pixels = cm.sum()
        pixel_accuracy = float(tp.sum() / max(1, total_pixels))

        # Boundary IoU
        b_iou_per_class = np.divide(
            self.boundary_intersections,
            self.boundary_unions,
            out=np.zeros_like(self.boundary_intersections),
            where=(self.boundary_unions > 0)
        )

        # Filter out classes with no ground truth or predictions
        present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0
        mIoU = float(iou_per_class[present].mean()) if present.any() else 0.0
        mean_dice = float(dice_per_class[present].mean()) if present.any() else 0.0
        mean_precision = float(precision_per_class[present].mean()) if present.any() else 0.0
        mean_recall = float(recall_per_class[present].mean()) if present.any() else 0.0
        mean_boundary_iou = float(b_iou_per_class[present].mean()) if present.any() else 0.0

        return {
            "mIoU": round(mIoU, 4),
            "mean_dice_f1": round(mean_dice, 4),
            "mean_precision": round(mean_precision, 4),
            "mean_recall": round(mean_recall, 4),
            "pixel_accuracy": round(pixel_accuracy, 4),
            "boundary_iou": round(mean_boundary_iou, 4),
            "per_class_iou": {c: round(float(iou_per_class[c]), 4) for c in range(self.num_classes)},
            "per_class_dice": {c: round(float(dice_per_class[c]), 4) for c in range(self.num_classes)},
            "confusion_matrix": cm.tolist()
        }

"""
run_final_78_benchmark.py
=========================
Completely self-contained, zero-leakage benchmark on all 233 held-out test frames.
Evaluates both SegFormer-B2 and DeepLabV3+ with advanced inference:
  - 3-Scale Multi-Scale Pyramid [0.85, 1.0, 1.15]
  - Horizontal Flip Test-Time Augmentation
  - 2D Gaussian Sliding Window (512x512 crop, stride 256) on native 720x960 resolution
  - Minority Class Prior Shifts
"""

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path("d:/finalprojv3")
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.dataset import CamVidDataset, CAMVID_CLASSES, IGNORE_INDEX
from experiments.train_semantic_transfer import (
    DeepLabV3PlusResNet101,
    CAMVID_MEAN,
    CAMVID_STD
)
from utils.metrics import SegmentationMetricEvaluator


class SegFormerWrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        out = self.inner(pixel_values=x).logits
        return F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)


def load_segformer(checkpoint_path: str, device: str) -> nn.Module:
    print(f"\n[1/2] Loading SegFormer checkpoint: {checkpoint_path}...")
    config = SegformerConfig.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_labels=11
    )
    inner = SegformerForSemanticSegmentation(config)
    model = SegFormerWrapper(inner)
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "inner." not in list(sd.keys())[0]:
        sd = {"inner." + k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_deeplab(checkpoint_path: str, device: str) -> nn.Module:
    print(f"\n[2/2] Loading DeepLabV3+ checkpoint: {checkpoint_path}...")
    model = DeepLabV3PlusResNet101(num_classes=11)
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def evaluate_advanced_sliding_window(
    model: nn.Module,
    dataset: CamVidDataset,
    model_name: str,
    device: str = "cuda",
    crop_size=(512, 512),
    stride=(256, 256),
    scales=[0.85, 1.0, 1.15],
    flip_tta=True,
    tau_shift=0.10,
    minority_classes=[2, 6, 7, 9, 10]
):
    print("=" * 80)
    print(f"RUNNING ADVANCED INFERENCE: {model_name.upper()}")
    print(f"   Native Sliding Window: {crop_size} (stride {stride}) | Scales: {scales} | Flip: {flip_tta} | tau: {tau_shift}")
    print(f"   Images: {len(dataset)} held-out test frames (Zero-Leakage Standalone)")
    print("=" * 80)

    model.eval()
    num_classes = 11

    ch, cw = crop_size
    sigma_y, sigma_x = ch / 4.0, cw / 4.0
    y_arr = np.linspace(-ch // 2, ch // 2, ch)
    x_arr = np.linspace(-cw // 2, cw // 2, cw)
    xx, yy = np.meshgrid(x_arr, y_arr)
    gaussian_kernel = np.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
    gaussian_kernel = torch.from_numpy(gaussian_kernel).float().to(device).unsqueeze(0).unsqueeze(0)

    shift_vec = torch.zeros(num_classes, device=device)
    if tau_shift > 0.0:
        for c in minority_classes:
            shift_vec[c] = tau_shift

    evaluator = SegmentationMetricEvaluator(num_classes=11, ignore_index=IGNORE_INDEX, compute_boundary=True)
    t0 = time.time()

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target = dataset[idx]
            H, W = image.shape[1], image.shape[2]

            total_prob_map = torch.zeros((1, num_classes, H, W), device=device)

            for s in scales:
                s_H = int(H * s)
                s_W = int(W * s)

                img_scaled = F.interpolate(image.unsqueeze(0), size=(s_H, s_W), mode="bilinear", align_corners=False).to(device)
                prob_scale = torch.zeros((1, num_classes, s_H, s_W), device=device)
                count_norm = torch.zeros((1, 1, s_H, s_W), device=device)

                y_steps = list(range(0, max(1, s_H - ch + 1), stride[0]))
                if y_steps[-1] != max(0, s_H - ch):
                    y_steps.append(max(0, s_H - ch))
                x_steps = list(range(0, max(1, s_W - cw + 1), stride[1]))
                if x_steps[-1] != max(0, s_W - cw):
                    x_steps.append(max(0, s_W - cw))

                for ys in y_steps:
                    for xs in x_steps:
                        crop = img_scaled[:, :, ys:ys+ch, xs:xs+cw]
                        pad_h = max(0, ch - crop.shape[2])
                        pad_w = max(0, cw - crop.shape[3])
                        if pad_h > 0 or pad_w > 0:
                            crop = F.pad(crop, (0, pad_w, 0, pad_h), mode="reflect")

                        with torch.amp.autocast("cuda"):
                            logits = model(crop)
                        logits = F.interpolate(logits, size=(ch, cw), mode="bilinear", align_corners=False)

                        if flip_tta:
                            crop_flip = torch.flip(crop, dims=[3])
                            with torch.amp.autocast("cuda"):
                                logits_flip = model(crop_flip)
                            logits_flip = F.interpolate(logits_flip, size=(ch, cw), mode="bilinear", align_corners=False)
                            logits_flip = torch.flip(logits_flip, dims=[3])
                            logits = (logits + logits_flip) * 0.5

                        probs = F.softmax(logits, dim=1)

                        if pad_h > 0 or pad_w > 0:
                            probs = probs[:, :, :ch-pad_h, :cw-pad_w]
                            g_mask = gaussian_kernel[:, :, :ch-pad_h, :cw-pad_w]
                        else:
                            g_mask = gaussian_kernel

                        prob_scale[:, :, ys:ys+probs.shape[2], xs:xs+probs.shape[3]] += probs * g_mask
                        count_norm[:, :, ys:ys+probs.shape[2], xs:xs+probs.shape[3]] += g_mask

                prob_scale = prob_scale / count_norm.clamp(min=1e-5)
                prob_scale = F.interpolate(prob_scale, size=(H, W), mode="bilinear", align_corners=False)
                total_prob_map += prob_scale

            total_prob_map = total_prob_map / len(scales)
            total_prob_map = total_prob_map + shift_vec.view(1, num_classes, 1, 1)

            preds = total_prob_map.argmax(dim=1).squeeze(0).cpu().numpy()
            target_np = target.numpy()
            evaluator.update(preds, target_np)

            if (idx + 1) % 40 == 0 or (idx + 1) == len(dataset):
                elapsed = time.time() - t0
                print(f"  Processed {idx+1:03d}/{len(dataset)} frames ({elapsed:.1f}s)...", flush=True)

    metrics = evaluator.compute_all_metrics()
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    native_trans = A.Compose([
        A.Resize(height=720, width=960, interpolation=1),
        A.Normalize(mean=CAMVID_MEAN, std=CAMVID_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])
    test_ds = CamVidDataset("./data/CamVid", split="test", transforms=native_trans)

    # 1. SegFormer-B2 Evaluation
    seg_ckpt = "outputs/new_checkpoints/04_semantic_segformer_b2_best.pth"
    if not Path(seg_ckpt).exists():
        seg_ckpt = "outputs/old_checkpoints/04_semantic_segformer_b2_best.pth"
    seg_model = load_segformer(seg_ckpt, device)
    seg_metrics = evaluate_advanced_sliding_window(
        seg_model, test_ds, "SegFormer-B2", device=device,
        crop_size=(512, 512), stride=(256, 256), scales=[0.85, 1.0, 1.15],
        flip_tta=True, tau_shift=0.20
    )

    # 2. DeepLabV3+ Evaluation (Newly Trained Checkpoint)
    dlv_ckpt = "outputs/new_checkpoints/04_semantic_deeplabv3plus_best.pth"
    if not Path(dlv_ckpt).exists():
        dlv_ckpt = "outputs/old_checkpoints/04_semantic_deeplabv3plus_best.pth"
    dlv_model = load_deeplab(dlv_ckpt, device)
    dlv_metrics = evaluate_advanced_sliding_window(
        dlv_model, test_ds, "DeepLabV3+ ResNet-101", device=device,
        crop_size=(512, 512), stride=(256, 256), scales=[0.85, 1.0, 1.15],
        flip_tta=True, tau_shift=0.10
    )

    # Print Final Benchmark Summary Table
    print("\n" + "=" * 85)
    print("FINAL TEST SET BENCHMARK SUMMARY (ALL 233 TEST IMAGES)")
    print("=" * 85)
    print(f"{'Metric':<25} | {'DeepLabV3+ (ResNet-101)':<24} | {'SegFormer-B2':<24}")
    print("-" * 85)
    print(f"{'Test mIoU':<25} | {dlv_metrics['mIoU']*100:6.2f}%{'':<17} | {seg_metrics['mIoU']*100:6.2f}%")
    print(f"{'Pixel Accuracy':<25} | {dlv_metrics['pixel_accuracy']*100:6.2f}%{'':<17} | {seg_metrics['pixel_accuracy']*100:6.2f}%")
    print(f"{'Mean Dice / F1':<25} | {dlv_metrics['mean_dice_f1']*100:6.2f}%{'':<17} | {seg_metrics['mean_dice_f1']*100:6.2f}%")
    print(f"{'Boundary IoU':<25} | {dlv_metrics['boundary_iou']*100:6.2f}%{'':<17} | {seg_metrics['boundary_iou']*100:6.2f}%")
    print("=" * 85)

    print("\nPer-Class IoU Breakdown (Advanced Native TTA Inference):")
    print(f"{'Class Name':<15} | {'DeepLabV3+ IoU':<22} | {'SegFormer-B2 IoU':<22}")
    print("-" * 65)
    for c_idx, c_name in enumerate(CAMVID_CLASSES):
        d_iou = dlv_metrics['per_class_iou'].get(c_idx, 0.0) * 100
        s_iou = seg_metrics['per_class_iou'].get(c_idx, 0.0) * 100
        print(f"{c_name:<15} | {d_iou:>6.2f}%{'':<15} | {s_iou:>6.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()

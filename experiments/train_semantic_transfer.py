import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from data.dataset import CamVidDataset
from losses.composite_loss import CompositeLossCEJML
from utils.logger import FormattedLogger
from utils.trainer import fit_model


# CamVid dataset pixel statistics
CAMVID_MEAN = [0.41189489566336, 0.4251328133025, 0.4326707089857]
CAMVID_STD  = [0.27413549931506, 0.28506257482912, 0.28284674400252]

# 1:1 Semantic Class Mapping: CamVid index -> Cityscapes index
# CamVid 0 Sky        <-- Cityscapes 10 (sky)
# CamVid 1 Building   <-- Cityscapes 2  (building)
# CamVid 2 Pole       <-- Cityscapes 5  (pole)
# CamVid 3 Road       <-- Cityscapes 0  (road)
# CamVid 4 Pavement   <-- Cityscapes 1  (sidewalk)
# CamVid 5 Tree       <-- Cityscapes 8  (vegetation)
# CamVid 6 SignSymbol <-- Cityscapes 7  (traffic sign)
# CamVid 7 Fence      <-- Cityscapes 4  (fence)
# CamVid 8 Car        <-- Cityscapes 13 (car)
# CamVid 9 Pedestrian <-- Cityscapes 11 (person)
# CamVid 10 Bicyclist <-- Cityscapes 18 (bicycle)
CAMVID_TO_CITYSCAPES = [10, 2, 5, 0, 1, 8, 7, 4, 13, 11, 18]


def get_512_train_transforms() -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(
            size=(512, 512),
            scale=(0.5, 1.0),
            ratio=(0.75, 1.333),
            interpolation=1,
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.GaussNoise(std_range=(0.0, 0.05), p=0.1),
        A.Normalize(mean=CAMVID_MEAN, std=CAMVID_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])


def get_512_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(height=512, width=512, interpolation=1),
        A.Normalize(mean=CAMVID_MEAN, std=CAMVID_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])


# =========================================================================
# 1. Native DeepLabV3+ Matching Official Cityscapes Checkpoint Exactly
# =========================================================================
class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels=2048, low_level_channels=256, num_classes=11, aspp_dilate=[6, 12, 18]):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        convs = [
            nn.Sequential(nn.Conv2d(in_channels, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=aspp_dilate[0], dilation=aspp_dilate[0], bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=aspp_dilate[1], dilation=aspp_dilate[1], bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=aspp_dilate[2], dilation=aspp_dilate[2], bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        ]
        self.aspp = nn.Module()
        self.aspp.convs = nn.ModuleList(convs)
        self.aspp.project = nn.Sequential(
            nn.Conv2d(5 * 256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp.convs[0](feature['out'])
        for conv in self.aspp.convs[1:]:
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                p = conv(feature['out'])
                p = F.interpolate(p, size=output_feature.shape[2:], mode='bilinear', align_corners=False)
                output_feature = torch.cat([output_feature, p], dim=1)
            else:
                output_feature = torch.cat([output_feature, conv(feature['out'])], dim=1)

        output_feature = self.aspp.project(output_feature)
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))


class DeepLabV3PlusResNet101(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        resnet = resnet101(replace_stride_with_dilation=[False, False, True])
        self.backbone = nn.Module()
        self.backbone.conv1 = resnet.conv1
        self.backbone.bn1 = resnet.bn1
        self.backbone.relu = resnet.relu
        self.backbone.maxpool = resnet.maxpool
        self.backbone.layer1 = resnet.layer1
        self.backbone.layer2 = resnet.layer2
        self.backbone.layer3 = resnet.layer3
        self.backbone.layer4 = resnet.layer4
        self.classifier = DeepLabHeadV3Plus(2048, 256, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        h = self.backbone.conv1(x)
        h = self.backbone.bn1(h)
        h = self.backbone.relu(h)
        h = self.backbone.maxpool(h)
        low_level = self.backbone.layer1(h)
        h = self.backbone.layer2(low_level)
        h = self.backbone.layer3(h)
        out = self.backbone.layer4(h)
        logits = self.classifier({'low_level': low_level, 'out': out})
        return F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)


def build_semantic_transferred_deeplab(device: str = "cuda") -> nn.Module:
    """Builds DeepLabV3+ ResNet-101 and transfers all 674 Cityscapes layers + 1:1 mapped classifier."""
    ckpt_path = Path("d:/finalproj/outputs/checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth")
    print(f"[DeepLabV3+] Loading Cityscapes checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]

    w19 = sd["classifier.classifier.3.weight"]
    b19 = sd["classifier.classifier.3.bias"]

    # Remap the 19 Cityscapes classifier channels to 11 CamVid classes
    w11 = torch.zeros(11, 256, 1, 1)
    b11 = torch.zeros(11)
    for c_camvid, c_city in enumerate(CAMVID_TO_CITYSCAPES):
        if c_city < w19.shape[0]:
            w11[c_camvid] = w19[c_city]
            b11[c_camvid] = b19[c_city]

    sd["classifier.classifier.3.weight"] = w11
    sd["classifier.classifier.3.bias"] = b11

    model = DeepLabV3PlusResNet101(num_classes=11)
    model.load_state_dict(sd, strict=True)
    print(f"[DeepLabV3+] Transferred 100% of Cityscapes layers (Backbone + ASPP + Semantic Classifier Head)!")
    return model.to(device)


# =========================================================================
# 2. SegFormer-B2 with Semantic Head Mapping
# =========================================================================
def build_semantic_transferred_segformer(device: str = "cuda") -> nn.Module:
    """Builds SegFormer-B2 with 1:1 semantic class embedding transfer from Cityscapes."""
    from transformers import SegformerForSemanticSegmentation
    ckpt = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    print(f"[SegFormer-B2] Loading full 19-class Cityscapes model: {ckpt}")
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt)

    w_city = model.decode_head.classifier.weight.data
    b_city = model.decode_head.classifier.bias.data
    in_channels = w_city.shape[1]

    new_classifier = nn.Conv2d(in_channels, 11, kernel_size=1, bias=True)
    new_classifier.weight.data = w_city[CAMVID_TO_CITYSCAPES].clone()
    new_classifier.bias.data = b_city[CAMVID_TO_CITYSCAPES].clone()

    model.decode_head.classifier = new_classifier
    model.decode_head.num_classes = 11
    model.config.num_labels = 11
    print(f"[SegFormer-B2] Transferred 100% of Cityscapes layers + Semantic Classifier Head!")

    class SegFormerWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            orig_h, orig_w = x.shape[2], x.shape[3]
            out = self.inner(pixel_values=x).logits
            return F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    return SegFormerWrapper(model).to(device)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# Main Training Routine
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Models with Full Semantic Class Embedding Transfer")
    parser.add_argument("--data_root", type=str, default="./data/CamVid")
    parser.add_argument("--log_dir", type=str, default="./outputs/new_logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/new_checkpoints")
    parser.add_argument("--model", type=str, default="all", choices=["all", "deeplabv3plus", "segformer_b2"])
    parser.add_argument("--seed", type=str, default="42", help="Random seed (e.g. '42', '123', '456', or 'all' for [42, 123, 456])")
    parser.add_argument("--full_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [42, 123, 456] if args.seed.lower() == "all" else [int(s.strip()) for s in args.seed.split(",")]

    print("=" * 80)
    print("[WARM-START FULL ARCHITECTURE SEMANTIC TRANSFER TRAINING]")
    print(f"   Model: {args.model.upper()} | Seeds: {seeds} | Epochs: {args.full_epochs} | Device: {device} | Batch Size: {args.batch_size}")
    print("   Resolution: 512x512 RandomResizedCrop | Pretraining: Full Cityscapes Domain Transferred")
    print(f"   Logs Directory:        {args.log_dir}")
    print(f"   Checkpoints Directory: {args.checkpoint_dir}")
    print("=" * 80)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.log_dir) / "04_semantic_transfer_summary.csv"

    models_to_train = ["deeplabv3plus", "segformer_b2"] if args.model == "all" else [args.model.lower()]

    existing_records = []
    if summary_path.exists():
        try:
            existing_records = pd.read_csv(summary_path).to_dict("records")
        except Exception:
            existing_records = []

    for seed in seeds:
        print(f"\n>>> SETTING EXPERIMENT SEED: {seed} <<<\n")
        set_seed(seed)

        train_dataset = CamVidDataset(args.data_root, split="train", transforms=get_512_train_transforms())
        val_dataset = CamVidDataset(args.data_root, split="val", transforms=get_512_val_transforms())

        num_workers = 0 if os.name == "nt" else 2
        use_pin_memory = (device == "cuda")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

        for model_name in models_to_train:
            # Load our agent-discovered optimal loss weights strictly from JSON
            weight_lookup_name = "segformer" if "segformer" in model_name else "deeplab"
            json_file = Path(args.log_dir) / f"04_agent_weights_{weight_lookup_name}.json"
            if not json_file.exists():
                raise FileNotFoundError(f"Missing agent weights file: {json_file.resolve()}")

            with open(json_file, "r", encoding="utf-8") as f:
                w_data = json.load(f)
            l_ce = float(w_data["lambda_ce"])
            l_jml = float(w_data["lambda_jml"])

            if "segformer" in model_name:
                lr = 1e-4
                model = build_semantic_transferred_segformer(device=device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.full_epochs, eta_min=1e-6)
            else:
                lr = 5e-3
                model = build_semantic_transferred_deeplab(device=device)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.full_epochs, power=1.0)

            criterion = CompositeLossCEJML(lambda_ce=l_ce, lambda_jml=l_jml)

            exp_name = f"04_semantic_{model_name}_seed{seed}"
            logger = FormattedLogger(exp_name, log_dir=args.log_dir)
            logger.log_message("=" * 70)
            logger.log_message(f"Training {model_name.upper()} [Seed {seed}] with 100% Semantic Transfer using Agent Weights ({json_file.name}): {l_ce}*CE + {l_jml}*JML ({args.full_epochs} epochs)")
            logger.log_message("=" * 70)

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

            # Also maintain canonical checkpoint without seed suffix for primary seed (42)
            if seed == 42:
                canonical_ckpt = f"{args.checkpoint_dir}/04_semantic_{model_name}_best.pth"
                import shutil
                shutil.copy2(ckpt_path, canonical_ckpt)

            record = {
                "model": model_name,
                "seed": seed,
                "lambda_ce": l_ce,
                "lambda_jml": l_jml,
                "best_val_miou": round(float(fit_res["best_val_miou"]), 4),
                "best_val_dice": round(float(fit_res["best_metrics"].get("mean_dice_f1", 0.0)), 4),
                "best_val_acc": round(float(fit_res["best_metrics"].get("pixel_accuracy", 0.0)), 4),
                "best_val_biou": round(float(fit_res["best_metrics"].get("boundary_iou", 0.0)), 4),
                "checkpoint": ckpt_path
            }
            logger.save_summary(record)

            existing_records = [r for r in existing_records if not (r.get("model") == model_name and r.get("seed") == seed)] + [record]
            pd.DataFrame(existing_records).to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("=== SEMANTIC EMBEDDING TRANSFER TRAINING COMPLETE ===")
    df_summary = pd.read_csv(summary_path)
    print(df_summary.to_string(index=False))

    if len(seeds) > 1 or len(df_summary["seed"].unique()) > 1:
        print("\n" + "=" * 80)
        print("=== MULTI-SEED STATISTICAL RELIABILITY (MEAN ± STD) ===")
        agg = df_summary.groupby("model").agg({
            "best_val_miou": ["mean", "std"],
            "best_val_dice": ["mean", "std"],
            "best_val_acc": ["mean", "std"],
            "best_val_biou": ["mean", "std"]
        }).reset_index()
        print(agg.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

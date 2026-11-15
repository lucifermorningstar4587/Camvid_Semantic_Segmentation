import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Standard 11 CamVid classes + void/background
CAMVID_CLASSES = [
    "Sky",          # 0
    "Building",     # 1
    "Pole",         # 2
    "Road",         # 3
    "Pavement",     # 4
    "Tree",         # 5
    "SignSymbol",   # 6
    "Fence",        # 7
    "Car",          # 8
    "Pedestrian",   # 9
    "Bicyclist",    # 10
]

IGNORE_INDEX = 255

# Standard RGB color palette for CamVid 11 classes (if masks are RGB)
COLOR_MAP = {
    (128, 128, 128): 0,   # Sky
    (128, 0, 0): 1,       # Building
    (192, 192, 128): 2,   # Pole
    (128, 64, 128): 3,    # Road
    (0, 0, 192): 4,       # Pavement / Sidewalk
    (128, 128, 0): 5,     # Tree
    (192, 128, 128): 6,   # SignSymbol
    (64, 64, 128): 7,     # Fence
    (64, 0, 128): 8,      # Car
    (64, 64, 0): 9,       # Pedestrian
    (0, 128, 192): 10,    # Bicyclist
    (0, 0, 0): IGNORE_INDEX, # Void / Unlabeled
}

INV_COLOR_MAP = {v: k for k, v in COLOR_MAP.items() if v != IGNORE_INDEX}
INV_COLOR_MAP[IGNORE_INDEX] = (0, 0, 0)


def get_training_transforms(height: int = 360, width: int = 480) -> A.Compose:
    """Robust augmentations to prevent overfitting during training."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-10, 10), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_validation_transforms(height: int = 360, width: int = 480) -> A.Compose:
    """Standard validation transforms without stochastic augmentations."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_native_transforms() -> A.Compose:
    """Native resolution transforms (720x960 or 360x480) for sliding window inference."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def rgb_to_mask(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert RGB color mask to 2D integer class mask (0..10, 255 for void)."""
    mask = np.full((rgb_arr.shape[0], rgb_arr.shape[1]), IGNORE_INDEX, dtype=np.int64)
    for color, cls_idx in COLOR_MAP.items():
        matches = np.all(rgb_arr == color, axis=-1)
        mask[matches] = cls_idx
    return mask


class CamVidDataset(Dataset):
    """
    CamVid PyTorch Dataset supporting official splits (train: 367, val: 101, test: 233).
    Supports:
      - data_root/{split}/images and data_root/{split}/masks
      - data_root/{split} and data_root/{split}_labels
      - 2D indexed masks (0..10, with 11/void -> 255) and RGB color masks.
    """
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transforms: Optional[A.Compose] = None,
        copy_paste_augmenter = None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.copy_paste_augmenter = copy_paste_augmenter

        # Priority directory patterns
        possible_img_dirs = [
            self.data_root / split / "images",
            self.data_root / split,
            self.data_root / f"{split}_images",
            self.data_root / "images" / split
        ]
        possible_lbl_dirs = [
            self.data_root / split / "masks",
            self.data_root / f"{split}_labels",
            self.data_root / f"{split}annot",
            self.data_root / "labels" / split,
            self.data_root / split / "labels"
        ]

        self.img_dir = next((d for d in possible_img_dirs if d.exists() and len(list(d.glob("*.png")) + list(d.glob("*.jpg"))) > 0), None)
        self.lbl_dir = next((d for d in possible_lbl_dirs if d.exists() and len(list(d.glob("*.png")) + list(d.glob("*.jpg"))) > 0), None)

        if self.img_dir is not None and self.lbl_dir is not None:
            self.images = sorted(list(self.img_dir.glob("*.png")) + list(self.img_dir.glob("*.jpg")))
            lbl_files = {f.stem.replace("_L", ""): f for f in (list(self.lbl_dir.glob("*.png")) + list(self.lbl_dir.glob("*.jpg")))}
            self.labels = []
            for img_p in self.images:
                key = img_p.stem.replace("_L", "")
                if key in lbl_files:
                    self.labels.append(lbl_files[key])
                else:
                    self.labels.append(self.lbl_dir / img_p.name)
        else:
            # Fallback search across directory tree
            all_imgs = sorted(list(self.data_root.glob(f"**/{split}*/**/*.png")) + list(self.data_root.glob(f"**/{split}*/**/*.jpg")))
            self.images = [p for p in all_imgs if "mask" not in p.parent.name and "label" not in p.parent.name and "annot" not in p.parent.name]
            self.labels = []
            all_lbls = {f.stem.replace("_L", ""): f for f in all_imgs if ("mask" in f.parent.name or "label" in f.parent.name or "annot" in f.parent.name)}
            for img_p in self.images:
                key = img_p.stem.replace("_L", "")
                if key in all_lbls:
                    self.labels.append(all_lbls[key])
                else:
                    self.labels.append(img_p)

        print(f"Loaded CamVid [{split}] split: {len(self.images)} images, {len(self.labels)} masks.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self.images[idx])
        lbl_path = str(self.labels[idx])

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)

        label_img = Image.open(lbl_path)
        label_np = np.array(label_img)

        # Handle RGB masks vs 2D indexed masks
        if label_np.ndim == 3 and label_np.shape[2] == 3:
            mask_np = rgb_to_mask(label_np)
        else:
            mask_np = label_np.astype(np.int64)
            # In CamVid benchmark, class 11 (or > 10) corresponds to void / unlabeled pixels
            mask_np[mask_np >= 11] = IGNORE_INDEX

        # Optional Class-Aware Copy-Paste augmentation during training
        if self.copy_paste_augmenter is not None and self.split == "train":
            image_np, mask_np = self.copy_paste_augmenter.apply(image_np, mask_np)

        if self.transforms:
            augmented = self.transforms(image=image_np, mask=mask_np)
            image_tensor = augmented["image"]
            mask_tensor = torch.as_tensor(augmented["mask"], dtype=torch.long)
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor


def compute_dataset_statistics(dataset: CamVidDataset) -> Dict[str, any]:
    """
    Computes class pixel distribution across the dataset to quantify class imbalance.
    Returns pixel counts, relative frequencies, and median-frequency class weights.
    """
    print(f"Computing class statistics on {dataset.split} set ({len(dataset)} images)...")
    pixel_counts = np.zeros(len(CAMVID_CLASSES), dtype=np.int64)
    total_valid_pixels = 0

    for i in range(len(dataset)):
        _, mask = dataset[i]
        mask_np = mask.numpy()
        for c in range(len(CAMVID_CLASSES)):
            cnt = int(np.sum(mask_np == c))
            pixel_counts[c] += cnt
            total_valid_pixels += cnt

    frequencies = (pixel_counts / max(1, total_valid_pixels)).tolist()
    # Median frequency balancing: weight_c = median(freq) / freq_c
    nonzero_freq = [f for f in frequencies if f > 0]
    median_freq = float(np.median(nonzero_freq)) if nonzero_freq else 1.0
    weights = [float(median_freq / f) if f > 0 else 0.0 for f in frequencies]

    stats = {
        "classes": CAMVID_CLASSES,
        "pixel_counts": pixel_counts.tolist(),
        "frequencies": frequencies,
        "class_weights": weights,
        "total_valid_pixels": int(total_valid_pixels)
    }
    return stats

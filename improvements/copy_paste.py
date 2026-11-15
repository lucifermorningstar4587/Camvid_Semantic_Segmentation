import random
import numpy as np
import cv2
from typing import Tuple, List, Optional

class ClassAwareCopyPaste:
    """
    Class-Aware Copy-Paste Data Augmentation.
    Targets minority/rare classes (e.g. Pedestrian, Bicyclist, Pole, SignSymbol)
    to actively alleviate severe class imbalance during training without synthetic artifacts.
    """
    def __init__(
        self,
        target_classes: List[int] = [2, 6, 9, 10], # Pole, SignSymbol, Pedestrian, Bicyclist
        p: float = 0.5,
        max_paste_objects: int = 3
    ):
        self.target_classes = set(target_classes)
        self.p = p
        self.max_paste_objects = max_paste_objects
        # Bank of donor instances: list of dicts with 'patch_rgb' and 'patch_mask'
        self.instance_bank = []

    def populate_bank(self, dataset, max_items: int = 200):
        """Pre-extract target class instances from training dataset."""
        print(f"Populating Class-Aware Copy-Paste bank targeting classes: {self.target_classes}...")
        for i in range(min(len(dataset), max_items)):
            img_path = str(dataset.images[i])
            lbl_path = str(dataset.labels[i])
            
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            lbl = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
            if lbl.ndim == 3:
                from data.dataset import rgb_to_mask
                lbl = rgb_to_mask(lbl)

            h, w = lbl.shape
            img = cv2.resize(img, (w, h))

            for cls_id in self.target_classes:
                cls_mask = (lbl == cls_id).astype(np.uint8)
                if cls_mask.sum() > 50: # valid size
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_mask)
                    for comp_idx in range(1, num_labels):
                        area = stats[comp_idx, cv2.CC_STAT_AREA]
                        if 100 < area < 40000: # suitable object size
                            x = stats[comp_idx, cv2.CC_STAT_LEFT]
                            y = stats[comp_idx, cv2.CC_STAT_TOP]
                            cw = stats[comp_idx, cv2.CC_STAT_WIDTH]
                            ch = stats[comp_idx, cv2.CC_STAT_HEIGHT]
                            
                            comp_mask = (labels[y:y+ch, x:x+cw] == comp_idx)
                            patch_rgb = img[y:y+ch, x:x+cw].copy()
                            patch_lbl = np.full((ch, cw), cls_id, dtype=np.int64)
                            
                            self.instance_bank.append({
                                "rgb": patch_rgb,
                                "mask": comp_mask,
                                "cls_id": cls_id
                            })
                            if len(self.instance_bank) >= 500:
                                break
        print(f"Successfully extracted {len(self.instance_bank)} minority class instances into Copy-Paste bank.")

    def apply(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Copy-Paste augmentation to an image and mask with probability p."""
        if len(self.instance_bank) == 0 or random.random() > self.p:
            return image, mask

        aug_image = image.copy()
        aug_mask = mask.copy()
        h, w, _ = aug_image.shape

        num_objects = random.randint(1, self.max_paste_objects)
        for _ in range(num_objects):
            donor = random.choice(self.instance_bank)
            patch_rgb = donor["rgb"]
            patch_mask = donor["mask"]
            cls_id = donor["cls_id"]

            ph, pw, _ = patch_rgb.shape
            if ph >= h or pw >= w:
                continue

            # Random placement in recipient image
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)

            # Paste object
            roi_img = aug_image[top:top+ph, left:left+pw]
            roi_mask = aug_mask[top:top+ph, left:left+pw]

            roi_img[patch_mask] = patch_rgb[patch_mask]
            roi_mask[patch_mask] = cls_id

            aug_image[top:top+ph, left:left+pw] = roi_img
            aug_mask[top:top+ph, left:left+pw] = roi_mask

        return aug_image, aug_mask

import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import cv2
from tqdm import tqdm
from typing import Tuple, Dict

def extract_pixel_features(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract multi-channel handcrafted features for traditional ML segmentation:
      - RGB channels (3)
      - HSV channels (3)
      - Normalized (x, y) spatial coordinates (2)
      - Sobel gradient magnitudes (2)
      - Gaussian blurred intensities (3)
    Total feature dimensions: 13 features per pixel.
    """
    h, w, _ = image_rgb.shape
    
    # RGB normalized
    rgb_norm = image_rgb.astype(np.float32) / 255.0
    
    # HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    
    # Spatial coordinates (x, y) normalized to [0, 1]
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_coords = (x_coords.astype(np.float32) / max(1, w - 1))[..., None]
    y_coords = (y_coords.astype(np.float32) / max(1, h - 1))[..., None]
    
    # Grayscale for gradients
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)[..., None]
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)[..., None]
    
    # Gaussian blur (texture context)
    blur = cv2.GaussianBlur(rgb_norm, (5, 5), sigmaX=1.5)
    
    # Stack all feature maps: (H, W, 13)
    feature_map = np.concatenate([rgb_norm, hsv, x_coords, y_coords, grad_x, grad_y, blur], axis=-1)
    return feature_map.reshape(-1, feature_map.shape[-1])


class TraditionalRandomForestBaseline:
    """
    Traditional Machine Learning Baseline for Semantic Segmentation:
    Pixel-level Random Forest Classifier on handcrafted spatial, color, and gradient features.
    """
    def __init__(self, n_estimators: int = 50, max_depth: int = 16, random_state: int = 42, n_jobs: int = -1):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced"
        )
        self.num_classes = 11

    def fit_from_dataset(self, dataset, max_samples_per_image: int = 1500):
        """Train Random Forest baseline using subsampled pixels with live progress reporting."""
        print(f"Extracting handcrafted features from {len(dataset)} training images...", flush=True)
        start_time = time.time()
        
        all_features = []
        all_labels = []
        
        for idx in tqdm(range(len(dataset)), desc="Feature Extraction (Train)", file=sys.stdout):
            img_path = str(dataset.images[idx])
            lbl_path = str(dataset.labels[idx])
            
            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            lbl = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
            if lbl.ndim == 3:
                from data.dataset import rgb_to_mask
                lbl = rgb_to_mask(lbl)
                
            h, w = lbl.shape
            img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            
            feats = extract_pixel_features(img_rgb)
            flat_lbl = lbl.reshape(-1)
            
            # Filter out void/ignore_index (>= 11 or 255)
            valid_mask = (flat_lbl >= 0) & (flat_lbl < self.num_classes)
            valid_feats = feats[valid_mask]
            valid_lbls = flat_lbl[valid_mask]
            
            if len(valid_lbls) > max_samples_per_image:
                sample_indices = np.random.choice(len(valid_lbls), max_samples_per_image, replace=False)
                valid_feats = valid_feats[sample_indices]
                valid_lbls = valid_lbls[sample_indices]
                
            all_features.append(valid_feats)
            all_labels.append(valid_lbls)
            
        X_train = np.vstack(all_features)
        y_train = np.concatenate(all_labels)
        
        print(f"\nFitting Random Forest ({self.rf.n_estimators} trees) on {len(y_train):,} pixels...", flush=True)
        self.rf.fit(X_train, y_train)
        fit_time = time.time() - start_time
        print(f"Random Forest fitted successfully in {fit_time:.1f}s.\n", flush=True)

    def predict_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Segment a single RGB image."""
        h, w, _ = image_rgb.shape
        feats = extract_pixel_features(image_rgb)
        pred_flat = self.rf.predict(feats)
        return pred_flat.reshape(h, w)

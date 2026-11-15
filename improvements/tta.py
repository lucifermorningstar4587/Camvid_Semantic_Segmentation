import torch
import torch.nn.functional as F
from typing import List

def multi_scale_tta_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    scales: List[float] = [0.75, 1.0, 1.25],
    flip: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Multi-Scale TTA (Test-Time Augmentation) Pyramid Evaluation:
    Computes ensemble predictions across a spatial pyramid of scales and horizontal reflections.
    Significantly enhances robustness against scale variance and orientation noise.
    
    image_tensor: [1, 3, H, W]
    Returns: [1, C, H, W] ensemble probability distribution
    """
    model.eval()
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    _, _, orig_h, orig_w = image_tensor.shape
    accumulated_probs = None
    num_passes = 0
    
    with torch.no_grad():
        for scale in scales:
            target_h = int(round(orig_h * scale / 32.0) * 32)
            target_w = int(round(orig_w * scale / 32.0) * 32)
            
            scaled_img = F.interpolate(image_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
            
            # Forward pass 1: normal scale
            logits = model(scaled_img)
            probs = F.softmax(logits, dim=1)
            probs_orig = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
            if accumulated_probs is None:
                accumulated_probs = probs_orig
            else:
                accumulated_probs += probs_orig
            num_passes += 1
            
            # Forward pass 2: horizontal flip
            if flip:
                scaled_flipped = torch.flip(scaled_img, dims=[3])
                logits_flipped = model(scaled_flipped)
                probs_flipped = F.softmax(logits_flipped, dim=1)
                # Unflip probabilities
                probs_unflipped = torch.flip(probs_flipped, dims=[3])
                probs_unflipped_orig = F.interpolate(probs_unflipped, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                accumulated_probs += probs_unflipped_orig
                num_passes += 1

    return accumulated_probs / float(num_passes)

import math
import numpy as np
import torch
import torch.nn.functional as F

def get_gaussian_window(patch_h: int, patch_w: int, sigma: float = 0.125, device: torch.device = "cpu") -> torch.Tensor:
    """Creates a 2D Gaussian weight window on specified device to smoothly blend overlapping patch predictions."""
    y = torch.linspace(-1, 1, patch_h, device=device)
    x = torch.linspace(-1, 1, patch_w, device=device)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    dist = (gx ** 2 + gy ** 2) / (2 * (sigma ** 2))
    window = torch.exp(-dist)
    window = window / window.max()
    return window.clamp(min=1e-4)


def sliding_window_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    patch_size: tuple = (240, 240),
    stride: tuple = (120, 120),
    device: str = None
) -> torch.Tensor:
    """
    Sliding Window Inference with 2D Gaussian Window Blending.
    Operates on full-resolution images without loss of high-frequency details.
    
    image_tensor: [1, 3, H, W]
    Returns: [1, num_classes, H, W] blended logits
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    target_dev = torch.device(device)
    model.eval()
    model = model.to(target_dev)
    image_tensor = image_tensor.to(target_dev)

    _, _, H, W = image_tensor.shape
    patch_h = min(patch_size[0], H)
    patch_w = min(patch_size[1], W)
    stride_y = min(stride[0], patch_h)
    stride_x = min(stride[1], patch_w)

    window = get_gaussian_window(patch_h, patch_w, device=target_dev).unsqueeze(0).unsqueeze(0)

    # Determine class count dynamically
    sample_patch = image_tensor[:, :, :patch_h, :patch_w]
    with torch.no_grad():
        sample_out = model(sample_patch)
        num_classes = sample_out.shape[1]

    full_logits = torch.zeros((1, num_classes, H, W), device=target_dev)
    count_weight = torch.zeros((1, 1, H, W), device=target_dev)

    y_steps = math.ceil((H - patch_h) / stride_y) + 1 if H > patch_h else 1
    x_steps = math.ceil((W - patch_w) / stride_x) + 1 if W > patch_w else 1

    for i in range(y_steps):
        for j in range(x_steps):
            y1 = min(i * stride_y, H - patch_h)
            x1 = min(j * stride_x, W - patch_w)
            y2 = y1 + patch_h
            x2 = x1 + patch_w

            patch = image_tensor[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                patch_out = model(patch)

            full_logits[:, :, y1:y2, x1:x2] += patch_out * window
            count_weight[:, :, y1:y2, x1:x2] += window

    normalized_logits = full_logits / count_weight.clamp(min=1e-5)
    return normalized_logits

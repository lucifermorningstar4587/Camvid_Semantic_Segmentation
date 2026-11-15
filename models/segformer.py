import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

class SegFormerB2(nn.Module):
    """
    SegFormer-B2 architecture with Mix Transformer (mit-b2) hierarchical backbone
    and lightweight all-MLP decoder.
    Supports:
      - 'imagenet': weights from nvidia/mit-b2
      - 'cityscapes': domain-transferred weights from nvidia/segformer-b2-finetuned-cityscapes-1024-1024
    """
    def __init__(self, num_classes: int = 11, pretraining: str = "imagenet"):
        super().__init__()
        self.num_classes = num_classes
        self.pretraining = pretraining.lower()

        if self.pretraining == "cityscapes":
            ckpt = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
            print(f"Initializing SegFormer-B2 from Cityscapes pretrained checkpoint: {ckpt}")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                ckpt,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            ckpt = "nvidia/mit-b2"
            print(f"Initializing SegFormer-B2 from ImageNet-1K pretrained backbone: {ckpt}")
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    ckpt,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            except Exception:
                # If mit-b2 standalone repo has only backbone config, initialize with SegformerConfig
                config = SegformerConfig.from_pretrained(ckpt)
                config.num_labels = num_classes
                self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, C, H, W]
        orig_h, orig_w = x.shape[2], x.shape[3]
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]
        
        # Bilinear interpolation to original input resolution
        logits_upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return logits_upsampled

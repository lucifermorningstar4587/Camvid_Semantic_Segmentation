import torch
import torch.nn as nn
from typing import Optional

class CrossEntropyLossWrapper(nn.Module):
    """
    Standard Cross-Entropy Loss with ignore_index=255 and optional class weights
    to penalize errors on minority classes (e.g., pedestrians, bicyclists).
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: int = 255):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W) with class indices in [0, C-1] and ignore_index=255
        """
        return self.criterion(logits, targets)

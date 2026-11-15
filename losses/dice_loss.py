import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    """
    Region-based Multi-Class Soft Dice Loss with smooth factor and ignore_index support.
    Addresses class imbalance by calculating harmonic overlap for each semantic class.
    """
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # Mask out ignore_index
        valid_mask = (targets != self.ignore_index)
        
        # Build one-hot encoding for targets
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0
        one_hot = F.one_hot(targets_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Apply valid mask to both
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask_expanded
        one_hot = one_hot * valid_mask_expanded
        
        # Calculate Dice per class
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        cardinality = torch.sum(probs + one_hot, dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Only average over classes present in ground truth
        class_present = (torch.sum(one_hot, dim=dims) > 0)
        if class_present.any():
            return 1.0 - dice_score[class_present].mean()
        return 1.0 - dice_score.mean()

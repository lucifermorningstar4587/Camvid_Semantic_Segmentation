import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for addressing extreme foreground-background and class imbalance.
    FL(p_t) = - (1 - p_t)^gamma * log(p_t)
    Reference: Lin et al., ICCV 2017.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return logits.sum() * 0.0

        targets_valid = targets[valid_mask]
        logits_valid = logits.permute(0, 2, 3, 1)[valid_mask] # [N_valid, C]

        log_p = F.log_softmax(logits_valid, dim=-1)
        p = torch.exp(log_p)

        # Gather target probabilities
        target_log_p = log_p.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        target_p = p.gather(1, targets_valid.unsqueeze(1)).squeeze(1)

        focal_weight = torch.pow(1.0 - target_p, self.gamma)
        loss = -focal_weight * target_log_p
        return loss.mean()

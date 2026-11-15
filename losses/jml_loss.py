import torch
import torch.nn as nn
import torch.nn.functional as F

class JaccardMetricLoss(nn.Module):
    """
    Jaccard Metric Loss (JML) from NeurIPS 2023:
    "Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union"
    by Wang et al. (GitHub: zifuwanggg/JDTLosses).

    Formulation:
      |p ∩ y| = (|p|_1 + |y|_1 - |p - y|_1) / 2
      |p ∪ y| = (|p|_1 + |y|_1 + |p - y|_1) / 2
      JML = 1 - (|p|_1 + |y|_1 - |p - y|_1) / (|p|_1 + |y|_1 + |p - y|_1 + eps)

    Properties:
      - Valid metric surrogate satisfying triangle inequality.
      - Seamlessly compatible with both hard one-hot targets and soft/distilled labels.
      - Directly optimizes fine-grained Intersection over Union (IoU).
    """
    def __init__(self, eps: float = 1e-7, ignore_index: int = 255):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Filter out pixels with ignore_index
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return logits.sum() * 0.0

        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0
        one_hot = F.one_hot(targets_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Apply valid mask
        valid_mask_exp = valid_mask.unsqueeze(1).expand_as(probs)
        p = probs * valid_mask_exp
        y = one_hot * valid_mask_exp

        # Flatten spatial dimensions per class: [C, B*H*W]
        p = p.permute(1, 0, 2, 3).contiguous().view(num_classes, -1)
        y = y.permute(1, 0, 2, 3).contiguous().view(num_classes, -1)

        # JML Metric formulation
        p_l1 = torch.sum(p, dim=-1)
        y_l1 = torch.sum(y, dim=-1)
        diff_l1 = torch.sum(torch.abs(p - y), dim=-1)

        # Numerator: 2 * |p ∩ y| = |p|_1 + |y|_1 - |p - y|_1
        # Denominator: 2 * |p ∪ y| = |p|_1 + |y|_1 + |p - y|_1
        intersection_term = p_l1 + y_l1 - diff_l1
        union_term = p_l1 + y_l1 + diff_l1 + self.eps

        jaccard_metric = intersection_term / union_term

        # Average only over present classes
        present_mask = (y_l1 > 0)
        if present_mask.any():
            loss = 1.0 - jaccard_metric[present_mask].mean()
        else:
            loss = 1.0 - jaccard_metric.mean()

        return loss

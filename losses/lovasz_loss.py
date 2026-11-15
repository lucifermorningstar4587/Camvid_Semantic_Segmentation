import torch
import torch.nn as nn
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension with respect to sorted errors.
    See Berman et al. (CVPR 2018).
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-15)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss on flattened predictions and targets.
    probas: [P, C] Probabilities at each pixel
    labels: [P] Ground truth labels (with ignore values already masked out)
    """
    if probas.numel() == 0:
        return probas * 0.0
    
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'total'] else range(C)
    
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present') and (fg.sum() == 0):
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
        
    if len(losses) == 0:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss: Direct tractable convex surrogate for multi-class Jaccard index.
    Reference: Berman, Triki, Blaschko, "The Lovász-Softmax loss", CVPR 2018.
    """
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        probas = F.softmax(logits, dim=1)
        # Permute to (B, H, W, C) -> flatten to (-1, C)
        B, C, H, W = probas.shape
        probas_flat = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.view(-1)
        
        valid = (targets_flat != self.ignore_index)
        if not valid.any():
            return logits.sum() * 0.0
            
        probas_valid = probas_flat[valid]
        targets_valid = targets_flat[valid]
        
        return lovasz_softmax_flat(probas_valid, targets_valid, classes='present')

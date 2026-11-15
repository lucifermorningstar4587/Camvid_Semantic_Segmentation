import torch
import torch.nn as nn
from losses.ce_loss import CrossEntropyLossWrapper
from losses.jml_loss import JaccardMetricLoss

class CompositeLossCEJML(nn.Module):
    """
    Composite Loss Function:
      L_total = lambda_ce * L_ce + lambda_jml * L_jml

    Addresses:
      - Class imbalance and metric mismatch (JML directly maximizes fine-grained IoU)
      - Calibration and smooth pixel optimization (CE ensures sharp probabilistic boundaries)

    The weights (lambda_ce, lambda_jml) are systematically determined by an automated
    hyperparameter search agent on the validation set, eliminating guesswork.
    """
    def __init__(self, lambda_ce: float = 0.5, lambda_jml: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.lambda_ce = float(lambda_ce)
        self.lambda_jml = float(lambda_jml)
        self.ce_loss = CrossEntropyLossWrapper(ignore_index=ignore_index)
        self.jml_loss = JaccardMetricLoss(ignore_index=ignore_index)

    def set_weights(self, lambda_ce: float, lambda_jml: float):
        self.lambda_ce = float(lambda_ce)
        self.lambda_jml = float(lambda_jml)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_ce = self.ce_loss(logits, targets)
        l_jml = self.jml_loss(logits, targets)
        return self.lambda_ce * l_ce + self.lambda_jml * l_jml

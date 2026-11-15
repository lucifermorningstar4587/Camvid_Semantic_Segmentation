import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List

class ClassPriorCalibrator:
    """
    Class Prior Calibration & Temperature Scaling Post-Processing.
    
    Mitigates class imbalance bias where deep networks over-predict dominant classes
    (Road, Sky, Building) and suppress critical rare classes (Pedestrian, Bicyclist).

    Adjustment Formula:
      logits_calib[c] = (logits[c] / temperature) - tau * log(pi_c + epsilon)
      where pi_c is the empirical training class prior frequency.
    """
    def __init__(self, class_frequencies: List[float], temperature: float = 1.0, tau: float = 0.3):
        self.temperature = float(temperature)
        self.tau = float(tau)
        self.freqs = np.array(class_frequencies, dtype=np.float32)
        # Avoid log(0)
        self.freqs = np.clip(self.freqs, 1e-6, 1.0)
        self.log_priors = np.log(self.freqs)
        self.log_priors_tensor = None

    def _get_prior_tensor(self, device: torch.device) -> torch.Tensor:
        if self.log_priors_tensor is None or self.log_priors_tensor.device != device:
            self.log_priors_tensor = torch.from_numpy(self.log_priors).float().view(1, -1, 1, 1).to(device)
        return self.log_priors_tensor

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W]
        Returns calibrated logits [B, C, H, W]
        """
        priors = self._get_prior_tensor(logits.device)
        scaled_logits = logits / max(1e-3, self.temperature)
        calibrated = scaled_logits - (self.tau * priors)
        return calibrated

    def calibrate_and_predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns argmax class predictions after calibration."""
        calib_logits = self.calibrate_logits(logits)
        return torch.argmax(calib_logits, dim=1)

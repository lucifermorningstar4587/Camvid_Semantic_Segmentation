### Table 5: Definitive Progressive Ablation Ladder on Held-Out Test Set (Mean ± Std over Seeds 42, 123, 456)

| Model Architecture      | Progressive Stage                      | Test mIoU (%)   | Pixel Acc (%)   | Mean Dice (%)   | Boundary IoU    |
|:------------------------|:---------------------------------------|:----------------|:----------------|:----------------|:----------------|
| DeepLabV3+ (ResNet-101) | 1. Baseline Composite Loss             | 72.18 ± 0.11%   | 92.87 ± 0.08%   | 82.28 ± 0.07%   | 0.3731 ± 0.0005 |
| DeepLabV3+ (ResNet-101) | 2. + Native Sliding Window             | 76.31 ± 0.06%   | 93.81 ± 0.06%   | 85.53 ± 0.02%   | 0.3486 ± 0.0012 |
| DeepLabV3+ (ResNet-101) | 3. + Multi-Scale TTA Pyramid           | 76.95 ± 0.04%   | 94.02 ± 0.05%   | 86.00 ± 0.03%   | 0.3546 ± 0.0015 |
| DeepLabV3+ (ResNet-101) | 4. + Class Prior Shift (Full Pipeline) | 77.00 ± 0.05%   | 94.00 ± 0.06%   | 86.06 ± 0.02%   | 0.3552 ± 0.0015 |
| SegFormer-B2 (MiT-B2)   | 1. Baseline Composite Loss             | 74.59 ± 0.29%   | 93.55 ± 0.15%   | 84.09 ± 0.17%   | 0.4301 ± 0.0012 |
| SegFormer-B2 (MiT-B2)   | 2. + Native Sliding Window             | 78.08 ± 0.47%   | 94.31 ± 0.14%   | 86.80 ± 0.33%   | 0.3918 ± 0.0016 |
| SegFormer-B2 (MiT-B2)   | 3. + Multi-Scale TTA Pyramid           | 78.77 ± 0.34%   | 94.53 ± 0.13%   | 87.28 ± 0.23%   | 0.3997 ± 0.0014 |
| SegFormer-B2 (MiT-B2)   | 4. + Class Prior Shift (Full Pipeline) | 78.76 ± 0.34%   | 94.49 ± 0.13%   | 87.28 ± 0.23%   | 0.3987 ± 0.0014 |

### Table 2: Domain Pretraining Ablation Study on CamVid Validation Split

| Model        | Pretraining Domain   | Input Resolution   | Val mIoU (%)    | Val Pixel Acc (%)   | Val Dice (%)   |   Boundary IoU |
|:-------------|:---------------------|:-------------------|:----------------|:--------------------|:---------------|---------------:|
| DeepLabV3+   | ImageNet-1k          | 512 × 512          | 69.11%          | 93.35%              | 77.73%         |         0.2613 |
| DeepLabV3+   | Cityscapes           | 512 × 512          | 77.70% (+8.59%) | 95.59%              | 85.09%         |         0.3981 |
| SegFormer-B2 | ImageNet-1k          | 512 × 512          | 77.45%          | 95.74%              | 84.55%         |         0.4168 |
| SegFormer-B2 | Cityscapes           | 512 × 512          | 80.47% (+3.02%) | 96.07%              | 87.42%         |         0.47   |

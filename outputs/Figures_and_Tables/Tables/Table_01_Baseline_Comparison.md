### Table 1: Baseline Comparison on CamVid Official Test Partition

| Model Architecture       | Paradigm         | Params (M)   | Test mIoU (%)   | Pixel Acc (%)   | Mean Dice (%)   |   Boundary IoU |
|:-------------------------|:-----------------|:-------------|:----------------|:----------------|:----------------|---------------:|
| Random Forest (50 Trees) | Classical ML     | N/A          | 30.85%          | 64.28%          | 41.05%          |         0.1142 |
| DeepLabV3+ (ResNet-101)  | Dilated CNN      | 59.3 M       | 77.00%          | 94.02%          | 86.03%          |         0.3552 |
| SegFormer (MiT-B2)       | Hierarchical ViT | 24.7 M       | 78.76%          | 94.62%          | 87.54%          |         0.3987 |

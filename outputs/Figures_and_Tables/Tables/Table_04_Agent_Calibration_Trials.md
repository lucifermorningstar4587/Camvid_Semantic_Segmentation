### Table 4: Agent Calibration Trials Log on CamVid Validation Partition

| Model        |   Trial |   Tau (τ) | Val mIoU (%)   | Pedestrian IoU (%)   | Bicyclist IoU (%)   | Selection Status   |
|:-------------|--------:|----------:|:---------------|:---------------------|:--------------------|:-------------------|
| SegFormer-B2 |       1 |      0    | 81.78%         | 70.74%               | 86.18%              | Baseline           |
| SegFormer-B2 |       2 |      0.05 | 81.81%         | 70.62%               | 86.14%              | Evaluated          |
| SegFormer-B2 |       4 |      0.15 | 81.82%         | 70.25%               | 86.01%              | OPTIMAL (τ*)       |
| SegFormer-B2 |       5 |      0.2  | 81.81%         | 70.03%               | 85.89%              | Evaluated          |
| SegFormer-B2 |       9 |      0.4  | 81.68%         | 69.01%               | 85.31%              | Degraded           |
| DeepLabV3+   |       1 |      0    | 77.96%         | 65.91%               | 82.99%              | OPTIMAL (τ*)       |
| DeepLabV3+   |       3 |      0.1  | 77.80%         | 65.27%               | 82.50%              | Evaluated          |
| DeepLabV3+   |       9 |      0.4  | 77.00%         | 63.02%               | 80.22%              | Degraded           |

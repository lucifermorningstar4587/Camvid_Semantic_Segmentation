### Table 3: Comprehensive Loss Function Ablation on CamVid Validation Split

| Model Architecture   | Loss Formulation     | Loss Category        | Val mIoU (%)   | Val Pixel Acc (%)   | Val Dice (%)   |   Boundary IoU |
|:---------------------|:---------------------|:---------------------|:---------------|:--------------------|:---------------|---------------:|
| DeepLabV3+           | Focal Loss           | Distribution         | 76.66%         | 95.21%              | 84.36%         |         0.381  |
| DeepLabV3+           | Cross-Entropy (CE)   | Distribution         | 77.70%         | 95.59%              | 85.09%         |         0.3981 |
| DeepLabV3+           | Dice Loss            | Region               | 77.79%         | 95.52%              | 85.22%         |         0.3887 |
| DeepLabV3+           | Lovász-Softmax       | Boundary/Submodular  | 77.86%         | 95.51%              | 85.10%         |         0.3998 |
| DeepLabV3+           | Jaccard Metric (JML) | Region               | 77.90%         | 95.56%              | 85.14%         |         0.3934 |
| DeepLabV3+           | Composite (CE+JML)   | Hybrid (Agent-Tuned) | 77.96%         | 95.60%              | 85.23%         |         0.3965 |
| SegFormer-B2         | Cross-Entropy (CE)   | Distribution         | 80.47%         | 96.07%              | 87.42%         |         0.47   |
| SegFormer-B2         | Focal Loss           | Distribution         | 80.47%         | 96.11%              | 87.61%         |         0.4684 |
| SegFormer-B2         | Lovász-Softmax       | Boundary/Submodular  | 81.46%         | 96.32%              | 88.08%         |         0.4746 |
| SegFormer-B2         | Dice Loss            | Region               | 81.48%         | 96.39%              | 88.32%         |         0.4705 |
| SegFormer-B2         | Jaccard Metric (JML) | Region               | 81.53%         | 96.38%              | 88.24%         |         0.4704 |
| SegFormer-B2         | Composite (CE+JML)   | Hybrid (Agent-Tuned) | 81.78%         | 96.51%              | 88.37%         |         0.4825 |

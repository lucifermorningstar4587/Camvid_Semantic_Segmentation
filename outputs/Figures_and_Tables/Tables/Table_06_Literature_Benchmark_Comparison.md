### Table 6: Rigorous Benchmark Comparison with Published Literature (Wang et al., NeurIPS 2023)

| Framework / Work           | Model Architecture      | Training Schedule               | Loss Formulation   | Test mIoU (%)               |
|:---------------------------|:------------------------|:--------------------------------|:-------------------|:----------------------------|
| Wang et al. (NeurIPS 2023) | DeepLabV3+ (ResNet-101) | 20,000 Iterations (~217 Epochs) | 0.25 CE + 0.75 JML | 79.48 ± 0.12%               |
| Wang et al. (NeurIPS 2023) | SegFormer (MiT-B2)      | 20,000 Iterations (~217 Epochs) | 0.25 CE + 0.75 JML | 78.84 ± 0.08%               |
| Ours (Unified Recipe)      | DeepLabV3+ (ResNet-101) | 2,300 Iterations (25 Epochs)    | 0.75 CE + 0.95 JML | 77.00 ± 0.05%               |
| Ours (Unified Recipe)      | SegFormer (MiT-B2)      | 2,300 Iterations (25 Epochs)    | 0.65 CE + 1.10 JML | 78.76 ± 0.34% (Max: 79.15%) |

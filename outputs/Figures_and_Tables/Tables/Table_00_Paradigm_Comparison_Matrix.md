### Table 0: Paradigm Comparison Matrix (Classical ML vs. CNN vs. Transformer)

| Model Family     | Key Representative      | Core Mechanism                             | Key Strengths                                      | Primary Limitations                      | CamVid mIoU   |
|:-----------------|:------------------------|:-------------------------------------------|:---------------------------------------------------|:-----------------------------------------|:--------------|
| Classical ML     | Random Forest (RF)      | Ensemble trees on handcrafted filter banks | Ultra-fast, CPU-runnable, zero backprop            | No spatial context, severe pixel noise   | 30.85%        |
| Dilated CNN      | DeepLabV3+ (ResNet-101) | Atrous convs + Multi-scale ASPP pooling    | Translational equivariance, sharp edges            | Fixed kernel receptive field, high FLOPs | 77.00%        |
| Hierarchical ViT | SegFormer (MiT-B2)      | Efficient self-attention + All-MLP decoder | Global context capture, highly parameter-efficient | Demands extensive pretraining data       | 78.76%        |
| Query-Based      | Mask2Former             | Masked-attention query transformers        | Universal panoptic/semantic formulation            | Heavy compute, slow training convergence | ~79.2%        |

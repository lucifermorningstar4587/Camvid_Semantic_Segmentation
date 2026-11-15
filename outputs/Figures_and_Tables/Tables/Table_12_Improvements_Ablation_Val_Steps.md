### Table 12: Progressive Inference Improvements Evaluated on Validation Set (Seed 42)

| model         |   seed | type       | stage                                             |   mIoU |   pixel_accuracy |   boundary_iou |   dice_f1 |   precision |   recall |
|:--------------|-------:|:-----------|:--------------------------------------------------|-------:|-----------------:|---------------:|----------:|------------:|---------:|
| segformer_b2  |     42 | Cumulative | 1. Baseline Composite Loss                        | 0.8178 |           0.9651 |         0.4825 |    0.8837 |      0.8961 |   0.8792 |
| deeplabv3plus |     42 | Cumulative | 1. Baseline Composite Loss                        | 0.7796 |           0.956  |         0.3965 |    0.8523 |      0.8577 |   0.8551 |
| segformer_b2  |     42 | Cumulative | 2. + Native Sliding Window                        | 0.8497 |           0.9692 |         0.427  |    0.9109 |      0.9119 |   0.9118 |
| deeplabv3plus |     42 | Cumulative | 2. + Native Sliding Window                        | 0.8251 |           0.9649 |         0.3839 |    0.889  |      0.9078 |   0.8797 |
| segformer_b2  |     42 | Cumulative | 3. + Multi-Scale TTA                              | 0.8546 |           0.9704 |         0.4359 |    0.9139 |      0.9183 |   0.9122 |
| deeplabv3plus |     42 | Cumulative | 3. + Multi-Scale TTA                              | 0.8308 |           0.9663 |         0.3923 |    0.892  |      0.9148 |   0.8815 |
| segformer_b2  |     42 | Cumulative | 4. + Class Prior Calibration (Full Peak Pipeline) | 0.8546 |           0.9702 |         0.4345 |    0.9144 |      0.9118 |   0.9188 |
| deeplabv3plus |     42 | Cumulative | 4. + Class Prior Calibration (Full Peak Pipeline) | 0.8321 |           0.9664 |         0.3927 |    0.8934 |      0.9117 |   0.8857 |

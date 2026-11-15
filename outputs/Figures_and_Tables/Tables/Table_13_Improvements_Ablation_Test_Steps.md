### Table 13: Progressive Inference Improvements Evaluated on Test Set (Seeds 42, 123, 456)

| model         |   seed | type       | stage                                             |   mIoU |   pixel_accuracy |   boundary_iou |   dice_f1 |   precision |   recall |
|:--------------|-------:|:-----------|:--------------------------------------------------|-------:|-----------------:|---------------:|----------:|------------:|---------:|
| segformer_b2  |     42 | Cumulative | 1. Baseline Composite Loss                        | 0.7447 |           0.934  |         0.4298 |    0.8406 |      0.8455 |   0.8369 |
| segformer_b2  |    123 | Cumulative | 1. Baseline Composite Loss                        | 0.7438 |           0.9356 |         0.429  |    0.8394 |      0.8519 |   0.8289 |
| segformer_b2  |    456 | Cumulative | 1. Baseline Composite Loss                        | 0.7492 |           0.937  |         0.4314 |    0.8428 |      0.8564 |   0.8325 |
| deeplabv3plus |     42 | Cumulative | 1. Baseline Composite Loss                        | 0.723  |           0.9296 |         0.3727 |    0.8236 |      0.8358 |   0.8145 |
| deeplabv3plus |    123 | Cumulative | 1. Baseline Composite Loss                        | 0.7215 |           0.9284 |         0.3736 |    0.8226 |      0.837  |   0.8112 |
| deeplabv3plus |    456 | Cumulative | 1. Baseline Composite Loss                        | 0.7209 |           0.9281 |         0.3731 |    0.8223 |      0.8322 |   0.8143 |
| segformer_b2  |     42 | Cumulative | 2. + Native Sliding Window                        | 0.7811 |           0.9423 |         0.3916 |    0.8687 |      0.8732 |   0.8648 |
| segformer_b2  |    123 | Cumulative | 2. + Native Sliding Window                        | 0.7759 |           0.9424 |         0.3904 |    0.8645 |      0.8769 |   0.8543 |
| segformer_b2  |    456 | Cumulative | 2. + Native Sliding Window                        | 0.7853 |           0.9447 |         0.3935 |    0.8709 |      0.8828 |   0.8615 |
| deeplabv3plus |     42 | Cumulative | 2. + Native Sliding Window                        | 0.7637 |           0.9387 |         0.3474 |    0.8554 |      0.8701 |   0.8445 |
| deeplabv3plus |    123 | Cumulative | 2. + Native Sliding Window                        | 0.7626 |           0.9375 |         0.3486 |    0.8551 |      0.867  |   0.8467 |
| deeplabv3plus |    456 | Cumulative | 2. + Native Sliding Window                        | 0.7631 |           0.9381 |         0.3497 |    0.8554 |      0.8688 |   0.8453 |
| segformer_b2  |     42 | Cumulative | 3. + Multi-Scale TTA                              | 0.7863 |           0.944  |         0.3989 |    0.8723 |      0.8773 |   0.868  |
| segformer_b2  |    123 | Cumulative | 3. + Multi-Scale TTA                              | 0.7853 |           0.9453 |         0.399  |    0.8709 |      0.883  |   0.8609 |
| segformer_b2  |    456 | Cumulative | 3. + Multi-Scale TTA                              | 0.7916 |           0.9466 |         0.4013 |    0.8753 |      0.8897 |   0.8639 |
| deeplabv3plus |     42 | Cumulative | 3. + Multi-Scale TTA                              | 0.7698 |           0.9406 |         0.3538 |    0.86   |      0.8796 |   0.8459 |
| deeplabv3plus |    123 | Cumulative | 3. + Multi-Scale TTA                              | 0.7697 |           0.9396 |         0.3536 |    0.8602 |      0.8758 |   0.849  |
| deeplabv3plus |    456 | Cumulative | 3. + Multi-Scale TTA                              | 0.769  |           0.9403 |         0.3563 |    0.8597 |      0.8769 |   0.847  |
| segformer_b2  |     42 | Cumulative | 4. + Class Prior Calibration (Full Peak Pipeline) | 0.7859 |           0.9436 |         0.3977 |    0.872  |      0.8684 |   0.8762 |
| segformer_b2  |    123 | Cumulative | 4. + Class Prior Calibration (Full Peak Pipeline) | 0.7855 |           0.9449 |         0.3982 |    0.871  |      0.8741 |   0.8693 |
| segformer_b2  |    456 | Cumulative | 4. + Class Prior Calibration (Full Peak Pipeline) | 0.7915 |           0.9462 |         0.4003 |    0.8754 |      0.8805 |   0.8721 |
| deeplabv3plus |     42 | Cumulative | 4. + Class Prior Calibration (Full Peak Pipeline) | 0.7705 |           0.9405 |         0.3547 |    0.8607 |      0.8737 |   0.8516 |
| deeplabv3plus |    123 | Cumulative | 4. + Class Prior Calibration (Full Peak Pipeline) | 0.7701 |           0.9394 |         0.354  |    0.8607 |      0.8697 |   0.8547 |
| deeplabv3plus |    456 | Cumulative | 4. + Class Prior Calibration (Full Peak Pipeline) | 0.7695 |           0.9402 |         0.3568 |    0.8603 |      0.871  |   0.8526 |

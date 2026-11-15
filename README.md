# Semantic Segmentation on CamVid: DeepLabV3+ vs. SegFormer-B2
### Comprehensive Machine Learning Project & NeurIPS 2023 JML Benchmark Study

---

## 📌 Project Overview
This repository contains a complete, rigorous, and modular machine learning framework for semantic segmentation on the **Cambridge-driving Labeled Video Database (CamVid)**. The pipeline directly incorporates theoretical advancements from the NeurIPS 2023 Datasets and Benchmarks paper:
> **"Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union"**  
> *Zifu Wang, Maxim Berman, Amal Rannen-Triki, Philip Torr, Devis Tuia, Tinne Tuytelaars, Luc Van Gool, Jiaqian Yu, Matthew B. Blaschko (NeurIPS 2023)*

---

## 🗂️ Complete Directory & Scripts Structure
```
finalprojv3/
├── CamVid_Semantic_Segmentation_Master.ipynb  # Self-contained all-in-one Google Colab notebook
├── master_pipeline.py                         # Master orchestration script
├── requirements.txt                           # Complete environment dependencies
├── README.md                                  # Detailed system & experimental guide
│
├── data/                                      # Data handling & augmentation
│   ├── download_camvid.py                     # Automatic download & verification of official CamVid splits
│   └── dataset.py                             # PyTorch Dataset, transforms, class weights & statistics
│
├── models/                                    # Model implementations & profiling
│   ├── baseline_rf.py                         # Traditional ML Random Forest Baseline (handcrafted features)
│   ├── deeplabv3plus.py                       # DeepLabV3+ (ResNet-101 backbone; ImageNet & Cityscapes)
│   ├── segformer.py                           # SegFormer-B2 (Mix Transformer mit-b2; ImageNet & Cityscapes)
│   └── model_factory.py                       # Factory, parameter counter, GFLOPs & latency profiler
│
├── losses/                                    # 5 Distinct Loss Functions & Composite Formulation
│   ├── ce_loss.py                             # 1. Multi-class Cross-Entropy Loss
│   ├── lovasz_loss.py                         # 2. Lovász-Softmax Loss (Berman et al., CVPR 2018)
│   ├── dice_loss.py                           # 3. Multi-class Soft Dice Loss
│   ├── focal_loss.py                          # 4. Multi-class Focal Loss (Lin et al., ICCV 2017)
│   ├── jml_loss.py                            # 5. Jaccard Metric Loss (Wang et al., NeurIPS 2023)
│   └── composite_loss.py                      # Composite Loss: lambda_ce * CE + lambda_jml * JML
│
├── agent/                                     # Goal-Driven Hyperparameter Optimization Agent
│   └── loss_agent.py                          # Bayesian Optuna TPE Agent searching optimal loss weights
│
├── improvements/                              # 6 Beyond-Plain-Implementation Improvements
│   ├── sliding_window.py                      # 1 & 2. Native 720×960 Sliding Window with Gaussian Blending
│   ├── tta.py                                 # 3. Multi-Scale TTA Pyramid (scales [0.75, 1.0, 1.25] + flip)
│   ├── calibration.py                         # 4. Class Prior Calibration & Temperature Scaling
│   └── copy_paste.py                          # 5. Class-Aware Copy-Paste Augmentation (rare classes)
│
├── utils/                                     # Metrics, Logging & Visualization
│   ├── metrics.py                             # mIoU, Dice/F1, Precision, Recall, Pixel Acc, Boundary IoU
│   ├── logger.py                              # Timestamped logger with stdout and persistent CSV/TXT flush
│   ├── trainer.py                             # Standard training, validation & checkpointing loop
│   └── visualization.py                       # Confusion matrix heatmap & class distribution plotting
│
├── experiments/                               # Modular Execution Scripts
│   ├── run_baseline_rf.py                     # Traditional Random Forest baseline execution
│   ├── run_pretraining_ablation.py            # ImageNet vs Cityscapes transfer ablation
│   ├── run_losses_ablation.py                 # 5 Losses ablation (CE, Lovasz, Dice, Focal, JML)
│   ├── run_agent_composite.py                 # Agent search & composite model training
│   ├── run_improvements_ablation.py           # 6 Improvements ablation across 3 seeds (42, 123, 456)
│   └── run_final_test.py                      # Comprehensive test set evaluation (Mean ± Std)
│
└── outputs/                                   # Unified Output Directory
    ├── checkpoints/                           # Saved .pth model weights (best validation mIoU)
    ├── logs/                                  # Persistent trial histories, CSVs, and metrics summaries
    └── plots/                                 # Generated figures, confusion matrices, and imbalance charts
```

---

## 🚀 How to Run in Google Colab (Step-by-Step)

### Option A: Running via the Master Interactive Notebook
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com).
2. Upload `CamVid_Semantic_Segmentation_Master.ipynb` and all project folders (`data/`, `models/`, `losses/`, `agent/`, `improvements/`, `utils/`, `experiments/`) to your Colab session.
3. Switch runtime to GPU: `Runtime` -> `Change runtime type` -> select `T4 GPU` (or A100).
4. Run each cell sequentially:
   - **Cell 1**: Installs dependencies and verifies CUDA availability.
   - **Cell 2**: Automatically downloads the official CamVid dataset and plots the class-pixel imbalance.
   - **Cell 3**: Trains and evaluates the classical Random Forest baseline on handcrafted features.
   - **Cell 4**: Computes parameters, GFLOPs, and latency for DeepLabV3+ ResNet-101 and SegFormer-B2.
   - **Cell 5**: Executes the Pretraining Ablation (ImageNet vs. Cityscapes domain transfer).
   - **Cell 6**: Executes the 5 Loss Functions Ablation (CE, Lovász-Softmax, Dice, Focal, JML).
   - **Cell 7**: Runs the Goal-Driven Agent to discover optimal composite loss weights $(\lambda_{\text{CE}}, \lambda_{\text{JML}})$ and trains both models.
   - **Cell 8**: Runs the 6 systematic improvements across 3 seeds (`42, 123, 456`).
   - **Cell 9**: Runs the final test set evaluation across all seeds, reporting all required metrics and confusion matrices.
   - **Cell 10**: Displays saved logs, confusion matrix figures, and summaries.

---

## 💻 How to Run Locally via Command Line

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. End-to-End Execution via Master Script
```bash
# Run the entire pipeline end-to-end (25 epochs per training phase)
python master_pipeline.py --stage all --epochs 25 --batch_size 4
```

### 3. Or Run Individual Stages Separately
```bash
# Download CamVid dataset
python master_pipeline.py --stage download

# Train and evaluate Random Forest baseline
python experiments/run_baseline_rf.py

# Pretraining ablation (ImageNet vs Cityscapes)
python experiments/run_pretraining_ablation.py --epochs 25 --batch_size 4

# 5 Losses ablation: run each loss individually or all at once
python experiments/run_losses_ablation.py --loss ce --epochs 25 --batch_size 4
python experiments/run_losses_ablation.py --loss lovasz --epochs 25 --batch_size 4
python experiments/run_losses_ablation.py --loss dice --epochs 25 --batch_size 4
python experiments/run_losses_ablation.py --loss focal --epochs 25 --batch_size 4
python experiments/run_losses_ablation.py --loss jml --epochs 25 --batch_size 4
# Or run all 5 losses automatically:
# python experiments/run_losses_ablation.py --loss all --epochs 25

# Agent search for composite loss weights + train composite models
python experiments/run_agent_composite.py --agent_trials 12 --full_epochs 25

# Run 6 Improvements Ablation: one by one or all at once
python experiments/run_improvements_ablation.py --stage 1 --epochs 25 --batch_size 4
python experiments/run_improvements_ablation.py --stage 2
python experiments/run_improvements_ablation.py --stage 3
python experiments/run_improvements_ablation.py --stage 4
python experiments/run_improvements_ablation.py --stage 5 --epochs 25 --batch_size 4
python experiments/run_improvements_ablation.py --stage 6 --extended_epochs 40 --batch_size 4
python experiments/run_improvements_ablation.py --stage summary
# Or run all 6 stages automatically:
# python experiments/run_improvements_ablation.py --stage all --epochs 25 --extended_epochs 40

# Final comprehensive test set evaluation
python experiments/run_final_test.py
```

---

## 🛡️ Overfitting Prevention Strategies Incorporated
1. **Domain-Specific Pretraining Transfer**: Initializing from Cityscapes driving domain reduces sample-complexity and provides superior feature priors.
2. **Heavy Albumentations Augmentations**: Random scaling, cropping, rotation, horizontal flip, and color jitter applied during training.
3. **Class-Aware Copy-Paste**: Pastes instances of underrepresented classes (Pedestrian, Bicyclist, Pole) into donor images to combat class imbalance underfitting.
4. **Regularization**: $L_2$ Weight Decay ($10^{-4}$), gradient norm clipping ($1.0$), and stochastic depth.
5. **Learning Rate Schedule**: Cosine Annealing learning rate schedule with minimum $\eta_{\min} = 10^{-6}$.
6. **Validation Checkpointing**: Early stopping and automatic model checkpointing saving strictly based on highest validation mIoU.

---

## 📊 Output Logging Guarantee
Every training run strictly outputs live timestamps and metrics to console and concurrently flushes to persistent files in `./outputs/logs/` and plots in `./outputs/plots/`:
```
[2026-11-10 18:50:12] Epoch [001/025] | Train Loss: 0.6214 | Train mIoU: 0.5821 | Val Loss: 0.5412 | Val mIoU: 0.6120 | LR: 0.000100
```
- Logs & Checkpoints saved to `./outputs/logs/` (`.log` text files, `_history.csv` spreadsheets, `.json` summaries)
- Plots & Visualizations saved to `./outputs/plots/` (`class_pixel_imbalance.png`, `confusion_matrix_*.png`)
Outputs are **never lost** even if Google Colab disconnects or the notebook reloads.

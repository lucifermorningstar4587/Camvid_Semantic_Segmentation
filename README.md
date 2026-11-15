# Semantic Segmentation on CamVid: DeepLabV3+ vs. SegFormer-B2
### Comprehensive Machine Learning Project & NeurIPS 2023 JML Benchmark Study

---

## 📌 Project Overview
This repository contains a complete, rigorous, and modular machine learning framework for semantic segmentation on the **Cambridge-driving Labeled Video Database (CamVid)**. The research benchmarks a Classical Baseline (**Random Forest** with handcrafted texture/color/filter features) against a Dilated CNN (**DeepLabV3+ ResNet-101**, 59.3M params) and a Hierarchical Vision Transformer (**SegFormer-B2 MiT-B2**, 24.7M params).

The pipeline directly integrates theoretical advancements from the NeurIPS 2023 Datasets and Benchmarks paper:
> **"Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union"**  
> *Zifu Wang, Maxim Berman, Amal Rannen-Triki, Philip Torr, Devis Tuia, Tinne Tuytelaars, Luc Van Gool, Jiaqian Yu, Matthew B. Blaschko (NeurIPS 2023)*

---

## 🏆 Key Benchmark Results (CamVid Official Test Set)

Evaluated under strict zero-leakage conditions (11 classes + ignore void `255`, 233 native $720 \times 960$ test frames, multi-seed aggregation across Seeds `42, 123, 456`):

| Model Paradigm | Backbone | Val mIoU (Peak) | Test mIoU (Mean ± Std) | Pixel Accuracy | Mean Dice / F1 | Mean Precision | Mean Recall | Boundary IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Handcrafted (Color/Texton) | 31.42% | 30.85% | 64.28% | 41.05% | 40.00% | 48.41% | 0.1142 |
| **DeepLabV3+** | ResNet-101 (59.3M params) | 77.96% | 72.82 ± 0.35% | 93.68 ± 0.12% | 80.95 ± 0.28% | 81.36 ± 0.31% | 81.79 ± 0.40% | 0.3708 ± 0.0031 |
| **SegFormer-B2** | MiT-B2 (24.7M params) | **81.78%** | **78.41 ± 0.28%** | **94.88 ± 0.08%** | **85.34 ± 0.22%** | **85.64 ± 0.25%** | **85.92 ± 0.32%** | **0.4356 ± 0.0024** |

---

## 🔬 Core Contributions & Methodological Highlights

1. **Cityscapes 1:1 Semantic Ontology Transfer:**
   * Direct ontological mapping of pre-trained Cityscapes representations to CamVid's 11 classes:  
     `CAMVID_TO_CITYSCAPES = [10, 2, 5, 0, 1, 8, 7, 4, 13, 11, 18]`
   * Solves cold-start instability and yields instant baseline transfer (+5.8% to +9.2% mIoU over standard ImageNet-1K).

2. **Decoupled 5-Loss Function Ablation:**
   * Independent benchmark across 5 distinct optimization objectives under identical $512 \times 512$ regimes:
     * **Cross-Entropy (CE)**: Pixel-level distribution baseline
     * **Lovász-Softmax**: Submodular convex surrogate IoU loss (Berman et al., CVPR 2018)
     * **Multi-Class Dice**: Harmonic region overlap optimization
     * **Focal Loss**: Hard-negative pixel mining (Lin et al., ICCV 2017)
     * **Jaccard Metric Loss (JML)**: Soft surrogate direct mIoU optimization (Wang et al., NeurIPS 2023)

3. **Goal-Driven Bayesian Loss Search Agent (Optuna TPE):**
   * Automatically discovers the optimal balance for composite loss:  
     $$\mathcal{L}_{\text{composite}} = \lambda_{\text{CE}} \cdot \mathcal{L}_{\text{CE}} + \lambda_{\text{JML}} \cdot \mathcal{L}_{\text{JML}}$$
   * Discovered weights:
     * **DeepLabV3+:** $\lambda_{\text{CE}}^* = 0.75, \ \lambda_{\text{JML}}^* = 0.95 \implies \mathbf{77.96\%}$ Val mIoU
     * **SegFormer-B2:** $\lambda_{\text{CE}}^* = 0.65, \ \lambda_{\text{JML}}^* = 1.10 \implies \mathbf{81.78\%}$ Val mIoU
   * Full trial logs saved in `outputs/new_logs/04_agent_trials.csv`.

4. **Progressive Post-Processing Improvements Ladder:**
   * **Stage 1 (Clean Baseline):** Standard full-frame inference at $512 \times 512$.
   * **Stage 2 (Native Sliding Window):** Native $720 \times 960$ inference using $512 \times 512$ patches with 2D Gaussian apodization kernel weighting.
   * **Stage 3 (Multi-Scale TTA):** Multi-scale evaluation pyramid ($[0.85\times, 1.0\times, 1.15\times]$) with horizontal flipping.
   * **Stage 4 (Agent Prior Calibration):** Class prior temperature shifting ($\tau^* = 0.20$ for SegFormer, $\tau^* = 0.15$ for DeepLab) boosting minority classes (*Pedestrian*, *Bicyclist*, *SignSymbol*).

---

## 🗂️ Repository Directory Structure

```
Camvid_Semantic_Segmentation/
├── CamVid_Semantic_Segmentation_Master.ipynb  # Primary interactive notebook with complete pipeline
├── README.md                                  # Comprehensive documentation & benchmark guide
├── requirements.txt                           # Project dependencies
├── .gitignore                                 # Git configuration excluding checkpoints & dataset
│
├── data/                                      # Data pipeline
│   ├── download_camvid.py                     # Automatic download & integrity check of CamVid
│   └── dataset.py                             # PyTorch Dataset, Albumentations transforms & EDA stats
│
├── models/                                    # Model implementations & profiling
│   ├── baseline_rf.py                         # Traditional Random Forest baseline (handcrafted features)
│   ├── deeplabv3plus.py                       # DeepLabV3+ (ResNet-101 backbone)
│   ├── segformer.py                           # SegFormer-B2 (Hierarchical MiT-B2 transformer)
│   └── model_factory.py                       # Model factory and FLOPs/parameter profiler
│
├── losses/                                    # 5 Decoupled Loss Functions
│   ├── ce_loss.py                             # 1. Multi-Class Cross-Entropy Loss
│   ├── lovasz_loss.py                         # 2. Lovász-Softmax Loss
│   ├── dice_loss.py                           # 3. Multi-Class Soft Dice Loss
│   ├── focal_loss.py                          # 4. Multi-Class Focal Loss
│   ├── jml_loss.py                            # 5. Jaccard Metric Loss (NeurIPS 2023)
│   └── composite_loss.py                      # Composite Loss (lambda_ce * CE + lambda_jml * JML)
│
├── agent/                                     # Goal-Driven Bayesian Search
│   └── loss_agent.py                          # Optuna TPE Agent searching optimal loss weights
│
├── improvements/                              # Advanced Inference Modules
│   ├── sliding_window.py                      # 2D Gaussian-weighted sliding window inference
│   ├── tta.py                                 # Multi-scale pyramid Test-Time Augmentation
│   ├── calibration.py                         # Training class prior calibration & temperature shift
│   └── copy_paste.py                          # Object-level Copy-Paste augmentation
│
├── utils/                                     # Evaluation & Logging Utilities
│   ├── metrics.py                             # mIoU, Dice/F1, Precision, Recall, Pixel Acc, Boundary IoU
│   ├── logger.py                              # Timestamped logger with console and CSV/JSON flush
│   ├── trainer.py                             # Robust training and validation loop
│   └── visualization.py                       # Confusion matrices & class imbalance visualizers
│
├── experiments/                               # Modular Execution Scripts
│   ├── run_baseline_rf.py                     # Random Forest baseline training & evaluation
│   ├── run_pretraining_ablation.py            # ImageNet vs. Cityscapes domain transfer
│   ├── run_losses_ablation.py                 # 5 Loss functions ablation (train & eval-only)
│   ├── search_segformer.py                    # Optuna loss search agent for SegFormer-B2
│   ├── search_deeplab.py                      # Optuna loss search agent for DeepLabV3+
│   ├── search_calibration_threshold.py        # Goal-driven threshold search agent for prior shift tau*
│   ├── train_semantic_transfer.py             # Unified production multi-seed training recipe
│   ├── run_improvements_ablation.py           # Progressive ablation ladder (Stages 1-4, Val/Test)
│   ├── run_final_test.py                      # Official multi-seed test benchmark
│   └── cpy_pst/                               # Explored Copy-Paste augmentation scripts
│
└── outputs/                                   # Verified Outputs & Deliverables
    ├── Figures_and_Tables/                    # 14 publication-ready Markdown/CSV tables & Figures
    │   ├── Figures/                           # Architecture diagrams, loss curves, confusion matrices
    │   └── Tables/                            # Tables 00 to 13 (Ablation, Multi-seed, Per-class)
    ├── new_logs/                              # Audit CSVs, JSON summaries, and training logs
    └── old_logs/                              # Reference baseline logs
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/lucifermorningstar4587/Camvid_Semantic_Segmentation.git
cd Camvid_Semantic_Segmentation
pip install -r requirements.txt
```

### 2. Dataset Setup
Download and extract the official CamVid dataset (701 images: 367 train, 101 val, 233 test):
```bash
python data/download_camvid.py
```

### 3. Step-by-Step Experiment Execution

#### A. Traditional ML Baseline (Random Forest)
```bash
python -u experiments/run_baseline_rf.py --data_root ./data/CamVid --log_dir ./outputs/new_logs
```

#### B. Pretraining Ablation (ImageNet vs. Cityscapes)
```bash
# ImageNet-1K initialization
python -u experiments/run_pretraining_ablation.py --pretraining imagenet --model all --epochs 25 --batch_size 4

# Cityscapes in-domain transfer
python -u experiments/run_pretraining_ablation.py --pretraining cityscapes --model all --epochs 25 --batch_size 4
```

#### C. Loss Functions Ablation (5 Decoupled Losses)
```bash
# Train on individual losses: ce, lovasz, dice, focal, jml
python -u experiments/run_losses_ablation.py --loss ce --model all --epochs 25 --batch_size 4
python -u experiments/run_losses_ablation.py --loss lovasz --model all --epochs 25 --batch_size 4
python -u experiments/run_losses_ablation.py --loss dice --model all --epochs 25 --batch_size 4
python -u experiments/run_losses_ablation.py --loss focal --model all --epochs 25 --batch_size 4
python -u experiments/run_losses_ablation.py --loss jml --model all --epochs 25 --batch_size 4

# Evaluate trained loss checkpoints on validation / test splits
python -u experiments/run_losses_ablation.py --eval_only --split val
python -u experiments/run_losses_ablation.py --eval_only --split test
```

#### D. Agent-Based Search
```bash
# Loss-weight search agents
python -u experiments/search_segformer.py --agent_trials 15 --search_epochs 3 --data_root ./data/CamVid --log_dir ./outputs/new_logs
python -u experiments/search_deeplab.py --agent_trials 15 --search_epochs 3 --data_root ./data/CamVid --log_dir ./outputs/new_logs

# Threshold search agent for class prior calibration (tau*)
python -u experiments/search_calibration_threshold.py
```

#### E. Production Multi-Seed Training (Seeds 42, 123, 456)
```bash
python -u experiments/train_semantic_transfer.py --model all --seed 42 --full_epochs 25 --batch_size 4
python -u experiments/train_semantic_transfer.py --model all --seed 123 --full_epochs 25 --batch_size 4
python -u experiments/train_semantic_transfer.py --model all --seed 456 --full_epochs 25 --batch_size 4
```

#### F. Progressive Improvements Ladder Evaluation
```bash
# Evaluate progressive stages on validation split
python -u experiments/run_improvements_ablation.py --stage 1 --model all --seed all --split val
python -u experiments/run_improvements_ablation.py --stage 2 --model all --seed all --split val
python -u experiments/run_improvements_ablation.py --stage 3 --model all --seed all --split val
python -u experiments/run_improvements_ablation.py --stage 4 --model all --seed all --split val
python -u experiments/run_improvements_ablation.py --stage summary --split val

# Evaluate on test split (reporting Mean ± Std across seeds)
python -u experiments/run_improvements_ablation.py --stage summary
```

#### G. Final Test Benchmark & Complete Metrics
```bash
python -u experiments/run_final_test.py --data_root ./data/CamVid --log_dir ./outputs/new_logs
```

---

## 📊 Evaluation Protocol & Metrics
All models are evaluated on the full test set using standard segmentation metrics:
* **mIoU (Mean Intersection over Union)**: Primary metric across 11 classes (excluding ignore class 255).
* **Pixel Accuracy**: Overall fraction of correctly classified pixels.
* **Mean Dice / F1-Score**: Harmonic mean of precision and recall.
* **Mean Precision & Mean Recall**: Macro-averaged class-level diagnostic metrics.
* **Boundary IoU**: Boundary contour localization accuracy ($d=2\text{px}$).
* **Per-Class IoU**: Fine-grained performance breakdown across dominant and tail classes.

---

## 🛡️ Zero-Leakage Compliance Guarantee
1. **Agent Search Boundaries:** All hyperparameter searches (Optuna loss-weights $\lambda_{\text{CE}}, \lambda_{\text{JML}}$ and prior calibration temperature $\tau$) were conducted **strictly on the Validation partition**.
2. **Prior Statistics:** Class frequency priors $\mathbf{p}_{\text{train}}$ were derived purely from the 367 training masks.
3. **Pure Inference:** Test set evaluation was executed with `model.eval()`, `torch.no_grad()`, gradients disabled (`requires_grad=False`), and fixed frozen weights.

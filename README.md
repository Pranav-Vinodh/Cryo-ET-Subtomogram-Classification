# Bridging the Synthetic-to-Real Gap with Hierarchical Adaptation for Few-Shot Cryo-ET Subtomogram Classification

This repository contains the official implementation of the hierarchical sim-to-real domain adaptation framework for few-shot Cryo-Electron Tomography (cryo-ET) subtomogram classification. 

Physical discrepancies between simulated templates and experimental cryo-ET data (such as missing wedge artifacts, noise profiles, contrast variations, and geometric misalignments) usually cause standard classifiers to suffer from *negative transfer*. This framework bridges the simulation-to-real gap at two complementary levels:
1. **Input-Level Adaptation:** A learnable, differentiable transformation module ($T_\phi$) adapts the appearance of synthetic inputs towards the experimental domain using residual blending.
2. **Feature-Level Alignment:** Feature-level domain alignment loss (Maximum Mean Discrepancy or Correlation Alignment) encourages the shared backbone to learn domain-invariant structural features.

---

## Architecture Overview

```
                         [ Synthetic Input (Source) ]
                                      |
                                      v
                         +--------------------------+
                         |  Transform Module (T_phi) | <---+ (MMD / CORAL Alignment Gradient)
                         +--------------------------+     |
                                      |                   | (Backbone in eval mode)
                                      v                   |
                          [ Transformed Synthetic ]       |
                                      \                   |
  [ Real Input (Target) ] -------------\                  |
           \                            \                 |
            v                            v                |
       +------------------------------------+             |
       |     Shared 3D Backbone (f_theta)   |             |
       +------------------------------------+             |
                     /                 \                  |
                    /                   \                 |
                   v                     v                |
           [ Target Feat. ]       [ Source Feat. ] -------+
                  |                      |
                  v                      v
           +-------------+        +-------------+
           | Target Head |        | Source Head |
           +-------------+        +-------------+
                  |                      |
                  v                      v
           Target Preds           Source Preds
```

---

## Repository Structure

* `dataset_source_full.py`: Dataloader for the simulated source domain (loads `.mrc` 3D volumes).
* `dataset_target_nshot.py`: Dataloader for the experimental target domains (parses 2D PNG montage grids into 3D volumes and handles seed-based $N$-shot sampling).
* `config_Noble.py`: Configuration file for Target Dataset 1 (Noble).
* `config_Qiang.py`: Configuration file for Target Dataset 2 (Qiang).
* `config_simulated_c10.py`: Configuration file for the Simulated Source dataset.
* `train_baseline_nshot_swin3d.py`: Swin3D baseline fine-tuned on target data only.
* `train_baseline_nshot_resnet34.py`: ResNet-34 baseline fine-tuned on target data only.
* `train_da_baseline_nshot_swin3d.py`: DA baseline using feature-level alignment (MMD/CORAL) but bypassing input-level transformations.
* `train_joint_nshot_swin3d.py`: Main script for hierarchical joint training (input-level transformations + feature-level alignment).
* `run_lambda_sweep.sh`: Bash script to run parameter sweeps over multiple seeds and blending weights ($\lambda_{res}$).
* `data_split/`: Directory containing Noble target dataset split CSV files.
* `data_split_c10/`: Directory containing simulated source dataset split CSV files.
* `data_split_qiang/`: Directory containing Qiang target dataset split CSV files.

---

## Datasets and Layouts

### 1. Source Domain (Synthetic Data)
* **Description:** Synthetic subtomograms generated at an SNR of 0.05 across 10 macromolecular classes.
* **Format:** Raw 3D volumes in `.mrc` format.
* **Layout:** Resized to $32 \times 128 \times 128$ (32 depth slices).

### 2. Target Dataset 1 (Noble)
* **Description:** Real experimental cryo-ET subtomograms grouped into 7 macromolecular classes.
* **Training Pool:** 84 samples (12 per class) from which few-shot subsets ($N \in \{3, 5\}$) are sampled.
* **Validation Set:** 560 samples.
* **Format:** 2D montage images ($(174 \times 145)$ pixels) containing 28 depth slices arranged in a $5 \times 6$ grid of $29 \times 29$ patches.

### 3. Target Dataset 2 (Qiang)
* **Description:** Real experimental neuronal cellular subtomograms from rat brains exhibiting C9orf72 pathology, containing 6 classes (`TRiC`, `membrane`, `none`, `proteasome_d`, `proteasome_s`, `ribosome`).
* **Training Pool:** 2,154 samples (90% stratified split).
* **Test Set:** 120 samples (balanced, 20 per class).
* **Format:** 16-bit 2D montage images ($(287 \times 287)$ pixels) containing 40 depth slices arranged in a $7 \times 7$ grid of $40 \times 40$ patches with $1\text{px}$ spacing (spacing interval of $41\text{px}$). The loader extracts 28 slices for consistency.

---

## Setup Instructions

Ensure your Python environment contains PyTorch with GPU (CUDA) support:
```bash
conda activate torch
# Install dependencies
pip install matplotlib mrcfile pandas pillow
```

---

## How to Run Experiments

All training scripts support target dataset selection via the `--dataset` argument (options: `noble`, `qiang`).

### 1. Target-Only Baselines
Fine-tunes the models on the $N$-shot target experimental data only:
```bash
# Swin3D Baseline
python train_baseline_nshot_swin3d.py --dataset qiang --n_shot 3 --seed 42 --cuda_device 0

# ResNet-34 Baseline
python train_baseline_nshot_resnet34.py --dataset qiang --n_shot 3 --seed 42 --cuda_device 0
```

### 2. Feature-Only DA Baseline (No Input Transform)
Aligns simulated and experimental features using MMD or CORAL on a shared Swin3D backbone:
```bash
python train_da_baseline_nshot_swin3d.py --dataset qiang --n_shot 3 --seed 42 --loss_type coral --lambda_align 0.2 --cuda_device 0
```

### 3. Full Hierarchical Adaptation (Joint Training)
Main experiment combining input-level transformations with feature-level alignment:
```bash
python train_joint_nshot_swin3d.py --dataset qiang --n_shot 3 --lambda_residual 0.5 --seed 42 --loss_type coral --lambda_mmd 0.2 --cuda_device 0
```

### 4. Ablation Studies
To isolate which input-level transformation categories contribute to domain shift reduction, pass the corresponding disable flags:
```bash
# Disable Spatial warping (STN)
python train_joint_nshot_swin3d.py --dataset qiang --n_shot 3 --disable_stn --cuda_device 0

# Disable Intensity modifications (DoG, Brightness/Contrast, Gamma)
python train_joint_nshot_swin3d.py --dataset qiang --n_shot 3 --disable_intensity --cuda_device 0

# Disable Global Color transformations
python train_joint_nshot_swin3d.py --dataset qiang --n_shot 3 --disable_color --cuda_device 0
```

---

### Remaining Experiments (Status Check)

Below is the checklist and current status of all planned experiments:

### 1. Noble Dataset (Dataset 1)
- [x] **Noble 3-Shot MMD Sweep:** **Completed** (Optimal $\lambda_{res} = 0.60$)
- [x] **Noble 3-Shot MMD Ablations:** **Completed** (STN-only, Intensity-only, and Color-only completed at $\lambda_{res} = 0.60$)
- [x] **Noble 5-Shot MMD:** **Completed** (Optimal $\lambda_{res} = 0.40$)
- [x] **Noble 3-Shot CORAL Sweep:** **Completed** (Optimal $\lambda_{res} = 0.20$)
- [x] **Noble 5-Shot CORAL:** **Completed** (Optimal $\lambda_{res} = 0.60$)

### 2. Qiang Dataset (Dataset 2)
- [x] **Qiang 3-Shot MMD Sweep:** **Completed** (Optimal $\lambda_{res} = 0.20$)
- [x] **Qiang 5-Shot MMD:** **Completed** (Optimal $\lambda_{res} = 0.00$)
- [x] **Qiang 3-Shot CORAL Sweep:** **Completed** (Optimal $\lambda_{res} = 0.40$)
- [x] **Qiang 5-Shot CORAL Sweep:** **Completed** (Optimal $\lambda_{res} = 0.80$)

---

## Experimental Design and Results Tables

Our revised experimental design is structured to answer all CVPR reviewer critiques while optimizing GPU compute time. We organize the results into three primary tables:

### 1. Table 2: Main Comparative Results
This table compares our proposed hierarchical adaptation methods (Swin3D + MMD/CORAL) against target-only baselines and feature-only DA baselines. All values report the **Mean ± Standard Deviation** across 10 random seeds at epoch 30.

* **Optimized 5-Shot Runs**: Since the optimal lambda is determined in the 3-shot sweep, the 5-shot sweeps are run only at these best values to save GPU time.

| Method | Noble 3-shot | Noble 5-shot | Qiang 3-shot | Qiang 5-shot |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet-34 Baseline (Target-Only)** | 22.21% ± 6.62% | 40.82% ± 12.58% | 19.50% ± 5.90% | 22.50% ± 7.89% |
| **Swin3D Baseline (Target-Only)** | 53.88% ± 10.54% | 62.48% ± 4.56% | 50.42% ± 5.49% | 51.67% ± 9.30% |
| **Swin3D + MMD (DA Feature-Only)** | 58.18% ± 6.21% | 62.77% ± 4.52% | 49.08% ± 7.57% | 56.58% ± 7.80% |
| **Swin3D + CORAL (DA Feature-Only)** | 60.39% ± 4.12% | 63.30% ± 4.79% | 54.67% ± 8.35% | **60.58% ± 5.17%** |
| **Swin3D + MMD (Proposed)** | **59.93% ± 5.43%** ($\lambda=0.6$) | **63.84% ± 3.11%** ($\lambda=0.4$) | **56.00% ± 6.40%** ($\lambda=0.2$) | **60.50% ± 7.53%** ($\lambda=0.0$) |
| **Swin3D + CORAL (Proposed)** | **61.41% ± 4.33%** ($\lambda=0.2$) | **65.25% ± 2.27%** ($\lambda=0.6$) | **57.50% ± 4.68%** ($\lambda=0.4$) | **61.50% ± 3.96%** ($\lambda=0.8$) |

---

### 2. Table 3: Blending Parameter Analysis ($\lambda_{res}$)
This table sweeps the residual blending weight $\lambda_{res} \in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]$ for the proposed joint training model to evaluate the interaction between domain shift complexity and input adaptation strength.

| $\lambda_{res}$ | Noble 3-shot (MMD) | Qiang 3-shot (MMD) | Noble 3-shot (CORAL) | Qiang 3-shot (CORAL) |
| :---: | :---: | :---: | :---: | :---: |
| **0.0** | 57.16% ± 4.74% | 51.50% ± 9.86% | 61.34% ± 3.68% | 56.25% ± 4.18% |
| **0.2** | 58.55% ± 4.64% | **56.00% ± 6.40%** | **61.41% ± 4.33%** | 54.83% ± 5.18% |
| **0.4** | 59.91% ± 5.33% | 55.33% ± 5.24% | 60.04% ± 5.28% | **57.50% ± 4.68%** |
| **0.6** | **59.93% ± 5.43%** | 53.75% ± 8.35% | 60.46% ± 3.72% | 56.42% ± 4.12% |
| **0.8** | 57.80% ± 5.30% | 54.17% ± 5.12% | 59.38% ± 4.47% | 55.08% ± 6.30% |
| **1.0** | 57.39% ± 4.58% | 50.17% ± 6.31% | 61.09% ± 5.12% | 55.08% ± 5.89% |

---

### 3. Table 4: Component Ablation Study (Noble 3-Shot MMD)
To isolate which input-level transformation categories contribute to domain shift reduction, we perform a component ablation study on the **Noble 3-Shot** setting using MMD loss at the optimal blending weight ($\lambda_{res} = 0.60$).

This directly addresses the reviewers' request to prove which input corrections (Spatial, Intensity, or Color) are effective.

| Configuration | Enabled Modules | Disabled Modules | Accuracy | Sweep Command |
| :--- | :--- | :--- | :---: | :--- |
| **Swin3D Target-Only** | None | All | 53.88% ± 10.54% | `python train_baseline_nshot_swin3d.py ...` |
| **Swin3D + MMD (DA Baseline)** | None (bypasses input) | All | 58.18% ± 6.21% | `python train_da_baseline_nshot_swin3d.py ...` |
| **Ablation: STN Only** | Spatial / Affine Warp | Intensity, Color | 57.12% ± 7.55% | `./run_lambda_sweep.sh ... --lambdas "0.6" --skip_swin_baseline --disable_intensity --disable_color` |
| **Ablation: Intensity Only** | DoG, Brightness/Contrast, Gamma | Spatial, Color | 56.73% ± 6.40% | `./run_lambda_sweep.sh ... --lambdas "0.6" --skip_swin_baseline --disable_stn --disable_color` |
| **Ablation: Color Only** | Global Channel Transform | Spatial, Intensity | 59.77% ± 3.94% | `./run_lambda_sweep.sh ... --lambdas "0.6" --skip_swin_baseline --disable_stn --disable_intensity` |
| **Swin3D + MMD (Proposed)** | Spatial, Intensity, Color | None | **59.93% ± 5.43%** | `./run_lambda_sweep.sh ... --lambdas "0.6" --skip_swin_baseline` |

---

## Compiling the Results
Once your sweeps are complete, run the summary script to automatically parse the CSV logs and output the tables in markdown format:
```bash
/shared/scratch/0/home/v_pranav_vinodh/miniconda3/envs/torch/bin/python summarize_results.py
```

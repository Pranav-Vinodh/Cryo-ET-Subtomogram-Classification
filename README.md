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

## Blending Parameter Analysis ($\lambda_{res}$)

The residual blending parameter $\lambda_{res} \in [0, 1]$ controls the strength of the learned input transformations ($x_{trans} = \lambda_{res} x_{orig} + (1 - \lambda_{res}) T(x_{orig})$):
* **$\lambda_{res} = 0.0$:** Applies full learnable input transformation.
* **$\lambda_{res} = 1.0$:** Bypasses input transformation entirely (acts as identity transform, relying solely on feature alignment).

As discussed in the paper:
* **Dataset 1 (Noble)** performs best at moderate-to-full input adaptation ($\lambda_{res} = 0.2$ or $0.0$) because the isolated macromolecules are clean and symmetric.
* **Dataset 2 (Qiang)** performs best at $\lambda_{res} = 1.0$ (bypassing input transformations). Because neuronal tomograms have complex backgrounds (membranes and crowded aggregates) and extreme class imbalance, the learnable parameters of the input transformation module are prone to overfitting when trained on very few target examples. Under these conditions, the identity transform acts as a robust regularizer.

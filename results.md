# Experimental Results

This file compiles the results for the few-shot Cryo-ET subtomogram classification experiments.

---

## 1. Noble 3-Shot MMD Sweep Results
* **Dataset:** Noble target dataset (7 classes, 3-shot training pool, 560 val samples)
* **Alignment Loss:** MMD (Maximum Mean Discrepancy)
* **Aggregation Method:** Epoch 30 performance across 10 random seeds

### Main Comparative Results

| Method | Accuracy | Best Blending Weight ($\lambda_{res}$) | Notes |
| :--- | :---: | :---: | :--- |
| **ResNet-34 Baseline (Target-Only)** | 61.18% ± 6.03% | N/A | Fine-tuned on real target data only |
| **Swin3D Baseline (Target-Only)** | 53.88% ± 10.54% | N/A | Fine-tuned on real target data only |
| **Swin3D + MMD (DA Feature-Only)** | 58.18% ± 6.21% | N/A | Feature alignment only (no input-level transforms) |
| **Swin3D + MMD (Proposed)** | **59.93% ± 5.43%** | **0.60** | **Hierarchical adaptation (Input + Feature alignment)** |

---

### Blending Parameter Sweep Analysis ($\lambda_{res}$)
Evaluates the proposed Swin3D + MMD joint training framework across different residual blending strengths $\lambda_{res} \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$. 
* **$\lambda_{res} = 0.0$**: Full learnable input transformation.
* **$\lambda_{res} = 1.0$**: Bypasses the input-level transformations completely (identity transform).

| Blending Weight ($\lambda_{res}$) | Accuracy (Noble 3-shot MMD) |
| :---: | :---: |
| **0.0** | 57.16% ± 4.74% |
| **0.2** | 58.55% ± 4.64% |
| **0.4** | 59.91% ± 5.33% |
| **0.6 (Best)** | **59.93% ± 5.43%** |
| **0.8** | 57.80% ± 5.30% |
| **1.0** | 57.39% ± 4.58% |

### Component Ablation Study (at optimal $\lambda_{res} = 0.60$)

This table isolates which input-level transformation categories contribute to domain shift reduction on the Noble 3-Shot MMD setting. All values report the **Mean ± Standard Deviation** across 10 random seeds at epoch 30.

| Configuration | Enabled Modules | Disabled Modules | Accuracy |
| :--- | :--- | :--- | :---: |
| **Swin3D Target-Only (Baseline)** | None | All | 53.88% ± 10.54% |
| **Swin3D + MMD (DA Baseline)** | None (bypasses input) | All | 58.18% ± 6.21% |
| **Ablation: STN Only** | Spatial / Affine Warp | Intensity, Color | 57.12% ± 7.55% |
| **Ablation: Intensity Only** | DoG, Brightness/Contrast, Gamma | Spatial, Color | 56.73% ± 6.40% |
| **Ablation: Color Only** | Global Channel Transform | Spatial, Intensity | 59.77% ± 3.94% |
| **Swin3D + MMD (Proposed)** | Spatial, Intensity, Color | None | **59.93% ± 5.43%** |

---

## 2. Other Sweeps (To Be Populated)
* Noble 5-Shot MMD
* Noble 3-Shot CORAL
* Noble 5-Shot CORAL
* Qiang 3-Shot MMD (In-Progress)
* Qiang 5-Shot MMD
* Qiang 3-Shot CORAL
* Qiang 5-Shot CORAL

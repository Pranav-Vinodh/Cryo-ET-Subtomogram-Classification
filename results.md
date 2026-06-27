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
| **ResNet-34 Baseline (Target-Only)** | 22.21% ± 6.62% | N/A | Fine-tuned on real target data only |
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

## 2. Qiang 3-Shot MMD Sweep Results
* **Dataset:** Qiang target dataset (6 classes, 3-shot training pool, 120 val samples)
* **Alignment Loss:** MMD (Maximum Mean Discrepancy)
* **Aggregation Method:** Epoch 30 performance across 10 random seeds

### Main Comparative Results

| Method | Accuracy | Best Blending Weight ($\lambda_{res}$) | Notes |
| :--- | :---: | :---: | :--- |
| **ResNet-34 Baseline (Target-Only)** | 19.50% ± 5.90% | N/A | Fine-tuned on real target data only |
| **Swin3D Baseline (Target-Only)** | 50.42% ± 5.49% | N/A | Fine-tuned on real target data only |
| **Swin3D + MMD (DA Feature-Only)** | 49.08% ± 7.57% | N/A | Feature alignment only (no input-level transforms) |
| **Swin3D + MMD (Proposed)** | **56.00% ± 6.40%** | **0.20** | **Hierarchical adaptation (Input + Feature alignment)** |

---

### Blending Parameter Sweep Analysis ($\lambda_{res}$)

| Blending Weight ($\lambda_{res}$) | Accuracy (Qiang 3-shot MMD) |
| :---: | :---: |
| **0.0** | 51.50% ± 9.86% |
| **0.2 (Best)** | **56.00% ± 6.40%** |
| **0.4** | 55.33% ± 5.24% |
| **0.6** | 53.75% ± 8.35% |
| **0.8** | 54.16% ± 5.12% |
| **1.0** | 50.17% ± 6.31% |

---

## 3. Noble 5-Shot MMD Sweep Results
* **Dataset:** Noble target dataset (7 classes, 5-shot training pool, 560 val samples)
* **Alignment Loss:** MMD (Maximum Mean Discrepancy)
* **Aggregation Method:** Epoch 30 performance across 10 random seeds

### Main Comparative Results

| Method | Accuracy | Best Blending Weight ($\lambda_{res}$) | Notes |
| :--- | :---: | :---: | :--- |
| **ResNet-34 Baseline (Target-Only)** | 40.82% ± 12.58% | N/A | Fine-tuned on real target data only |
| **Swin3D Baseline (Target-Only)** | 62.48% ± 4.56% | N/A | Fine-tuned on real target data only |
| **Swin3D + MMD (DA Feature-Only)** | 62.77% ± 4.52% | N/A | Feature alignment only (no input-level transforms) |
| **Swin3D + MMD (Proposed)** | **63.84% ± 3.11%** | **0.40** | **Hierarchical adaptation (Input + Feature alignment)** |

---

### Blending Parameter Sweep Analysis ($\lambda_{res}$)

| Blending Weight ($\lambda_{res}$) | Accuracy (Noble 5-shot MMD) |
| :---: | :---: |
| **0.0** | 62.86% ± 3.96% |
| **0.2** | 62.91% ± 2.86% |
| **0.4 (Best)** | **63.84% ± 3.11%** |
| **0.6** | 63.50% ± 4.35% |
| **0.8** | 62.96% ± 2.22% |
| **1.0** | 60.71% ± 5.28% |

---

## 4. Qiang 5-Shot MMD Sweep Results
* **Dataset:** Qiang target dataset (6 classes, 5-shot training pool, 120 val samples)
* **Alignment Loss:** MMD (Maximum Mean Discrepancy)
* **Aggregation Method:** Epoch 30 performance across 10 random seeds

### Main Comparative Results

| Method | Accuracy | Best Blending Weight ($\lambda_{res}$) | Notes |
| :--- | :---: | :---: | :--- |
| **ResNet-34 Baseline (Target-Only)** | 22.50% ± 7.89% | N/A | Fine-tuned on real target data only |
| **Swin3D Baseline (Target-Only)** | 51.67% ± 9.30% | N/A | Fine-tuned on real target data only |
| **Swin3D + MMD (DA Feature-Only)** | 56.58% ± 7.80% | N/A | Feature alignment only (no input-level transforms) |
| **Swin3D + MMD (Proposed)** | **60.50% ± 7.53%** | **0.00** | **Hierarchical adaptation (Input + Feature alignment)** |

---

### Blending Parameter Sweep Analysis ($\lambda_{res}$)

| Blending Weight ($\lambda_{res}$) | Accuracy (Qiang 5-shot MMD) |
| :---: | :---: |
| **0.0 (Best)** | **60.50% ± 7.53%** |
| **0.2** | 58.83% ± 6.31% |
| **0.4** | 56.08% ± 9.83% |
| **0.6** | 58.58% ± 6.40% |
| **0.8** | 53.50% ± 7.76% |
| **1.0** | 56.50% ± 6.46% |

---

## 5. Noble 3-Shot CORAL Sweep Results
* **Dataset:** Noble target dataset (7 classes, 3-shot training pool, 560 val samples)
* **Alignment Loss:** CORAL (Correlation Alignment)
* **Aggregation Method:** Epoch 30 performance across 10 random seeds

### Main Comparative Results

| Method | Accuracy | Best Blending Weight ($\lambda_{res}$) | Notes |
| :--- | :---: | :---: | :--- |
| **ResNet-34 Baseline (Target-Only)** | 22.21% ± 6.62% | N/A | Fine-tuned on real target data only |
| **Swin3D Baseline (Target-Only)** | 53.88% ± 10.54% | N/A | Fine-tuned on real target data only |
| **Swin3D + CORAL (DA Feature-Only)** | 60.39% ± 4.12% | N/A | Feature alignment only (no input-level transforms) |
| **Swin3D + CORAL (Proposed)** | **61.41% ± 4.33%** | **0.20** | **Hierarchical adaptation (Input + Feature alignment)** |

---

### Blending Parameter Sweep Analysis ($\lambda_{res}$)

| Blending Weight ($\lambda_{res}$) | Accuracy (Noble 3-shot CORAL) |
| :---: | :---: |
| **0.0** | 61.34% ± 3.68% |
| **0.2 (Best)** | **61.41% ± 4.33%** |
| **0.4** | 60.04% ± 5.28% |
| **0.6** | 60.46% ± 3.72% |
| **0.8** | 59.38% ± 4.47% |
| **1.0** | 61.09% ± 5.12% |

---

## 6. Noble 5-Shot CORAL Sweep Results
* **Dataset:** Noble target dataset (7 classes, 5-shot training pool, 560 val samples)
* **Alignment Loss:** CORAL (Correlation Alignment)
* **Aggregation Method:** Epoch 30 performance across 10 random seeds

### Main Comparative Results

| Method | Accuracy | Best Blending Weight ($\lambda_{res}$) | Notes |
| :--- | :---: | :---: | :--- |
| **ResNet-34 Baseline (Target-Only)** | 40.82% ± 12.58% | N/A | Fine-tuned on real target data only |
| **Swin3D Baseline (Target-Only)** | 62.48% ± 4.56% | N/A | Fine-tuned on real target data only |
| **Swin3D + CORAL (DA Feature-Only)** | 63.30% ± 4.79% | N/A | Feature alignment only (no input-level transforms) |
| **Swin3D + CORAL (Proposed)** | **65.25% ± 2.27%** | **0.60** | **Hierarchical adaptation (Input + Feature alignment)** |

---

### Blending Parameter Sweep Analysis ($\lambda_{res}$)

| Blending Weight ($\lambda_{res}$) | Accuracy (Noble 5-shot CORAL) |
| :---: | :---: |
| **0.0** | 64.59% ± 2.73% |
| **0.2** | 65.04% ± 3.30% |
| **0.4** | 62.43% ± 10.01% |
| **0.6 (Best)** | **65.25% ± 2.27%** |
| **0.8** | 64.00% ± 4.42% |
| **1.0** | 64.39% ± 3.12% |

---

## 7. Qiang 3-Shot CORAL Sweep Results
* **Dataset:** Qiang target dataset (6 classes, 3-shot training pool, 120 val samples)
* **Alignment Loss:** CORAL (Correlation Alignment)
* **Aggregation Method:** Epoch 30 performance across 10 random seeds

### Main Comparative Results

| Method | Accuracy | Best Blending Weight ($\lambda_{res}$) | Notes |
| :--- | :---: | :---: | :--- |
| **ResNet-34 Baseline (Target-Only)** | 19.50% ± 5.90% | N/A | Fine-tuned on real target data only |
| **Swin3D Baseline (Target-Only)** | 50.42% ± 5.49% | N/A | Fine-tuned on real target data only |
| **Swin3D + CORAL (DA Feature-Only)** | 54.67% ± 8.35% | N/A | Feature alignment only (no input-level transforms) |
| **Swin3D + CORAL (Proposed)** | **57.50% ± 4.68%** | **0.40** | **Hierarchical adaptation (Input + Feature alignment)** |

---

### Blending Parameter Sweep Analysis ($\lambda_{res}$)

| Blending Weight ($\lambda_{res}$) | Accuracy (Qiang 3-shot CORAL) |
| :---: | :---: |
| **0.0** | 56.25% ± 4.18% |
| **0.2** | 54.83% ± 5.18% |
| **0.4 (Best)** | **57.50% ± 4.68%** |
| **0.6** | 56.42% ± 4.12% |
| **0.8** | 55.08% ± 6.30% |
| **1.0** | 55.08% ± 5.89% |

---

## 8. Qiang 5-Shot CORAL Sweep Results
* **Dataset:** Qiang target dataset (6 classes, 5-shot training pool, 120 val samples)
* **Alignment Loss:** CORAL (Correlation Alignment)
* **Aggregation Method:** Epoch 30 performance across 10 random seeds

### Main Comparative Results

| Method | Accuracy | Best Blending Weight ($\lambda_{res}$) | Notes |
| :--- | :---: | :---: | :--- |
| **ResNet-34 Baseline (Target-Only)** | 22.50% ± 7.89% | N/A | Fine-tuned on real target data only |
| **Swin3D Baseline (Target-Only)** | 51.67% ± 9.30% | N/A | Fine-tuned on real target data only |
| **Swin3D + CORAL (DA Feature-Only)** | 60.58% ± 5.17% | N/A | Feature alignment only (no input-level transforms) |
| **Swin3D + CORAL (Proposed)** | **61.50% ± 3.96%** | **0.80** | **Hierarchical adaptation (Input + Feature alignment)** |

---

### Blending Parameter Sweep Analysis ($\lambda_{res}$)

| Blending Weight ($\lambda_{res}$) | Accuracy (Qiang 5-shot CORAL) |
| :---: | :---: |
| **0.0** | 60.92% ± 6.88% |
| **0.2** | 60.50% ± 6.05% |
| **0.4** | 58.83% ± 6.40% |
| **0.6** | 60.42% ± 5.61% |
| **0.8 (Best)** | **61.50% ± 3.96%** |
| **1.0** | 59.50% ± 4.91% |

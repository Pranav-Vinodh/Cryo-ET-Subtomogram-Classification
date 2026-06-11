# Draft Paper Revisions: Addressing CVPR Review Comments

This document contains LaTeX-ready draft sections to be integrated into the research paper: **"Bridging the Synthetic-to-Real Gap with Hierarchical Adaptation for Few-Shot Cryo-ET Subtomogram Classification"**. These updates directly address the comments from Reviewer zujd and Reviewer kdhq.

---

## Revision 1: Softening the Novelty Claim & Strengthening Related Work
*Target Section: Section 1 (Introduction) & Section 2 (Related Work)*

### Introduction (Softening the "First" Claim)
**Old text (or concept):**
> "To the best of our knowledge, we propose the first synthetic-to-real adaptation framework for cryo-ET subtomogram classification..."

**Revised Text:**
```latex
% --- INTRODUCTION UPDATE ---
We propose a novel hierarchical synthetic-to-real adaptation framework for cryo-ET subtomogram classification in the low-data (few-shot) regime. While transfer learning and unsupervised feature alignment have been explored in cryo-ET, our framework is the first to combine a learnable, differentiable input-level transformation module with feature-level domain alignment to systematically bridge the simulation-to-experiment gap. 
```

### Related Work (Positioning relative to Yu et al. and Cryo-Shift)
**Revised Text:**
```latex
% --- RELATED WORK UPDATE ---
\subsection{Domain Adaptation in Cryo-ET}
Domain discrepancy remains a core challenge in cryo-ET, especially when leveraging simulated data. Prior works have attempted to mitigate this discrepancy at different levels. Yu et al.~\cite{yu2021domain} proposed a few-shot domain adaptation method for cross-tomogram and cross-dataset subtomogram classification, relying primarily on adversarial training to align feature representations. Similarly, Cryo-Shift~\cite{cryoshift2022} introduced an unsupervised domain adaptation framework that aligns simulated and experimental subtomograms in a low-dimensional space via domain-specific normalization and data randomization. 

In contrast to these approaches, our method operates in a supervised few-shot setting and introduces a hierarchical adaptation pipeline. Rather than relying solely on feature-level alignment, we implement a learnable input-level transformation module that directly models and corrects physical discrepancies (such as geometric misalignments, noise characteristics, and contrast variations) before the features are aligned. This dual-level alignment prevents the backbone from overfitting to simulation-specific artifacts.
```

---

## Revision 2: Detailed Dataset Specifications
*Target Section: Section 4.1 (Datasets)*

**Revised Text:**
```latex
% --- DATASETS SECTION UPDATE ---
\subsection{Target (Experimental) Datasets}
To evaluate our few-shot classification performance, we utilize two distinct experimental target domains:

\textbf{Dataset 1 (Noble):} 
Introduced by Noble et al.~\cite{noble2018elife}, this dataset consists of experimental cryo-ET subtomograms grouped into 7 macromolecular classes (e.g., ribosomes, proteasomes, and storage proteins). Each volume consists of 28 depth slices. In our experimental setup, we crop these from the original 2D montage images (composed of a $5 \times 6$ patch grid of $29 \times 29$ pixels) and resize each slice to $128 \times 128$ pixels for compatibility with Kinetics-pretrained backbones. The target training pool consists of 84 labeled samples (12 per class), from which we sample $n \in \{3, 5\}$ shots for training. Testing is conducted on a held-out set of 560 samples.

\textbf{Dataset 2 (Qiang):}
Obtained from neuronal cellular tomograms of rat brains exhibiting C9orf72 poly-GA pathology, as compiled by Guo et al.~\cite{guo2018cell}. This dataset comprises 6 highly imbalanced macromolecular classes: membrane, ribosome, TRiC, double-cap proteasome, single-cap proteasome, and an empty/unidentified ``none'' class. The dataset is extremely class-imbalanced, with sample counts ranging from 80 (ribosome) to 1,043 (double-cap proteasome), totaling 2,394 volumes. Each volume contains 28 depth slices reconstructed from 2D montages. The training pool contains 2,154 samples (90\% stratified split), from which few-shot subsets are extracted, and evaluation is carried out on a balanced test set of 120 samples.
```

---

## Revision 3: Explanation of Dataset 2 $\lambda_{res} = 1.0$ (Ablation Analysis)
*Target Section: Section 4.4 (Effect of Residual Connection Strength)*

**Revised Text:**
```latex
% --- ABLATION & LAMBDA DISCUSSION UPDATE ---
\subsection{Interpretation of Dataset-Specific Residual Behavior}
As shown in Table~3, the optimal residual blending parameter $\lambda_{res}$ exhibits contrasting behaviors between the two datasets. While Dataset 1 (Noble) achieves its highest accuracy with moderate ($\lambda_{res} = 0.2$) or full ($\lambda_{res} = 0.0$) input-level transformations, Dataset 2 (Qiang) performs best at $\lambda_{res} = 1.0$, which bypasses the learnable input transformations entirely and relies solely on feature-level alignment (CORAL/MMD).

This difference highlights how domain shift severity and dataset properties interact with learnable input adaptation. Dataset 2 is characterized by extreme class imbalance and severe structural heterogeneity (e.g., complex neuronal aggregates and membrane boundaries). Under these conditions, the learnable parameters of the transformation module (such as the 3D affine warp and intensity corrections) are prone to overfitting and gradient instability when trained on only 3 or 5 target examples. Consequently, the identity transform ($\lambda_{res} = 1.0$) acts as a robust regularizer, preserving the structure of the simulated data while domain alignment is handled entirely in the feature space. Conversely, Dataset 1 contains cleaner, more symmetric isolated macromolecules, allowing the model to successfully learn joint input-level warping and frequency filtering without overfitting. This finding demonstrates that the hierarchical components should be tuned dynamically based on the complexity and noise level of the target domain.
```

---

## Revision 4: Guide for Tomogram Visualizations
*Target Section: Appendix / Figure 1 (Visualizing Cryo-ET Data)*

**Recommended Figure Design:**
```latex
% --- SUGGESTED LATEX FIGURE CODE FOR TOMOGRAM SAMPLES ---
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/cryoet_data_samples.pdf}
  \caption{Visual comparison of simulated (source) and experimental (target) subtomograms. (a) Simulated ribosome subtomograms (SNR = 0.05) exhibit uniform background noise and clear structural features. (b) Noble dataset (Target 1) subtomograms show low SNR and contrast variations. (c) Qiang dataset (Target 2) neuronal subtomograms exhibit severe noise, missing wedge distortions, and crowded backgrounds containing membranes and aggregates.}
  \label{fig:cryoet_samples}
\end{figure}
```
*Note: We recommend generating a mock visualization using the Python script `visuals_generation.py` to plot horizontal slices from the datasets to create `figures/cryoet_data_samples.pdf` for the final paper submission.*

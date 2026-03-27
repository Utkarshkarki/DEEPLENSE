# DeepLense GSoC 2026 🌌

This repository contains my solutions for the **DeepLense Google Summer of Code (GSoC) 2026** evaluation tests. The project focuses on the application of deep learning techniques to identify and analyze strong gravitational lensing events.

## 🎯 Evaluated Tasks

The following tests from the GSoC application guidelines have been completed and fully documented within this repository:

1. **Test I: Multi-Class Classification (Common Test)**
2. **Test VII: Physics-Guided Machine Learning (Domain-Specific)**
3. **Test VIII: Diffusion Models for Lensing Generation (Domain-Specific)**

---

## 📂 Repository Structure

All code, models, and evaluation notebooks are organized within the `src/` directory:

```text
DEEPLENSE/
├── requirements.txt
└── src/
    ├── dataset/                             # Data loaders and preprocessing
    ├── checkpoints/                         # Saved model weights (.pth)
    ├── results/                             # Saved plots, ROC curves, and generated grids
    │
    ├── multi_class_classification.ipynb     # Solution for Test I
    │
    ├── pinn_model/                          # Source code for Test VII (Physics-Guided ML)
    ├── pinn_classification.ipynb            # Evaluation notebook for Test VII
    │
    ├── diffusion_model/                     # Source code for Test VIII (Diffusion Models)
    └── diffusion_model.ipynb                # Evaluation notebook for Test VIII
```

---

## 🧪 Detailed Implementations

### 1. Test I: Multi-Class Classification
**Notebook:** `src/multi_class_classification.ipynb`
* **Objective:** Classify lensing images into three categories (no substructure, spherical, vortex).
* **Approach:** Implemented a robust **ResNet-18** baseline trained on 150x150 single-channel images.
* **Outputs:** Includes ROC curves, AUC scores, and confusion matrices to establish a quantitative baseline for subsequent experiments.

### 2. Test VII: Physics-Guided ML
**Notebook:** `src/pinn_classification.ipynb`
* **Objective:** Improve sample efficiency and interpretability using physics-informed neural networks.
* **Approach:** Developed the **Lensiformer**, a novel architecture integrating standard self-attention (Shifted Patch Tokenization) with an `InverseLensLayer`. This layer explicitly encodes the lens equation (computing the deflection angle $\alpha$ and surface mass density $\kappa$).
* **Strategy:** Used Curriculum Learning to stabilize the physics-informed loss term during early epochs.
* **Outputs:** Demonstrates improved AUC performance over the strong ResNet-18 baseline, particularly on edge cases.

### 3. Test VIII: Diffusion Models
**Notebook:** `src/diffusion_model.ipynb`
* **Objective:** Generate realistic, high-fidelity mock gravitational lensing images unconditionally.
* **Approach:** Designed a **Denoising Diffusion Probabilistic Model (DDPM)** using a U-Net backbone with attention at the 16x16 resolution and a cosine noise schedule (to better preserve structural macro-features).
* **Evaluation:** Implemented domain-standard metrics including:
  * **Fréchet Inception Distance (FID)**
  * **1D Power Spectrum Analysis** (comparing the spatial frequencies of generated vs. real lenses)
  * **Pixel Intensity Distribution**
* **Outputs:** Includes denoising trajectories, epoch-wise generation grids, and quantitative comparisons. The DDPM weights utilize an Exponential Moving Average (EMA) for superior sample quality.

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/[Your-Username]/[Your-Repo-Name].git
   cd [Your-Repo-Name]
   ```

2. **Install Dependencies:**
   It is recommended to use a virtual environment (`venv` or `conda`).
   ```bash
   pip install -r requirements.txt
   ```

3. **View the Results:**
   All notebooks have been pre-executed. You can open them directly on GitHub or locally via Jupyter to view the metrics, plots, and strategic analysis:
   ```bash
   jupyter notebook src/
   ```

---

## ✒️ Author
[Your Name / GitHub Username]  
*Prospective GSoC 2026 Contributor for DeepLense*
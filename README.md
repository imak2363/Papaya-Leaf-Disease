# Papaya-Leaf-Disease
# 🌿 SwinGhost-ClustNet: An Explainable Deep Ensemble Model for Papaya Leaf Disease Detection

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-99.25%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ROC--AUC-1.000-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/F1--Score-99.25%25-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Published-ScienceDirect-red?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S2772375526000481">
    📄 Read the Paper (ScienceDirect)
  </a>
</p>

> **Official implementation** of the paper:  
> **"SwinGhost-ClustNet: An Explainable Deep Ensemble Model for Papaya Leaf Disease Detection and Field Deployment in Bangladeshi Agriculture"**  
> *Published in Smart Agricultural Technology — Elsevier (2026)*

---

## 📌 Overview

Early detection of papaya leaf disease is critical for smallholder farmers in Bangladesh, who face **39.9% post-harvest losses** and widespread pesticide misuse. This work proposes **SwinGhost-ClustNet**, a hybrid deep ensemble that combines:

- 🔭 **Swin Transformer** — for capturing global context and long-range dependencies
- 👻 **GhostNet** — for efficient local texture feature extraction
- 🔗 **Cross-Attention Fusion** — for merging multi-scale feature representations
- 🔵 **K-Means Clustering (k=8)** — for pseudo-label generation to improve generalization with limited labeled data

The system also integrates a **Flask-based deployment API** providing real-time diagnostics and pesticide recommendations in Bengali, tailored for field use by smallholder farmers.

---

## 🏆 Key Results

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | **99.25%** |
| Precision     | **99.28%** |
| Recall        | **99.25%** |
| F1-Score      | **99.25%** |
| ROC-AUC       | **1.000**  |

- Tested on **1,401 images** (held-out test set)
- **+2.10% improvement** over base models (ablation-validated)
- K-Means silhouette score: **0.67** (k=8)

---

## 🧠 Explainability (XAI)

To build farmer trust and ensure clinically meaningful predictions, three Grad-CAM-based visualization methods were applied:

| XAI Method   | IoU Score |
|--------------|-----------|
| Grad-CAM++   | **0.72**  |
| Layer-CAM    | **0.68**  |
| Grad-CAM     | **0.65**  |

These methods verify that model decisions are anchored to **disease-affected leaf areas** (lesions, webbing patterns), not spurious image artifacts.

---

## 📂 Dataset

The model was trained and evaluated on a curated dataset of **9,342 papaya leaf images** collected from **8 districts** of Bangladesh.

| Property        | Details                           |
|-----------------|-----------------------------------|
| Total Images    | 9,342                             |
| Classes         | 8                                 |
| Districts       | 8 (across Bangladesh)             |
| Collection      | Field-collected, real-world conditions |

**Disease Classes:**
1. 🍂 Anthracnose
2. 🦠 Bacterial Spot
3. ✅ Healthy
4. 🌀 Leaf Curl
5. 🐛 Mealybug
6. 🕷️ Mite
7. 🌿 Mosaic
8. 🔴 Ring Spot

> 📦 **Dataset Reference:** BDPapayaLeaf — *A dataset of papaya leaf for disease detection, classification, and analysis*
> Dataset available in the following link: https://www.kaggle.com/datasets/shourav123/enlarged-data
---

## 🏗️ Model Architecture

```
Input Image (224×224×3)
        │
   ┌────┴────┐
   │         │
Swin      GhostNet
Transformer  (Local
(Global    Textures)
Context)
   │         │
   └────┬────┘
        │
  Cross-Attention
      Fusion
        │
  K-Means Pseudo
   Supervision
   (k=8, clusters)
        │
  Classification
  Head (8 classes)
        │
     Output +
  Grad-CAM / XAI
```

---

## 🚀 Flask Deployment API

A lightweight **Flask-based web API** was built for field deployment, providing:

- ✅ Real-time disease diagnosis from leaf images
- 📊 Confidence scores with a **>70% threshold criterion**
- 💊 Pesticide recommendations in **Bengali language**
- 🌐 Accessible without requiring domain expertise

---

## 🛠️ Requirements

```bash
Python >= 3.8
torch >= 1.12
torchvision
timm                  # Swin Transformer
numpy
pandas
scikit-learn          # K-Means clustering
matplotlib
opencv-python
Flask
grad-cam              # Grad-CAM, Grad-CAM++, Layer-CAM
Pillow
tqdm
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
SwinGhost-ClustNet/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── swin_transformer.py       # Swin Transformer backbone
│   ├── ghostnet.py               # GhostNet backbone
│   ├── cross_attention.py        # Cross-attention fusion module
│   └── swinghost_clustnet.py     # Full ensemble model
│
├── clustering/
│   └── kmeans_pseudo_label.py    # K-Means pseudo-labeling (k=8)
│
├── xai/
│   ├── gradcam.py                # Grad-CAM visualization
│   ├── gradcampp.py              # Grad-CAM++ visualization
│   └── layercam.py               # Layer-CAM visualization
│
├── deployment/
│   ├── app.py                    # Flask API
│   ├── templates/                # HTML templates
│   └── static/                   # CSS/JS assets
│
├── notebooks/
│   ├── training.ipynb
│   ├── evaluation.ipynb
│   └── xai_visualization.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── gradcam_samples/
│
├── requirements.txt
├── train.py
├── evaluate.py
└── README.md
```

---

## ⚙️ Usage

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/SwinGhost-ClustNet.git
cd SwinGhost-ClustNet
```

### 2. Prepare the Dataset

Place the dataset in the `data/` folder following the structure:

```
data/
├── train/
│   ├── anthracnose/
│   ├── bacterial_spot/
│   ├── healthy/
│   └── ...
├── val/
└── test/
```

### 3. Train the Model

```bash
python train.py --epochs 50 --batch_size 32 --lr 0.0001
```

### 4. Evaluate the Model

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### 5. Generate XAI Visualizations

```bash
python xai/gradcam.py --image path/to/leaf_image.jpg --checkpoint checkpoints/best_model.pth
```

### 6. Run the Flask API

```bash
cd deployment
python app.py
```

Then open `http://localhost:5000` in your browser to use the disease detection interface.

---

## 📊 Ablation Study Summary

| Configuration                       | Accuracy |
|-------------------------------------|----------|
| Swin Transformer only               | ~97.15%  |
| GhostNet only                       | ~96.80%  |
| Swin + GhostNet (no fusion)         | ~97.90%  |
| + Cross-Attention Fusion            | ~98.60%  |
| + K-Means Pseudo-Labels             | **99.25%** |

---

## 📖 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{ejaz2026swinghost,
  title     = {SwinGhost-ClustNet: An Explainable Deep Ensemble Model for Papaya Leaf Disease Detection and Field Deployment in Bangladeshi Agriculture},
  author    = {Md. Sabbir Ejaz and others},
  journal   = {Smart Agricultural Technology},
  publisher = {Elsevier},
  year      = {2026},
  doi       = {10.1016/j.atech.2026.100481},
  url       = {https://www.sciencedirect.com/science/article/pii/S2772375526000481}
}
```

> ⚠️ Please verify the exact author list, volume, and DOI from the published paper before finalizing the citation.

---

## 🔮 Future Work

As outlined in the paper, planned future directions include:

- [ ] Severity scoring and dosage-based pesticide recommendations
- [ ] Multi-label classification for co-occurring diseases
- [ ] Model pruning to reduce size below 100 MB for edge deployment
- [ ] Cloud deployment on AWS / GCP
- [ ] Multi-crop validation beyond papaya

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Farmers and agricultural workers across 8 districts of Bangladesh who contributed to dataset collection
- The open-source communities behind [timm](https://github.com/huggingface/pytorch-image-models), [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam), and Flask

---

<p align="center">Made with ❤️ for Bangladeshi agriculture</p>

# 🚀 CIFAR-10 Image Classification — ANN vs CNN

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

> A complete deep learning project comparing **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** on the CIFAR-10 dataset — built for learning, interviewing, and portfolio demonstration.

---

## 📌 Project Overview

| Property | Detail |
|---|---|
| **Dataset** | CIFAR-10 — 60,000 color images, 10 classes, 32×32×3 pixels |
| **Models** | ANN · CNN · CNN + Data Augmentation |
| **Framework** | TensorFlow / Keras 2.20 |
| **Hardware** | Google Colab T4 GPU |
| **Best Accuracy** | ~80% (CNN + Augmentation) |

---

## 📁 Repository Structure

```
cifar10-ann-cnn/
│
├── CIFAR10_ANN_CNN_Full.ipynb     ← Main notebook (run this in Colab)
├── README.md                      ← You are here
├── requirements.txt               ← Python dependencies
│
├── outputs/                       ← Auto-generated when notebook runs
│   ├── cifar10_samples.png        ← Sample images grid
│   ├── learning_curves.png        ← Train/val accuracy & loss curves
│   ├── confusion_ANN.png          ← ANN confusion matrix
│   ├── confusion_CNN.png          ← CNN confusion matrix
│   ├── confusion_CNN_Aug.png      ← Augmented CNN confusion matrix
│   ├── per_class_accuracy.png     ← Per-class accuracy bar chart
│   ├── misclassified.png          ← 32 misclassified examples
│   ├── feature_maps.png           ← CNN intermediate activations
│   └── comparison_dashboard.png  ← Final model comparison
│
└── saved_models/                  ← Auto-saved during training
    ├── best_cnn.keras
    └── best_aug_cnn.keras
```

---

## 🗂️ Dataset — CIFAR-10

**Source:** [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)  
Loaded automatically via `tf.keras.datasets.cifar10.load_data()`

| Property | Value |
|---|---|
| Total images | 60,000 |
| Training set | 50,000 images |
| Test set | 10,000 images |
| Image size | 32 × 32 × 3 (RGB) |
| Classes | 10 (balanced — 6,000 per class) |
| Input to ANN | 3,072 (flattened) |
| Input to CNN | (32, 32, 3) |

### 10 Classes

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | ✈ Airplane | 5 | 🐶 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐴 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚛 Truck |

### Preprocessing
```python
# Normalize pixels from [0, 255] → [0.0, 1.0]
x_train_norm = x_train.astype('float32') / 255.0

# ANN: flatten spatial dims → 1D vector
x_train_flat = x_train_norm.reshape(50000, 3072)

# CNN: keep full (32, 32, 3) tensor
```

---

## 🧠 Models

### Model 1 — ANN (Artificial Neural Network)

```
Input(3072) → Dense(512)+BN+ReLU+Dropout(0.3)
            → Dense(256)+BN+ReLU+Dropout(0.3)
            → Dense(128)+ReLU
            → Dense(10)+Softmax
```

| Property | Value |
|---|---|
| Parameters | ~1.7 Million |
| Test Accuracy | ~52% |
| Limitation | Flattening destroys spatial structure |
| Regularization | Dropout(0.3), BatchNorm, L2(1e-4) |

---

### Model 2 — CNN (Convolutional Neural Network)

```
Input(32,32,3)
 → Conv2D(32,3×3)+BN+ReLU → Conv2D(32,3×3)+BN+ReLU → MaxPool → Dropout(0.25)
 → Conv2D(64,3×3)+BN+ReLU → Conv2D(64,3×3)+BN+ReLU → MaxPool → Dropout(0.25)
 → Conv2D(128,3×3)+BN+ReLU → MaxPool → Dropout(0.25)
 → Flatten → Dense(256)+BN+ReLU+Dropout(0.5)
 → Dense(10)+Softmax
```

| Property | Value |
|---|---|
| Parameters | ~320 K |
| Test Accuracy | ~75% |
| Advantage | Weight sharing, spatial awareness |
| Regularization | Dropout(0.25/0.5), BatchNorm, L2(1e-4) |

---

### Model 3 — CNN + Data Augmentation

Same as CNN but with an augmentation pipeline as the first layer:

```python
layers.RandomFlip("horizontal")   # Mirror left-right
layers.RandomRotation(0.1)        # ±10° rotation
layers.RandomZoom(0.1)            # ±10% zoom
layers.RandomContrast(0.1)        # ±10% brightness contrast
```

| Property | Value |
|---|---|
| Parameters | ~850 K |
| Test Accuracy | ~80% |
| Advantage | Invariant to flips, rotations, scale |

---

## 📊 Results Summary

| Model | Test Accuracy | Test Loss | Parameters | Spatial Aware | Augmentation |
|---|---|---|---|---|---|
| ANN | ~52% | ~1.40 | 1.7 M | ❌ | ❌ |
| CNN | ~75% | ~0.72 | 320 K | ✅ | ❌ |
| CNN + Aug | ~80% | ~0.60 | 850 K | ✅ | ✅ |

> CNN achieves **+23% accuracy** over ANN with **5× fewer parameters**.  
> Adding augmentation pushes another **+5%** with zero extra data.

---

## ⚙️ Training Configuration

| Setting | ANN | CNN | CNN + Aug |
|---|---|---|---|
| Optimizer | Adam(1e-3) | Adam(1e-3) | Adam(1e-3) |
| Loss | Sparse Categorical CE | Sparse Categorical CE | Sparse Categorical CE |
| Batch Size | 128 | 64 | 64 |
| Max Epochs | 30 | 50 | 60 |
| EarlyStopping patience | 5 | 8 | 10 |
| ReduceLR patience | 3 | 4 | 5 |
| ReduceLR factor | 0.5 | 0.5 | 0.5 |
| Validation split | 10% | 10% | 10% |

---

## 📓 Notebook Sections

| # | Section | Description |
|---|---|---|
| 1 | Setup & Imports | TF, NumPy, sklearn, seaborn, reproducibility seeds |
| 2 | Load & Explore | Dataset shapes, class distribution, sample image grid |
| 3 | Preprocessing | Normalize, flatten for ANN, rationale explained |
| 4 | ANN Model | Build, compile, train with callbacks, evaluate |
| 5 | CNN Model | 3 conv blocks, train, evaluate |
| 6 | CNN + Augmentation | RandomFlip/Rotation/Zoom/Contrast pipeline |
| 7 | Learning Curves | All 3 models — accuracy & loss on one chart |
| 8 | Full Evaluation | Classification report + confusion matrix heatmap |
| 9 | Misclassifications | 32 wrong predictions visualized |
| 10 | Feature Maps | CNN conv layer activations for a test image |
| 11 | Dashboard | Accuracy bars, loss bars, gain vs baseline, val curves |
| 12 | Conclusions | Key takeaways, concept summary, next steps |

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `CIFAR10_ANN_CNN_Full.ipynb`
3. Set runtime: `Runtime → Change runtime type → T4 GPU`
4. Click `Runtime → Run All`

### Option B — Local

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/cifar10-ann-cnn.git
cd cifar10-ann-cnn

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook CIFAR10_ANN_CNN_Full.ipynb
```

---

## 🔑 Key Concepts

### Why CNN beats ANN on images
ANN flattens a 32×32×3 image into 3,072 independent numbers — the network has no idea that pixel (5,5) and pixel (6,5) are neighbors. CNN's 3×3 filter scans across the image, treating nearby pixels as a group, learning edges → textures → object parts hierarchically. That spatial inductive bias is why CNN gets ~75% with 5× fewer parameters.

### Regularization used
- **Dropout** — randomly zeroes neuron outputs during training, forcing the network to learn redundant representations
- **Batch Normalization** — normalizes each layer's output distribution, enabling higher learning rates and faster convergence
- **L2 Weight Decay** — penalizes large weights, producing smoother decision boundaries

### Training tricks
- **EarlyStopping** — halts training when validation loss stops improving, saving the best weights automatically
- **ReduceLROnPlateau** — cuts learning rate by 50% when training plateaus, helping escape local minima
- **ModelCheckpoint** — saves the best model weights to disk at every epoch

---

## 🔮 Next Steps to Improve

| Technique | Expected Gain |
|---|---|
| Transfer Learning (VGG16 / ResNet50) | 88–92% |
| MixUp / CutOut augmentation | +2–3% |
| Cosine LR annealing with warmup | +1–2% |
| Label smoothing | +0.5–1% |
| Ensemble (CNN + CNN+Aug) | +1–2% |

---

## 📚 References

- [CIFAR-10 Dataset — Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/api_docs)
- [Batch Normalization — Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167)
- [Dropout — Srivastava et al. (2014)](https://jmlr.org/papers/v15/srivastava14a.html)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built for learning deep learning fundamentals. Contributions and ⭐ stars are welcome!*

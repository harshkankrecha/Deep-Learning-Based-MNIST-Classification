# MNIST Image Classification & Hardware Benchmarking

An end-to-end computer vision project implementing a custom **TinyVGG** Convolutional Neural Network (CNN) from scratch using **PyTorch**. This project focuses on high-accuracy digit classification, hardware acceleration profiling (CPU vs. GPU/CUDA), and deep diagnostic evaluation.

## 🚀 Key Features
* **PyTorch-Native Architecture:** From-scratch implementation of the TinyVGG architecture optimized for grayscale digit recognition.
* **Hardware Benchmarking:** Comparative performance profiling demonstrating a **3.16x training acceleration** using NVIDIA CUDA.
* **Optimized Data Pipeline:** Mini-batched ($batch\_size=32$) and shuffled training pipelines via PyTorch DataLoaders.
* **Advanced Evaluation:** Multi-class confusion matrix generation and prediction diagnostics leveraging **Torchmetrics**.

---

## 📊 Performance & Benchmarking Results

### ⏱️ Hardware Acceleration Profile
Moving the training loop from the CPU to the GPU yielded an immediate **3.16x performance speedup**, drastically lowering training latency.

| Hardware | Total Training Time (Seconds) | Training Execution Speedup |
| :--- | :--- | :--- |
| **CPU** | 116.20s | 1.0x (Baseline) |
| **GPU (NVIDIA CUDA)** | **36.80s** | **3.16x Faster** |

### 🎯 Model Accuracy
* **Peak Test Accuracy:** **98.55%**

### 🔍 Confusion Matrix Analysis
The model's classification errors were isolated using a `Torchmetrics` multi-class confusion matrix. The diagnostics reveal high precision across all classes, with expected minor structural ambiguities among morphologically similar digits (e.g., distinguishing complex geometric strokes between `4`, `7`, and `9`).

---

## 🛠️ Tech Stack & Dependencies
* **Core Framework:** PyTorch
* **Evaluation Metrics:** Torchmetrics
* **Data Visualization:** Matplotlib
* **Scientific Computing:** NumPy, Pandas

---

## 🧬 Model Architecture (TinyVGG)

The network is composed of repeating convolutional blocks utilizing small $3 \times 3$ kernel filters, ReLU activations, and MaxPool operations to downsample spatial dimensions while building a rich hierarchy of feature maps:

```text
Input (1x28x28) 
   │
   ├── [Conv Block 1] ──► Conv2d (10 filters) ──► ReLU ──► Conv2d (10 filters) ──► ReLU ──► MaxPool2d
   │
   ├── [Conv Block 2] ──► Conv2d (10 filters) ──► ReLU ──► Conv2d (10 filters) ──► ReLU ──► MaxPool2d
   │
   └── [Classifier]   ──► Flatten ──► Linear Layer (10 output classes) ──► Logits

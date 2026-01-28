# 🍎 Fruit Recognition Using Deep Learning

A deep learning project for recognizing 10 different types of fruits from images using a Convolutional Neural Network (CNN) trained entirely from scratch.

## 📋 Project Overview

This project implements a fruit recognition system that can classify 10 different fruit types using a CNN architecture trained from scratch (no transfer learning). The system includes both a training pipeline and a web application for real-time fruit classification.

### Supported Fruits

1. 🍎 Apple
2. 🥑 Avocado
3. 🍌 Banana
4. 🍋 Lemon
5. 🥭 Mango
6. 🍊 Orange
7. 🍈 Papaya
8. 🍍 Pineapple
9. 🍅 Tomato
10. 🍉 Watermelon

## 🎯 Project Goals

- Design and implement a CNN architecture suitable for fruit recognition
- Train a CNN model entirely from scratch (no pre-trained weights)
- Implement advanced data augmentation techniques including MixUp
- Achieve high classification accuracy on a custom dataset
- Build a web application for real-time fruit recognition
- Evaluate model performance with comprehensive metrics

## 📁 Project Structure

```
Deep_learning/
├── dataset/
│   ├── train/          # Training images (70%)
│   ├── validation/     # Validation images (15%)
│   └── test/           # Test images (15%)
├── notebooks/          # Jupyter notebooks for exploration
├── src/
│   ├── data_preprocessing.py  # Data loading utilities
│   ├── model.py               # Model architecture definitions
│   ├── train.py               # Training script (CNN from scratch)
│   └── evaluate.py            # Evaluation script
├── models/             # Saved trained models
│   ├── cnn_from_scratch_*_best.h5
│   ├── cnn_from_scratch_*_final.h5
│   ├── cnn_from_scratch_info_*.json
│   └── cnn_from_scratch_history_*.png
├── results/            # Evaluation results and plots
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── evaluation_results.json
├── templates/          # Web app templates
│   └── index.html
├── static/            # Static files for web app
├── uploads/          # Temporary upload directory
├── app.py            # Flask web application
├── start_training.sh  # Training script launcher
├── start_web_app.sh   # Web app launcher
├── requirements.txt  # Python dependencies
├── paper.tex         # Research paper (LaTeX)
└── README.md         # This file
```
## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- TensorFlow 2.x
- Flask (for web application)

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Deep_learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** For HEIC image support (Apple devices), install pillow-heif:
```bash
pip install pillow-heif
```

### Dataset Structure

The project uses a custom dataset organized as follows:

```
dataset/
├── train/
│   ├── Apple/
│   ├── Avocado/
│   └── ... (10 classes)
├── validation/
│   ├── Apple/
│   └── ...
└── test/
    ├── Apple/
    └── ...
```

The dataset used in this project contains approximately 4000 images across 10 fruit classes, split into 70% training, 15% validation, and 15% test sets.

## 💻 Usage

### 1. Data Preprocessing

Split your raw images into the required train/validation/test structure:

```bash
python -c "from src.data_preprocessing import FruitDataset; dataset = FruitDataset(data_dir='dataset', img_size=(192, 192)); dataset.split_dataset(source_dir='source_images')"
```

### 2. Training (CNN from Scratch)

The system is optimized for training a high-accuracy model from scratch using MixUp and Cosine Decay.

#### **Easy Start (Recommended)**
For Windows users:
```powershell
.\start_training.ps1
```
For Linux/Mac users:
```bash
./start_training.sh
```

#### **Manual Execution**
```bash
python src/train.py \
    --data_dir dataset \
    --epochs 50 \
    --batch_size 32 \
    --img_size 192 192 \
    --initial_lr 3e-4 \
    --save_dir models
```

### 3. Evaluation

Evaluate your trained model and generate detailed metrics:

```bash
python src/evaluate.py \
    --model_path models/cnn_from_scratch_YYYYMMDD_HHMMSS_best.h5 \
    --data_dir dataset \
    --img_size 192 192 \
    --save_dir results \
    --visualize
```

This generates:
- ✅ Test accuracy and loss
- ✅ Classification report (Precision, Recall, F1)
- ✅ Confusion matrix visualization
- ✅ Per-class accuracy plots
- ✅ Sample prediction visualizations

### 4. Web Application

Launch the real-time recognition interface:

#### **Easy Start**
For Windows: `.\start_web_app.ps1` | For Linux: `./start_web_app.sh`

#### **Manual Execution**
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

## 📊 Model Architecture

The `Stronger_CNN` architecture used in this project features:
- **5 Convolutional Blocks**: Increasing filters from 32 to 320.
- **Advanced Regularization**: MixUp augmentation, Dropout (0.45), and L2 Weight Decay.
- **Optimized Training**: Cosine learning rate decay and Global Average Pooling.

## 📈 Results

Typical results on the custom dataset:
- **Best Validation Accuracy**: ~85%
- **Test Accuracy**: ~80%

## 👥 Team Members

[Add your team members' names here]

---
**Note**: This project follows a professional deep learning pipeline including dataset splitting, normalization, and comprehensive evaluation.

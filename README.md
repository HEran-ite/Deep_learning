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

### Dataset

The project uses a custom dataset of fruit images. The dataset should be organized as:

```
dataset/
├── train/
│   ├── Apple/
│   ├── Avocado/
│   ├── Banana/
│   └── ...
├── validation/
│   ├── Apple/
│   ├── Avocado/
│   └── ...
└── test/
    ├── Apple/
    ├── Avocado/
    └── ...
```

The dataset used in this project contains approximately 4000 images across 10 fruit classes, split into 70% training, 15% validation, and 15% test sets.

## 💻 Usage

### 1. Training the CNN Model

Train a CNN model from scratch using the improved training script:

#### Quick Start (Using Shell Script)
```bash
./start_training.sh
```

#### Manual Training
```bash
python3 src/train.py \
    --data_dir dataset \
    --epochs 50 \
    --batch_size 32 \
    --img_size 192 192 \
    --initial_lr 0.0002 \
    --save_dir models
```

**Training Parameters:**
- `--data_dir`: Path to dataset directory (default: `dataset`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--img_size`: Image dimensions height width (default: 192 192)
- `--initial_lr`: Initial learning rate (default: 0.0002)
- `--save_dir`: Directory to save models (default: `models`)

**Training Features:**
- Cosine decay learning rate scheduling
- MixUp data augmentation (α=0.2)
- Advanced data augmentation (random flip, rotation, zoom, translation)
- Normalization layer adapted on training data
- Early stopping based on validation accuracy
- Model checkpointing (saves best model)

### 2. Model Architecture

The CNN architecture consists of:

- **Input**: 192×192×3 RGB images
- **Data Augmentation**: RandomFlip, RandomRotation(0.03), RandomZoom(0.08), RandomTranslation(0.03, 0.03)
- **Preprocessing**: Rescaling(1./255) + Normalization layer
- **Convolutional Blocks**:
  - Block 1: 2× Conv2D(32) + BatchNorm + ReLU + MaxPool
  - Block 2: 2× Conv2D(64) + BatchNorm + ReLU + MaxPool
  - Block 3: 2× Conv2D(128) + BatchNorm + ReLU + MaxPool
  - Block 4: 2× Conv2D(256) + BatchNorm + ReLU + MaxPool
  - Block 5: Conv2D(320) + BatchNorm + ReLU + MaxPool
- **Classifier Head**:
  - GlobalAveragePooling2D
  - Dropout(0.45)
  - Dense(384) + L2 regularization
  - Dropout(0.45)
  - Dense(10, softmax)

**Total Parameters**: ~2-3M (moderate-sized model)

### 3. Evaluation

Evaluate your trained model:

```bash
python3 src/evaluate.py \
    --model_path models/cnn_from_scratch_YYYYMMDD_HHMMSS_best.h5 \
    --data_dir dataset \
    --img_size 192 192 \
    --batch_size 32 \
    --save_dir results \
    --visualize
```

This generates:
- Test accuracy and loss
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Per-class accuracy plot
- Sample prediction visualizations

### 4. Web Application

Run the Flask web application for real-time fruit recognition:

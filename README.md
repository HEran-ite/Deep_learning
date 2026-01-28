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
Run the Flask web application for real-time fruit recognition:

#### Quick Start (Using Shell Script)
```bash
./start_web_app.sh
```

#### Manual Start
```bash
python3 app.py
```

Then open your browser and navigate to:
- `http://localhost:5000` or
- `http://127.0.0.1:5000`

**Features:**
- 📤 Upload images (JPG, PNG, HEIC supported)
- 📷 Live camera capture
- 🎯 Real-time fruit classification
- 📊 Top-3 predictions with confidence scores
- 🎨 Modern, responsive UI

**Supported Image Formats:**
- JPEG/JPG
- PNG
- HEIC/HEIF (Apple devices)

**File Size Limit:** 32MB

### 5. Single Image Prediction (Python)

```python
from src.evaluate import predict_single_image
import json

# Load class names from model info
with open('models/cnn_from_scratch_info_YYYYMMDD_HHMMSS.json') as f:
    model_info = json.load(f)
    class_names = model_info['class_names']

# Predict
predicted_class, confidence, top3 = predict_single_image(
    'models/cnn_from_scratch_YYYYMMDD_HHMMSS_best.h5',
    'path/to/image.jpg',
    class_names,
    img_size=(192, 192)
)

print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
print("Top 3 predictions:", top3)
```

## 📊 Results

### Model Performance

The best model achieved the following performance:

- **Test Accuracy**: 80.17%
- **Validation Accuracy**: 85.83%
- **Test Loss**: 1.04

**Training Configuration:**
- Epochs: 70
- Batch Size: 32
- Image Size: 192×192
- Initial Learning Rate: 2×10⁻⁴ (cosine decay)
- Optimizer: Adam
- Loss: Categorical Crossentropy

**Key Techniques:**
- MixUp augmentation (α=0.2)
- Advanced data augmentation
- Normalization layer adapted on training data
- L2 regularization (weight decay: 2×10⁻⁴)
- Dropout (0.45) in classifier head
- Early stopping (patience: 10 epochs)

### Per-Class Performance

The model shows varying performance across fruit classes:
- **Best performing**: Watermelon (100% accuracy)
- **Strong performance**: Avocado (88% F1-score), Papaya (80% F1-score)
- **Challenging classes**: Lemon, Mango (lower recall)

Detailed per-class metrics are available in `results/evaluation_results.json` and visualized in `results/per_class_accuracy.png`.

## 🔧 Configuration

### Key Hyperparameters

- **Epochs**: 50-70 (use early stopping)
- **Batch Size**: 32 (reduce to 16 if OOM)
- **Learning Rate**: 2×10⁻⁴ to 3×10⁻⁴ (with cosine decay)
- **Image Size**: 192×192 (optimal for this dataset)
- **MixUp α**: 0.2 (standard)
- **Dropout**: 0.45 (classifier head)
- **Weight Decay**: 2×10⁻⁴ (L2 regularization)

### Training Tips

1. **Monitor Training**: Check `logs/` directory for training logs
2. **Early Stopping**: Model automatically stops if validation accuracy doesn't improve for 10 epochs
3. **Best Model**: Always use `*_best.h5` for evaluation (not `*_final.h5`)
4. **GPU Recommended**: Training is faster on GPU, but works on CPU too

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce `batch_size` to 16 or 8
- Reduce `img_size` to 128×128 (may affect accuracy)

### Low Accuracy
- Collect more training data (aim for 100+ images per class)
- Increase data augmentation strength
- Train for more epochs (with early stopping)
- Adjust learning rate

### Slow Training
- Use GPU if available (TensorFlow will auto-detect)
- Reduce image size (but may affect accuracy)
- Reduce batch size

### Web App Issues
- **Port 5000 in use**: Run `lsof -ti:5000 | xargs kill -9` or use `start_web_app.sh`
- **Model not loading**: Ensure model files exist in `models/` directory
- **HEIC not working**: Install `pillow-heif`: `pip install pillow-heif`

## 📝 Research Paper

The project includes a comprehensive research paper (`paper.tex`) covering:
- Introduction and problem statement
- Related work and literature review
- Dataset collection methodology
- Model architecture and training methodology
- Experimental results and analysis
- Challenges, limitations, and future work

Compile the paper using LaTeX:
```bash
pdflatex paper.tex
```

## 📚 Technical Details

### Data Augmentation

**During Training:**
- MixUp (α=0.2): Mixes pairs of images and labels
- Random horizontal flip
- Random rotation (±3%)
- Random zoom (±8%)
- Random translation (±3%)

**During Inference:**
- No augmentation (deterministic predictions)

### Training Pipeline

1. Load and preprocess images (resize to 192×192)
2. Create tf.data datasets with one-hot encoding
3. Adapt normalization layer on training data
4. Apply MixUp augmentation to training set
5. Train with cosine decay learning rate
6. Monitor validation accuracy for early stopping
7. Save best model based on validation accuracy

### Model Selection

The best model is selected based on:
- **Primary metric**: Validation accuracy
- **Checkpointing**: Saves model with highest validation accuracy
- **Early stopping**: Stops training if no improvement for 10 epochs

## 🎓 Educational Value

This project demonstrates:
- CNN architecture design from scratch
- Advanced data augmentation (MixUp)
- Learning rate scheduling (cosine decay)
- Model regularization (L2, dropout)
- Proper train/validation/test splits
- Comprehensive evaluation metrics
- Web application deployment

## 📄 License

[Add your license information here]

## 👥 Authors

[Add your name/team members here]

## 🙏 Acknowledgments

- Dataset: Kaggle - "4000 Images of Local Fruits in Ethiopia"
- TensorFlow/Keras for deep learning framework
- Flask for web application framework

---

**Note**: This project focuses exclusively on CNN training from scratch. No transfer learning or pre-trained models are used, demonstrating that effective fruit recognition can be achieved through careful architecture design and training strategies.


# Basket of Fruit Recognition Using Deep Learning

A deep learning project for recognizing 10 different types of fruits from images using Convolutional Neural Networks (CNNs).

## 📋 Project Overview

This project implements a fruit recognition system that can classify 10 different fruit types:
1. Banana
2. Apple
3. Orange
4. Mango
5. Avocado
6. Papaya
7. Pineapple
8. Lemon
9. Watermelon
10. Tomato


## 🎯 Project Goals

- Collect and organize a custom dataset of fruit images
- Design and implement CNN models (from scratch and transfer learning)
- Train models to achieve high classification accuracy
- Evaluate model performance with comprehensive metrics
- Document the entire process

## 📁 Project Structure

```
Basket-of-Fruit-Recognition/
├── dataset/
│   ├── train/          # Training images (70%)
│   │   ├── banana/
│   │   ├── apple/
│   │   └── ...
│   ├── validation/     # Validation images (15%)
│   └── test/           # Test images (15%)
├── notebooks/          # Jupyter notebooks for exploration
├── src/
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── model.py               # Model architectures
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── models/             # Saved trained models
├── results/            # Evaluation results and plots
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Basket-of-Fruit-Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Collection

**IMPORTANT:** You must collect your own images. Do not use pre-existing datasets.

1. Take photos of each fruit type using your phone camera
2. Aim for 80-120 images per fruit class (800-1200 total images)
3. Vary:
   - Lighting conditions (daylight, indoor)
   - Backgrounds (table, basket, hand)
   - Angles (top, side, close-up)
   - Distance (near/far)
   - Include both single fruits and mixed baskets

4. Organize images in a source directory:
```
source_images/
├── banana/
├── apple/
├── orange/
└── ...
```

5. Split dataset using the preprocessing script (see Usage section)

## 💻 Usage

### 1. Data Preprocessing

Split your collected images into train/validation/test sets:

```bash
python -c "from src.data_preprocessing import FruitDataset; dataset = FruitDataset(data_dir='dataset', img_size=(192, 192)); dataset.split_dataset(source_dir='source_images')"
```

### 2. Training

#### Option A: CNN from Scratch (Optimized)

For Windows users:
```powershell
.\start_training.ps1
```

For Linux/Mac users:
```bash
./start_training.sh
```

Or manually:
```bash
python src/train.py \
    --data_dir dataset \
    --epochs 50 \
    --batch_size 32 \
    --img_size 192 192 \
    --initial_lr 3e-4 \
    --save_dir models
```

#### Option B: Transfer Learning

```bash
python src/train_transfer.py \
    --model_type transfer \
    --base_model MobileNetV2 \
    --data_dir dataset \
    --epochs 30 \
    --batch_size 32 \
    --freeze_base
```
*(Note: Create a separate script for transfer learning if needed, or update train.py to support it)*

### 3. Evaluation

Evaluate your trained model:

```bash
cd src
python evaluate.py \
    --model_path ../models/fruit_model_YYYYMMDD_HHMMSS_best.h5 \
    --data_dir ../dataset \
    --save_dir ../results \
    --visualize
```

### 4. Single Image Prediction

```python
from src.evaluate import predict_single_image
import json

# Load class names from model info
with open('models/model_info_YYYYMMDD_HHMMSS.json') as f:
    model_info = json.load(f)
    class_names = model_info['class_names']

# Predict
predicted_class, confidence, top3 = predict_single_image(
    'models/fruit_model_YYYYMMDD_HHMMSS_best.h5',
    'path/to/image.jpg',
    class_names
)

print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
print("Top 3 predictions:", top3)
```

## 📊 Model Architectures

### Transfer Learning Models

- **MobileNetV2**: Lightweight, fast, good for mobile deployment
- **ResNet50**: Deeper network, higher accuracy
- **EfficientNetB0**: Balanced efficiency and accuracy

### CNN from Scratch

- 4 Convolutional blocks with MaxPooling
- 2 Dense layers with Dropout
- Output layer with Softmax activation

## 📈 Results

After training and evaluation, you'll get:

- Training history plots (accuracy and loss)
- Confusion matrix
- Per-class accuracy
- Classification report (precision, recall, F1-score)
- Sample prediction visualizations

Results are saved in the `results/` directory.

## 📝 Report Structure

Your written report should include:

1. **Abstract**: Brief summary of the project
2. **Introduction**: Problem statement and objectives
3. **Related Work**: Brief literature review
4. **Dataset Collection Method**: How you collected images
5. **Data Preprocessing**: Image preprocessing steps
6. **Model Architecture**: Detailed model design
7. **Training & Evaluation**: Training process and hyperparameters
8. **Results**: Performance metrics and analysis
9. **Challenges & Limitations**: Issues faced and constraints
10. **Conclusion & Future Work**: Summary and improvements

## 🔧 Configuration

Key hyperparameters you can adjust:

- `epochs`: Number of training epochs (15-30 recommended)
- `batch_size`: Batch size (16, 32, or 64)
- `learning_rate`: Learning rate (0.001 default)
- `img_size`: Image dimensions (224x224 default)
- `freeze_base`: Whether to freeze base model (transfer learning)

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce `batch_size` (try 16 or 8)
- Reduce `img_size` (try 128x128)

### Low Accuracy
- Collect more training data
- Increase data augmentation
- Try different base models
- Adjust learning rate

### Slow Training
- Use GPU if available
- Reduce image size
- Use MobileNetV2 (lightweight)

## 📚 References

- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/
- Transfer Learning Guide: https://www.tensorflow.org/tutorials/images/transfer_learning

## 👥 Team Members

[Add your team members' names here]

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

[Add any acknowledgments here]

---

**Note**: Remember to collect your own data! This project requires original image collection as per assignment requirements.


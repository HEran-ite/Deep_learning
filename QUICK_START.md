# Quick Start Guide

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Your Data

Take photos of 10 fruit types (80-120 images per fruit):
- Banana, Apple, Orange, Mango, Avocado, Papaya, Pineapple, Lemon, Strawberry, Grape

Organize them in a source directory:
```
source_images/
├── banana/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── apple/
└── ...
```

### 3. Split Your Dataset

Run this Python script to split your data:

```python
from src.data_preprocessing import FruitDataset

dataset = FruitDataset(data_dir='dataset')
dataset.split_dataset(
    source_dir='source_images',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### 4. Explore Your Data (Optional)

Open `notebooks/data_exploration.ipynb` in Jupyter to visualize your dataset.

### 5. Train Your Model

#### Option A: Transfer Learning (Recommended - Faster & Better Accuracy)

```bash
cd src
python train.py \
    --model_type transfer \
    --base_model MobileNetV2 \
    --data_dir ../dataset \
    --epochs 30 \
    --batch_size 32 \
    --freeze_base
```

#### Option B: CNN from Scratch (Better for Learning)

```bash
cd src
python train.py \
    --model_type scratch \
    --data_dir ../dataset \
    --epochs 30 \
    --batch_size 32
```

### 6. Evaluate Your Model

After training, find your best model in the `models/` directory and evaluate it:

```bash
cd src
python evaluate.py \
    --model_path ../models/fruit_model_YYYYMMDD_HHMMSS_best.h5 \
    --data_dir ../dataset \
    --save_dir ../results \
    --visualize
```

### 7. Check Results

Results will be saved in `results/`:
- `confusion_matrix.png` - Confusion matrix visualization
- `per_class_accuracy.png` - Accuracy per fruit class
- `sample_predictions.png` - Sample predictions with images
- `evaluation_results.json` - Detailed metrics

### 8. Write Your Report

Use `REPORT_TEMPLATE.md` as a guide for your written report.

---

## Common Issues

**Out of Memory?**
- Reduce batch_size: `--batch_size 16`
- Reduce image size: `--img_size 128 128`

**Low Accuracy?**
- Collect more training data
- Try different base models (ResNet50, EfficientNetB0)
- Adjust learning rate: `--learning_rate 0.0001`

**Need Help?**
- Check the full README.md for detailed documentation
- Review the code comments in each script


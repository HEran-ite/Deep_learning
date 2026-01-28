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

### 6. Evaluate Your Model

After training, find your best model in the `models/` directory and evaluate it using the evaluation script:

```bash
python src/evaluate.py \
    --model_path models/cnn_from_scratch_YYYYMMDD_HHMMSS_best.h5 \
    --data_dir dataset \
    --img_size 192 192 \
    --save_dir results \
    --visualize
```

### 7. Run the Web Application

Launch the interactive interface to test your model:

#### **Easy Start**
For Windows: `.\start_web_app.ps1`
For Linux/Mac: `./start_web_app.sh`

#### **Manual Execution**
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

### 8. Write Your Report

Use `REPORT_TEMPLATE.md` as a guide for your written report.

---

## Common Issues

**Out of Memory?**
- Reduce batch_size: `--batch_size 16`
- Reduce image size: `--img_size 160 160`

**Low Accuracy?**
- Collect more training data
- Ensure images are correctly organized in `dataset/`
- Check training logs in `models/` or `logs/`

**Need Help?**
- Check the full `README.md` for detailed documentation
- Review the code comments in each script


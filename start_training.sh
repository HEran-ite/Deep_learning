#!/bin/bash
# Start CNN from scratch training (high accuracy)

cd "$(dirname "$0")"

echo "🚀 Starting CNN from Scratch Training"
echo "======================================"
echo ""

# Check dataset
if [ ! -d "dataset/train" ]; then
    echo "❌ Dataset not found at dataset/train/"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start training
echo "📊 Training Configuration:"
echo "   - Model: CNN from Scratch"
echo "   - Architecture: 4 Conv blocks (32→64→128→256)"
echo "   - Epochs: 50"
echo "   - Batch Size: 32"
echo "   - Learning Rate: 3e-4 (cosine decay)"
echo "   - Image Size: 192x192"
echo "   - Augmentation: MixUp + Advanced Transforms"
echo ""
echo "Starting training..."
echo ""

python3 src/train_improved.py \
    --data_dir dataset \
    --epochs 50 \
    --batch_size 32 \
    --img_size 192 192 \
    --initial_lr 3e-4 \
    --save_dir models \
    2>&1 | tee "logs/cnn_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Training completed!"
echo "📁 Check models/ directory for saved models"

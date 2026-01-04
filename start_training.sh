#!/bin/bash
# Training script with logging

cd "$(dirname "$0")"
mkdir -p logs

echo "🚀 Starting ResNet CNN Training..."
echo "📝 Logs will be saved to: logs/cnn_training_*.log"
echo ""

python3 src/train.py \
    --model_type scratch \
    --data_dir dataset \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --img_size 224 224 \
    --save_dir models \
    2>&1 | tee "logs/cnn_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ CNN Training completed!"
echo ""
echo "🚀 Starting Transfer Learning Training..."
echo "📝 Logs will be saved to: logs/transfer_training_*.log"
echo ""

python3 src/train.py \
    --model_type transfer \
    --base_model MobileNetV2 \
    --data_dir dataset \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --img_size 224 224 \
    --save_dir models \
    2>&1 | tee "logs/transfer_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ All training completed!"
echo "📊 Models saved in: models/"
echo "📝 Logs saved in: logs/"


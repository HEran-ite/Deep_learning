# Start CNN from scratch training (high accuracy) for Windows PowerShell

Write-Host "🚀 Starting CNN from Scratch Training" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""

# Check dataset
if (-Not (Test-Path "dataset/train")) {
    Write-Host "❌ Dataset not found at dataset/train/" -ForegroundColor Red
    exit 1
}

# Create logs directory
if (-Not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/cnn_training_$($timestamp).log"

# Start training
Write-Host "📊 Training Configuration:"
Write-Host "   - Model: CNN from Scratch"
Write-Host "   - Architecture: 5 Conv blocks (32→64→128→256→320)"
Write-Host "   - Epochs: 50"
Write-Host "   - Batch Size: 32"
Write-Host "   - Learning Rate: 3e-4 (cosine decay)"
Write-Host "   - Image Size: 192x192"
Write-Host "   - Augmentation: MixUp + Advanced Transforms"
Write-Host ""
Write-Host "Starting training..."
Write-Host ""

python src/train.py `
    --data_dir dataset `
    --epochs 50 `
    --batch_size 32 `
    --img_size 192 192 `
    --initial_lr 3e-4 `
    --save_dir models | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "✅ Training completed!" -ForegroundColor Green
Write-Host "📁 Check models/ directory for saved models"

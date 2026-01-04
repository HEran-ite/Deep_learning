"""
Model Comparison Script
Train and compare ResNet CNN from scratch vs Transfer Learning
"""

import os
import argparse
from train import train_model


def compare_models(data_dir='dataset',
                  epochs=30,
                  batch_size=32,
                  learning_rate=0.001,
                  img_size=(224, 224),
                  save_dir='models'):
    """
    Train and compare both models
    
    Args:
        data_dir: Root directory of dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        img_size: Image size (height, width)
        save_dir: Directory to save models
    """
    
    print("="*60)
    print("MODEL COMPARISON: ResNet CNN vs Transfer Learning")
    print("="*60)
    
    # Train ResNet CNN from scratch
    print("\n" + "="*60)
    print("TRAINING 1: ResNet CNN from Scratch")
    print("="*60)
    model1, history1, dataset1 = train_model(
        model_type='scratch',
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        img_size=img_size,
        save_dir=save_dir
    )
    
    print("\n" + "="*60)
    print("TRAINING 1 COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {max(history1.history['val_accuracy']):.4f}")
    print(f"Final Training Accuracy: {history1.history['accuracy'][-1]:.4f}")
    print(f"Model Parameters: {model1.count_params():,}")
    
    # Train Transfer Learning model
    print("\n" + "="*60)
    print("TRAINING 2: Transfer Learning (MobileNetV2)")
    print("="*60)
    model2, history2, dataset2 = train_model(
        model_type='transfer',
        base_model='MobileNetV2',
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        img_size=img_size,
        freeze_base=False,  # Fine-tuning for better performance
        save_dir=save_dir
    )
    
    print("\n" + "="*60)
    print("TRAINING 2 COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {max(history2.history['val_accuracy']):.4f}")
    print(f"Final Training Accuracy: {history2.history['accuracy'][-1]:.4f}")
    print(f"Model Parameters: {model2.count_params():,}")
    
    # Comparison Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<30} {'ResNet CNN':<20} {'Transfer Learning':<20}")
    print("-"*60)
    print(f"{'Best Val Accuracy':<30} {max(history1.history['val_accuracy']):.4f} ({max(history1.history['val_accuracy'])*100:.2f}%){'':<10} {max(history2.history['val_accuracy']):.4f} ({max(history2.history['val_accuracy'])*100:.2f}%)")
    print(f"{'Final Train Accuracy':<30} {history1.history['accuracy'][-1]:.4f} ({history1.history['accuracy'][-1]*100:.2f}%){'':<10} {history2.history['accuracy'][-1]:.4f} ({history2.history['accuracy'][-1]*100:.2f}%)")
    print(f"{'Model Parameters':<30} {model1.count_params():,}{'':<10} {model2.count_params():,}")
    print(f"{'Best Val Loss':<30} {min(history1.history['val_loss']):.4f}{'':<10} {min(history2.history['val_loss']):.4f}")
    
    accuracy_diff = max(history2.history['val_accuracy']) - max(history1.history['val_accuracy'])
    print(f"\n{'Accuracy Difference':<30} {accuracy_diff:.4f} ({accuracy_diff*100:.2f}% better for Transfer Learning)")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    if max(history2.history['val_accuracy']) > max(history1.history['val_accuracy']):
        print("✅ Transfer Learning performs better (as expected for small datasets)")
        print(f"   Transfer Learning is {accuracy_diff*100:.2f}% more accurate")
    else:
        print("✅ ResNet CNN performs better (unusual but possible)")
        print(f"   ResNet CNN is {abs(accuracy_diff)*100:.2f}% more accurate")
    
    print("\nBoth models saved in:", save_dir)
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Compare ResNet CNN vs Transfer Learning')
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Root directory of dataset')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (height width)')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    compare_models(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=tuple(args.img_size),
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()


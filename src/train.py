"""
Training Script
Main script to train the fruit recognition model
"""

import os
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from data_preprocessing import FruitDataset
from model import build_cnn_from_scratch, build_transfer_learning_model, compile_model


def train_model(model_type='transfer', 
                base_model='MobileNetV2',
                data_dir='dataset',
                epochs=30,
                batch_size=32,
                learning_rate=0.001,
                img_size=(224, 224),
                freeze_base=True,
                save_dir='models',
                fine_tune_layers=None):
    """
    Train the fruit recognition model
    
    Args:
        model_type: 'transfer' or 'scratch'
        base_model: Base model name for transfer learning ('MobileNetV2', 'ResNet50', 'EfficientNetB0')
        data_dir: Root directory of dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        img_size: Image size (height, width)
        freeze_base: Whether to freeze base model (for transfer learning)
        save_dir: Directory to save model and results
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include model type in filename for clarity
    model_type_label = 'cnn' if model_type == 'scratch' else 'transfer'
    base_model_label = f'_{base_model}' if model_type == 'transfer' else ''
    model_save_path = os.path.join(save_dir, f'{model_type_label}{base_model_label}_model_{timestamp}')
    
    # Load dataset
    print("Loading dataset...")
    dataset = FruitDataset(data_dir, img_size=img_size, batch_size=batch_size)
    
    # Use moderate augmentation for CNN from scratch (stable, exam-safe)
    augmentation_strength = 'aggressive' if model_type == 'scratch' else 'normal'
    print(f"Using {augmentation_strength} data augmentation for {model_type} model")
    
    train_gen, val_gen, test_gen = dataset.create_data_generators(
        use_augmentation=True,
        augmentation_strength=augmentation_strength
    )
    
    num_classes = len(dataset.class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {dataset.class_names}")
    
    # Build model with optimized configurations
    print(f"\nBuilding {model_type} model...")
    if model_type == 'transfer':
        # For transfer learning, use fine-tuning for better performance
        model, _ = build_transfer_learning_model(
            base_model_name=base_model,
            input_shape=(*img_size, 3),
            num_classes=num_classes,
            freeze_base=freeze_base,
            fine_tune_layers=fine_tune_layers
        )
        # Use lower learning rate for transfer learning
        initial_lr = learning_rate if freeze_base else learning_rate * 0.1
    else:
        # CNN from scratch - use optimized architecture
        model = build_cnn_from_scratch(
            input_shape=(*img_size, 3),
            num_classes=num_classes
        )
        # Use lower learning rate for ResNet architecture
        initial_lr = learning_rate * 0.5
    
    # Compile model with stable settings for CNN from scratch
    model = compile_model(model, learning_rate=initial_lr)
    
    # CRITICAL CHECK: Verify output layer matches number of classes
    output_shape = model.output_shape
    if output_shape[-1] != num_classes:
        raise ValueError(f"CRITICAL ERROR: Model output size ({output_shape[-1]}) doesn't match number of classes ({num_classes})!")
    print(f"✅ Model output size verified: {output_shape[-1]} classes")
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Optimized callbacks for better training
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path + '_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if model_type == 'scratch' else 'val_accuracy',
            patience=5 if model_type == 'scratch' else 10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive reduction
            patience=3,  # Less patience
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(save_dir, f'{model_type_label}{base_model_label}_training_log_{timestamp}.csv')
        )
    ]
    
    # No cosine annealing - use ReduceLROnPlateau only for stability
    
    # Calculate steps per epoch
    steps_per_epoch = train_gen.samples // batch_size
    validation_steps = val_gen.samples // batch_size
    
    # No class weights - keep it simple and stable
    class_weights_dict = None
    
    # Train model
    print("\nStarting training...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Initial learning rate: {initial_lr}")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Save final model
    model.save(model_save_path + '_final.h5')
    print(f"\nModel saved to: {model_save_path}")
    
    # Save training history
    history_path = os.path.join(save_dir, f'{model_type_label}{base_model_label}_training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in values] for k, values in history.history.items()}, f)
    
    # Plot training history
    plot_training_history(history, save_dir, timestamp, model_type_label, base_model_label)
    
    # Save model info
    model_info = {
        'model_type': model_type,
        'base_model': base_model if model_type == 'transfer' else None,
        'num_classes': num_classes,
        'class_names': dataset.class_names,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'img_size': img_size,
        'timestamp': timestamp
    }
    
    info_path = os.path.join(save_dir, f'{model_type_label}{base_model_label}_model_info_{timestamp}.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return model, history, dataset


def plot_training_history(history, save_dir, timestamp, model_type_label='', base_model_label=''):
    """Plot and save training history graphs"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    filename = f'{model_type_label}{base_model_label}_training_history_{timestamp}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    print(f"Training history plot saved to: {save_dir}/{filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Fruit Recognition Model')
    parser.add_argument('--model_type', type=str, default='transfer',
                       choices=['transfer', 'scratch'],
                       help='Model type: transfer learning or from scratch')
    parser.add_argument('--base_model', type=str, default='MobileNetV2',
                       choices=['MobileNetV2', 'ResNet50', 'EfficientNetB0'],
                       help='Base model for transfer learning')
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
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze base model weights (for transfer learning)')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save model')
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model_type,
        base_model=args.base_model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=tuple(args.img_size),
        freeze_base=args.freeze_base,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()


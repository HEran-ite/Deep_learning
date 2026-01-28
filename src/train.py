
import os
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

# Disable JIT for stability
tf.config.optimizer.set_jit(False)



def create_datasets(data_dir, img_size=(192, 192), batch_size=32, seed=42):
    """Create datasets matching notebook exactly"""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")
    
    # Get class names
    class_names = sorted(
        [d for d in os.listdir(train_dir) 
         if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")],
        key=str.lower
    )
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    
    # Build raw datasets
    train_ds_raw = keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=seed
    )
    
    val_ds_raw = keras.utils.image_dataset_from_directory(
        val_dir,
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )
    
    test_ds_raw = keras.utils.image_dataset_from_directory(
        test_dir,
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Convert to one-hot
    def to_one_hot(images, labels):
        images = tf.cast(images, tf.float32)
        return images, tf.one_hot(labels, depth=num_classes)
    
    # MixUp function
    def sample_beta(alpha, shape):
        g1 = tf.random.gamma(shape, alpha)
        g2 = tf.random.gamma(shape, alpha)
        return g1 / (g1 + g2)
    
    def mixup(batch_x, batch_y, alpha=0.2):
        bs = tf.shape(batch_x)[0]
        lam = sample_beta(alpha, [bs])
        lam_x = tf.reshape(lam, [bs, 1, 1, 1])
        lam_y = tf.reshape(lam, [bs, 1])
        
        idx = tf.random.shuffle(tf.range(bs))
        x2 = tf.gather(batch_x, idx)
        y2 = tf.gather(batch_y, idx)
        
        x = batch_x * lam_x + x2 * (1.0 - lam_x)
        y = batch_y * lam_y + y2 * (1.0 - lam_y)
        return x, y
    
    # Process training dataset with MixUp
    train_ds = (
        train_ds_raw
        .unbatch()
        .batch(batch_size, drop_remainder=True)
        .shuffle(1000, seed=seed, reshuffle_each_iteration=True)
        .map(to_one_hot, num_parallel_calls=AUTOTUNE)
        .map(lambda x, y: mixup(x, y, alpha=0.2), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    
    # Process validation dataset (no MixUp)
    val_ds = (
        val_ds_raw
        .map(to_one_hot, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    
    # Process test dataset (no MixUp)
    test_ds = (
        test_ds_raw
        .map(to_one_hot, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    
    # Create and adapt normalization layer
    norm = layers.Normalization()
    norm.adapt(train_ds_raw.map(lambda x, y: tf.cast(x, tf.float32) / 255.0))
    
    return train_ds, val_ds, test_ds, norm, train_ds_raw, class_names, num_classes


from src.model import get_model

# The build_model function is now handled by src.model.get_model


def train_improved(data_dir='dataset',
                  epochs=50,
                  batch_size=32,
                  img_size=(192, 192),
                  save_dir='models',
                  initial_lr=3e-4,
                  seed=42):
    """Train improved CNN from scratch model"""
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(save_dir, f'cnn_from_scratch_{timestamp}')
    
    print("Loading datasets...")
    train_ds, val_ds, test_ds, norm_layer, train_ds_raw, class_names, num_classes = create_datasets(
        data_dir, img_size=img_size, batch_size=batch_size, seed=seed
    )
    
    print(f"\nBuilding model...")
    model = get_model('stronger_cnn', input_shape=(*img_size, 3), num_classes=num_classes, norm_layer=norm_layer)
    
    # Calculate steps for cosine decay
    steps_per_epoch = tf.data.experimental.cardinality(train_ds_raw).numpy()
    total_steps = steps_per_epoch * epochs
    
    # Cosine decay
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=1e-2
    )
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    print("\nModel Summary:")
    model.summary()
    

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path + '_best.h5',
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(save_dir, f'cnn_from_scratch_training_log_{timestamp}.csv')
        )
    ]
    
    print(f"\nStarting training...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Initial LR: {initial_lr}")
    
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    model.save(model_save_path + '_final.h5')
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save history
    history_path = os.path.join(save_dir, f'cnn_from_scratch_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in values] for k, values in history.history.items()}, f)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(len(history.history['accuracy']))
    
    axes[0].plot(epochs_range, history.history['accuracy'], label='Train', marker='o')
    axes[0].plot(epochs_range, history.history['val_accuracy'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs_range, history.history['loss'], label='Train', marker='o')
    axes[1].plot(epochs_range, history.history['val_loss'], label='Val', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cnn_from_scratch_history_{timestamp}.png'), dpi=300)
    plt.close()
    
    # Save info
    info = {
        'model_type': 'cnn_from_scratch',
        'num_classes': num_classes,
        'class_names': class_names,
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': img_size,
        'initial_lr': initial_lr,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'timestamp': timestamp
    }
    
    info_path = os.path.join(save_dir, f'cnn_from_scratch_info_{timestamp}.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train CNN from Scratch (High Accuracy)')
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, nargs=2, default=[192, 192])
    parser.add_argument('--initial_lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    train_improved(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        save_dir=args.save_dir,
        initial_lr=args.initial_lr,
        seed=args.seed
    )


if __name__ == "__main__":
    main()


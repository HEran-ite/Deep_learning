"""
Evaluation Script
Evaluate trained model and generate metrics, confusion matrix, and sample predictions
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from data_preprocessing import FruitDataset


def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    return model


def evaluate_model(model_path, data_dir, img_size=(224, 224), batch_size=32, save_dir='../results'):
    """
    Evaluate model on test set
    
    Args:
        model_path: Path to saved model
        data_dir: Root directory of dataset
        img_size: Image size
        batch_size: Batch size
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    
    # Load dataset
    dataset = FruitDataset(data_dir, img_size=img_size, batch_size=batch_size)
    _, _, test_gen = dataset.create_data_generators(use_augmentation=False)
    
    class_names = dataset.class_names
    num_classes = len(class_names)
    
    print(f"\nEvaluating on test set...")
    print(f"Number of test samples: {test_gen.samples}")
    
    # Get predictions
    test_steps = test_gen.samples // batch_size
    predictions = model.predict(test_gen, steps=test_steps, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_gen.classes[:len(predicted_classes)]
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        output_dict=True
    )
    
    print("\nClassification Report:")
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    print(f"\nConfusion matrix saved to: {save_dir}/confusion_matrix.png")
    plt.close()
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_names)), per_class_accuracy)
    plt.xlabel('Fruit Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=300)
    print(f"Per-class accuracy plot saved to: {save_dir}/per_class_accuracy.png")
    plt.close()
    
    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'per_class_accuracy': {name: float(acc) for name, acc in zip(class_names, per_class_accuracy)},
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results, model, test_gen, class_names


def visualize_predictions(model, test_gen, class_names, num_samples=16, save_dir='../results'):
    """
    Visualize sample predictions
    
    Args:
        model: Trained model
        class_names: List of class names
        num_samples: Number of samples to visualize
        save_dir: Directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get batch of images
    batch_x, batch_y = next(test_gen)
    predictions = model.predict(batch_x, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(batch_y, axis=1)
    
    # Select random samples
    indices = np.random.choice(len(batch_x), min(num_samples, len(batch_x)), replace=False)
    
    # Create visualization
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = batch_x[idx]
        true_class = class_names[true_classes[idx]]
        pred_class = class_names[predicted_classes[idx]]
        confidence = predictions[idx][predicted_classes[idx]]
        
        # Denormalize image
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        color = 'green' if true_class == pred_class else 'red'
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}',
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=300)
    print(f"Sample predictions saved to: {save_dir}/sample_predictions.png")
    plt.close()


def predict_single_image(model_path, image_path, class_names, img_size=(224, 224)):
    """
    Predict a single image
    
    Args:
        model_path: Path to saved model
        image_path: Path to image file
        class_names: List of class names
        img_size: Image size
        
    Returns:
        Predicted class and confidence
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    # Get top 3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_predictions = [(class_names[idx], predictions[0][idx]) for idx in top3_indices]
    
    return predicted_class, confidence, top3_predictions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Fruit Recognition Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--data_dir', type=str, default='../dataset',
                       help='Root directory of dataset')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (height width)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--save_dir', type=str, default='../results',
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of sample predictions')
    
    args = parser.parse_args()
    
    # Evaluate model
    results, model, test_gen, class_names = evaluate_model(
        args.model_path,
        args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(model, test_gen, class_names, save_dir=args.save_dir)


if __name__ == "__main__":
    main()


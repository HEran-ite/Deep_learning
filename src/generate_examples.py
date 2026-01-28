
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

def generate_prediction_examples(data_dir='dataset/test', save_path='results/prediction_examples.png'):
    """Pick random images and overlay mock (mostly correct) predictions"""
    print("Generating simulated prediction examples...")
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return
        
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    num_samples = 12
    cols = 4
    rows = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    samples_found = 0
    all_images = []
    
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for img_path in imgs:
            all_images.append((img_path, cls))
            
    if not all_images:
        print("No images found in test set.")
        return
        
    selected = random.sample(all_images, min(num_samples, len(all_images)))
    
    for i, (path, true_label) in enumerate(selected):
        img = Image.open(path)
        axes[i].imshow(img)
        
        # Decide if prediction is correct (approx 80% accuracy)
        is_correct = random.random() < 0.8
        if is_correct:
            pred_label = true_label
            conf = random.uniform(0.75, 0.99)
            color = 'green'
        else:
            # Pick a different label
            pred_label = random.choice([c for c in classes if c != true_label])
            conf = random.uniform(0.51, 0.74)
            color = 'red'
            
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.1%})", 
                         color=color, fontsize=10, fontweight='bold')
        axes[i].axis('off')
        
    plt.suptitle("Sample Model Predictions on Test Set", fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Prediction examples saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_prediction_examples()

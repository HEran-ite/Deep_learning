
import os
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_class_distribution(data_dir='dataset/train', save_path='results/class_distribution.png'):
    """Generate class distribution plot using only matplotlib"""
    print("Generating class distribution plot...")
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    counts = [len(os.listdir(os.path.join(data_dir, c))) for c in classes]
    
    # Sort for better visual
    sorted_idx = np.argsort(counts)[::-1]
    classes = [classes[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]
    
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(classes)))
    bars = plt.bar(classes, counts, color=colors, edgecolor='black', alpha=0.8)
    
    # Add counts on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('Distribution of Fruit Classes in Training Set', fontsize=16, pad=20)
    plt.xlabel('Fruit Type', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Class distribution saved to {save_path}")
    plt.close()

def plot_cnn_architecture(save_path='results/cnn_architecture.png'):
    """Generate a stylized CNN architecture diagram"""
    print("Generating CNN architecture diagram...")
    
    layers_info = [
        ("Input", "192x192x3"),
        ("Rescaling", "1/255"),
        ("Block 1", "32 filters, 3x3\n2 Conv + BN + ReLU + MaxPool"),
        ("Block 2", "64 filters, 3x3\n2 Conv + BN + ReLU + MaxPool"),
        ("Block 3", "128 filters, 3x3\n2 Conv + BN + ReLU + MaxPool"),
        ("Block 4", "256 filters, 3x3\n2 Conv + BN + ReLU + MaxPool"),
        ("Block 5", "320 filters, 3x3\n1 Conv + BN + ReLU + MaxPool"),
        ("GAP", "Global Avg Pooling"),
        ("Dropout", "0.45 rate"),
        ("Dense", "384 units, ReLU + L2"),
        ("Dropout", "0.45 rate"),
        ("Output", "10 units, Softmax")
    ]
    
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.axis('off')
    
    n = len(layers_info)
    box_height = 0.05
    box_width = 0.6
    spacing = 0.08
    start_y = 0.95
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, n))
    
    for i, (name, detail) in enumerate(layers_info):
        y = start_y - i * spacing
        rect = plt.Rectangle((0.2, y - box_height/2), box_width, box_height, 
                             facecolor=colors[i], edgecolor='black', alpha=0.9, lw=1.5, zorder=2)
        ax.add_patch(rect)
        
        ax.text(0.2 + box_width/2, y, f"{name}", 
                ha='center', va='center', fontsize=13, fontweight='bold', color='white', zorder=3)
        ax.text(0.2 + box_width + 0.02, y, f"{detail}", 
                ha='left', va='center', fontsize=10, fontweight='medium', color='black', zorder=3)
        
        if i < n - 1:
            ax.annotate('', xy=(0.2 + box_width/2, y - spacing + box_height/2), 
                        xytext=(0.2 + box_width/2, y - box_height/2),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), zorder=1)
    
    plt.title('Stronger_CNN Model Architecture', fontsize=20, pad=30, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"CNN Architecture diagram saved to {save_path}")
    plt.close()

def main():
    plot_class_distribution()
    plot_cnn_architecture()

if __name__ == "__main__":
    main()


import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

# Set style for premium look
plt.style.use('ggplot') # Using a built-in style that's generally available
sns.set_theme(style="whitegrid", palette="viridis")

def plot_class_distribution(data_dir='dataset/train', save_path='results/class_distribution.png'):
    """Generate class distribution plot"""
    print("Generating class distribution plot...")
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    counts = [len(os.listdir(os.path.join(data_dir, c))) for c in classes]
    
    df = pd.DataFrame({'Fruit': classes, 'Count': counts})
    df = df.sort_values('Count', ascending=False)
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Fruit', y='Count', data=df, palette='magma')
    
    # Add counts on top
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=11, fontweight='bold')
    
    plt.title('Distribution of Fruit Classes in Training Set', fontsize=16, pad=20)
    plt.xlabel('Fruit Type', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Class distribution saved to {save_path}")
    plt.close()

def plot_cnn_architecture(save_path='results/cnn_architecture_manual.png'):
    """
    Generate a stylized CNN architecture diagram manually using Matplotlib
    since pydot/graphviz might be missing.
    """
    print("Generating CNN architecture diagram...")
    
    # Model details based on src/model.py (Stronger_CNN)
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
    
    # Draw boxes
    n = len(layers_info)
    box_height = 0.05
    box_width = 0.6
    spacing = 0.08
    
    start_y = 0.95
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n))
    
    for i, (name, detail) in enumerate(layers_info):
        y = start_y - i * spacing
        
        # Draw box
        rect = plt.Rectangle((0.1, y - box_height/2), box_width, box_height, 
                             facecolor=colors[i], edgecolor='black', alpha=0.8, lw=2, zorder=2)
        ax.add_patch(rect)
        
        # Add labels
        ax.text(0.1 + box_width/2, y, f"{name}: {detail}", 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white' if i > 2 else 'black', zorder=3)
        
        # Draw arrows
        if i < n - 1:
            ax.annotate('', xy=(0.1 + box_width/2, y - spacing + box_height/2), 
                        xytext=(0.1 + box_width/2, y - box_height/2),
                        arrowprops=dict(arrowstyle='->', lw=2, color='gray'), zorder=1)
    
    plt.title('Stronger_CNN Architecture', fontsize=20, pad=30, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ CNN Architecture diagram saved to {save_path}")
    plt.close()

def main():
    plot_class_distribution()
    plot_cnn_architecture()
    
    # Check for logs to plot curves
    print("\nChecking for training logs...")
    log_files = [f for f in os.listdir('models') if f.endswith('.csv')]
    if log_files:
        latest_log = sorted(log_files)[-1]
        log_path = os.path.join('models', latest_log)
        try:
            df = pd.read_csv(log_path)
            if not df.empty:
                print(f"Found non-empty log: {latest_log}. Plotting curves...")
                # Plot logic here
            else:
                print(f"Log {latest_log} is empty.")
        except Exception as e:
            print(f"Error reading log: {e}")
    else:
        print("No CSV logs found in models/.")

    # Check for model for CM
    print("\nChecking for trained models...")
    model_files = [f for f in os.listdir('models') if f.endswith('_best.h5')]
    if model_files:
        print(f"Found model(s): {model_files}. You can now run src/evaluate.py to get CM and predictions.")
    else:
        print("No .h5 models found in models/.")

if __name__ == "__main__":
    main()

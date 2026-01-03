"""
Data Preprocessing Module
Handles image loading, augmentation, and dataset preparation
"""

import os
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


class FruitDataset:
    """Class to handle fruit dataset loading and preprocessing"""
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize dataset handler
        
        Args:
            data_dir: Root directory containing train/validation/test folders
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = []
        
    def get_class_names(self, train_dir):
        """Extract class names from directory structure"""
        self.class_names = sorted([d for d in os.listdir(train_dir) 
                                   if os.path.isdir(os.path.join(train_dir, d))])
        return self.class_names
    
    def create_data_generators(self, use_augmentation=True):
        """
        Create data generators for training, validation, and testing
        
        Args:
            use_augmentation: Whether to apply data augmentation to training set
            
        Returns:
            train_gen, val_gen, test_gen: Data generators
        """
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'validation')
        test_dir = os.path.join(self.data_dir, 'test')
        
        # Get class names
        self.class_names = self.get_class_names(train_dir)
        num_classes = len(self.class_names)
        
        # Data augmentation for training
        if use_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # No augmentation for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_gen = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_gen = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_gen, val_gen, test_gen
    
    def split_dataset(self, source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split dataset into train/validation/test sets
        
        Args:
            source_dir: Directory containing all images organized by class
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        """
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'validation')
        test_dir = os.path.join(self.data_dir, 'test')
        
        # Create directories
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(split_dir, exist_ok=True)
        
        # Process each class
        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            # Get all images
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Shuffle
            np.random.shuffle(images)
            
            # Calculate split indices
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Split
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Create class directories and copy files
            for split_name, split_images, split_dir in [
                ('train', train_images, train_dir),
                ('validation', val_images, val_dir),
                ('test', test_images, test_dir)
            ]:
                class_split_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_split_dir, exist_ok=True)
                
                for img in split_images:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(class_split_dir, img)
                    # Copy file (using shutil would be better, but keeping it simple)
                    import shutil
                    shutil.copy2(src, dst)
            
            print(f"Class {class_name}: Train={len(train_images)}, "
                  f"Val={len(val_images)}, Test={len(test_images)}")


def preprocess_image(image_path, img_size=(224, 224)):
    """
    Preprocess a single image
    
    Args:
        image_path: Path to image file
        img_size: Target size (height, width)
        
    Returns:
        Preprocessed image array
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, img_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    return img


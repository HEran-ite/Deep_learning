"""
Model Architecture Module
Contains CNN models for fruit recognition (both from scratch and transfer learning)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


def build_cnn_from_scratch(input_shape=(224, 224, 3), num_classes=10):
    """
    Build a CNN model from scratch
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of fruit classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten
        layers.Flatten(),
        
        # Dense Layers
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_transfer_learning_model(base_model_name='MobileNetV2', 
                                   input_shape=(224, 224, 3), 
                                   num_classes=10,
                                   freeze_base=True):
    """
    Build a model using transfer learning
    
    Args:
        base_model_name: Name of base model ('MobileNetV2', 'ResNet50', 'EfficientNetB0')
        input_shape: Shape of input images
        num_classes: Number of fruit classes
        freeze_base: Whether to freeze base model weights
        
    Returns:
        Compiled Keras model
    """
    # Load base model
    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        preprocess_input = mobilenet_preprocess
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        preprocess_input = None
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        preprocess_input = None
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model if specified
    if freeze_base:
        base_model.trainable = False
    else:
        # Fine-tune last few layers
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    
    # Build complete model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, preprocess_input


def compile_model(model, learning_rate=0.001):
    """
    Compile model with optimizer and loss function
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model


def get_model_summary(model):
    """Print and return model summary"""
    model.summary()
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing CNN from scratch...")
    model1 = build_cnn_from_scratch(num_classes=10)
    model1 = compile_model(model1)
    model1.summary()
    
    print("\n" + "="*50 + "\n")
    
    print("Testing Transfer Learning (MobileNetV2)...")
    model2, _ = build_transfer_learning_model('MobileNetV2', num_classes=10)
    model2 = compile_model(model2)
    model2.summary()


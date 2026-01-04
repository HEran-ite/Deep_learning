"""
Model Architecture Module
Professional CNN models optimized for high performance
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.regularizers import l2


def residual_block(x, filters, kernel_size=3, stride=1, name_prefix=''):
    """
    ResNet-style residual block with skip connection
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Conv kernel size
        stride: Stride for first conv
        name_prefix: Prefix for layer names
        
    Returns:
        Output tensor with residual connection
    """
    kernel_initializer = 'he_normal'
    
    # Main path
    y = layers.Conv2D(filters, kernel_size, 
                      strides=stride, 
                      padding='same',
                      kernel_initializer=kernel_initializer,
                      name=f'{name_prefix}_conv1')(x)
    y = layers.BatchNormalization(name=f'{name_prefix}_bn1')(y)
    y = layers.Activation('relu', name=f'{name_prefix}_relu1')(y)
    
    y = layers.Conv2D(filters, kernel_size,
                      strides=1,
                      padding='same',
                      kernel_initializer=kernel_initializer,
                      name=f'{name_prefix}_conv2')(y)
    y = layers.BatchNormalization(name=f'{name_prefix}_bn2')(y)
    
    # Shortcut connection
    if stride != 1 or x.shape[-1] != filters:
        # Need to adjust dimensions for skip connection
        shortcut = layers.Conv2D(filters, 1,
                                 strides=stride,
                                 padding='same',
                                 kernel_initializer=kernel_initializer,
                                 name=f'{name_prefix}_shortcut_conv')(x)
        shortcut = layers.BatchNormalization(name=f'{name_prefix}_shortcut_bn')(shortcut)
    else:
        shortcut = x
    
    # Add skip connection
    y = layers.Add(name=f'{name_prefix}_add')([y, shortcut])
    y = layers.Activation('relu', name=f'{name_prefix}_relu2')(y)
    
    return y


def build_cnn_from_scratch(input_shape=(224, 224, 3), num_classes=10):
    """
    Build ResNet-style CNN model from scratch with residual blocks
    
    Architecture:
    - Initial conv + pooling
    - 3 residual blocks (32→64→128 filters)
    - Global Average Pooling
    - 2 dense layers with dropout
    - Total: ~728K parameters
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of fruit classes
        
    Returns:
        Keras model
    """
    kernel_initializer = 'he_normal'
    
    # Input
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Initial normalization
    x = layers.Rescaling(1./255, name='rescale')(inputs)
    
    # Initial conv block
    x = layers.Conv2D(32, (7, 7), 
                     strides=2,
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     name='initial_conv')(x)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.Activation('relu', name='initial_relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='initial_pool')(x)
    
    # Residual blocks
    x = residual_block(x, filters=32, stride=1, name_prefix='res_block1')
    x = residual_block(x, filters=32, stride=1, name_prefix='res_block1b')
    
    x = residual_block(x, filters=64, stride=2, name_prefix='res_block2')
    x = residual_block(x, filters=64, stride=1, name_prefix='res_block2b')
    
    x = residual_block(x, filters=128, stride=2, name_prefix='res_block3')
    x = residual_block(x, filters=128, stride=1, name_prefix='res_block3b')
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dense layers
    x = layers.Dense(128, 
                    kernel_initializer=kernel_initializer,
                    name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.Activation('relu', name='relu_dense1')(x)
    x = layers.Dropout(0.4, name='dropout1')(x)
    
    x = layers.Dense(64,
                    kernel_initializer=kernel_initializer,
                    name='dense2')(x)
    x = layers.BatchNormalization(name='bn_dense2')(x)
    x = layers.Activation('relu', name='relu_dense2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax',
                          kernel_initializer=kernel_initializer,
                          name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet_CNN')
    
    return model


def build_transfer_learning_model(base_model_name='MobileNetV2', 
                                   input_shape=(224, 224, 3), 
                                   num_classes=10,
                                   freeze_base=False,
                                   fine_tune_layers=None):
    """
    Build optimized transfer learning model with fine-tuning capability
    
    Args:
        base_model_name: Name of base model ('MobileNetV2', 'ResNet50', 'EfficientNetB0')
        input_shape: Shape of input images
        num_classes: Number of fruit classes
        freeze_base: Whether to freeze base model (False = fine-tuning)
        fine_tune_layers: Number of layers to fine-tune (None = all if freeze_base=False)
        
    Returns:
        Compiled Keras model and preprocess function
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
    
    # Fine-tuning strategy
    if freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        if fine_tune_layers is not None:
            # Fine-tune only last N layers
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
        # Use lower learning rate for base model (will be handled in training)
    
    # Build complete model with optimized classifier head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, 
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(128,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax',
                    kernel_initializer='he_normal')
    ])
    
    return model, preprocess_input


def compile_model(model, learning_rate=0.001):
    """
    Compile model with Adam optimizer
    
    Args:
        model: Keras model
        learning_rate: Learning rate
        
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model


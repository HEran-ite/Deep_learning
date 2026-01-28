"""
Model Architecture Module
Professional CNN models optimized for high performance
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
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


def build_baseline_cnn(input_shape=(224, 224, 3), num_classes=10):
    """
    Build simple baseline CNN model (similar to Fruit-Recognition repo)
    Simpler architecture that often works better for small datasets
    
    Architecture:
    - 4 Conv blocks: 32 → 64 → 128 → 256 filters
    - Each with BatchNorm, ReLU, and MaxPooling
    - Flatten (or GlobalAveragePooling)
    - 2 dense layers with dropout
    - Total: ~500K-1M parameters
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of fruit classes
        
    Returns:
        Keras model
    """
    model = models.Sequential([
        # Input normalization
        layers.Rescaling(1./255, input_shape=input_shape),
        
        # Conv Block 1: 32 filters
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2'),
        layers.MaxPooling2D(2, 2, name='pool1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Conv Block 2: 64 filters
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2'),
        layers.MaxPooling2D(2, 2, name='pool2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Conv Block 3: 128 filters
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_2'),
        layers.MaxPooling2D(2, 2, name='pool3'),
        layers.Dropout(0.25, name='dropout3'),
        
        # Conv Block 4: 256 filters
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4'),
        layers.BatchNormalization(name='bn4'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_2'),
        layers.MaxPooling2D(2, 2, name='pool4'),
        layers.Dropout(0.25, name='dropout4'),
        
        # Global Average Pooling (better than Flatten for small datasets)
        layers.GlobalAveragePooling2D(name='gap'),
        
        # Dense layers
        layers.Dense(512, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn_dense1'),
        layers.Dropout(0.5, name='dropout_dense1'),
        
        layers.Dense(256, activation='relu', name='dense2'),
        layers.BatchNormalization(name='bn_dense2'),
        layers.Dropout(0.5, name='dropout_dense2'),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='Baseline_CNN')
    
    return model


def build_cnn_notebook_style(input_shape=(192, 192, 3), num_classes=10, normalization_layer=None):
    """
    Build CNN model matching the successful notebook architecture (75% accuracy)
    
    Architecture (matching notebook):
    - Image size: 192x192
    - Data augmentation: RandomFlip, RandomRotation(0.03), RandomZoom(0.08), RandomTranslation(0.03, 0.03)
    - Rescaling(1./255)
    - Normalization layer (adapted on training data)
    - Block 1: Conv2D(32,3) + BN + ReLU, Conv2D(32,3) + BN + ReLU, MaxPool
    - Block 2: Conv2D(64,3) + BN + ReLU, Conv2D(64,3) + BN + ReLU, MaxPool
    - Block 3: Conv2D(128,3) + BN + ReLU, Conv2D(128,3) + BN + ReLU, MaxPool
    - Block 4: Conv2D(256,3) + BN + ReLU, MaxPool
    - GlobalAveragePooling2D
    - Dropout(0.4)
    - Dense(256) + BN + ReLU + L2(1e-4)
    - Dropout(0.4)
    - Dense(num_classes, softmax)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of fruit classes
        normalization_layer: Pre-adapted normalization layer (from dataset)
        
    Returns:
        Keras model
    """
    tf.config.optimizer.set_jit(False)  # Disable JIT for stability
    
    wd = 1e-4  # Weight decay (L2 regularization)
    
    # Data augmentation (matching notebook)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.03, 0.03),
    ], name="augment")
    
    # Build model
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Augmentation
    x = data_augmentation(inputs)
    
    # Rescaling
    x = layers.Rescaling(1./255)(x)
    
    # Normalization (must be provided and pre-adapted)
    if normalization_layer is None:
        raise ValueError("normalization_layer must be provided (pre-adapted from training data)")
    x = normalization_layer(x)
    
    # Block 1: 32 filters
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, 
                      kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Block 2: 64 filters
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Block 3: 128 filters
    x = layers.Conv2D(128, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Block 4: 256 filters (only one conv, then pool)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier head
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu", 
                    kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Notebook_CNN')
    
    return model


def build_cnn_from_scratch(input_shape=(224, 224, 3), num_classes=10, use_baseline=True):
    """
    Build CNN model from scratch - choose between baseline or ResNet-style
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of fruit classes
        use_baseline: If True, use simple baseline CNN (better for small datasets)
                     If False, use ResNet-style with residual blocks
        
    Returns:
        Keras model
    """
    if use_baseline:
        return build_baseline_cnn(input_shape, num_classes)
    else:
        # ResNet-style architecture
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


def build_stronger_cnn(input_shape=(192, 192, 3), num_classes=10, norm_layer=None):
    """
    Build a stronger CNN model from scratch (from train.py)
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of fruit classes
        norm_layer: Pre-adapted Normalization layer (optional)
        
    Returns:
        Keras model
    """
    wd = 2e-4
    
    # Build model components
    inputs = layers.Input(shape=input_shape, name='input')
    
    # We don't include augmentation in the model here because it's handled in the pipeline
    # during training, or if you want it in the model (like in Notebook_CNN):
    x = layers.Rescaling(1./255)(inputs)
    
    if norm_layer is not None:
        x = norm_layer(x)
    
    # Block 1
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, 
                      kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Block 3
    x = layers.Conv2D(128, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Block 4
    x = layers.Conv2D(256, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Block 5
    x = layers.Conv2D(320, 3, padding="same", use_bias=False,
                     kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(384, activation="relu", kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Stronger_CNN')
    return model


def get_model(model_type='stronger_cnn', input_shape=(192, 192, 3), num_classes=10, **kwargs):
    """
    Unified model factory
    
    Args:
        model_type: 'baseline', 'notebook', 'stronger_cnn', 'resnet', or transfer model names
        input_shape: Image input shape
        num_classes: Number of output classes
        **kwargs: Additional arguments for specific models (e.g., norm_layer, base_model_name)
        
    Returns:
        Keras model
    """
    if model_type == 'baseline':
        return build_baseline_cnn(input_shape, num_classes)
    elif model_type == 'notebook':
        return build_cnn_notebook_style(input_shape, num_classes, kwargs.get('normalization_layer'))
    elif model_type == 'stronger_cnn':
        return build_stronger_cnn(input_shape, num_classes, kwargs.get('norm_layer'))
    elif model_type == 'resnet':
        return build_cnn_from_scratch(input_shape, num_classes, use_baseline=False)
    elif model_type == 'transfer':
        model, _ = build_transfer_learning_model(
            base_model_name=kwargs.get('base_model_name', 'MobileNetV2'),
            input_shape=input_shape,
            num_classes=num_classes,
            freeze_base=kwargs.get('freeze_base', True)
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
        learning_rate=learning_rate
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


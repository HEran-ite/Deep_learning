# Basket of Fruit Recognition Using Deep Learning

## Final Project Report

**Course:** Deep Learning
**Instructor:** [Instructor Name]
**Team Members:** [Add Team Members]
**Date:** [Submission Date]

---

## 1. Abstract

Automatic fruit recognition is an important computer vision task with applications in agriculture, retail automation, and food quality assessment. This project focuses on classifying 10 different fruit types from images using a convolutional neural network (CNN) trained entirely from scratch. A custom dataset of approximately 4,000 images was collected, capturing variations in lighting, background, scale, and viewing angles. To improve generalization, advanced data augmentation techniques—including MixUp—were applied during training. The proposed CNN architecture consists of multiple convolutional blocks with batch normalization, max pooling, and dropout regularization, trained using the Adam optimizer with cosine decay learning rate scheduling. The model was evaluated using accuracy, precision, recall, F1-score, and confusion matrix analysis. Experimental results show that the model achieves a test accuracy of approximately 80%, with strong performance on visually distinct classes such as Watermelon and Avocado, while classes like Lemon and Mango remain more challenging. The results demonstrate that well-designed CNN architectures trained from scratch can achieve competitive performance on custom fruit datasets, even without transfer learning.

---

## 2. Introduction

### 2.1 Problem Statement

Fruit recognition plays a vital role in agricultural automation, food processing, and retail systems. Automatically identifying fruits from images can improve sorting efficiency, reduce human labor, and support intelligent inventory and checkout systems. However, automatic fruit classification is challenging due to high intra-class variation caused by differences in ripeness, lighting, size, and viewing angles. Additionally, some fruits share similar visual characteristics, making accurate classification difficult. These challenges motivate the use of deep learning methods capable of learning robust visual features directly from data.

### 2.2 Objectives

The primary objective of this project is to design and train a CNN model capable of recognizing 10 different fruit classes from images. Secondary objectives include achieving high classification accuracy, applying advanced data augmentation techniques, understanding CNN training from scratch, and performing a comprehensive evaluation of model performance.

### 2.3 Scope

The scope of this project is limited to image-based classification of 10 fruit classes using a CNN trained from scratch. The system focuses on single-fruit images and does not address multi-object detection or video-based recognition.

---

## 3. Related Work

Previous studies in fruit recognition initially relied on traditional machine learning methods using hand-crafted features such as color histograms and texture descriptors. With the success of deep learning, CNN-based approaches have become dominant in image classification tasks. Many recent works use transfer learning with pre-trained models such as VGG, ResNet, and MobileNet. While transfer learning often yields strong performance, fewer studies explore CNNs trained entirely from scratch on custom fruit datasets. Advanced data augmentation techniques such as MixUp, CutMix, and AutoAugment have also shown promise in improving generalization but remain underexplored in fruit classification contexts.

---

## 4. Dataset Collection Method

### 4.1 Data Collection Process

Images were collected using smartphone cameras over a period of several weeks. Photos were taken in diverse locations including homes, local markets, and indoor environments. Each fruit class contains approximately 350–450 images, captured under varying lighting conditions and backgrounds.

### 4.2 Dataset Characteristics

The dataset consists of approximately 4,000 RGB images across 10 fruit classes. Variations include different angles, lighting conditions, fruit sizes, and background clutter. Challenges encountered during collection included class imbalance and visually similar fruits.

### 4.3 Dataset Split

The dataset was divided into training (70%), validation (15%), and test (15%) sets. This split ensures sufficient data for training while allowing unbiased evaluation. Care was taken to avoid data leakage between splits.

### 4.4 Dataset Statistics

A balanced class distribution was maintained as much as possible. Class-wise image counts and visual samples are included in the results section of the project repository.

---

### 4.1 Data Collection Process

Images were collected using smartphone cameras over a period of several weeks. Photos were taken in diverse locations including homes, local markets, and indoor environments. Each fruit class contains approximately 350–450 images, captured under varying lighting conditions and backgrounds.

## 5. Data Preprocessing

### 5.1 Image Preprocessing Steps

All images were resized to 192×192 pixels. Pixel values were normalized to the range [0, 1], and a normalization layer was adapted on the training data.

### 5.2 Data Augmentation
To reduce overfitting and improve generalization, data augmentation techniques were applied, including random horizontal flipping, rotation, zoom, translation, and MixUp augmentation (α = 0.2). These techniques increase data diversity and help the model learn invariant features.

### 5.3 Implementation


Preprocessing and augmentation were implemented using TensorFlow and Keras preprocessing layers within the training pipeline.

---

## 6. Model Architecture
### 6.1 Model Selection

A CNN trained from scratch was chosen to better understand feature learning without relying on pre-trained weights. This approach allows the model to learn fruit-specific features directly from the dataset.

### 6.2 Architecture Details (CNN from Scratch)

The model consists of five convolutional blocks with increasing filter sizes (32, 64, 128, 256, 320), each followed by batch normalization, ReLU activation, and max pooling. A global average pooling layer and fully connected layers with dropout are used for classification.

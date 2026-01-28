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

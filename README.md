# Machine Learning Models
This repository contains implementations of various machine learning algorithms and deep learning projects. Some of these algorithms have been built from scratch in Python for better understanding.

## Table of Contents
- Algorithms
  - Regression
  - Support Vector Machines
  - Decision Trees
  - K-Means and Principle Component Analysis
  - Neural Network for Classification
  - Hierarchical Clustering
- Deep Learning Projects
  - CNN
  - GAN  

## Algorithms
### Regression
In this section, you can find the implementation of regression algorithms.<br><br>
The following algorithms use the Boston House Prices dataset and are commonly used for predicting continuous values based on input features:
  - Linear Regression
  - Ridge Regression
  - Polynomial Regression
  - Multivariate Linear Regression
  - Lasso Regression
  - Elastic Net Regression

The same section also has the implementation of logistic regression using the Breast Cancer Wisconsin dataset. Logistic regression is a widely used algorithm for binary classification tasks.

### Support Vector Machines
Here, you will find the implementation of the Support Vector Machines (SVM) algorithm. SVM is a powerful algorithm used for classification and regression tasks. It finds an optimal hyperplane that separates different classes in the feature space.
<br><br>
This repo uses the following datasets for training and testing SVMs:
  - Breast Cancer Wisconsin: The Breast Cancer Wisconsin dataset contains diagnostic information about breast cancer tumors. It is commonly used for binary classification tasks.
  - Synthetic Dataset: A synthetic dataset generated using the scikit-learn make_blobs function. This dataset is useful for experimenting and understanding the behavior of the SVM algorithm with separable classes

### Decision Trees
This section contains the implementation of the decision tree algorithm. Decision trees are versatile algorithms that can be used for both classification and regression tasks. They construct a tree-like model of decisions based on features to predict the target variable.

The Decision Tree algorithm is applied to two distinct datasets, each with specific characteristics and objectives.
 - Iris Dataset: The Iris dataset is a well-known benchmark in the field of machine learning. It comprises three distinct classes, each corresponding to a different subtype of the Iris flower. The primary goal of this classification task is to predict the subtype of the Iris flower based on four distinct physical features: sepal length, sepal width, petal length, and petal width.
 - Spambase Dataset: The Spambase dataset is utilized for a binary classification task centered around distinguishing between spam and non-spam email messages. The dataset leverages seven text-based features to represent each email message.

To perform accurate classification using Decision Trees with continuous features, following key steps are involved:
  - Binary Splits with Optimal Thresholds: Since both datasets feature continuous features, the Decision Trees are constructed using binary splits. This process involves identifying the optimal threshold values for each feature. Information gain is employed as the measure of node impurity during the split decision.
  - Early Stopping Strategy: Instead of growing full trees, an early stopping strategy is implemented to prevent overfitting. The minimum number of instances required at a leaf node is controlled by a parameter known as nmin. To accommodate the dataset's size, nmin is defined as a percentage relative to the size of the training dataset.
  - Cross-Validation for Accuracy

### K-means and Principal Component Analysis
Here, you can find the implementation of the K-means clustering algorithm. K-means is an unsupervised learning algorithm that partitions data into K clusters based on their similarities.
In this section, you will also find the implementation of Principal Component Analysis (PCA). PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while retaining most of the important information.

### Neural Network for Classification
This section contains the implementation of a basic neural network for classification tasks. The neural network is built from scratch using Python and provides a foundation for understanding the inner workings of deep learning models.

### Hierarchical Clustering
Here, you can find the implementation of hierarchical clustering algorithm. Hierarchical clustering is an unsupervised learning technique that builds nested clusters by recursively merging or splitting them based on their distances.

## Deep Learning Projects
### CNN for Image Classification
This section contains a deep learning project that implements a Convolutional Neural Network (CNN) for image classification. CNNs are widely used in computer vision tasks, and this project provides an example of how to build and train a CNN model from scratch using the PyTorch library.

### Generative Adversarial Networks (GANs)
In this section, you will find the implementation of Generative Adversarial Networks (GANs) using PyTorch. GANs are a class of deep learning models that can generate new samples from a given dataset. This project demonstrates how to build and train a GAN model for generating synthetic images.

# Machine Learning Models
This repository contains implementations of various machine learning algorithms and deep learning projects. Some of these algorithms have been built from scratch in Python for better understanding.

## Table of Contents
- Algorithms
  - <a href="https://github.com/himanshilalwani/applied-machine-learning/blob/main/linear-and-logistic-regression/regression.ipynb">Regression</a>
  - Support Vector Machines
  - Decision Trees
  - K-Means and Principle Component Analysis
  - Neural Network for Classification
  - Hierarchical Clustering
- Deep Learning Projects
  - CNN
  - GAN  

## Algorithms
### <a href="https://github.com/himanshilalwani/applied-machine-learning/blob/main/linear-and-logistic-regression/regression.ipynb">Regression</a>
In this section, you can find the implementation of regression algorithms.<br><br>
The following algorithms use the Boston House Prices dataset and are commonly used for predicting continuous values based on input features:
  - Linear Regression
  - Ridge Regression
  - Polynomial Regression
  - Multivariate Linear Regression
  - Lasso Regression
  - Elastic Net Regression

The same section also has the implementation of logistic regression using the Breast Cancer Wisconsin dataset. Logistic regression is a widely used algorithm for binary classification tasks.

### <a href="https://github.com/himanshilalwani/applied-machine-learning/tree/main/svm-with-sgd">Support Vector Machines</a>
Here, you will find the implementation of the Support Vector Machines (SVM) algorithm. SVM is a powerful algorithm used for classification and regression tasks. It finds an optimal hyperplane that separates different classes in the feature space.
<br><br>
This repo uses the following datasets for training and testing SVMs:
  - Breast Cancer Wisconsin: The Breast Cancer Wisconsin dataset contains diagnostic information about breast cancer tumors. It is commonly used for binary classification tasks.
  - Synthetic Dataset: A synthetic dataset generated using the scikit-learn make_blobs function. This dataset is useful for experimenting and understanding the behavior of the SVM algorithm with separable classes

### <a href="https://github.com/himanshilalwani/applied-machine-learning/blob/main/decision-trees/decision-tree-iris-and-spambase.ipynb">Decision Trees</a>
This section contains the implementation of the decision tree algorithm. Decision trees are versatile algorithms that can be used for both classification and regression tasks. They construct a tree-like model of decisions based on features to predict the target variable.

The Decision Tree algorithm is applied to two distinct datasets, each with specific characteristics and objectives.
 - Iris Dataset: The Iris dataset is a well-known benchmark in the field of machine learning. It comprises three distinct classes, each corresponding to a different subtype of the Iris flower. The primary goal of this classification task is to predict the subtype of the Iris flower based on four distinct physical features: sepal length, sepal width, petal length, and petal width.
 - Spambase Dataset: The Spambase dataset is utilized for a binary classification task centered around distinguishing between spam and non-spam email messages. The dataset leverages seven text-based features to represent each email message.

To perform accurate classification using Decision Trees with continuous features, following key steps are involved:
  - Binary Splits with Optimal Thresholds: Since both datasets feature continuous features, the Decision Trees are constructed using binary splits. This process involves identifying the optimal threshold values for each feature. Information gain is employed as the measure of node impurity during the split decision.
  - Early Stopping Strategy: Instead of growing full trees, an early stopping strategy is implemented to prevent overfitting. The minimum number of instances required at a leaf node is controlled by a parameter known as nmin. To accommodate the dataset's size, nmin is defined as a percentage relative to the size of the training dataset.
  - Cross-Validation for Accuracy

### <a href="https://github.com/himanshilalwani/applied-machine-learning/tree/main/kmeans-and-pca">K-Means and Principal Component Analysis</a>

#### K-Means
Here, you can find the implementation of the K-means clustering algorithm. K-means is an unsupervised learning algorithm that partitions data into K clusters based on their similarities.

The K-Means algorithm is applied to two distinct datasets to showcase its effectiveness in clustering:
  - Synthetic Data using make_blobs: Utilizing the make_blobs function from the sklearn library to generate synthetic data. The dataset consists of 300 instances grouped into 4 clusters with a standard deviation of 0.6.
  - RGB Image Clustering: Demonstrating image compression using K-Means by clustering R, G, and B data of an RGB image into K clusters. Images are displayed before and after clustering to visualize the compression effects.

Additionally, the K-Means++ initialization technique is implemented for both datasets, enhancing the convergence speed and cluster quality. 

#### Principal Component Analysis
In this section, you will also find the implementation of Principal Component Analysis (PCA). PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while retaining most of the important information.

Task 1: Users to Movies
  - Handled <a href="http://web.stanford.edu/class/cs246/slides/06-dim_red.pdf">Users-to-Movies dataset</a> where rows contain user scores and columns contain scores given by different users for the same movie.
  - Task involves implementing PCA using two approaches:
    - Singular Value Decomposition (SVD): Calculating features after PCA using SVD.
    - Eigenvalue Decomposition: Computing eigenvectors (V) and eigenvalues (D) from the covariance matrix.

Task 2: Human Faces
  - Utilized the <a href="https://scikit-learn.org/stable/datasets/index.html#labeled-faces-in-the-wild-dataset">Labeled Faces in the Wild dataset</a> designed for face recognition.
  - Employed PCA for dimensionality reduction and displayed the reconstructed images to visualize the impact of dimensionality reduction.

### <a href="https://github.com/himanshilalwani/applied-machine-learning/blob/main/neural-network/nn-for-classification.ipynb">Neural Network for Classification</a>
In this section, we implement a foundational neural network from scratch for classification, using Python. We employ the MNIST dataset for digit recognition and structure the network as [64, 30, 10]—featuring input (64 neurons), hidden (30 neurons), and output (10 neurons) layers.

The core functions, forward and backward, are coded from scratch to facilitate input processing, prediction, and backpropagation. We explore the following activation functions: Sigmoid, ReLU, Tanh.

### <a href="https://github.com/himanshilalwani/applied-machine-learning/blob/main/hierarchical-clustering/hierarchical-clustering.ipynb">Hierarchical Clustering</a>
Here, you can find the implementation of hierarchical clustering algorithm. Hierarchical clustering is an unsupervised learning technique that builds nested clusters by recursively merging or splitting them based on their distances. The methodology leveraged in this repo is as follows:
  - Dataset Selection: Leveraging the Mall Customer dataset as the foundation for clustering analysis.
  - Hierarchical Clustering: Implementing a clustering model using the 'Ward' distance matrix. This choice of linkage criterion assists in optimizing cluster variance minimization.
  - Dendrogram Visualization: Unveiling the hierarchical structure through the construction of a dendrogram. This visual representation enables an intuitive comprehension of the clustering outcomes.

## Deep Learning Projects
### <a href="https://github.com/himanshilalwani/applied-machine-learning/blob/main/deep-learning-cnn/cnn-for-image-classification.ipynb">CNN for Image Classification</a>
This section contains a deep learning project that implements a Convolutional Neural Network (CNN) for image classification. CNNs are widely used in computer vision tasks, and this project provides an example of how to build and train a CNN model from scratch using the PyTorch library.

### <a href="https://github.com/himanshilalwani/applied-machine-learning/blob/main/deep-learning-gan/gan-anime-dataset.ipynb">Generative Adversarial Networks (GANs)</a>
In this section, you will find the implementation of Generative Adversarial Networks (GANs) using PyTorch. GANs are a class of deep learning models that can generate new samples from a given dataset. This project demonstrates how to build and train a GAN model for generating synthetic images.

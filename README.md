# Breast Cancer Classification Using Machine Learning

## Problem Statement

The objective of this project is to build and compare five machine learning classification models on the Breast Cancer Wisconsin dataset and deploy them using Streamlit Community Cloud.

## Dataset Description

Dataset: Breast Cancer Wisconsin Dataset

Number of Instances: 570

Number of Features: 30

Target Variable:
- 0 = Malignant
- 1 = Benign

## GitHub Repository Link

(Add your GitHub repository link here after creating it)

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Evaluation Metrics Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| Decision Tree | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8930 |
| KNN | 0.9474 | 0.9577 | 0.9577 | 0.9577 | 0.9577 | 0.8930 |
| Naive Bayes | 0.9649 | 0.9974 | 0.9589 | 0.9859 | 0.9722 | 0.9220 |
| SVM | 0.9825 | 0.9974 | 0.9726 | 1.0000 | 0.9861 | 0.9630 |

## Observations

| ML Model | Observation |
|-----------|------------|
| Logistic Regression | Strong baseline model with excellent accuracy and interpretability. |
| Decision Tree | Simple and interpretable but slightly lower performance. |
| KNN | Good performance but sensitive to neighborhood selection. |
| Naive Bayes | Fast and effective despite strong independence assumptions. |
| SVM | Achieved the highest overall performance among all models. |

## Overall Winner

Support Vector Machine (SVM)

## Streamlit Features

- Upload Test CSV
- Select Model
- View Evaluation Metrics
- View Classification Report
- View Confusion Matrix
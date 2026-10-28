# Music Genre Classification Using CNN

## Overview

This project develops a Music Genre Classification System using a Convolutional Neural Network (CNN). It processes audio files, converts them into Mel spectrograms, and predicts the music genre using deep learning.

The system uses the GTZAN music genre dataset and includes audio preprocessing, CNN model training, performance evaluation, uploaded-audio prediction, and Grad-CAM-based explainable AI.

## Objectives

- Automatically classify music into different genres.
- Extract audio features using Mel spectrograms.
- Train a CNN model using the GTZAN dataset.
- Evaluate classification performance using accuracy, a confusion matrix, and a classification report.
- Predict the genre of uploaded music files.
- Visualize model attention using Grad-CAM.

## Technologies Used

- Python
- TensorFlow and Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- KaggleHub
- Google Colab
- Jupyter Notebook

## Dataset

The project uses the GTZAN music genre collection.

Dataset: https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection

The dataset contains music samples from ten genres:

1. Blues
2. Classical
3. Country
4. Disco
5. Hip-hop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock

The audio files are processed to extract Mel spectrogram features for CNN training and classification.

## Project Workflow

### 1. Dataset Download

The GTZAN dataset is downloaded using KaggleHub and organized into genre-specific folders.

### 2. Audio Preprocessing

Audio files are loaded using Librosa and processed at a consistent sample rate. Audio segments of up to 30 seconds are used for feature extraction.

### 3. Mel Spectrogram Extraction

Mel spectrograms represent the distribution of audio energy across time and frequency. They are converted into decibel values and used as input features for the CNN.

### 4. Dataset Preparation

The extracted features are converted into numerical arrays and associated with their corresponding genre labels.

The data is split into training and testing sets for model development and evaluation.

### 5. CNN Model Training

A Sequential CNN model is developed using TensorFlow and Keras.

The model learns patterns from Mel spectrograms to distinguish between different music genres.

Training configuration:

- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Evaluation metric: Accuracy
- Epochs: 30
- Batch size: 32

### 6. Model Evaluation

The model is evaluated using:

- Test accuracy
- Test loss
- Training and validation accuracy plots
- Confusion matrix
- Classification report
- Precision, recall, and F1-score

These metrics help assess the model's performance across the ten music genres.

### 7. Music Genre Prediction

The notebook includes functionality to upload an audio file, preprocess it, generate its Mel spectrogram, and predict the corresponding music genre using the trained CNN model.

### 8. Explainable AI Using Grad-CAM

Grad-CAM generates heatmaps that help visualize the spectrogram regions contributing to the CNN's prediction.

This provides insight into the model's behavior and supports the interpretation of its classification results.

## Repository Structure

```text
music-genre-classification-cnn/
│
├── Information_Visualization.ipynb
├── README.md
├── requirements.txt
└── .gitignore

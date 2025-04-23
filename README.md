# Driver Drowsiness Detection using CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning model that detects drowsy drivers using a hybrid CNN-LSTM architecture on sequential image data.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [License](#license)

## Overview
This project implements a hybrid CNN-LSTM model to analyze sequences of driver images (5 frames) and classify them as:
- Drowsy (0)
- Alert (1)

The model achieves **98.4% validation accuracy** by combining:
- **CNN layers** for spatial feature extraction
- **LSTM layer** for temporal pattern recognition

## Dataset
The [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/rakibuleceruet/driver-drowsiness-dataset-ddd) contains:
- 41,793 RGB images (128x128)
- 2 classes (Drowsy/Alert)
- Split into:
  - Train: 33,434 images (80%)
  - Validation: 6,268 images (15%)
  - Test: 2,091 images (5%)


## Model Architecture
```python
model = Sequential([
    TimeDistributed(Conv2D(8, (3,3), activation='relu'), input_shape=(5, 128, 128, 3)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D(2,2)),
    # ... 2 more Conv blocks ...
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(16),
    Dense(1, activation='sigmoid')
])

Key Specifications:

Parameters: 9,537 (37.25 KB)

Optimizer: Adam (lr=0.0001)

Loss: Binary Crossentropy

Batch Size: 32

Results
After 20 epochs:

Metric	Training	Validation
Accuracy	92.98%	98.39%
Loss	0.2373	0.1307

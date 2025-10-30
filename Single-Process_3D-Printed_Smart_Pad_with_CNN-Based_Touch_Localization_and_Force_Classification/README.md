# Single-Process 3D-Printed Smart Pad with CNN-Based Touch Localization and Force Classification

This repository contains the code and resources for a machine learning system that performs touch localization and force classification on a 3D-printed smart pad using Convolutional Neural Networks (CNNs).

## Overview

The project implements a deep learning approach to analyze sensor data from a single-process 3D-printed smart pad, enabling accurate detection of touch location and applied force classification.

## Repository Structure

```
.
├── data.py                      # Data loading and preprocessing utilities
├── model.py                     # CNN model architecture definitions
├── training.py                  # Model training scripts
├── model_training.ipynb         # Interactive notebook for model training
├── model_inference.ipynb        # Interactive notebook for model inference
├── requirements.txt             # Python package dependencies
├── pyproject.toml              # Project configuration
├── dataset/
│   └── 1000Hz/                 # Sensor data at 1000Hz sampling rate
└── weights/                     # Pre-trained model weights
    ├── model_1_sensors_1000Hz_epoch055.pt
    ├── model_2_sensors_1000Hz_epoch053.pt
    ├── model_3_sensors_1000Hz_epoch055.pt
    └── model_4_sensors_1000Hz_epoch065.pt
```

## How to Use

### Prerequisites

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Or if using `pyproject.toml`:

```bash
pip install -e .
```

### Training a Model

1. **Using Jupyter notebook:**
   Open and run `model_training.ipynb` for an interactive training experience.

### Running Inference

1. **Using Jupyter notebook:**
   Open and run `model_inference.ipynb` to test the pre-trained models.

### Using Pre-trained Models

Pre-trained model weights are available in the `weights` directory. Load them using the functions in `model.py`.

## Dataset

The sensor data is located in the `dataset/1000Hz` directory, containing measurements sampled at 1000Hz from the smart pad sensors.

## License

Please refer to the associated research paper for citation and usage guidelines.

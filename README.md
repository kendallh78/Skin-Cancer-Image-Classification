# Skin Cancer Classification Model

A deep learning model for classifying skin lesions as benign or malignant, using state-of-the-art preprocessing techniques and transfer learning.

## Overview

This project implements a skin lesion classification pipeline that:
- Preprocesses dermoscopic images to remove artifacts (hair, borders)
- Uses transfer learning with EfficientNet-B0 or ResNet18
- Implements multiple techniques to prevent overfitting and improve performance
- Provides detailed metrics, visualizations and threshold optimization

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Training Techniques](#training-techniques)
- [Results Visualization](#results-visualization)
- [Performance Metrics](#performance-metrics)

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/kendallh78/skin-cancer-classification.git
cd skin-cancer-classification
pip install -r requirements.txt
```

### Requirements

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.20.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
albumentations>=1.0.0
matplotlib>=3.4.0
tqdm>=4.60.0
pillow>=8.3.0
pandas>=1.3.0
# For CUDA support and mixed precision training (USE_AMP)
nvidia-apex>=0.1.0; sys_platform != 'darwin' and platform_machine != 'arm64'
```

## Project Structure

```
├── config.py             # Configuration parameters
├── main.py               # Entry point for the pipeline
├── model.py              # Model architecture definitions
├── preprocessing.py      # Image preprocessing functions
├── training.py           # Model training and testing functions
├── postprocess.py        # Threshold optimization and metrics 
├── visualization.py      # Plotting and visualization functions
├── requirements.txt      # Required packages
└── README.md             # This file
```

## Configuration

Key parameters can be modified in `config.py`:

```python
# Main parameters
BATCH_SIZE = 64
USE_EFFICIENT_NET = True  # set to False to use ResNet18
NUM_EPOCHS = 20 
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224  # input image size
PATIENCE = 5    # early stopping patience

# Data paths
DATA_DIR = '/medical_images'  
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
```

Expected directory structure:
```
dataset/
├── train/
│   ├── benign/
│   │   ├── image1.jpg
│   │   └── ...
│   └── malignant/
│       ├── image1.jpg
│       └── ...
└── test/
    ├── benign/
    │   ├── image1.jpg
    │   └── ...
    └── malignant/
        ├── image1.jpg
        └── ...
```

## Usage

Run the full pipeline with:

```bash
python main.py
```

This will:
1. Preprocess all images with hair removal and border detection
2. Train the model with early stopping
3. Evaluate performance on the test set
4. Generate visualizations and metrics

## Preprocessing Pipeline

The preprocessing steps include:
- **Hair removal**: Uses morphological operations to detect and remove hair artifacts
- **Border detection**: Identifies and eliminates artificial borders/rulers
- **Data augmentation**: Applies transforms like rotation, flipping, and color jittering
- **Normalization**: Standardizes pixel values for neural network input

Example of preprocessing stages:
1. Original image → Hair mask generation → Hair removal
2. Border detection → Final preprocessed image

## Model Architecture

The project supports two backbone architectures:

1. **EfficientNet-B0** (default):
   - Lightweight and efficient architecture
   - Pretrained on ImageNet with selective layer freezing
   - Fine-tuned final layers for skin lesion classification

2. **ResNet18**:
   - Alternative backbone if EfficientNet is not desired
   - Modified with dropout and batch normalization layers

## Training Techniques

Advanced techniques implemented:
- **Mixup augmentation**: Blends samples and labels to improve generalization
- **Early stopping**: Prevents overfitting by monitoring validation performance
- **CosineAnnealingLR**: Learning rate scheduling for better convergence
- **Mixed precision training**: Uses FP16 calculations when available
- **Label smoothing**: Prevents overconfidence in predictions

## Results Visualization

The pipeline generates:
- Training and validation loss/accuracy curves
- ROC curve and AUC score visualization
- Confusion matrix and classification report
- Sample predictions with original and preprocessed images

## Performance Metrics

Comprehensive metrics reported:
- Accuracy, precision, recall, F1-score
- ROC AUC and PR AUC
- Optimized threshold for better classification
- Per-class performance metrics

## Inspiration for Model Choice
- https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-022-00793-7

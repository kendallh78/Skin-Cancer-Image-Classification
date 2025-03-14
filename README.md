## Cancer Object Classification Pipeline
# This repository contains code for preprocessing satellite imagery from the XView dataset for use with YOLO object detection models.
## Overview
# The preprocessing pipeline handles the following tasks:

Class filtering to focus on specific target classes
Image processing (tiling or padding)
Train/test dataset splitting
Configuration file updates for YOLO training

Installation
bashCopygit clone https://github.com/yourusername/xview-preprocessing.git
cd xview-preprocessing
pip install -r requirements.txt
Project Structure
Copy├── preprocessing/
│   ├── tiling.py
│   ├── padding.py
│   ├── class_filtering.py
├── util/
│   ├── util.py
├── preprocessed_datasets/
├── xview_dataset_raw/
│   ├── train_images/
│   ├── xView_train.geojson
├── data_config.yaml
├── run_preprocessing.py
Usage
Basic Usage
pythonCopyfrom preprocessing_pipeline import run_preprocessing

params = {
    'skip_step': False,
    'clear_data': True,
    'target_classes': [1, 2, 3],  # Class IDs to keep
    'approach': 'TILING',         # 'TILING' or 'PADDING'
    'imgsz': 640,                 # Target image size
    'stride': 320,                # For tiling approach
    'train_split': 80             # Percentage for training set
}

run_preprocessing(params)
Preprocessing Approaches
Tiling
Splits large satellite images into smaller tiles of size imgsz with a stride of stride.
pythonCopyparams = {
    'approach': 'TILING',
    'imgsz': 640,
    'stride': 320
}
Padding
Pads images to reach the target size imgsz.
pythonCopyparams = {
    'approach': 'PADDING',
    'imgsz': 640
}
Output
The pipeline creates the following directory structure:
Copypreprocessed_datasets/
├── filtered_labels.geojson
├── baseline_unsplit/
│   ├── images/
│   ├── labels/
├── baseline-<approach>-<max_dim>-<imgsz>/
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   ├── val/
│   │   ├── images/
│   │   ├── labels/

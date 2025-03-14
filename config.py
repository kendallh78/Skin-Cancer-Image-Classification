import os
import torch


class Config:
    BATCH_SIZE = 64
    PIN_MEMORY = True
    USE_EFFICIENT_NET = True
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 1e-4
    IMG_SIZE = 224
    DATA_DIR = 'Medical Images'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES = ["benign", "malignant"]
    NUM_CLASSES = len(CLASSES)
    SAVE_MODEL_PATH = 'skin_cancer_model.pth'
    PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')  # new directory for preprocessed images
    NUM_WORKERS = 4  # workers for DataLoader
    PATIENCE = 5  # for early stopping for time constraints
    UNFREEZE_EPOCH = 5
    USE_AMP = torch.cuda.is_available()  # mixed precision training
    LABEL_SMOOTHING = 0.1
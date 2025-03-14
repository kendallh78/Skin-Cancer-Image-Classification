import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import concurrent.futures
from sklearn.cluster import KMeans
import pickle


def create_data_lists(data_dir):
    image_paths = []
    labels = []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        label = 0 if class_name == 'benign' else 1

        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(label)

    return image_paths, labels


def remove_hair(image):
    """ Removes hair from the image using morphological operations and inpainting. """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Kernel for morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

    # Blackhat operation to find hair-like structures
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Binary thresholding to create hair mask
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Use inpainting to remove hair
    cleaned_image = cv2.inpaint(image, thresh, 3, cv2.INPAINT_TELEA)

    return cleaned_image, thresh


def create_border_mask(image, k=3):
    """ Uses K-Means to segment lesion and replace background with dominant color. """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the largest contour (assumed to be the lesion)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    # Extract only background pixels for clustering
    background_pixels = image[mask == 0].reshape(-1, 3)

    if background_pixels.shape[0] > 0:
        kmeans = KMeans(n_clusters=min(k, len(background_pixels)), random_state=42, n_init=10)
        kmeans.fit(background_pixels)
        dominant_color = np.median(kmeans.cluster_centers_, axis=0).astype(np.uint8)
    else:
        dominant_color = np.array([255, 255, 255])  # Default to white if no background found

    # Ensure correct shape
    background_filled = np.full_like(image, dominant_color)
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Ensure mask has 3 channels

    final_img = np.where(mask_3ch == 255, image, background_filled)

    return final_img


def preprocess_image(image):
    """ Applies hair removal and lesion segmentation for preprocessing. """
    no_hair_img, hair_mask = remove_hair(image)

    border_mask = create_border_mask(no_hair_img)

    border_mask = cv2.resize(border_mask, (no_hair_img.shape[1], no_hair_img.shape[0]))

    # convert to binary if needed
    gray_mask = cv2.cvtColor(border_mask, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

    masked_img = cv2.bitwise_and(no_hair_img, no_hair_img, mask=binary_mask)

    return masked_img, hair_mask, border_mask


def preprocess_and_save(img_path, output_dir, img_size=224):
    rel_path = os.path.basename(img_path)
    out_path = os.path.join(output_dir, rel_path)

    if os.path.exists(out_path):
        return out_path

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processed_img, _, _ = preprocess_image(image)

    # Resize image
    processed_img = cv2.resize(processed_img, (img_size, img_size))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, processed_img)

    return out_path


def batch_preprocess_images(image_paths, output_dir, img_size=224, n_workers=4):
    os.makedirs(output_dir, exist_ok=True)

    processed_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_path = {
            executor.submit(preprocess_and_save, path, output_dir, img_size): path
            for path in image_paths
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(image_paths), desc="Preprocessing images"):
            path = future_to_path[future]
            try:
                processed_paths.append(future.result())
            except Exception as e:
                print(f"Error processing {path}: {e}")

    return processed_paths


# Albumentations transforms
def get_transforms(img_size=224):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_test_transform


class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


def get_data_loaders(config):
    print("Loading data...")
    train_paths, train_labels = create_data_lists(config.TRAIN_DIR)
    test_paths, test_labels = create_data_lists(config.TEST_DIR)

    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

    print("Preprocessing images...")
    print("Preprocessing training images...")
    processed_train_paths = batch_preprocess_images(train_paths, os.path.join(config.PREPROCESSED_DIR, 'train'),
                                                    config.IMG_SIZE, config.NUM_WORKERS)

    print("Preprocessing validation images...")
    processed_val_paths = batch_preprocess_images(val_paths, os.path.join(config.PREPROCESSED_DIR, 'val'),
                                                  config.IMG_SIZE, config.NUM_WORKERS)

    print("Preprocessing test images...")
    processed_test_paths = batch_preprocess_images(test_paths, os.path.join(config.PREPROCESSED_DIR, 'test'),
                                                   config.IMG_SIZE, config.NUM_WORKERS)
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)

    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)

    print(f"✅ Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print(f"⚠️ Overlap between Train & Val: {len(overlap_train_val)} images")
    print(f"⚠️ Overlap between Train & Test: {len(overlap_train_test)} images")
    print(f"⚠️ Overlap between Val & Test: {len(overlap_val_test)} images")

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("🚨 WARNING: Data leakage detected! Fix your dataset splitting.")
        exit()
    else:
        print("✅ No data leakage detected.")

    train_transform, val_test_transform = get_transforms(config.IMG_SIZE)

    train_dataset = SkinLesionDataset(processed_train_paths, train_labels, transform=train_transform)
    val_dataset = SkinLesionDataset(processed_val_paths, val_labels, transform=val_test_transform)
    test_dataset = SkinLesionDataset(processed_test_paths, test_labels, transform=val_test_transform)

    return {
        'train_loader': DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                                   num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY),
        'val_loader': DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                 num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY),  # ✅ Added
        'test_loader': DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                  num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY),
    }



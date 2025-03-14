import matplotlib.pyplot as plt
import numpy as np
import cv2
from preprocessing import preprocess_image


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()


def visualize_examples(image_paths, true_labels, predictions, probabilities, classes, num_examples=5):

    # indices of correct and incorrect predictions
    correct_indices = np.where(np.array(predictions) == np.array(true_labels))[0]
    incorrect_indices = np.where(np.array(predictions) != np.array(true_labels))[0]

    # select random examples from each
    if len(correct_indices) > 0:
        correct_samples = np.random.choice(correct_indices, min(num_examples, len(correct_indices)), replace=False)
    else:
        correct_samples = []

    if len(incorrect_indices) > 0:
        incorrect_samples = np.random.choice(incorrect_indices, min(num_examples, len(incorrect_indices)),
                                             replace=False)
    else:
        incorrect_samples = []

    # plot correct predictions
    if len(correct_samples) > 0:
        plt.figure(figsize=(15, 3 * len(correct_samples)))
        for i, idx in enumerate(correct_samples):
            img_path = image_paths[idx]
            true_label = classes[true_labels[idx]]
            pred_label = classes[predictions[idx]]
            prob = probabilities[idx]

            # load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_img = img.copy()

            processed_img, hair_mask, border_mask = preprocess_image(img)

            # display images
            plt.subplot(len(correct_samples), 3, i * 3 + 1)
            plt.imshow(original_img)
            plt.title(f'Original: {true_label}')
            plt.axis('off')

            plt.subplot(len(correct_samples), 3, i * 3 + 2)
            plt.imshow(hair_mask, cmap='gray')
            plt.title('Hair Mask')
            plt.axis('off')

            plt.subplot(len(correct_samples), 3, i * 3 + 3)
            plt.imshow(processed_img)
            plt.title(f'Processed: {pred_label} ({prob:.2f})')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('correct_predictions.png')
        plt.close()

    # plot incorrect predictions
    if len(incorrect_samples) > 0:
        plt.figure(figsize=(15, 3 * len(incorrect_samples)))
        for i, idx in enumerate(incorrect_samples):
            img_path = image_paths[idx]
            true_label = classes[true_labels[idx]]
            pred_label = classes[predictions[idx]]
            prob = probabilities[idx]

            # load and preprocess the image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_img = img.copy()

            processed_img, hair_mask, border_mask = preprocess_image(img)

            # display images
            plt.subplot(len(incorrect_samples), 3, i * 3 + 1)
            plt.imshow(original_img)
            plt.title(f'Original: {true_label}')
            plt.axis('off')

            plt.subplot(len(incorrect_samples), 3, i * 3 + 2)
            plt.imshow(hair_mask, cmap='gray')
            plt.title('Hair Mask')
            plt.axis('off')

            plt.subplot(len(incorrect_samples), 3, i * 3 + 3)
            plt.imshow(processed_img)
            plt.title(f'Processed: {pred_label} ({prob:.2f})')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('incorrect_predictions.png')
        plt.close()
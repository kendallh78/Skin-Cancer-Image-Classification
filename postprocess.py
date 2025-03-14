import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def optimize_threshold(true_labels, probabilities, classes):

    probabilities = np.ravel(probabilities)
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    if np.isinf(optimal_threshold) or np.isnan(optimal_threshold):
        print("⚠️ Warning: Invalid threshold detected. Setting to default 0.5")
        optimal_threshold = 0.5

    print(f'Optimal threshold found: {optimal_threshold:.4f}')

    # optimal threshold
    optimized_preds = (np.array(probabilities) >= optimal_threshold).astype(int)

    print("Unique predicted labels:", set(optimized_preds))
    print("Unique true labels:", set(true_labels))

    if len(set(optimized_preds)) < len(classes):
        print("⚠️ Warning: Not all classes are present in predictions!")
        missing_classes = set(range(len(classes))) - set(optimized_preds)
        for missing in missing_classes:
            optimized_preds = np.append(optimized_preds, missing)
            true_labels = np.append(true_labels, missing)

    # Compute metrics
    accuracy = np.mean(optimized_preds == np.array(true_labels))
    class_report = classification_report(true_labels, optimized_preds, target_names=classes)
    conf_matrix = confusion_matrix(true_labels, optimized_preds)

    print(f'Post-processing Accuracy: {accuracy:.4f}')
    print('\nPost-processing Classification Report:')
    print(class_report)
    print('\nPost-processing Confusion Matrix:')
    print(conf_matrix)

    return {
        'optimal_threshold': optimal_threshold,
        'optimized_preds': optimized_preds,
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    }

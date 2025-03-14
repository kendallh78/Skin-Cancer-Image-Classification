import torch
import numpy as np
import os
import time

from config import Config
from preprocessing import get_data_loaders
from model import SkinLesionModel
from training import train_model, test_model, setup_training
from postprocess import optimize_threshold
from visualization import plot_training_curves, plot_roc_curve, visualize_examples


def main():
    # random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = False  # for speed
    torch.backends.cudnn.benchmark = True  # for speed

    start_time = time.time()
    print("Starting skin lesion classification pipeline...")
    print(f"Using device: {Config.DEVICE}")

    # preprocessed directory if it doesn't exist
    os.makedirs(Config.PREPROCESSED_DIR, exist_ok=True)

    # data loaders and paths
    data = get_data_loaders(Config)

    # model initialization
    print(f"Initializing {'EfficientNet' if Config.USE_EFFICIENT_NET else 'ResNet18'} model...")
    model = SkinLesionModel(num_classes=Config.NUM_CLASSES,
                            model_type='efficient' if Config.USE_EFFICIENT_NET else 'resnet18')
    model = model.to(Config.DEVICE)

    # count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")

    # setup training components
    criterion, optimizer = setup_training(Config, model)

    # train model
    print("Starting model training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, data['train_loader'], data['val_loader'], criterion, optimizer,
        Config.NUM_EPOCHS, Config.DEVICE, Config.SAVE_MODEL_PATH, Config
    )

    # plot the training curves
    print("Plotting training curves...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # load best model for testing
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(Config.SAVE_MODEL_PATH))

    # test model
    print("Testing model...")
    test_results = test_model(model, data['test_loader'], criterion, Config.DEVICE, Config.CLASSES, Config)

    # plot ROC curve
    print("Plotting ROC curve...")
    plot_roc_curve(test_results['fpr'], test_results['tpr'], test_results['auc'])

    # post-processing -- threshold optimization
    print("Performing threshold optimization...")
    post_results = optimize_threshold(test_results['true_labels'], test_results['probabilities'], Config.CLASSES)

    # visualization
    print("Visualizing prediction examples...")
    visualize_examples(data['test_paths'], test_results['true_labels'],
                       test_results['predictions'], test_results['probabilities'], Config.CLASSES)

    # elapsed time calc
    elapsed_time = time.time() - start_time
    print(f"Pipeline completed successfully in {elapsed_time:.2f} seconds!")

    return test_results, post_results


if __name__ == "__main__":
    main()
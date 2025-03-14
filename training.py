import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, save_path, config):

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    patience_counter = 0

    # learning rate scheduler to help with time constraints
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]") as train_bar:
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero parameter gradients
                optimizer.zero_grad()


                # forward pass with mixed precision
                if config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # backward and optimize with scaler
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # standard forward and backward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                # calculate loss and stats and such
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=loss.item())

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total if total > 0 else 0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]") as val_bar:
                for inputs, labels in val_bar:
                    inputs, labels = inputs.to(device), labels.to(device)

                    if config.USE_AMP:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    val_bar.set_postfix(loss=loss.item())

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        # save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f'Model saved with val accuracy: {val_epoch_acc:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'Early stopping patience: {patience_counter}/{config.PATIENCE}')

        # early stopping, 20 epochs too long, stop at 5 if no significant change
        if patience_counter >= config.PATIENCE:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    return train_losses, val_losses, train_accs, val_accs


def test_model(model, test_loader, criterion, device, classes, config):

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as test_bar:
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                if config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)

                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

                test_bar.set_postfix(loss=loss.item())

    test_loss = test_loss / len(test_loader.dataset)

    print("Unique predicted labels:", set(all_preds))
    print("Unique true labels:", set(all_labels))
    print("Length of all_preds:", len(all_preds))
    print("Length of all_labels:", len(all_labels))

    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, \
        average_precision_score


    print("Unique predicted labels:", set(all_preds))
    print("Unique true labels:", set(all_labels))

    if len(set(all_preds)) < len(classes):  # Ensure all classes are predicted
        print("⚠️ Warning: Not all classes are present in predictions!")

    # metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    class_report = classification_report(all_labels, all_preds, target_names=classes)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}')
    print('\nClassification Report:')
    print(class_report)
    print('\nConfusion Matrix:')
    print(conf_matrix)

    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'pr_auc': pr_auc,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'test_loss': test_loss,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels
    }


def setup_training(config, model):
    # cross entropy loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    return criterion, optimizer
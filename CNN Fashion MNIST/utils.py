# utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_dataloaders(batch_size=64, val_split=0.1, num_workers=2):
    """
    Returns DataLoaders for training, validation, and test sets
    using the Fashion-MNIST dataset.
    """
    # Normalisation values for Fashion-MNIST (grayscale: single channel)
    # These center the image tensors around 0 with values in [-1, 1]
    mean, std = 0.5, 0.5

    # Define transformations for training data:
    # - RandomHorizontalFlip and RandomRotation are for data augmentation
    # - ToTensor converts PIL images to PyTorch tensors
    # - Normalize standardises pixel values
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )

    # Define transformations for validation and test sets:
    # No augmentation; only tensor conversion and normalisation
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    )

    # Load training dataset with training transforms
    train_dataset = datasets.FashionMNIST(
        root="data",  # Directory to store data
        train=True,  # Load training set
        download=True,  # Download if not already present
        transform=train_transform,
    )

    # Load test dataset with test transforms
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,  # Load test set
        download=True,
        transform=test_transform,
    )

    # Split training data into training and validation subsets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders to feed data into the model in batches
    # - shuffle=True for training to randomise batches each epoch
    # - num_workers=2 uses 2 subprocesses for faster loading
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on the given dataset.
    Returns average loss and accuracy.
    """
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_total += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = loss_total / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(
    model, train_loader, val_loader, device, epochs=10, lr=0.001, weight_decay=1e-4
):
    """
    Trains the given model and evaluates it on the validation set.

    Args:
        model: PyTorch model (nn.Module)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch.device ('cuda' or 'cpu')
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularisation factor

    Returns:
        model: Trained model (best performing if validation used)
        history: Dictionary of loss and accuracy per epoch
    """
    # Move model to device (GPU or CPU)
    model.to(device)

    # Define loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Store loss/accuracy per epoch
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        print(
            f"Epoch {epoch:02d}: "
            f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}"
        )

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, history


def evaluate_on_test(model, test_loader, device, print_report=True):
    """
    Evaluates the model on the test set.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: torch.device
        print_report: Whether to print classification report

    Returns:
        test_loss: Average cross-entropy loss on test set
        test_acc: Overall accuracy
        y_true: Ground truth labels
        y_pred: Model predictions
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_loss = total_loss / total
    test_acc = correct / total

    if print_report:
        print(f"\nTest Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    return test_loss, test_acc, y_true, y_pred


def plot_confusion_matrix(
    y_true, y_pred, class_names=None, normalise=True, figsize=(8, 6)
):
    """
    Plots a confusion matrix as a heatmap.

    Args:
        y_true: List or array of ground truth labels.
        y_pred: List or array of predicted labels.
        class_names: Optional list of class names for axis labels.
        normalise: If True, show percentages instead of raw counts.
        figsize: Size of the matplotlib figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalise rows to get percentages
    if normalise:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.round(cm * 100, 2)  # Show as percentages

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalise else "d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,  # type: ignore
        yticklabels=class_names,  # type: ignore
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalised Confusion Matrix" if normalise else "Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_training_history(history, figsize=(12, 5)):
    """
    Plots training and validation loss and accuracy over epochs.

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss',
                 'train_acc', 'val_acc' (output of train_model).
        figsize: Tuple specifying the figure size.
    """
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=figsize)

    # ---- Loss subplot ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    # ---- Accuracy subplot ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model(
    num_classes: int,
    backbone: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create a transfer learning model for face mask classification.
    """
    # 1. Load backbone with or without pretrained weights
    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # 2. Replace final FC layer (classifier head)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

    # 3. Optionally freeze backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict]:
    """
    Train a classification model and return the trained model and training history.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        DataLoader for the training set.
    val_loader : DataLoader
        DataLoader for the validation set.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimiser.
    device : str or torch.device or None
        Device to train on. If None, will pick 'cuda' if available else 'cpu'.

    Returns
    -------
    model : nn.Module
        The trained model (on the given device).
    history : dict
        Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc',
        each mapped to a list of values per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        # -------------------------
        # TRAINING PHASE
        # -------------------------
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()

            outputs = model(images)  # shape: (batch_size, num_classes) # forward pass
            loss = criterion(outputs, labels)

            loss.backward()  # backward pass
            optimiser.step()  # update weights

            running_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        epoch_train_loss = running_loss / running_total
        epoch_train_acc = running_correct / running_total

        # -------------------------
        # VALIDATION PHASE
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"Train loss: {epoch_train_loss:.4f} acc: {epoch_train_acc:.3f} | "
            f"Val loss: {epoch_val_loss:.4f} acc: {epoch_val_acc:.3f}"
        )

    return model, history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device | None = None,
    class_names: list[str] | None = None,
):
    """
    Evaluate a classification model on a given DataLoader and print metrics.

    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate.
    data_loader : DataLoader
        DataLoader for the evaluation set (e.g. test set).
    device : str or torch.device or None
        Device to run evaluation on. If None, use 'cuda' if available else 'cpu'.
    class_names : list of str or None
        Optional list of class names in the order of their indices.

    Returns
    -------
    y_true : list[int]
        Ground-truth label indices.
    y_pred : list[int]
        Predicted label indices.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # forward pass
            _, preds = outputs.max(1)  # predicted indices

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # Print metrics
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return y_true, y_pred

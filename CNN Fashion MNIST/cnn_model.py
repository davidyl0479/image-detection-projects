# cnn_model.py
#
# Simple CNN models for Fashion-MNIST:
# - FashionMNISTCNN:        baseline model
# - FashionMNISTCNNDeeper:  deeper (more conv layers)
# - FashionMNISTCNNWider:   wider (more channels per layer)

import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionMNISTCNN(nn.Module):
    """
    Baseline convolutional neural network for Fashion-MNIST.

    Input:  (batch_size, 1, 28, 28)
    Output: (batch_size, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ----- Convolutional feature extractor -----
        # Block 1: 1 → 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Shared max-pooling layer (2×2)
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout for convolutional layers
        self.dropout_conv = nn.Dropout2d(p=0.25)

        # After 3×(conv+pool) the spatial size is 3×3, with 128 channels:
        # 28 → 14 → 7 → 3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)  # (B, 1, 28, 28) → (B, 32, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 32, 14, 14)
        x = self.dropout_conv(x)

        # Block 2
        x = self.conv2(x)  # (B, 64, 14, 14)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 64, 7, 7)
        x = self.dropout_conv(x)

        # Block 3
        x = self.conv3(x)  # (B, 128, 7, 7)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 128, 3, 3)
        x = self.dropout_conv(x)

        # Flatten to vector
        x = torch.flatten(x, start_dim=1)  # (B, 128*3*3)

        # Classifier
        x = self.fc1(x)  # (B, 256)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)  # (B, num_classes)

        return x


class FashionMNISTCNNDeeper(nn.Module):
    """
    Deeper CNN: more convolutional layers (2 per block).

    Same channel sizes as the baseline (32, 64, 128),
    but each block has two conv layers before pooling.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Block 1: 1 → 32 → 32
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)

        # Block 2: 32 → 64 → 64
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        # Block 3: 64 → 128 → 128
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(p=0.25)

        # Spatial size is still 3×3 after 3 pools
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1 (two convs then pool)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Block 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


class FashionMNISTCNNWider(nn.Module):
    """
    Wider CNN: more channels per layer (64, 128, 256).

    Same structure as the baseline, but each block has
    more feature maps, increasing capacity per layer.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Block 1: 1 → 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Block 2: 64 → 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Block 3: 128 → 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(p=0.25)

        # After 3 pools, spatial size is 3×3, channels = 256
        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)  # (B, 1, 28, 28) → (B, 64, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 64, 14, 14)
        x = self.dropout_conv(x)

        # Block 2
        x = self.conv2(x)  # (B, 128, 14, 14)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 128, 7, 7)
        x = self.dropout_conv(x)

        # Block 3
        x = self.conv3(x)  # (B, 256, 7, 7)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 256, 3, 3)
        x = self.dropout_conv(x)

        x = torch.flatten(x, start_dim=1)  # (B, 256*3*3)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """
    Utility to count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

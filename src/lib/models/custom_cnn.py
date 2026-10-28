"""
Custom CNN for face embeddings (Model A of notebook 2).

Lives in a module, not in the notebook, so that the backend can import the class
when loading a full .pth model. Input: aligned RGB face (N, 3, 112, 112) scaled
to [-1, 1], the same as face_pipeline.preprocess.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv 3x3 + BatchNorm + ReLU, then halve the resolution."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """
    5 conv blocks (112 -> 3 px) + global average pooling + embedding layer.
    forward() returns the embedding; the classifier is only used for training.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.pool(self.features(x)))

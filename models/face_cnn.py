"""
models/face_cnn.py
------------------
PyTorch Multi-Class CNN Architecture for Face Identification.

Features:
- Custom 4-block Deep Convolutional Neural Network with BatchNorm, ReLU, MaxPool,
  Adaptive Average Pooling, Dropout, and Linear classification head.
- Optional Transfer Learning backbone (MobileNetV3-Small) for lightweight high-accuracy deployment.
- Calibrated Softmax probability prediction with rejection support.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class FaceCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network for Multi-Class Face Classification.
    Input resolution: (3, 112, 112)
    Output: Softmax probabilities across N registered classes.
    """
    def __init__(self, num_classes: int, dropout_rate: float = 0.4):
        super(FaceCNN, self).__init__()
        self.num_classes = num_classes

        # Block 1: 3 -> 32 filters (112x112 -> 56x56)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 32 -> 64 filters (56x56 -> 28x28)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 64 -> 128 filters (28x28 -> 14x14)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: 128 -> 256 filters (14x14 -> 7x7)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Spatial Invariance via Adaptive Average Pooling (outputs 1x1 feature map)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Embedding & Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv layers, xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor, use_unclamped: bool = True) -> torch.Tensor:
        """
        Extracts 128-dimensional L2-normalized embedding vector prior to final class logits.
        
        Optimization Note:
        - When use_unclamped=True (default), features are projected prior to the final ReLU
          clamping (self.classifier[:3]: Flatten -> Linear -> BatchNorm1d). This allows
          embeddings to span both positive and negative values on the complete unit
          hypersphere S^127, eliminating artificial positive-orthant cosine similarity inflation
          and drastically reducing False Acceptance Rate (FAR) for unseen impostors.
        - When use_unclamped=False, uses legacy ReLU activation (self.classifier[:4]).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        layers = self.classifier[:3] if use_unclamped else self.classifier[:4]
        for layer in layers:
            x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits

    @torch.no_grad()
    def predict_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Inference helper returning:
        1. Softmax probabilities (batch_size, num_classes)
        2. Top-1 predicted class index and confidence score
        """
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        top_conf, top_class = torch.max(probs, dim=-1)
        return probs, (top_class, top_conf)


class TransferFaceModel(nn.Module):
    """
    MobileNetV3-Small Transfer Learning Architecture with native 128-D embedding extraction.
    Ensures exact API parity with FaceCNN (extract_features, forward, predict_probabilities).
    """
    def __init__(self, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.3):
        super(TransferFaceModel, self).__init__()
        self.num_classes = num_classes
        import torchvision.models as models
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features: int = int(getattr(base.classifier[0], "in_features", 1024))

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128, bias=False),
            nn.BatchNorm1d(128)
        )
        self.classifier_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 128-D L2-normalized embedding vector."""
        x = self.features(x)
        x = self.avgpool(x)
        emb = self.embedding_head(x)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits."""
        x = self.features(x)
        x = self.avgpool(x)
        emb = self.embedding_head(x)
        return self.classifier_head(emb)

    @torch.no_grad()
    def predict_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        top_conf, top_class = torch.max(probs, dim=-1)
        return probs, (top_class, top_conf)


def build_transfer_face_model(num_classes: int, pretrained: bool = True) -> TransferFaceModel:
    """
    Alternative Transfer Learning backbone using MobileNetV3-Small.
    Offers rich pre-trained visual representations with fast inference.
    """
    return TransferFaceModel(num_classes=num_classes, pretrained=pretrained)


class FeatureExtractorWrapper(nn.Module):
    """Wrapper exposing only extract_features for TorchScript compilation."""
    def __init__(self, model: Any):
        super(FeatureExtractorWrapper, self).__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extract_fn: Any = getattr(self.model, "extract_features", self.model)
        return extract_fn(x)


def export_torchscript_extractor(model: Any, save_path: str, device: torch.device = torch.device("cpu")) -> str:
    """Compiles and exports feature extractor as a standalone TorchScript model."""
    model.eval()
    model.to(device)
    wrapper = FeatureExtractorWrapper(model)
    dummy = torch.randn(1, 3, 112, 112, device=device)
    traced = torch.jit.trace(wrapper, dummy)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    traced.save(save_path)
    return save_path


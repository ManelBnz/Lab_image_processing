"""
models/inception.py
──────────────────────────────────────────────────────────────────────────────
Inception V3 with ImageNet pretrained weights for binary classification.
Transfer learning: use pretrained feature extractor, fine-tune last layers.

Note: Inception V3 expects 299×299 input by default, but works with other
sizes thanks to AdaptiveAvgPool2d.
"""

import torch
import torch.nn as nn
from torchvision import models


class Inception(nn.Module):
    """
    Pretrained Inception V3 adapted for binary classification.

    Args:
        in_channels  : number of input channels (3 for RGB)
        dropout_rate : dropout before the final linear layer
    """

    def __init__(self, in_channels: int = 3, dropout_rate: float = 0.5, **kwargs):
        super().__init__()

        # Load pretrained Inception V3 (aux_logits must stay True with pretrained weights)
        self.backbone = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        )
        # Disable aux logits after loading
        self.backbone.aux_logits = False
        self.backbone.AuxLogits = None

        # Freeze early layers (up to Mixed_5d)
        freeze_until = "Mixed_5d"
        frozen = True
        for name, param in self.backbone.named_parameters():
            if freeze_until in name:
                frozen = False
            if frozen:
                param.requires_grad = False

        # Replace the classification head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model  = Inception(in_channels=3)
    dummy  = torch.randn(2, 3, 299, 299)
    output = model(dummy)
    print("Output shape:", output.shape)  # expected: (2, 1)

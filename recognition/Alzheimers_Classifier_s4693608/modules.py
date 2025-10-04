import torch
import torch.nn as nn
import timm

class AlzheimersClassifier(nn.Module):
    """
    ConvNeXt-based classifier for Alzheimer's (AD) vs Normal Control (NC).

    This class wraps a pretrained ConvNeXt model from the `timm` library and modifies it to:
      - Accept grayscale input images (1 channel) by duplicating them into 3 channels,
        since ConvNeXt expects RGB input.
      - Replace the final classification head with a linear layer outputting 2 classes
        (Alzheimer's disease = 1, Normal Control = 0).

    Attributes:
    model : timm.models.ConvNeXt
        The ConvNeXt backbone model with a modified classifier head.

    Methods:
    forward(x: torch.Tensor) -> torch.Tensor
        Performs a forward pass through the network.
        Takes grayscale input of shape [B, 1, H, W], duplicates channels,
        and outputs logits of shape [B, 2].
    """
    def __init__(self, model_name="convnext_tiny", num_classes=2, pretrained=True):
        super().__init__()
        # Load pretrained ConvNeXt backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Replace classifier head (for 2 classes instead of 1000)
        self.model.reset_classifier(num_classes=num_classes)

    def forward(self, x):
        # duplicate x channels ([B, 1, H, W] -> [B, 3, H, W])
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)
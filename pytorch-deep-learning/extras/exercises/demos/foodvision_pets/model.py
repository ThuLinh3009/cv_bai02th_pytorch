"""Builds the EffNetB0 feature extractor used by the FoodVision Pets demo."""
import torch
import torchvision
from torch import nn


def create_effnetb0(num_classes: int, device: str = "cpu"):
  weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
  transforms = weights.transforms()

  model = torchvision.models.efficientnet_b0(weights=weights).to(device)
  for param in model.features.parameters():
    param.requires_grad = False

  model.classifier = nn.Sequential(
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=1280, out_features=num_classes)
  ).to(device)

  return model, transforms

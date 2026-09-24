"""
Contains a from-scratch PyTorch implementation of the ViT-Base architecture
(Dosovitskiy et al. 2020, "An Image is Worth 16x16 Words"), built on top of
torch.nn.TransformerEncoderLayer.

Example usage:
  from vit import ViT
  model = ViT(num_classes=3)
"""
import torch
from torch import nn


class PatchEmbedding(nn.Module):
  """Turns a 2D input image into a 1D sequence of learnable patch embeddings."""
  def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768):
    super().__init__()
    self.patch_size = patch_size
    self.patcher = nn.Conv2d(in_channels=in_channels,
                             out_channels=embedding_dim,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0)
    self.flatten = nn.Flatten(start_dim=2, end_dim=3)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    image_resolution = x.shape[-1]
    assert image_resolution % self.patch_size == 0, \
        f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
    x_patched = self.patcher(x)
    x_flattened = self.flatten(x_patched)
    return x_flattened.permute(0, 2, 1)


class ViT(nn.Module):
  """Creates a Vision Transformer (ViT-Base) architecture."""
  def __init__(self,
              img_size: int = 224,
              in_channels: int = 3,
              patch_size: int = 16,
              num_transformer_layers: int = 12,
              embedding_dim: int = 768,
              mlp_size: int = 3072,
              num_heads: int = 12,
              attn_dropout: float = 0,
              mlp_dropout: float = 0.1,
              embedding_dropout: float = 0.1,
              num_classes: int = 1000):
    super().__init__()

    assert img_size % patch_size == 0, "Image size must be divisible by patch size."
    self.num_patches = (img_size * img_size) // patch_size**2

    self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
    self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True)
    self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)

    encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                nhead=num_heads,
                                                dim_feedforward=mlp_size,
                                                dropout=mlp_dropout,
                                                activation="gelu",
                                                batch_first=True,
                                                norm_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                      num_layers=num_transformer_layers)

    self.classifier = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim, out_features=num_classes)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch_size = x.shape[0]
    class_token = self.class_embedding.expand(batch_size, -1, -1)

    x = self.patch_embedding(x)
    x = torch.cat((class_token, x), dim=1)
    x = self.position_embedding + x
    x = self.embedding_dropout(x)

    x = self.transformer_encoder(x)
    return self.classifier(x[:, 0])

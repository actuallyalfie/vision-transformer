import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # TODO: create a linear layer to project patches to embed_dim
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        
        # TODO: split into patches
        # TODO: flatten each patch
        # TODO: project through linear layer
        
        # return shape: (batch, num_patches, embed_dim)
        pass
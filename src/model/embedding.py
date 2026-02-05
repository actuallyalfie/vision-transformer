import torch

class PatchEmbedding(torch.nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = torch.nn.Linear(patch_size * patch_size * in_channels, embed_dim)
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        
        # Split into patches
        x = x.unfold(2, self.patch_size, self.patch_size) # unfold height
        x = x.unfold(3, self.patch_size, self.patch_size) # unfold width

        # Rearrange dimensions
        x = x.permute(0, 2, 3, 1, 4, 5)

        # Flatten each patch
        x = x.reshape(x.shape[0], -1, self.patch_size * self.patch_size * 3)

        # Project through the linear layer
        x = self.projection(x)
        return x
        
        # return shape: (batch, num_patches, embed_dim)
#MLP-Mixed Model python
#0.12412762641906738
#torch.Size([1, 16, 17, 8])
import torch
import time
import torch.nn as nn

class MLPMixerblock(nn.Module):
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MLPMixerblock, self).__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(num_patches),
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches),)
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, hidden_dim),)

    def forward(self, x):
        x = x+self.token_mixing(x.permute(0,2,1)).permute(0,2,1)
        x = x+self.channel_mixing(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, input_channels, image_size, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, num_blocks):
        super(MLPMixer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.num_patches = num_patches
        self.patch_embeddings = nn.Conv2d(input_channels, hidden_dim, kernel_size=patch_size, stride = patch_size)
        self.mixer_blocks = nn.Sequential(*[MLPMixerblock(num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Conv2d(hidden_dim, input_channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embeddings(x)
        h_patches = x.shape[2]
        w_patches = x.shape[3]
        x = x.flatten(2).transpose(1,2)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        # batch_size, num_patches, hidden_dim = x.size()
        # new_h, new_w = self.image_size[0] //self.patch_size[0], self.image_size[1]//self.patch_size[1]
        x = x.transpose(1,2).reshape(batch_size, -1, h_patches, w_patches)
        x = self.fc_out(x)
        return x

image_size = (102, 64)
patch_size = (6, 8)
model = MLPMixer(input_channels=16, image_size=image_size, patch_size=patch_size, hidden_dim=128, tokens_mlp_dim=256, channels_mlp_dim=512, num_blocks=4)

input_tensor = torch.randn(1, 16, 102, 64)
s_time = time.time()
output_tensor = model(input_tensor)
e_time = time.time()
print(e_time - s_time)
output_tensor.shape

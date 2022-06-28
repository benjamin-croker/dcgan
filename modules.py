import torch
import torch.nn as nn


def _init_weights(module: nn.Module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, n_channels: int, n_latent: int, n_feat_map_gen: int, **kwargs):
        super(Generator, self).__init__()

        self._layers = nn.Sequential(
            # input size: (batch, n_latent, 1, 1)
            nn.ConvTranspose2d(
                n_latent, n_feat_map_gen * 8,
                kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(n_feat_map_gen * 8),
            nn.ReLU(inplace=True),

            # conv1 size: (batch, n_feat_map_gen * 8, 4, 4)
            nn.ConvTranspose2d(
                n_feat_map_gen * 8, n_feat_map_gen * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_feat_map_gen * 4),
            nn.ReLU(inplace=True),

            # conv2 size: (batch, n_feat_map_gen * 4, 8, 8)
            nn.ConvTranspose2d(
                n_feat_map_gen * 4, n_feat_map_gen * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_feat_map_gen * 2),
            nn.ReLU(inplace=True),

            # conv3 size: (batch, n_feat_map_gen * 2, 16, 16)
            nn.ConvTranspose2d(
                n_feat_map_gen * 2, n_feat_map_gen * 1,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_feat_map_gen * 1),
            nn.ReLU(inplace=True),

            # conv4 size: (batch, n_feat_map_gen , 32, 32)
            nn.ConvTranspose2d(
                n_feat_map_gen, n_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_channels),
            nn.Tanh()
            # output size: (batch, n_channels, 64, 64)
        )
        self.apply(_init_weights)

    def forward(self, input):
        return self._layers(input)


class Discriminator(nn.Module):
    def __init__(self, n_channels: int, n_feat_map_dis: int, **kwargs):
        super(Discriminator, self).__init__()

        self._layers = nn.Sequential(
            # input size: (batch, n_channel, 64, 64)
            nn.Conv2d(
                n_channels, n_feat_map_dis,
                kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # conv1 size: (batch, n_feat_map_dis, 32, 32)
            nn.Conv2d(
                n_feat_map_dis, n_feat_map_dis * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_feat_map_dis * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # conv2 size: (batch, n_feat_map_dis * 2, 16, 16)
            nn.Conv2d(
                n_feat_map_dis * 2, n_feat_map_dis * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_feat_map_dis * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # conv3 size: (batch, n_feat_map_dis * 4, 8, 8)
            nn.Conv2d(
                n_feat_map_dis * 4, n_feat_map_dis * 8,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_feat_map_dis * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # conv4 size: (batch, n_feat_map_dis*8, 4, 4)
            nn.Conv2d(
                n_feat_map_dis * 8, 1,
                kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.Sigmoid()
            # output size: (batch, 1, 1, 1)
        )
        self.apply(_init_weights)
    
    def forward(self, input):
        return self._layers(input)

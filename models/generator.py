"""Generator model based off of SRGAN

Classes
-------
Generator()
    Generator network, based off of the SRGAN paper
ResidualBlock()
    Single residual block of the Generator, as defined in the paper
PixelShuffleBlock(in_channels)
    Single pixel shuffle block of the Generator, as defined in the paper
"""

import torch
import torch.nn as nn
from collections import OrderedDict

class Generator(nn.Module):
    """Generator network, based off of the SRGAN paper
    found here: https://arxiv.org/abs/1609.04802
    ...
    
    Attributes
    ----------
    pre_residual : torch.nn.Sequential
        Convolutional block with PReLU activation, first layers of the network
    residual_section : torch.nn.Sequential
        Residual blocks of the network
    post_residual : torch.nn.Sequential
        Convolutional layer followed by batch norm
    pixel_shuffle_1 : PixelShuffleBlock
        First pixel shuffle block of the network, upsamples by factor of 2
    pixel_shuffle_2 : PixelShuffleBlock
        Second pixel shuffle block of the network, upsamples by factor of 2
    
    Methods
    -------
    foward(x)
        Performs foward pass on input x
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.pre_residual = nn.Sequential(nn.Conv2d(2, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())
        self.residual_section = nn.Sequential(*[ResidualBlock() for _ in range(12)]) # Original SRGAN architecture uses 16 residual blocks
        self.post_residual = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64))
        self.pixel_shuffle_1 = PixelShuffleBlock(64)
        self.pixel_shuffle_2 = PixelShuffleBlock(64)
        self.final_conv = nn.Conv2d(64, 2, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        """Performs foward pass on input x
        Parameters
        ----------
        x : torch.Tensor
            Network input
        
        Returns
        -------
        torch.Tensor
            Model output
        """

        out = self.pre_residual(x)
        recorded = out # Record here for elementwise sum later
        out = self.residual_section(out)
        out = self.post_residual(out)
        out = torch.add(recorded, out) # Elementwise sum as specified in the paper
        out = self.pixel_shuffle_1(out)
        out = self.pixel_shuffle_2(out)
        out = self.final_conv(out)

        return out

class ResidualBlock(nn.Module):
    """Single residual block of the Generator, as defined in the paper
    found here: https://arxiv.org/abs/1609.04802
    ...
    
    Attributes
    ----------
    channels : int
        Number of channels in the convolutional layers
    
    Methods
    -------
    foward(x)
        Performs foward pass on input x
    """

    def __init__(self):
        super(ResidualBlock, self).__init__()
        channels = 64
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False) #TODO: check padding and bias
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False) #TODO: check padding and bias
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Performs foward pass on input x
        Parameters
        ----------
        x : torch.Tensor
            Network input
        
        Returns
        -------
        torch.Tensor
            Model output
        """

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.add(x, out) # Elementwise sum as specified in the paper

        return out

class PixelShuffleBlock(nn.Module):
    """Single pixel shuffle block of the Generator, as defined in the paper
    found here: https://arxiv.org/abs/1609.04802
    ...
    
    Attributes
    ----------
    in_channels : int
        Number of in channels to convolutional block
    
    Methods
    -------
    foward(x)
        Performs foward pass on input x
    """

    def __init__(self, in_channels):
        super(PixelShuffleBlock, self).__init__()
        out_channels = 256
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) #TODO: check padding and bias
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        """Performs foward pass on input x
        Parameters
        ----------
        x : torch.Tensor
            Network input
        
        Returns
        -------
        torch.Tensor
            Model output
        """
        
        out = self.conv1(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)

        return out

if __name__ == '__main__':
    G = Generator()
    print(G)
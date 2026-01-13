"""
U-Net Architecture for HRF Segmentation
Standard U-Net design pattern with separate upsampling inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    
    CRITICAL: Takes TWO separate inputs:
    - x_deeper: from the deeper layer (needs upsampling)
    - x_skip: from the encoder skip connection
    
    Process:
    1. Upsample x_deeper to match x_skip spatial size
    2. Concatenate [x_skip, upsampled_x_deeper]
    3. Apply DoubleConv to reduce channels
    """
    def __init__(self, in_channels_deeper, in_channels_skip, out_channels, bilinear=False):
        super().__init__()
        
        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After concat, we have in_channels_skip + in_channels_deeper channels
            self.conv = DoubleConv(in_channels_skip + in_channels_deeper, out_channels)
        else:
            # Use transposed convolution
            # Reduce deeper channels by half during upsampling
            self.up = nn.ConvTranspose2d(
                in_channels_deeper, 
                in_channels_deeper // 2, 
                kernel_size=2, 
                stride=2
            )
            # After concat, we have in_channels_skip + in_channels_deeper//2 channels
            self.conv = DoubleConv(in_channels_skip + in_channels_deeper // 2, out_channels)

    def forward(self, x_deeper, x_skip):
        """
        Args:
            x_deeper: Feature map from deeper layer (to be upsampled)
            x_skip: Feature map from encoder (skip connection)
        """
        # Upsample the deeper feature map
        x_deeper = self.up(x_deeper)

        # Handle potential spatial dimension mismatch
        # This can happen with odd-sized inputs
        diffY = x_skip.size()[2] - x_deeper.size()[2]
        diffX = x_skip.size()[3] - x_deeper.size()[3]

        # Pad the upsampled feature map if needed
        x_deeper = F.pad(x_deeper, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        # Concatenate skip connection with upsampled features
        x = torch.cat([x_skip, x_deeper], dim=1)
        
        # Apply convolutions
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to produce output"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for segmentation
    
    Implementation details:
    - Encoder-decoder structure with skip connections
    - Up modules take separate inputs (not pre-concatenated)
    - Proper channel handling throughout
    - Works with any input size
    
    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes (1 for binary segmentation)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
        base_filters: Number of filters in first layer (default: 64)
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, base_filters=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_filters = base_filters

        # ========== ENCODER PATH ==========
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)           # 64 → 128
        self.down2 = Down(base_filters * 2, base_filters * 4)       # 128 → 256
        self.down3 = Down(base_filters * 4, base_filters * 8)       # 256 → 512
        self.down4 = Down(base_filters * 8, base_filters * 16)      # 512 → 1024

        # ========== DECODER PATH ==========
        # CRITICAL: Up(in_channels_deeper, in_channels_skip, out_channels)
        # Each Up module reduces channels while upsampling
        
        # up4: takes 1024 from bottleneck, 512 from encoder → outputs 512
        self.up4 = Up(base_filters * 16, base_filters * 8, base_filters * 8, bilinear)
        
        # up3: takes 512 from up4, 256 from encoder → outputs 256
        self.up3 = Up(base_filters * 8, base_filters * 4, base_filters * 4, bilinear)
        
        # up2: takes 256 from up3, 128 from encoder → outputs 128
        self.up2 = Up(base_filters * 4, base_filters * 2, base_filters * 2, bilinear)
        
        # up1: takes 128 from up2, 64 from encoder → outputs 64
        self.up1 = Up(base_filters * 2, base_filters, base_filters, bilinear)
        
        # ========== OUTPUT ==========
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        """
        Forward pass with proper skip connections
        
        Process:
        1. Encoder: Store skip connections at each level
        2. Bottleneck: Deepest processing
        3. Decoder: Upsample and merge with skip connections
        
        Args:
            x: Input tensor (batch_size, n_channels, height, width)
            
        Returns:
            Output tensor (batch_size, n_classes, height, width)
        """
        # ========== ENCODER PATH ==========
        x1 = self.inc(x)         # Initial conv, no downsampling
        x2 = self.down1(x1)      # 1st downsampling
        x3 = self.down2(x2)      # 2nd downsampling
        x4 = self.down3(x3)      # 3rd downsampling
        x5 = self.down4(x4)      # 4th downsampling (bottleneck)

        # ========== DECODER PATH ==========
        # CRITICAL: Pass x_deeper and x_skip as SEPARATE arguments
        # Not pre-concatenated!
        
        # Upsample from bottleneck, merge with skip from down3
        x = self.up4(x5, x4)     # (1024, x4:512) → 512
        
        # Upsample, merge with skip from down2
        x = self.up3(x, x3)      # (512, x3:256) → 256
        
        # Upsample, merge with skip from down1
        x = self.up2(x, x2)      # (256, x2:128) → 128
        
        # Upsample, merge with skip from initial conv
        x = self.up1(x, x1)      # (128, x1:64) → 64
        
        # ========== OUTPUT ==========
        logits = self.outc(x)    # 64 → n_classes
        
        return logits


def count_parameters(model):
    """Count total and trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable




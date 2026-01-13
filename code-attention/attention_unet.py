"""
Attention U-Net Architecture for HRF Segmentation

This module implements the Attention U-Net, which introduces Attention Gates (AGs)
into the standard U-Net skip connections.

Key Features:
1. Attention Gates: Learn to suppress irrelevant background regions (like retinal layers)
   and focus on salient features (HRF spots).
2. Same interface as standard UNet for easy swapping.
3. Minimal parameter overhead compared to standard U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import DoubleConv, Down, OutConv

class AttentionBlock(nn.Module):
    """
    Attention Gate (AG)
    
    Filters the features from the skip connection (x) using the gating signal (g)
    from the coarser scale.
    
    Args:
        F_g: Number of channels in the gating signal (g)
        F_l: Number of channels in the skip connection (x)
        F_int: Number of intermediate channels (usually F_l // 2)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from coarser scale (deeper layer)
            x: Skip connection features from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize g1 to match x1 if needed (should handle slight mismatches due to pooling)
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UpAttention(nn.Module):
    """
    Upscaling then DoubleConv, WITH Attention Gate
    """
    def __init__(self, in_channels_deeper, in_channels_skip, out_channels, bilinear=False):
        super().__init__()
        
        # 1. Attention Gate
        # F_g = in_channels_deeper (gating signal)
        # F_l = in_channels_skip (skip connection)
        # F_int = in_channels_skip // 2
        self.attention = AttentionBlock(F_g=in_channels_deeper, F_l=in_channels_skip, F_int=in_channels_skip // 2)
        
        # 2. Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After concat, we have in_channels_skip + in_channels_deeper
            self.conv = DoubleConv(in_channels_skip + in_channels_deeper, out_channels)
        else:
            # Transposed conv
            self.up = nn.ConvTranspose2d(in_channels_deeper, in_channels_deeper // 2, kernel_size=2, stride=2)
            # After concat, we have in_channels_skip + in_channels_deeper//2
            self.conv = DoubleConv(in_channels_skip + in_channels_deeper // 2, out_channels)
            
    def forward(self, x_deeper, x_skip):
        # 1. Apply Attention Gate to skip connection
        # Note: We use x_deeper as the gating signal. 
        # Some implementations upsample x_deeper first, others don't.
        # Standard AG uses the coarser signal (x_deeper) to gate the finer signal (x_skip).
        
        # We upsample x_deeper first for the concatenation path, 
        # but for the attention gate, we can use the pre-upsampled version (g) 
        # or the upsampled version. 
        # The standard implementation usually uses the upsampled version as 'g' 
        # OR projects the non-upsampled 'g' to match 'x'.
        # Our AttentionBlock handles resizing internally if needed.
        
        # Let's upsample first to match dimensions for concatenation anyway
        x_upsampled = self.up(x_deeper)
        
        # Handle padding for dimension mismatch (same as standard UNet)
        diffY = x_skip.size()[2] - x_upsampled.size()[2]
        diffX = x_skip.size()[3] - x_upsampled.size()[3]

        x_upsampled = F.pad(x_upsampled, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        
        # Apply Attention
        # g = x_deeper (coarser signal, AttentionBlock handles resizing)
        # x = x_skip
        x_skip_attended = self.attention(g=x_deeper, x=x_skip)
        
        # Concatenate
        x = torch.cat([x_skip_attended, x_upsampled], dim=1)
        
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture.
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, base_filters=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_filters = base_filters

        # ========== ENCODER PATH (Same as UNet) ==========
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16)

        # ========== DECODER PATH (With Attention) ==========
        # up4: takes 1024, 512 -> 512
        self.up4 = UpAttention(base_filters * 16, base_filters * 8, base_filters * 8, bilinear)
        
        # up3: takes 512, 256 -> 256
        self.up3 = UpAttention(base_filters * 8, base_filters * 4, base_filters * 4, bilinear)
        
        # up2: takes 256, 128 -> 128
        self.up2 = UpAttention(base_filters * 4, base_filters * 2, base_filters * 2, bilinear)
        
        # up1: takes 128, 64 -> 64
        self.up1 = UpAttention(base_filters * 2, base_filters, base_filters, bilinear)
        
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with Attention
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        logits = self.outc(x)
        return logits

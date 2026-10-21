import torch
import torch.nn as nn
 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()

        padding = dilation if kernel_size == 3 else 0

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MHPM(nn.Module):
    def __init__(self, in_channels=768, out_channels=256):
        super().__init__()

        # Reduce input channels
        self.reduce = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=1
        )
                # Multi-scale convolution branches
        self.branch1 = ConvBlock(
            out_channels, out_channels,
            kernel_size=1, dilation=1
        )

        self.branch2 = ConvBlock(
            out_channels, out_channels,
            kernel_size=3, dilation=1
        )

        self.branch3 = ConvBlock(
            out_channels, out_channels,
            kernel_size=3, dilation=2
        )

        self.branch4 = ConvBlock(
            out_channels, out_channels,
            kernel_size=3, dilation=3
        )
                # Fuse the four multi-scale branches
        self.fuse = ConvBlock(
            out_channels * 4,
            out_channels,
            kernel_size=1
        )
                # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # Reduce channels
        x = self.reduce(x)

        # Multi-scale processing
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
                # Concatenate multi-scale features
        multi_scale = torch.cat([b1, b2, b3, b4], dim=1)

        # Fuse features
        fused = self.fuse(multi_scale)

        # Residual connection
        fused = fused + x

        # Channel attention
        attention = self.channel_attention(fused)
        output = fused * attention

        return output

# ablation_models.py
"""
Ablation Study Models for U-Net Comparison
All models follow the same architecture but with different components:
- UNet: Baseline (standard convolutions, no residuals) – uses transposed convolution
- DSC-UNet: Depthwise Separable Convolutions – uses bilinear upsampling
- SA-UNet: Depthwise Separable Convolutions + Shuffle Attention – uses bilinear upsampling
- IR-UNet: Inverted Residual Blocks (no attention) – uses bilinear upsampling (user-provided)
- SA-IRUNet: Inverted Residual Blocks + Shuffle Attention – uses bilinear upsampling (user-provided)

Upsampling strategy:
- UNet: transposed convolution (ConvTranspose2d)
- Others: bilinear upsampling + 1×1 convolution (following user's implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from thop import profile


# -----------------------------------------------------------------------------
# Shuffle Attention Layer (SA) – User-provided
# -----------------------------------------------------------------------------
class ShuffleAttention(nn.Module):
    """Shuffle Attention (SA) module that replaces SE block.
    
    Args:
        channel: Number of input channels
        groups: Number of groups for channel shuffling
    """
    def __init__(self, channel, groups=64):
        super(ShuffleAttention, self).__init__()
        self.groups = groups
        assert channel % (2 * groups) == 0, f"Channel {channel} must be divisible by {2 * groups}"
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # Channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # Spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out


# -----------------------------------------------------------------------------
# Depthwise Separable Convolution (for DSC-UNet and SA-UNet)
# -----------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: Depthwise + Pointwise"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=None):
        super(DepthwiseSeparableConv, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# -----------------------------------------------------------------------------
# Inverted Residual Block (with Shuffle Attention) – User-provided
# -----------------------------------------------------------------------------
class InvertedResidualBlockWithSA(nn.Module):
    """Inverted residual block with Shuffle Attention (used in SAIRUNet)"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 expand_ratio=6, dropout_rate=0.05):
        super(InvertedResidualBlockWithSA, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = [k // 2 for k in kernel_size]
        
        layers = []
        # 1x1 expansion
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1,
                                   padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                               stride=stride, padding=padding,
                               groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # Shuffle Attention
        if hidden_dim >= 4:
            max_groups = min(64, hidden_dim // 2)
            for g in range(max_groups, 0, -1):
                if (hidden_dim // 2) % g == 0:
                    groups = g
                    break
            else:
                groups = 1
            layers.append(ShuffleAttention(hidden_dim, groups=groups))
        
        # Projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.dropout(out)
        if self.use_residual:
            out = out + identity
        return out


# -----------------------------------------------------------------------------
# Inverted Residual Block (without Attention) – User-provided
# -----------------------------------------------------------------------------
class InvertedResidualBlock(nn.Module):
    """Inverted residual block without attention (used in IRUNet)"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 expand_ratio=6, dropout_rate=0.05):
        super(InvertedResidualBlock, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = [k // 2 for k in kernel_size]
        
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1,
                                   padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                               stride=stride, padding=padding,
                               groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.dropout(out)
        if self.use_residual:
            out = out + identity
        return out


# -----------------------------------------------------------------------------
# Model 1: UNet (Baseline – Standard Convolutions, No Residuals)
# Uses transposed convolution for upsampling.
# -----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.05, kernel_size=5):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class UNet(nn.Module):
    """Standard U-Net with regular convolutions (Baseline – No Residuals)"""
    def __init__(self, in_channels=3, out_channels=6, n_filters=64, dropout=0.05, kernel_size=5):
        super(UNet, self).__init__()
        
        self.enc1 = ConvBlock(in_channels, n_filters * 1, dropout=dropout, kernel_size=kernel_size)
        self.enc2 = ConvBlock(n_filters * 1, n_filters * 2, dropout=dropout, kernel_size=kernel_size)
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4, dropout=dropout, kernel_size=kernel_size)
        self.enc4 = ConvBlock(n_filters * 4, n_filters * 8, dropout=dropout, kernel_size=kernel_size)
        
        self.dec1 = ConvBlock(n_filters * 4 + n_filters * 4, n_filters * 4,
                              dropout=dropout, kernel_size=kernel_size)
        self.dec2 = ConvBlock(n_filters * 2 + n_filters * 2, n_filters * 2,
                              dropout=dropout, kernel_size=kernel_size)
        self.dec3 = ConvBlock(n_filters * 1 + n_filters * 1, n_filters * 1,
                              dropout=dropout, kernel_size=kernel_size)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Transposed convolution (only in baseline)
        self.up1 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4,
                                      kernel_size=kernel_size, stride=2,
                                      padding=kernel_size//2, output_padding=1)
        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2,
                                      kernel_size=kernel_size, stride=2,
                                      padding=kernel_size//2, output_padding=1)
        self.up3 = nn.ConvTranspose2d(n_filters * 2, n_filters * 1,
                                      kernel_size=kernel_size, stride=2,
                                      padding=kernel_size//2, output_padding=1)
        
        self.output = nn.Conv2d(n_filters * 1, out_channels, kernel_size=1, stride=1)
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        pool1 = self.dropout_layer(pool1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        pool2 = self.dropout_layer(pool2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        pool3 = self.dropout_layer(pool3)
        
        enc4 = self.enc4(pool3)
        
        up1 = self.up1(enc4)
        if up1.shape[2:] != enc3.shape[2:]:
            up1 = F.interpolate(up1, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        up1 = torch.cat([up1, enc3], dim=1)
        up1 = self.dropout_layer(up1)
        dec1 = self.dec1(up1)
        
        up2 = self.up2(dec1)
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, enc2], dim=1)
        up2 = self.dropout_layer(up2)
        dec2 = self.dec2(up2)
        
        up3 = self.up3(dec2)
        if up3.shape[2:] != enc1.shape[2:]:
            up3 = F.interpolate(up3, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        up3 = torch.cat([up3, enc1], dim=1)
        up3 = self.dropout_layer(up3)
        dec3 = self.dec3(up3)
        
        out = self.output(dec3)
        return out


# -----------------------------------------------------------------------------
# Helper for bilinear upsampling (used in DSCUNet and SAUNet)
# -----------------------------------------------------------------------------
def up_block(in_channels, out_channels, kernel_size):
    """Upsample (bilinear) + Conv2d (1x1) for channel reduction, followed by BN+ReLU.
       Similar to the user's implementation for IR/SAIR models.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# -----------------------------------------------------------------------------
# Model 2: DSC-UNet (Depthwise Separable Convolution U-Net)
# Uses bilinear upsampling + 1x1 conv.
# -----------------------------------------------------------------------------
class DSCConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.05, kernel_size=5):
        super(DSCConvBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=kernel_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class DSCUNet(nn.Module):
    """U-Net with Depthwise Separable Convolutions"""
    def __init__(self, in_channels=3, out_channels=6, n_filters=64, dropout=0.05, kernel_size=5):
        super(DSCUNet, self).__init__()
        
        self.enc1 = DSCConvBlock(in_channels, n_filters * 1, dropout=dropout, kernel_size=kernel_size)
        self.enc2 = DSCConvBlock(n_filters * 1, n_filters * 2, dropout=dropout, kernel_size=kernel_size)
        self.enc3 = DSCConvBlock(n_filters * 2, n_filters * 4, dropout=dropout, kernel_size=kernel_size)
        self.enc4 = DSCConvBlock(n_filters * 4, n_filters * 8, dropout=dropout, kernel_size=kernel_size)
        
        self.dec1 = DSCConvBlock(n_filters * 4 + n_filters * 4, n_filters * 4,
                                 dropout=dropout, kernel_size=kernel_size)
        self.dec2 = DSCConvBlock(n_filters * 2 + n_filters * 2, n_filters * 2,
                                 dropout=dropout, kernel_size=kernel_size)
        self.dec3 = DSCConvBlock(n_filters * 1 + n_filters * 1, n_filters * 1,
                                 dropout=dropout, kernel_size=kernel_size)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bilinear upsampling + 1x1 conv
        self.up1 = up_block(n_filters * 8, n_filters * 4, kernel_size)
        self.up2 = up_block(n_filters * 4, n_filters * 2, kernel_size)
        self.up3 = up_block(n_filters * 2, n_filters * 1, kernel_size)
        
        self.output = nn.Conv2d(n_filters * 1, out_channels, kernel_size=1, stride=1)
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        pool1 = self.dropout_layer(pool1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        pool2 = self.dropout_layer(pool2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        pool3 = self.dropout_layer(pool3)
        
        enc4 = self.enc4(pool3)
        
        up1 = self.up1(enc4)
        if up1.shape[2:] != enc3.shape[2:]:
            up1 = F.interpolate(up1, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        up1 = torch.cat([up1, enc3], dim=1)
        up1 = self.dropout_layer(up1)
        dec1 = self.dec1(up1)
        
        up2 = self.up2(dec1)
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, enc2], dim=1)
        up2 = self.dropout_layer(up2)
        dec2 = self.dec2(up2)
        
        up3 = self.up3(dec2)
        if up3.shape[2:] != enc1.shape[2:]:
            up3 = F.interpolate(up3, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        up3 = torch.cat([up3, enc1], dim=1)
        up3 = self.dropout_layer(up3)
        dec3 = self.dec3(up3)
        
        out = self.output(dec3)
        return out


# -----------------------------------------------------------------------------
# Model 3: SA-UNet (Depthwise Separable + Shuffle Attention)
# Uses bilinear upsampling + 1x1 conv.
# -----------------------------------------------------------------------------
class SAConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.05, kernel_size=5):
        super(SAConvBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=kernel_size)
        
        # Shuffle Attention
        if out_channels >= 4:
            max_groups = min(64, out_channels // 2)
            for g in range(max_groups, 0, -1):
                if (out_channels // 2) % g == 0:
                    groups = g
                    break
            else:
                groups = 1
            self.attention = ShuffleAttention(out_channels, groups=groups)
        else:
            self.attention = nn.Identity()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x

class SAUNet(nn.Module):
    """U-Net with Depthwise Separable Convolutions and Shuffle Attention"""
    def __init__(self, in_channels=3, out_channels=6, n_filters=64, dropout=0.05, kernel_size=5):
        super(SAUNet, self).__init__()
        
        self.enc1 = SAConvBlock(in_channels, n_filters * 1, dropout=dropout, kernel_size=kernel_size)
        self.enc2 = SAConvBlock(n_filters * 1, n_filters * 2, dropout=dropout, kernel_size=kernel_size)
        self.enc3 = SAConvBlock(n_filters * 2, n_filters * 4, dropout=dropout, kernel_size=kernel_size)
        self.enc4 = SAConvBlock(n_filters * 4, n_filters * 8, dropout=dropout, kernel_size=kernel_size)
        
        self.dec1 = SAConvBlock(n_filters * 4 + n_filters * 4, n_filters * 4,
                                dropout=dropout, kernel_size=kernel_size)
        self.dec2 = SAConvBlock(n_filters * 2 + n_filters * 2, n_filters * 2,
                                dropout=dropout, kernel_size=kernel_size)
        self.dec3 = SAConvBlock(n_filters * 1 + n_filters * 1, n_filters * 1,
                                dropout=dropout, kernel_size=kernel_size)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bilinear upsampling + 1x1 conv
        self.up1 = up_block(n_filters * 8, n_filters * 4, kernel_size)
        self.up2 = up_block(n_filters * 4, n_filters * 2, kernel_size)
        self.up3 = up_block(n_filters * 2, n_filters * 1, kernel_size)
        
        self.output = nn.Conv2d(n_filters * 1, out_channels, kernel_size=1, stride=1)
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        pool1 = self.dropout_layer(pool1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        pool2 = self.dropout_layer(pool2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        pool3 = self.dropout_layer(pool3)
        
        enc4 = self.enc4(pool3)
        
        up1 = self.up1(enc4)
        if up1.shape[2:] != enc3.shape[2:]:
            up1 = F.interpolate(up1, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        up1 = torch.cat([up1, enc3], dim=1)
        up1 = self.dropout_layer(up1)
        dec1 = self.dec1(up1)
        
        up2 = self.up2(dec1)
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, enc2], dim=1)
        up2 = self.dropout_layer(up2)
        dec2 = self.dec2(up2)
        
        up3 = self.up3(dec2)
        if up3.shape[2:] != enc1.shape[2:]:
            up3 = F.interpolate(up3, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        up3 = torch.cat([up3, enc1], dim=1)
        up3 = self.dropout_layer(up3)
        dec3 = self.dec3(up3)
        
        out = self.output(dec3)
        return out


# -----------------------------------------------------------------------------
# Model 4: IR-UNet (Inverted Residual U-Net without Attention)
# Uses user-provided implementation (bilinear upsampling).
# -----------------------------------------------------------------------------
class IRUNet(nn.Module):
    """
    Inverted Residual U-Net (IR-UNet) – no attention.
    Uses bilinear upsampling.
    """
    def __init__(self, in_channels=3, out_channels=6, n_filters=64,
                 dropout=0.05, kernel_size=5, expand_ratio=6):
        super(IRUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        
        # Encoder
        self.enc1 = InvertedResidualBlock(in_channels, n_filters * 1,
                                          kernel_size=kernel_size, stride=1,
                                          expand_ratio=1, dropout_rate=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = InvertedResidualBlock(n_filters * 1, n_filters * 2,
                                          kernel_size=kernel_size, stride=1,
                                          expand_ratio=expand_ratio, dropout_rate=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = InvertedResidualBlock(n_filters * 2, n_filters * 4,
                                          kernel_size=kernel_size, stride=1,
                                          expand_ratio=expand_ratio, dropout_rate=dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = InvertedResidualBlock(n_filters * 4, n_filters * 8,
                                          kernel_size=kernel_size, stride=1,
                                          expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # Decoder with bilinear upsampling
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 8, n_filters * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(inplace=True)
        )
        self.dec5 = InvertedResidualBlock(n_filters * 4 + n_filters * 4, n_filters * 4,
                                          kernel_size=kernel_size, stride=1,
                                          expand_ratio=expand_ratio, dropout_rate=dropout)
        
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 4, n_filters * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.dec6 = InvertedResidualBlock(n_filters * 2 + n_filters * 2, n_filters * 2,
                                          kernel_size=kernel_size, stride=1,
                                          expand_ratio=expand_ratio, dropout_rate=dropout)
        
        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 2, n_filters * 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 1),
            nn.ReLU(inplace=True)
        )
        self.dec7 = InvertedResidualBlock(n_filters * 1 + n_filters * 1, n_filters * 1,
                                          kernel_size=kernel_size, stride=1,
                                          expand_ratio=expand_ratio, dropout_rate=dropout)
        
        self.final_conv = nn.Conv2d(n_filters * 1, out_channels, kernel_size=1, stride=1)
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
        for module in [self.up5, self.up6, self.up7]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        pool1 = self.dropout_layer(pool1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        pool2 = self.dropout_layer(pool2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        pool3 = self.dropout_layer(pool3)
        
        enc4 = self.enc4(pool3)
        
        up5 = self.up5(enc4)
        up5 = torch.cat([up5, enc3], dim=1)
        up5 = self.dropout_layer(up5)
        dec5 = self.dec5(up5)
        
        up6 = self.up6(dec5)
        up6 = torch.cat([up6, enc2], dim=1)
        up6 = self.dropout_layer(up6)
        dec6 = self.dec6(up6)
        
        up7 = self.up7(dec6)
        up7 = torch.cat([up7, enc1], dim=1)
        up7 = self.dropout_layer(up7)
        dec7 = self.dec7(up7)
        
        out = self.final_conv(dec7)
        return out


# -----------------------------------------------------------------------------
# Model 5: SA-IRUNet (Inverted Residual + Shuffle Attention)
# Uses user-provided implementation (bilinear upsampling).
# -----------------------------------------------------------------------------
class SAIRUNet(nn.Module):
    """
    Shuffle Attention Inverted Residual U-Net (SA-IRUNet)
    Uses bilinear upsampling and includes Shuffle Attention in the bottleneck.
    """
    def __init__(self, in_channels=3, out_channels=6, n_filters=64,
                 dropout=0.05, kernel_size=5, expand_ratio=6):
        super(SAIRUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        
        # Encoder
        self.enc1 = InvertedResidualBlockWithSA(in_channels, n_filters * 1,
                                                kernel_size=kernel_size, stride=1,
                                                expand_ratio=1, dropout_rate=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = InvertedResidualBlockWithSA(n_filters * 1, n_filters * 2,
                                                kernel_size=kernel_size, stride=1,
                                                expand_ratio=expand_ratio, dropout_rate=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = InvertedResidualBlockWithSA(n_filters * 2, n_filters * 4,
                                                kernel_size=kernel_size, stride=1,
                                                expand_ratio=expand_ratio, dropout_rate=dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = InvertedResidualBlockWithSA(n_filters * 4, n_filters * 8,
                                                kernel_size=kernel_size, stride=1,
                                                expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # Decoder with bilinear upsampling
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 8, n_filters * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(inplace=True)
        )
        self.dec5 = InvertedResidualBlockWithSA(n_filters * 4 + n_filters * 4, n_filters * 4,
                                                kernel_size=kernel_size, stride=1,
                                                expand_ratio=expand_ratio, dropout_rate=dropout)
        
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 4, n_filters * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.dec6 = InvertedResidualBlockWithSA(n_filters * 2 + n_filters * 2, n_filters * 2,
                                                kernel_size=kernel_size, stride=1,
                                                expand_ratio=expand_ratio, dropout_rate=dropout)
        
        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 2, n_filters * 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 1),
            nn.ReLU(inplace=True)
        )
        self.dec7 = InvertedResidualBlockWithSA(n_filters * 1 + n_filters * 1, n_filters * 1,
                                                kernel_size=kernel_size, stride=1,
                                                expand_ratio=expand_ratio, dropout_rate=dropout)
        
        self.final_conv = nn.Conv2d(n_filters * 1, out_channels, kernel_size=1, stride=1)
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
        for module in [self.up5, self.up6, self.up7]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        pool1 = self.dropout_layer(pool1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        pool2 = self.dropout_layer(pool2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        pool3 = self.dropout_layer(pool3)
        
        enc4 = self.enc4(pool3)
        
        up5 = self.up5(enc4)
        up5 = torch.cat([up5, enc3], dim=1)
        up5 = self.dropout_layer(up5)
        dec5 = self.dec5(up5)
        
        up6 = self.up6(dec5)
        up6 = torch.cat([up6, enc2], dim=1)
        up6 = self.dropout_layer(up6)
        dec6 = self.dec6(up6)
        
        up7 = self.up7(dec6)
        up7 = torch.cat([up7, enc1], dim=1)
        up7 = self.dropout_layer(up7)
        dec7 = self.dec7(up7)
        
        out = self.final_conv(dec7)
        return out


# -----------------------------------------------------------------------------
# Model Metrics Functions (unchanged from previous)
# -----------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_block_wise_summary(model, dummy_input, device='cpu'):
    block_info = []
    blocks = ['enc1', 'enc2', 'enc3', 'enc4', 'dec1', 'dec2', 'dec3']
    def hook_fn(block_name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                block_info.append({
                    'block': block_name,
                    'type': module.__class__.__name__,
                    'input_shape': list(input[0].shape) if input else [],
                    'output_shape': list(output.shape)
                })
        return hook
    hooks = []
    for block_name in blocks:
        if hasattr(model, block_name):
            hooks.append(getattr(model, block_name).register_forward_hook(hook_fn(block_name)))
    with torch.no_grad():
        _ = model(dummy_input)
    for hook in hooks:
        hook.remove()
    return block_info

def calculate_model_metrics(model, input_shape, device='cpu'):
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(1, *input_shape).to(device)
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs = flops / 2
    except Exception as e:
        print(f"  Profile error: {e}")
        flops = 0
        macs = 0
        params = sum(p.numel() for p in model.parameters())
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    activation_memory = 0
    def hook_fn(module, input, output):
        nonlocal activation_memory
        if isinstance(output, torch.Tensor):
            activation_memory += output.nelement() * output.element_size()
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                               nn.BatchNorm2d, nn.ReLU, nn.Sequential)):
            hooks.append(module.register_forward_hook(hook_fn))
    with torch.no_grad():
        _ = model(dummy_input)
    for hook in hooks:
        hook.remove()
    
    layer_stats = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d, nn.MaxPool2d, nn.Dropout2d)):
            params_layer = sum(p.numel() for p in module.parameters())
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and module.weight is not None:
                if hasattr(module, 'stride'):
                    if isinstance(module.stride, tuple):
                        stride_h, stride_w = module.stride
                    else:
                        stride_h = stride_w = module.stride
                    out_h = dummy_input.shape[2] // stride_h if stride_h > 0 else dummy_input.shape[2]
                    out_w = dummy_input.shape[3] // stride_w if stride_w > 0 else dummy_input.shape[3]
                else:
                    out_h = dummy_input.shape[2]
                    out_w = dummy_input.shape[3]
                flops_layer = params_layer * out_h * out_w * 2
            else:
                flops_layer = 0
            if params_layer > 0:
                layer_stats.append({
                    'name': name if name else 'root',
                    'type': module.__class__.__name__,
                    'parameters': params_layer,
                    'flops': flops_layer
                })
    if flops == 0 and layer_stats:
        flops = sum(layer['flops'] for layer in layer_stats)
        macs = flops / 2
    
    block_summary = get_block_wise_summary(model, dummy_input, device)
    metrics = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'non_trainable_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'total_parameters_millions': sum(p.numel() for p in model.parameters()) / 1e6,
        'flops': flops,
        'flops_giga': flops / 1e9,
        'macs': macs,
        'macs_giga': macs / 1e9,
        'model_size_mb': model_size_mb,
        'activation_memory_mb': activation_memory / (1024 ** 2),
        'parameter_memory_mb': param_size / (1024 ** 2),
        'buffer_memory_mb': buffer_size / (1024 ** 2),
        'total_memory_mb': (param_size + buffer_size + activation_memory) / (1024 ** 2),
        'layer_breakdown': layer_stats,
        'block_summary': block_summary
    }
    return metrics

def print_block_wise_summary(block_summary):
    print("\n" + "="*100)
    print("BLOCK-WISE ARCHITECTURE SUMMARY")
    print("="*100)
    print(f"\n{'Block':<15} {'Type':<25} {'Input Shape':<30} {'Output Shape':<30}")
    print("-"*100)
    for block in block_summary:
        block_name = block['block']
        block_type = block['type'][:25]
        input_shape = ' x '.join([str(x) for x in block['input_shape'][1:]]) if block['input_shape'] else 'N/A'
        output_shape = ' x '.join([str(x) for x in block['output_shape'][1:]]) if block['output_shape'] else 'N/A'
        print(f"{block_name:<15} {block_type:<25} {input_shape:<30} {output_shape:<30}")
    print("="*100 + "\n")

def print_model_metrics(metrics, model_type):
    print("\n" + "="*80)
    print(f"MODEL METRICS SUMMARY - {model_type}")
    print("="*80)
    print_block_wise_summary(metrics['block_summary'])
    print("\n[1] PARAMETER SUMMARY")
    print("-"*80)
    print(f"  Total parameters:        {metrics['total_parameters']:,} ({metrics['total_parameters_millions']:.2f}M)")
    print(f"  Trainable parameters:    {metrics['trainable_parameters']:,}")
    print(f"  Non-trainable parameters:{metrics['non_trainable_parameters']:,}")
    print("\n[2] COMPUTATIONAL COST")
    print("-"*80)
    print(f"  FLOPs:                   {metrics['flops']:,} ({metrics['flops_giga']:.2f} G)")
    print(f"  MACs:                    {metrics['macs']:,} ({metrics['macs_giga']:.2f} G)")
    print("\n[3] MEMORY USAGE (Batch Size = 1)")
    print("-"*80)
    print(f"  Model parameters:        {metrics['parameter_memory_mb']:.2f} MB")
    print(f"  Buffers:                 {metrics['buffer_memory_mb']:.2f} MB")
    print(f"  Activations:             {metrics['activation_memory_mb']:.2f} MB")
    print(f"  Total memory:            {metrics['total_memory_mb']:.2f} MB")
    print("\n[4] LAYER-WISE BREAKDOWN (All Layers)")
    print("-"*120)
    print(f"  {'#':<5} {'Layer Name':<55} {'Type':<20} {'Params':>12} {'FLOPs(M)':>12}")
    print("-"*120)
    sorted_layers = sorted(metrics['layer_breakdown'], key=lambda x: x['parameters'], reverse=True)
    for idx, layer in enumerate(sorted_layers, 1):
        layer_name = layer['name']
        if len(layer_name) > 55:
            layer_name = layer_name[:52] + "..."
        params_m = layer['parameters'] / 1e6 if layer['parameters'] > 0 else 0
        flops_m = layer['flops'] / 1e6 if layer['flops'] > 0 else 0
        print(f"  {idx:<5} {layer_name:<55} {layer['type']:<20} {params_m:>11.3f}M {flops_m:>11.2f}M")
    print("-"*120)
    print("\n[5] PARAMETER DISTRIBUTION BY LAYER TYPE")
    print("-"*80)
    type_stats = {}
    for layer in metrics['layer_breakdown']:
        layer_type = layer['type']
        if layer_type not in type_stats:
            type_stats[layer_type] = {'params': 0, 'count': 0}
        type_stats[layer_type]['params'] += layer['parameters']
        type_stats[layer_type]['count'] += 1
    print(f"  {'Layer Type':<30} {'Count':>10} {'Total Params':>20} {'Percentage':>15}")
    print("-"*80)
    total_params = metrics['total_parameters']
    for layer_type in sorted(type_stats.keys(), key=lambda x: type_stats[x]['params'], reverse=True):
        stats = type_stats[layer_type]
        percentage = (stats['params'] / total_params) * 100 if total_params > 0 else 0
        print(f"  {layer_type:<30} {stats['count']:>10} {stats['params']:>20,} {percentage:>14.2f}%")
    print("="*80 + "\n")

def save_model_metrics(metrics, save_path, model_type):
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL METRICS SUMMARY - {model_type}\n")
        f.write("="*80 + "\n\n")
        f.write("BLOCK-WISE ARCHITECTURE SUMMARY\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Block':<15} {'Type':<25} {'Input Shape':<30} {'Output Shape':<30}\n")
        f.write("-"*100 + "\n")
        for block in metrics['block_summary']:
            block_name = block['block']
            block_type = block['type'][:25]
            input_shape = ' x '.join([str(x) for x in block['input_shape'][1:]]) if block['input_shape'] else 'N/A'
            output_shape = ' x '.join([str(x) for x in block['output_shape'][1:]]) if block['output_shape'] else 'N/A'
            f.write(f"{block_name:<15} {block_type:<25} {input_shape:<30} {output_shape:<30}\n")
        f.write("-"*100 + "\n\n")
        f.write("PARAMETERS:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Total parameters:        {metrics['total_parameters']:,} ({metrics['total_parameters_millions']:.2f}M)\n")
        f.write(f"  Trainable parameters:    {metrics['trainable_parameters']:,}\n")
        f.write(f"  Non-trainable parameters: {metrics['non_trainable_parameters']:,}\n\n")
        f.write("COMPUTATIONAL COST:\n")
        f.write("-"*80 + "\n")
        f.write(f"  FLOPs:                   {metrics['flops']:,} ({metrics['flops_giga']:.2f} G)\n")
        f.write(f"  MACs:                    {metrics['macs']:,} ({metrics['macs_giga']:.2f} G)\n\n")
        f.write("MEMORY USAGE:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Model parameters:        {metrics['parameter_memory_mb']:.2f} MB\n")
        f.write(f"  Buffers:                 {metrics['buffer_memory_mb']:.2f} MB\n")
        f.write(f"  Activations:             {metrics['activation_memory_mb']:.2f} MB\n")
        f.write(f"  Total memory:            {metrics['total_memory_mb']:.2f} MB\n\n")
        f.write("LAYER-WISE BREAKDOWN (All Layers):\n")
        f.write("-"*120 + "\n")
        f.write(f"{'#':<5} {'Layer Name':<55} {'Type':<20} {'Params':>15} {'FLOPs':>15}\n")
        f.write("-"*120 + "\n")
        sorted_layers = sorted(metrics['layer_breakdown'], key=lambda x: x['parameters'], reverse=True)
        for idx, layer in enumerate(sorted_layers, 1):
            layer_name = layer['name']
            if len(layer_name) > 55:
                layer_name = layer_name[:52] + "..."
            f.write(f"{idx:<5} {layer_name:<55} {layer['type']:<20} {layer['parameters']:>15,} {layer['flops']:>15,}\n")
        f.write("-"*120 + "\n\n")
        f.write("PARAMETER DISTRIBUTION BY LAYER TYPE:\n")
        f.write("-"*80 + "\n")
        type_stats = {}
        for layer in metrics['layer_breakdown']:
            layer_type = layer['type']
            if layer_type not in type_stats:
                type_stats[layer_type] = {'params': 0, 'count': 0}
            type_stats[layer_type]['params'] += layer['parameters']
            type_stats[layer_type]['count'] += 1
        f.write(f"{'Layer Type':<30} {'Count':>10} {'Total Params':>20} {'Percentage':>15}\n")
        f.write("-"*80 + "\n")
        total_params = metrics['total_parameters']
        for layer_type in sorted(type_stats.keys(), key=lambda x: type_stats[x]['params'], reverse=True):
            stats = type_stats[layer_type]
            percentage = (stats['params'] / total_params) * 100 if total_params > 0 else 0
            f.write(f"{layer_type:<30} {stats['count']:>10} {stats['params']:>20,} {percentage:>14.2f}%\n")
        f.write("="*80 + "\n")
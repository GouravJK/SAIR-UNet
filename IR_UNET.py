import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# -----------------------------------------------------------------------------
# Shuffle Attention Layer (SA) - Fixed Version
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
        # Ensure channel is divisible by 2*groups
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
        # Reshape: [b, c, h, w] -> [b, groups, c//groups, h, w]
        x = x.reshape(b, groups, c // groups, h, w)
        # Permute: [b, groups, c//groups, h, w] -> [b, c//groups, groups, h, w]
        x = x.permute(0, 2, 1, 3, 4)
        # Reshape back: [b, c, h, w]
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Reshape for group processing
        x = x.reshape(b * self.groups, -1, h, w)
        # Split into two parts for channel and spatial attention
        x_0, x_1 = x.chunk(2, dim=1)

        # Channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # Spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # Concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        
        # Apply channel shuffle
        out = self.channel_shuffle(out, 2)
        return out

# -----------------------------------------------------------------------------
# Inverted Residual Block with Shuffle Attention
# -----------------------------------------------------------------------------
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, 
                 expand_ratio=6, dropout_rate=0.05):
        super(InvertedResidualBlock, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Padding
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = [k // 2 for k in kernel_size]
        
        # Inverted residual block: Expand -> Depthwise -> Shuffle Attention -> Project
        layers = []
        
        # 1x1 Pointwise convolution (expansion)
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, 
                         padding=0, bias=False)
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                     stride=stride, padding=padding, 
                     groups=hidden_dim, bias=False)
        )
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # Shuffle Attention (replaces SE block)
        # Adjust groups based on channel size
        if hidden_dim >= 4:  # Need at least 4 channels for proper grouping
            # Find the largest divisor of hidden_dim//2 that's <= 64
            max_groups = min(64, hidden_dim // 2)
            # Find a suitable number of groups that divides hidden_dim//2
            for g in range(max_groups, 0, -1):
                if (hidden_dim // 2) % g == 0:
                    groups = g
                    break
            else:
                groups = 1
            layers.append(ShuffleAttention(hidden_dim, groups=groups))
        else:
            # If channels are too small, skip attention
            pass
        
        # 1x1 Pointwise convolution (projection)
        layers.append(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, 
                     padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # He normal initialization
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
# Shuffle Attention U-Net with Inverted Residuals (SA-IRUNet)
# Dynamic input and output channels
# -----------------------------------------------------------------------------
class SAIRUNet(nn.Module):
    """
    Shuffle Attention Inverted Residual U-Net (SA-IRUNet)
    A lightweight U-Net variant where SE blocks in inverted residuals are 
    replaced with Shuffle Attention mechanism.
    
    Uses bilinear upsampling instead of transpose convolutions for better 
    spatial preservation and fewer artifacts.
    
    Args:
        in_channels: Number of input channels (dynamic)
        out_channels: Number of output channels (dynamic)
        n_filters: Base number of filters (default: 64)
        dropout: Dropout rate (default: 0.05)
        kernel_size: Kernel size for convolutions (default: 5)
        expand_ratio: Expansion factor for inverted residuals (default: 6)
    """
    def __init__(self, in_channels, out_channels, n_filters=64, 
                 dropout=0.05, kernel_size=5, expand_ratio=6):
        super(SAIRUNet, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        
        # ---------- Encoder ----------
        # First block: no expansion since we start from small channels
        self.enc1 = InvertedResidualBlock(in_channels, n_filters * 1, 
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=1, dropout_rate=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 64 -> 32
        
        self.enc2 = InvertedResidualBlock(n_filters * 1, n_filters * 2,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 32 -> 16
        
        self.enc3 = InvertedResidualBlock(n_filters * 2, n_filters * 4,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)   # 16 -> 8
        
        self.enc4 = InvertedResidualBlock(n_filters * 4, n_filters * 8,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # ---------- Decoder with Bilinear Upsampling ----------
        # up5: 8 -> 16, concat with enc3 (both 256 channels)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 8, n_filters * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(inplace=True)
        )
        self.dec5 = InvertedResidualBlock(n_filters * 4 + n_filters * 4, n_filters * 4,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # up6: 16 -> 32, concat with enc2 (128 channels)
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 4, n_filters * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.dec6 = InvertedResidualBlock(n_filters * 2 + n_filters * 2, n_filters * 2,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # up7: 32 -> 64, concat with enc1 (64 channels)
        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 2, n_filters * 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 1),
            nn.ReLU(inplace=True)
        )
        self.dec7 = InvertedResidualBlock(n_filters * 1 + n_filters * 1, n_filters * 1,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # Final convolution - dynamically set output channels
        self.final_conv = nn.Conv2d(n_filters * 1, out_channels, kernel_size=1, stride=1)
        
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize final conv
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
        
        # Initialize upsampling convs
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
        # Encoder
        enc1 = self.enc1(x)          # 64x64
        pool1 = self.pool1(enc1)     # 32x32
        pool1 = self.dropout_layer(pool1)
        
        enc2 = self.enc2(pool1)      # 32x32
        pool2 = self.pool2(enc2)     # 16x16
        pool2 = self.dropout_layer(pool2)
        
        enc3 = self.enc3(pool2)      # 16x16
        pool3 = self.pool3(enc3)     # 8x8
        pool3 = self.dropout_layer(pool3)
        
        enc4 = self.enc4(pool3)      # 8x8
        
        # Decoder with bilinear upsampling
        up5 = self.up5(enc4)         # 16x16
        up5 = torch.cat([up5, enc3], dim=1)  # Concatenate with encoder 3
        up5 = self.dropout_layer(up5)
        dec5 = self.dec5(up5)        # 16x16
        
        up6 = self.up6(dec5)         # 32x32
        up6 = torch.cat([up6, enc2], dim=1)  # Concatenate with encoder 2
        up6 = self.dropout_layer(up6)
        dec6 = self.dec6(up6)        # 32x32
        
        up7 = self.up7(dec6)         # 64x64
        up7 = torch.cat([up7, enc1], dim=1)  # Concatenate with encoder 1
        up7 = self.dropout_layer(up7)
        dec7 = self.dec7(up7)        # 64x64
        
        out = self.final_conv(dec7)
        return out

# -----------------------------------------------------------------------------
# Inverted Residual Block (Without Attention)
# -----------------------------------------------------------------------------
class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block without attention mechanism.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depthwise convolution
        stride: Stride for depthwise convolution
        expand_ratio: Expansion factor for inverted residual
        dropout_rate: Dropout rate
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, 
                 expand_ratio=6, dropout_rate=0.05):
        super(InvertedResidualBlock, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Padding
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = [k // 2 for k in kernel_size]
        
        # Inverted residual block: Expand -> Depthwise -> Project
        layers = []
        
        # 1x1 Pointwise convolution (expansion)
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, 
                         padding=0, bias=False)
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                     stride=stride, padding=padding, 
                     groups=hidden_dim, bias=False)
        )
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # 1x1 Pointwise convolution (projection)
        layers.append(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, 
                     padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # He normal initialization
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
# IR UNet (Inverted Residual U-Net without Attention)
# -----------------------------------------------------------------------------
class IR_UNet(nn.Module):
    """
    Inverted Residual U-Net (IR-UNet)
    A lightweight U-Net variant using inverted residual blocks without any 
    attention mechanism for faster inference and fewer parameters.
    
    Uses bilinear upsampling instead of transpose convolutions for better 
    spatial preservation and fewer artifacts.
    
    Args:
        in_channels: Number of input channels (dynamic)
        out_channels: Number of output channels (dynamic)
        n_filters: Base number of filters (default: 64)
        dropout: Dropout rate (default: 0.05)
        kernel_size: Kernel size for convolutions (default: 5)
        expand_ratio: Expansion factor for inverted residuals (default: 6)
    """
    def __init__(self, in_channels, out_channels, n_filters=64, 
                 dropout=0.05, kernel_size=5, expand_ratio=6):
        super(IRUNet, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        
        # ---------- Encoder ----------
        # First block: no expansion since we start from small channels
        self.enc1 = InvertedResidualBlock(in_channels, n_filters * 1, 
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=1, dropout_rate=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 64 -> 32
        
        self.enc2 = InvertedResidualBlock(n_filters * 1, n_filters * 2,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 32 -> 16
        
        self.enc3 = InvertedResidualBlock(n_filters * 2, n_filters * 4,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)   # 16 -> 8
        
        self.enc4 = InvertedResidualBlock(n_filters * 4, n_filters * 8,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # ---------- Decoder with Bilinear Upsampling ----------
        # up5: 8 -> 16, concat with enc3 (both 256 channels)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 8, n_filters * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(inplace=True)
        )
        self.dec5 = InvertedResidualBlock(n_filters * 4 + n_filters * 4, n_filters * 4,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # up6: 16 -> 32, concat with enc2 (128 channels)
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 4, n_filters * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.dec6 = InvertedResidualBlock(n_filters * 2 + n_filters * 2, n_filters * 2,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # up7: 32 -> 64, concat with enc1 (64 channels)
        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_filters * 2, n_filters * 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 1),
            nn.ReLU(inplace=True)
        )
        self.dec7 = InvertedResidualBlock(n_filters * 1 + n_filters * 1, n_filters * 1,
                                         kernel_size=kernel_size, stride=1,
                                         expand_ratio=expand_ratio, dropout_rate=dropout)
        
        # Final convolution - dynamically set output channels
        self.final_conv = nn.Conv2d(n_filters * 1, out_channels, kernel_size=1, stride=1)
        
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize final conv
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
        
        # Initialize upsampling convs
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
        # Encoder
        enc1 = self.enc1(x)          # 64x64
        pool1 = self.pool1(enc1)     # 32x32
        pool1 = self.dropout_layer(pool1)
        
        enc2 = self.enc2(pool1)      # 32x32
        pool2 = self.pool2(enc2)     # 16x16
        pool2 = self.dropout_layer(pool2)
        
        enc3 = self.enc3(pool2)      # 16x16
        pool3 = self.pool3(enc3)     # 8x8
        pool3 = self.dropout_layer(pool3)
        
        enc4 = self.enc4(pool3)      # 8x8
        
        # Decoder with bilinear upsampling
        up5 = self.up5(enc4)         # 16x16
        up5 = torch.cat([up5, enc3], dim=1)  # Concatenate with encoder 3
        up5 = self.dropout_layer(up5)
        dec5 = self.dec5(up5)        # 16x16
        
        up6 = self.up6(dec5)         # 32x32
        up6 = torch.cat([up6, enc2], dim=1)  # Concatenate with encoder 2
        up6 = self.dropout_layer(up6)
        dec6 = self.dec6(up6)        # 32x32
        
        up7 = self.up7(dec6)         # 64x64
        up7 = torch.cat([up7, enc1], dim=1)  # Concatenate with encoder 1
        up7 = self.dropout_layer(up7)
        dec7 = self.dec7(up7)        # 64x64
        
        out = self.final_conv(dec7)
        return out



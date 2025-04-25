import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_filters, growth_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, growth_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(num_filters + growth_channels, growth_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(num_filters + 2 * growth_channels, growth_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(num_filters + 3 * growth_channels, growth_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(num_filters + 4 * growth_channels, num_filters, kernel_size=3, padding=1, stride=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_filters, growth_channels=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(num_filters, growth_channels)
        self.RDB2 = ResidualDenseBlock(num_filters, growth_channels)
        self.RDB3 = ResidualDenseBlock(num_filters, growth_channels)
    
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, growth_channels, num_blocks):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, stride=1)
        self.RRDB_trunk = nn.Sequential(
            *[RRDB(num_filters, growth_channels) for _ in range(num_blocks)]
        )
        self.trunk_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, stride=1)
        
        # Upsampling layers
        self.upconv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, stride=1)
        self.upconv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, stride=1)
        self.HRconv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, stride=1)
        self.conv_last = nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1, stride=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.HRconv(fea))
        out = self.conv_last(fea)

        return out
      
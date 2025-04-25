import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu(self.bn1(out))
        out = self.conv2(out)
        out = self.lrelu(self.bn2(out))
        
        return out

class VGGDiscriminator(nn.Module):
    def __init__(self, in_channels, num_features, input_size=128):
        super().__init__()
        self.input_size = input_size
        assert self.input_size == 128, (f'input size must be 128, but received {input_size}')

        self.conv0_0 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, stride=1)
        self.conv0_1 = nn.Conv2d(num_features, num_features, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_features)

        self.conv1 = ConvBlock(num_features, num_features * 2)
        self.conv2 = ConvBlock(num_features * 2, num_features * 4)
        self.conv3 = ConvBlock(num_features * 4, num_features * 8)
        self.conv4 = ConvBlock(num_features * 8, num_features * 8)

        self.ln1 = nn.Linear(num_features * 8 * 4 * 4, 100)
        self.ln2 = nn.Linear(100, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv0_0(x))
        out = self.lrelu(self.bn0_1(self.conv0_1(out)))

        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = out.view(out.size(0), -1)
        out = self.lrelu(self.ln1(out))
        out = self.ln2(out)

        return out
      
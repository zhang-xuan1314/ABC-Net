import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
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
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv2(self.drop(self.activation(self.bn(self.conv1(x)))))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, heads=[1, 21, 5, 1, 4, 2]):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.heads = heads
        self.s = torch.nn.Parameter(torch.randn(10)/100)
        self.inc1 = DoubleConv(in_channels, 16, kernel_size=3)
        self.inc2 = DoubleConv(16, 16, kernel_size=3)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.inc3 = DoubleConv(64, 64, kernel_size=3)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 128)
        self.dconv1 = DoubleConv(128, 128)
        self.dconv2 = DoubleConv(128, 128)
        self.out_modules = nn.ModuleList()
        for head in self.heads:
            self.out_modules.append(OutConv(128, head))

    def forward(self, x):
        x1 = self.inc1(x)
        x1 = self.inc2(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.inc3(x3)

        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.dconv1(x)
        x = self.dconv2(x)
        out = []
        for out_module in self.out_modules:
            out.append(out_module(x))
        return out


if __name__ == '__main__':
    import sys

    # sys.path.extend('/home/user/.local/lib/python3.7/site-packages')
    # import inplace_abn
    net = UNet(in_channels=3)
    x = torch.rand(1, 3, 480, 480)
    y = net(x)
    print(y[0].shape)
    total_nums = 0
    for p in net.parameters():
        total_nums += p.numel()
    print(total_nums)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = x * self.channel_att(x)
        x_out = x_out * self.spatial_att(x_out)
        return x_out

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        def down_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(CBAM(out_channels))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(CBAM(out_channels))
            return nn.Sequential(*layers)

        self.down1 = down_block(1, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)

        self.up1 = up_block(512, 512, dropout=0.5)
        self.up2 = up_block(512 + 512, 256, dropout=0.5)
        self.up3 = up_block(256 + 256, 128)
        self.up4 = up_block(128 + 128, 64)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.down1(x)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e5 = self.down5(e4)

        d1 = self.up1(e5)
        d1 = torch.cat([d1, e4], dim=1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        out = self.up5(d4)
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()

        def disc_block(in_channels, out_channels, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.block1 = disc_block(4, 64, normalize=False)
        self.block2 = disc_block(64, 128)
        self.block3 = disc_block(128, 256)
        self.cbam = CBAM(256)
        self.block4 = disc_block(256, 512)
        self.final_conv = nn.Conv2d(512, 1, 3, stride=1, padding=1)

    def forward(self, gray_input, color_input):
        x = torch.cat([gray_input, color_input], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.cbam(x)
        x = self.block4(x)
        x = self.final_conv(x)
        if self.training:
            noise = torch.randn_like(x)
            x = x + noise
        return x
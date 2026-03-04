import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -------------------------
# Basic Conv Block
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        return x1, x2, x3, x4


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        d1 = self.up1(x4)
        d1 = self.dec1(torch.cat([d1, x3], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, x1], dim=1))

        return self.final(d3)


# -------------------------
# Siamese U-Net
# -------------------------
class SiameseUNet(nn.Module):
    def __init__(self):
        super(SiameseUNet, self).__init__()

        self.encoder = Encoder(3)
        self.decoder = Decoder()

    def forward(self, t1, t2):
        # Extract features
        t1_x1, t1_x2, t1_x3, t1_x4 = self.encoder(t1)
        t2_x1, t2_x2, t2_x3, t2_x4 = self.encoder(t2)

        # Feature difference
        x1 = torch.abs(t1_x1 - t2_x1)
        x2 = torch.abs(t1_x2 - t2_x2)
        x3 = torch.abs(t1_x3 - t2_x3)
        x4 = torch.abs(t1_x4 - t2_x4)

        # Decode
        out = self.decoder(x1, x2, x3, x4)

        return torch.sigmoid(out)


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))   # /2
        x1 = self.layer1(self.maxpool(x0))        # /4
        x2 = self.layer2(x1)                      # /8
        x3 = self.layer3(x2)                      # /16
        x4 = self.layer4(x3)                      # /32
        return x0, x1, x2, x3, x4


class TransformerFusion(nn.Module):
    def __init__(self, dim=512, heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B, HW, C
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ChangeFormerLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.fusion = TransformerFusion(dim=512, heads=8, mlp_dim=1024, dropout=0.1)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec0 = DoubleConv(128, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, t1, t2):
        t1_x0, t1_x1, t1_x2, t1_x3, t1_x4 = self.encoder(t1)
        t2_x0, t2_x1, t2_x2, t2_x3, t2_x4 = self.encoder(t2)

        x0 = torch.abs(t1_x0 - t2_x0)
        x1 = torch.abs(t1_x1 - t2_x1)
        x2 = torch.abs(t1_x2 - t2_x2)
        x3 = torch.abs(t1_x3 - t2_x3)
        x4 = torch.abs(t1_x4 - t2_x4)

        x4 = self.fusion(x4)

        d3 = self.up3(x4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([d0, x0], dim=1))

        out = self.final(self.final_up(d0))
        return torch.sigmoid(out)


def build_model(arch: str):
    name = (arch or "siamese_unet").lower()
    if name == "changeformer_lite":
        return ChangeFormerLite()
    return SiameseUNet()

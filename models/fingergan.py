import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(feature, feature, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = feature
        
        # Decoder
        decoder_in_channels = []
        enc_channels = features.copy()
        for i in range(len(features) - 1, 0, -1):
            decoder_in_channels.append(features[i] + features[i-1])
        for idx, feature in enumerate(reversed(features[:-1])):
            in_ch = decoder_in_channels[idx]
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, feature, 2, 2, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(feature, feature, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2)
                )
            )
        
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def center_crop(self, enc_feat, target_shape):
        _, _, h, w = enc_feat.shape
        th, tw = target_shape
        y1 = (h - th) // 2
        x1 = (w - tw) // 2
        return enc_feat[:, :, y1:y1+th, x1:x1+tw]

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)

        # Decoder
        for i, decode in enumerate(self.decoder):
            skip = skip_connections[-i-2]
            if skip.shape[2:] != x.shape[2:]:
                skip = self.center_crop(skip, x.shape[2:])
            x = decode(torch.cat([x, skip], dim=1))

        return self.sigmoid(self.final_conv(x))

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        
        for feature in features:
            layers.extend([
                nn.Conv2d(in_channels, feature, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature),
                nn.LeakyReLU(0.2)
            ])
            in_channels = feature
        
        layers.extend([
            nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ])
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
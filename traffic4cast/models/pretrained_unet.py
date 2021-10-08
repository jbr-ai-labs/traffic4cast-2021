import torch
import torch.nn as nn

from traffic4cast.models.baseline_unet import UNetConvBlock
from segmentation_models_pytorch.encoders import get_encoder


class UNetAlteredUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetAlteredUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1), )

        self.conv_block = UNetConvBlock(out_size * 2, out_size, padding,
                                        batch_norm)

    def forward(self, x, bridge):  # noqa
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class PretrainedEncoderUNet(nn.Module):
    def __init__(
            self, encoder='resnext50_32x4d', in_channels=12 * 8,
            n_classes=6 * 8, up_mode="upconv", depth=6):
        super(PretrainedEncoderUNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        assert encoder in ("resnext50_32x4d", "densenet169", "densenet201",
                           "efficientnet-b3", "efficientnet-b5")
        fmap_channels = {
            "resnext50_32x4d": [96, 64, 256, 512, 1024, 2048],
            "densenet169": [96, 64, 256, 512, 1280, 1664],
            "densenet201": [96, 64, 256, 512, 1792, 1920],
            "efficientnet-b5": [96, 48, 40, 64, 176, 512],
            "efficientnet-b3": [96, 40, 32, 48, 136, 384],
        }

        self.channels = fmap_channels[encoder][:depth]

        self.down_path = get_encoder(encoder, in_channels=in_channels,
                                     weights="imagenet", depth=depth-1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(len(self.channels) - 1)):
            self.up_path.append(
                UNetAlteredUpBlock(self.channels[i + 1], self.channels[i],
                                   up_mode, True, True))

        self.last = nn.Conv2d(self.channels[0], n_classes, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        blocks = self.down_path(x)
        x = blocks[-1]

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])

        return self.last(x)

from __future__ import division
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()

        channels = [120, 120, 120]
        # channels = [160, 160, 160]

        kernels = [opt.conv_kernel_size] * 3
        init_channel = 10  # 18
        self.last_channel = channels[-1]
        self.gc1 = nn.Sequential(
            nn.Conv2d(in_channels=init_channel, out_channels=channels[0], kernel_size=kernels[0], stride=1, padding=kernels[0]//2),
            nn.InstanceNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[0], kernel_size=kernels[0], stride=1, padding=kernels[0] // 2),
            nn.InstanceNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.gc2 = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernels[1], stride=2, padding=kernels[1]//2),
            nn.InstanceNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(2, 2),
        )

        self.gc3 = nn.Sequential(
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernels[2], stride=2, padding=kernels[2]//2),
            nn.InstanceNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(2, 2),
        )

    def encode2(self, x):
        return [x, x, x]

    def encode(self, x):

        outputs = []
        # outputs.append(x)
        x = self.gc1(x)  # [1, 40, 128, 128]
        outputs.append(x)
        # print(x.shape)
        x = self.gc2(x)  # [1, 80, 64, 64]
        outputs.append(x)
        # print(x.shape)
        x = self.gc3(x)  # [1, 160, 32, 32]
        outputs.append(x)
        return outputs

    def forward(self, x):
        outputs = []
        # outputs.append(x)
        x = self.gc1(x)  # [1, 40, 128, 128]
        outputs.append(x)
        # print(x.shape)
        x = self.gc2(x)  # [1, 80, 64, 64]
        outputs.append(x)
        # print(x.shape)
        x = self.gc3(x)  # [1, 160, 32, 32]
        outputs.append(x)
        return outputs


def init_conv_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        # torch.nn.init.xavier_normal_(m.weight)
        # m.weight.data.fill_(1.0)
        # print(m.weight.data.min(), m.weight.data.max(), )


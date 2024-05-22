from torch import nn
import torch



# 定义 Fire 模块 (Squeeze + Expand)
class Fire(nn.Module):
    def __init__(self, in_ch, squeeze_ch, e1_ch, e3_ch):  # 声明 Fire 模块的超参数
        super(Fire, self).__init__()
        # Squeeze, 1x1 卷积
        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1)
        # # Expand, 1x1 卷积
        self.expand1 = nn.Conv2d(squeeze_ch, e1_ch, kernel_size=1)
        # Expand, 3x3 卷积
        self.expand3 = nn.Conv2d(squeeze_ch, e3_ch, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.squeeze(x))
        x = torch.cat([self.activation(self.expand1(x)),
                       self.activation(self.expand3(x))], dim=1)
        return x

# 定义简化的 SqueezeNet 模型类 2
class SqueezeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 3x32x32 -> 64x32x32
            nn.ReLU(inplace=True),
            Fire(64, 16, 64, 64),  # 64x32x32 -> 128x32x32
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 128x32x32 -> 128x16x16
            Fire(128, 32, 64, 64),  # 128x16x16 -> 128x16x16
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 128x16x16 -> 128x8x8
            Fire(128, 64, 128, 128),  # 128x8x8 -> 256x8x8
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 256x8x8 -> 256x4x4
            Fire(256, 64, 256, 256)  # 256x4x4 -> 512x4x4
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Conv2d(512, self.num_classes, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1)),  # 512x4x4 -> 10x1x1
        # )

    def forward(self, x):
        x = self.features(x)  # torch.Size([1, 512, 4, 4])
        # x = self.classifier(x)  # torch.Size([1, 10, 1, 1])
        x = torch.flatten(x, 1)
        return x

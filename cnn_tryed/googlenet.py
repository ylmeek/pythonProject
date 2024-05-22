# import packages
import torch
import torchvision


# Define BasicConv2d
class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


# Define InceptionAux.
class InceptionAux(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(2 * 2 * 128, 256))
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.avgpool(x)
        out = self.conv(out)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, 0.5, training=self.training)
        out = torch.nn.functional.relu(self.fc1(out), inplace=True)
        out = torch.nn.functional.dropout(out, 0.5, training=self.training)
        out = self.fc2(out)
        return out


# Define Inception.
class Inception(torch.nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = torch.nn.Sequential(BasicConv2d(in_channels, ch3x3red, kernel_size=1),
                                           BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1))
        self.branch3 = torch.nn.Sequential(BasicConv2d(in_channels, ch5x5red, kernel_size=1),
                                           BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2))
        self.branch4 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                           BasicConv2d(in_channels, pool_proj, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# Define GooLeNet.
class GoogLeNet(torch.nn.Module):
    def __init__(self, num_classes=10, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = BasicConv2d(3, 64, kernel_size=4, stride=2, padding=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.4)
        self.fc = torch.nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 32 x 32
        x = self.conv1(x)
        # # N x 64 x 18 x 18
        x = self.maxpool1(x)
        # # N x 64 x 9 x 9
        x = self.conv2(x)
        # # N x 64 x 9 x 9
        x = self.conv3(x)
        # # N x 192 x 9 x 9
        x = self.maxpool2(x)
        #
        # # N x 192 x 8 x 8
        x = self.inception3a(x)
        # # N x 256 x 8 x 8
        x = self.inception3b(x)
        # # N x 480 x 8 x 8
        x = self.maxpool3(x)
        # N x 480 x 4 x 4
        x = self.inception4a(x)
        # # N x 512 x 4 x 4
        #
        # if self.training and self.aux_logits:  # eval model lose this layer
        #     aux1 = self.aux1(x)
        #
        x = self.inception4b(x)
        # # N x 512 x 4 x 4
        x = self.inception4c(x)
        # # N x 512 x 4 x 4
        x = self.inception4d(x)
        # # N x 528 x 4 x 4
        # if self.training and self.aux_logits:  # eval model lose this layer
        #     aux2 = self.aux2(x)
        #
        x = self.inception4e(x)
        # # N x 832 x 4 x 4
        x = self.maxpool4(x)
        # # N x 832 x 2 x 2
        x = self.inception5a(x)
        # # N x 832 x 2 x 2
        x = self.inception5b(x)
        # # N x 1024 x 2 x 2
        #
        x = self.avgpool(x)
        # # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        # #x = self.fc(x)
        # # N x 10 (num_classes)
        #x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


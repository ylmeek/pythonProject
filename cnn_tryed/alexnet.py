import torch
from torch import nn
from torch.nn.functional import  interpolate
from torch import optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        print(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


input_image = Image.open('E:\pythonProject\image.jpeg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    # batch 1 , channel 3, length * width 224 * 224
    #print(input_batch.shape)
    #torch.Size([1, 3, 224, 224])

model = AlexNet()

    # move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

tempt = dict(model.features.named_children())
a = tempt['5']
print(a)
conv1 = dict(model.features.named_children())['0']
localw = conv1.weight.cpu().clone()
print("total of number of filter : ", len(localw))
num = 10
for i in range(len(localw)):
    localw0 = localw[i]
            #print(localw0.shape)
            # mean of 3 channel.
            #localw0 = torch.mean(localw0,dim=0)
            # there should be 3(3 channels) 11 * 11 filter.
    plt.figure(figsize=(20, 17))
    if (len(localw0)) > 1:
        for idx, filer in enumerate(localw0):
            plt.subplot(9, 9, idx+1)
            plt.axis('off')
            plt.imshow(filer[ :, :].detach(),cmap='gray')
            plt.show()

    else:
            plt.subplot(9, 9, idx+1)
            plt.axis('off')
            plt.imshow(localw0[0, :, :].detach(),cmap='gray')
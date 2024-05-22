import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.alex = torchvision.models.alexnet(torchvision.models.AlexNet_Weights.DEFAULT)
        self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        print()


    def forward(self, x):
        x = self.alex.features(x)
        x = self.alex.classifier(x)
        return x


model = CNN()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_image = Image.open('E:\pythonProject\image.jpeg')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
with torch.no_grad():
    output = model(input_batch)
    output_cpu = output.cpu()
    print(output)
    a = output.tolist()
    print(a)


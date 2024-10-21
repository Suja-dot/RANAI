#Resnet Model python
import torch
import time
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2= nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

    
class SimpleResNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleResNet, self).__init__()
        self.layer1 = BasicBlock(input_channels, 16)
        self.layer2 = BasicBlock(16, output_channels)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = SimpleResNet(input_channels=16, output_channels=16)
input_tensor = torch.rand(1, 16, 102, 64)
s_time = time.time()
output_tensor = model(input_tensor)
e_time = time.time()
print(e_time-s_time)
output_tensor.shape

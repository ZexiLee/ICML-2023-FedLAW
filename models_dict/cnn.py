import torch.nn as nn
import torch.nn.functional as F
import torch

import logging
from contextlib import contextmanager

import torch
import torch.nn as nn
import torchvision
from six import add_metaclass
from torch.nn import init
import copy
import math
from .reparam_function import ReparamModule

class CNNCifar100(nn.Module):
    def __init__(self):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 12/8,10


class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(3, 32, 3)) #0
        self.maxpool = nn.MaxPool2d(2, 2)
        # self.layers.append(nn.MaxPool2d(2, 2)) # 1
        self.layers.append(nn.Conv2d(32, 64, 3)) # 2
        self.layers.append(nn.Conv2d(64, 64, 3)) #3
        self.layers.append(nn.Linear(64 * 4 * 4, 64))#4
        self.layers.append(nn.Linear(64, 10)) #5

    def forward(self, x):
        x = self.maxpool(F.relu(self.layers[0](x)))
        x = self.maxpool(F.relu(self.layers[1](x)))
        x = F.relu(self.layers[2](x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.layers[3](x))
        x = self.layers[4](x)
        return x  # 12/8,10

class CNNCifar100_fedlaw(ReparamModule):
    def __init__(self):
        super(CNNCifar100_fedlaw, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 12/8,10


class CNNCifar10_fedlaw(ReparamModule):
    def __init__(self):
        super(CNNCifar10_fedlaw, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(3, 32, 3)) #0
        self.maxpool = nn.MaxPool2d(2, 2)
        # self.layers.append(nn.MaxPool2d(2, 2)) # 1
        self.layers.append(nn.Conv2d(32, 64, 3)) # 2
        self.layers.append(nn.Conv2d(64, 64, 3)) #3
        self.layers.append(nn.Linear(64 * 4 * 4, 64))#4
        self.layers.append(nn.Linear(64, 10)) #5

    def forward(self, x):
        x = self.maxpool(F.relu(self.layers[0](x)))
        x = self.maxpool(F.relu(self.layers[1](x)))
        x = F.relu(self.layers[2](x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.layers[3](x))
        x = self.layers[4](x)
        return x  # 12/8,10

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(       #(1*28*28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),    #(16*28*28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#(16*14*14)
        )
        self.conv2 = nn.Sequential(  # 16*14*14
            nn.Conv2d(16,32,5,1,2),  #32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32*7*7
        )
        self.out = nn.Linear(32*7*7,10)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   #(batch,32,7,7)
        x = x.view(x.size(0),-1) #(batch,32*7*7)
        feature = x
        output = self.out(x)
        return output


class LeNet5_fedlaw(ReparamModule):
    def __init__(self):
        super(LeNet5_fedlaw,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(       #(1*28*28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),    #(16*28*28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#(16*14*14)
        )
        self.conv2 = nn.Sequential(  # 16*14*14
            nn.Conv2d(16,32,5,1,2),  #32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32*7*7
        )
        self.out = nn.Linear(32*7*7,10)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   #(batch,32,7,7)
        x = x.view(x.size(0),-1) #(batch,32*7*7)
        feature = x
        output = self.out(x)
        return output


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(28*28, 200))
        self.layers.append(nn.Linear(200, 200))
        self.layers.append(nn.Linear(200, 10))

    def forward(self, x): # x: (batch, )
        x = x.reshape(-1, 28 * 28)
        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        x = self.layers[2](x)
        return x

class MLP_fedlaw(ReparamModule):
    def __init__(self):
        super(MLP_fedlaw, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(28*28, 200))
        self.layers.append(nn.Linear(200, 200))
        self.layers.append(nn.Linear(200, 10))

    def forward(self, x): # x: (batch, )
        x = x.reshape(-1, 28 * 28)
        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        x = self.layers[2](x)
        return x
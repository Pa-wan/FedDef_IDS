import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


# 网络架构
class Net(nn.Module):

    def __init__(self, input_size, hidden_ratio=0.75):
        super(Net, self).__init__()
        self.e1 = nn.Linear(input_size, ceil(input_size*hidden_ratio))
        self.d1 = nn.Linear(ceil(input_size*hidden_ratio), input_size)

    def forward(self, x):
        x = F.sigmoid(self.e1(x))
        x = F.sigmoid(self.d1(x))
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class polynom_act(nn.Module):

    def __init__(self, alpha=None, beta=None, c=None):
        super(polynom_act, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        return (self.alpha * (x ** 2) + self.beta * x + self.c)


class HENet(nn.Module):
    def __init__(self, num_classes, filters=128):
        super(HENet, self).__init__()
        first_f = filters
        second_f = filters * 2
        third_f = filters * 4
        self.conv1 = nn.Conv1d(3, first_f, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(first_f)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.act1 = polynom_act()
        self.avg_pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(first_f, second_f, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(second_f)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()
        self.act2 = polynom_act()
        self.avg_pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(second_f, third_f, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(third_f)
        self.bn3.weight.data.fill_(1)
        self.bn3.bias.data.zero_()
        self.act3 = polynom_act()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(third_f, num_classes)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 3, 480))
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.avg_pool1(out)
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avg_pool2(out)
        out = self.act3(self.bn3(self.conv3(out)))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

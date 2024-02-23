import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False

# Convolutional Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_normal_(self.conv.weight)
    
    def forward(self, x):
        return F.relu(self.batchNorm(self.conv(x)))


# Primary Capsule Layer
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=4, in_channels=32, out_channels=32, kernel_size=3, num_routes=32 * 4 * 4):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0)
            for _ in range(num_capsules)])

        self.init_weights()
        
    def init_weights(self):
        for conv in self.capsules:
            nn.init.xavier_normal_(conv.weight)
        
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return self.squash(u)

    # Non-Linear activation function: Squash
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class HighLevelCaps(nn.Module):
    def __init__(self, num_capsules=4, num_routes=32 * 4 * 4, in_channels=4, out_channels=16):
        super(HighLevelCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        # self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, num_routes, num_capsules, out_channels, in_channels)))
        # self.W = nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, num_routes, num_capsules, out_channels, in_channels)))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        # Routing algorithm
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


# Fully connected output layer
class FC(nn.Module):
    def __init__(self, input_width=8, input_height=8, input_channel=1, hc_out_channels=16, num_capsules=10):
        super(FC, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.hc_out_channels = hc_out_channels
        self.output_layer = nn.Linear(self.hc_out_channels*num_capsules, 3)
        
        # self.output_layer = nn.Sequential(nn.Linear(self.hc_out_channels*num_capsules, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.4), nn.Linear(512, 3))
            
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_normal_(self.output_layer.weight)
            
    def forward(self, x):
        return self.output_layer(x.view(-1, self.hc_out_channels*10))


class OxyNet(nn.Module):
    def __init__(self, config=None):
        super(OxyNet, self).__init__()
        
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels, config.pc_kernel_size, config.pc_num_routes)
            self.hc_capsules = HighLevelCaps(config.hc_num_capsules, config.hc_num_routes, config.hc_in_channels, config.hc_out_channels)
            self.fc = FC(config.input_width, config.input_height, config.cnn_in_channels, config.hc_out_channels, config.hc_num_capsules)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.hc_capsules = HighLevelCaps()
            self.fc = FC()
        
        self.ce_loss = nn.CrossEntropyLoss()

    
    def forward(self, data):
        output = self.hc_capsules(self.primary_capsules(self.conv_layer(data)))
        return self.fc(output)

    def loss(self, data, target):
        return self.ce_loss(data, target)
        
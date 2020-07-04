# Credit to https://github.com/Simon4Yan/eSPGAN/blob/master/py-spgan/models/models_spgan.py

import functools
import torch.nn as nn
import torch.nn.functional as F


def generator(pretrained=False):
    if not pretrained:
        model = Generator()
        return model

def discriminator(pretrained=False):
    if not pretrained:
        model = Discriminator()
        return model

def metric_net(pretrained=False):
    if not pretrained:
        model = Metric_Net()
        return model


class Generator(nn.Module):
    def __init__(self, dim=64):
        super(Generator, self).__init__()

        conv_bn_relu = conv_norm_act
        dconv_bn_relu = dconv_norm_act

        self.G = nn.Sequential(nn.ReflectionPad2d(3),
                                conv_bn_relu(3, dim * 1, 7, 1),
                                conv_bn_relu(dim * 1, dim * 2, 3, 2, 1),
                                conv_bn_relu(dim * 2, dim * 4, 3, 2, 1),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                dconv_bn_relu(dim * 4, dim * 2, 3, 2, 1, 1),
                                dconv_bn_relu(dim * 2, dim * 1, 3, 2, 1, 1),
                                nn.ReflectionPad2d(3),
                                nn.Conv2d(dim, 3, 7, 1),
                                nn.Tanh())
    def forward(self, x):
        return self.G(x)


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()

        lrelu = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        conv_bn_lrelu = functools.partial(conv_norm_act, relu=lrelu)

        self.D = nn.Sequential(nn.Conv2d(3, dim, 4, 2, 1), nn.LeakyReLU(0.2),
                                conv_bn_lrelu(dim * 1, dim * 2, 4, 2, 1),
                                conv_bn_lrelu(dim * 2, dim * 4, 4, 2, 1),
                                conv_bn_lrelu(dim * 4, dim * 8, 4, 1, (1, 2)),
                                nn.Conv2d(dim * 8, 1, 4, 1, (2, 1)))
    def forward(self, x):
        return self.D(x)


class Metric_Net(nn.Module):
    def __init__(self, dim=64):
        super(Metric_Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),
            Conv_Relu_Pool(dim, dim * 2),
            Conv_Relu_Pool(dim * 2, dim * 4))
            # Conv_Relu_Pool(dim * 4, dim * 8))

        self.fc1 = nn.Linear(2048, dim * 2, bias=None)
        self.relu1 = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dim * 2, dim, bias=None)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)

        return x


class Conv_Relu_Pool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_Relu_Pool, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2)
        )
    def forward(self, x):
        return self.layer(x)

def conv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm = functools.partial(nn.InstanceNorm2d, affine=False), relu=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        norm(out_dim),
        relu())

def dconv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                   output_padding=0, norm = functools.partial(nn.InstanceNorm2d, affine=False), relu=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=False),
        norm(out_dim),
        relu())

class ResiduleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResiduleBlock, self).__init__()

        conv_bn_relu = conv_norm_act

        self.layers = nn.Sequential(nn.ReflectionPad2d(1),
                                conv_bn_relu(in_dim, out_dim, 3, 1),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(out_dim, out_dim, 3, 1),
                                nn.InstanceNorm2d(out_dim))
    def forward(self, x):
        return x + self.layers(x)
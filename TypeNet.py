from Resnet import ResNet
from Resnet import Bottleneck
from SENET import SEBottleneck
from CBAM import CBAMBottleneck
import torch.nn as nn
from Densenet import *
from SKNet import SKNet
from torchvision.models import resnext101_32x8d


def SKNet101(num_classes=1):
    model = SKNet(num_classes, [3, 4, 23, 3])
    return model


def SEResnet50(num_classes=1):
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def SEResnet101(num_classes=1):
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model


def CBAMResnet50(num_classes=1):
    model = ResNet(CBAMBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def CBAMResnet101(num_classes=1):
    model = ResNet(CBAMBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def Resnet101(num_classes=1):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model

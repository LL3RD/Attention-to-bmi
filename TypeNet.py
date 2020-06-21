from Resnet import ResNet
from Resnet import Bottleneck
from SENET import SEBottleneck
from CBAM import CBAMBottleneck
import torch.nn as nn
from Densenet import *
from SKNet import SKNet
import torchvision.models as models


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


def AlexNet(num_classes=1):
    model = models.alexnet(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features,  num_features // 2),
        nn.ReLU(inplace=True),
        nn.Linear(num_features // 2, num_features // 4),
        nn.ReLU(inplace=True),
        nn.Linear(num_features // 4, num_features // 8),
        nn.ReLU(inplace=True),
        nn.Linear(num_features // 8, num_classes),
    )
    return model


def VGG16(num_classes=1):
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, num_classes),
    )
    return model


def GoogleNet(num_classes=1):
    model = models.googlenet(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, 128),
        nn.ReLU(True),
        nn.Linear(128, num_classes)
    )
    return model


def MobileNet(num_classes=1):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, 128),
        nn.ReLU(True),
        nn.Linear(128, num_classes)
    )
    return model


def Resnext101(num_classes=1):
    model = models.resnext101_32x8d(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, num_classes),
    )
    return model


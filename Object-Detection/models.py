import torch
from torch import nn
from torchvision import models

# Types:
#     1 : Simple Concatenation + ClassifierHead
#     2 : EO + (lambda * SAR) + ClassifierHead
#     3 : EO * (lambda * SAR) + ClassifierHead
#     4 : Simple Concatenation + 2 Layer MLP

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=10, type=1):
        super(ClassifierHead, self).__init__()
        self.type = type
        self.input_dim = input_dim
        self.num_classes = num_classes
        if type == 1:
            self.fc = nn.Linear(input_dim, num_classes)
        if type == 2:
            self.lamda = nn.Parameter(torch.ones(input_dim))
            self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, eo, sar):
        if self.type == 1:
            combined = torch.cat((eo, sar), 1)
            out = self.fc(combined)
        if self.type == 2:
            out = self.fc(eo + self.lamda * sar)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, last_hidden_dim=512, pretrained=False, freeze_backbone=False):
        super(ResNet18, self).__init__()
        self.model1 = models.resnet18(pretrained=pretrained)
        if freeze_backbone:
            for param in self.model1:
                param.requires_grad = False
        self.model1.fc = nn.Linear(last_hidden_dim, num_classes)

    def forward(self, x):
        return self.model1(x)


class MultiModalResNet18(nn.Module):
    def __init__(self, num_classes=10, last_hidden_dim=512, pretrained=False, freeze_backbone=False, type=1):
        super(MultiModalResNet18, self).__init__()
        self.num_classes = num_classes
        self.type = type
        self.model1 = models.resnet18(pretrained=pretrained)
        self.model2 = models.resnet18(pretrained=pretrained)
        if freeze_backbone:
            for param in self.model1:
                param.requires_grad = False
            for param in self.model2:
                param.requires_grad = False
        # del self.model1.fc
        # del self.model2.fc
        self.model1.fc = nn.Identity()
        self.model2.fc = nn.Identity()
        if type == 1:
            self.fc = ClassifierHead(input_dim=2*last_hidden_dim, num_classes=num_classes, type=self.type)
        if self.type == 2:
            self.fc = ClassifierHead(input_dim=last_hidden_dim, num_classes=num_classes, type=self.type)

    def forward(self, eo, sar):
        eo = self.model1(eo)
        sar = self.model2(sar)
        # print(eo.shape, sar.shape)
        # combined = torch.cat((eo, sar), 1)
        out = self.fc(eo, sar)
        return out, eo, sar
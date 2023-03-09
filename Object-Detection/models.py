import torch
from torch import nn
from torchvision import models

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

    def __init__(self, num_classes=10, last_hidden_dim=512, pretrained=False, freeze_backbone=False):
        super(MultiModalResNet18, self).__init__()
        self.model1 = models.resnet18(pretrained=pretrained)
        self.model2 = models.resnet18(pretrained=pretrained)
        del self.model1.fc
        del self.model2.fc
        if freeze_backbone:
            for param in self.model1:
                param.requires_grad = False
            for param in self.model2:
                param.requires_grad = False
        self.fc = nn.Linear(2*last_hidden_dim, num_classes)

    def forward(self, eo, sar):
        eo = self.model1(eo)
        sar = self.model2(sar)
        combined = torch.cat((eo, sar), 1)
        out = self.fc(combined)
        return out, eo, sar
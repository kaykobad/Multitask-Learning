import torch
from torch import nn
from torchvision import models

# Types:
#     1 : Simple Concatenation + ClassifierHead
#     2 : EO + (lambda * SAR) + ClassifierHead
#     3 : EO * (lambda * SAR) + ClassifierHead
#     4 : Simple Concatenation + 2 Layer MLP

class ClassifierType:
    linear = "Linear"
    mlp = "MLP"

class MLPClassifierHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_classes=10, type=1):
        super(MLPClassifierHead, self).__init__()
        self.type = type
        self.input_dim = input_dim
        self.num_classes = num_classes

        if type == 2 or type == 3:
            self.lamda = nn.Parameter(torch.ones(input_dim))
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, eo, sar):
        if self.type == 1:
            combined = torch.cat((eo, sar), 1)
            out = self.fc(combined)
        if self.type == 2:
            out = self.fc(eo + self.lamda * sar)
        if self.type == 3:
            out = self.fc(eo * (self.lamda * sar))
        return out


class ClassifierHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=10, type=1):
        super(ClassifierHead, self).__init__()
        self.type = type
        self.input_dim = input_dim
        self.num_classes = num_classes
        if type == 1:
            self.fc = nn.Linear(input_dim, num_classes)
        if type == 2 or type == 3:
            self.lamda = nn.Parameter(torch.ones(input_dim))
            self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, eo, sar):
        if self.type == 1:
            combined = torch.cat((eo, sar), 1)
            out = self.fc(combined)
        if self.type == 2:
            out = self.fc(eo + self.lamda * sar)
        if self.type == 3:
            out = self.fc(eo * (self.lamda * sar))
        return out


class CommonSpaceProjection(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, out_dim=256):
        super(CommonSpaceProjection, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, eo, sar):
        out1 = self.projection(eo)
        out2 = self.projection(sar)
        return out1, out2



class ResNet18(nn.Module):
    def __init__(self, num_classes=10, last_hidden_dim=512, pretrained=False, freeze_backbone=False):
        super(ResNet18, self).__init__()
        # print(pretrained)
        self.pretrained = pretrained
        self.pretrained_weights = "IMAGENET1K_V1" if pretrained else None
        self.model1 = models.resnet18(weights=self.pretrained_weights)
        if freeze_backbone:
            for param in self.model1:
                param.requires_grad = False
        self.model1.fc = nn.Linear(last_hidden_dim, num_classes)

    def forward(self, x):
        return self.model1(x)


# classifier can be either "Linear" of "MLP"
# Separate backbone for each modality
class MultiModalResNet18(nn.Module):
    def __init__(
        self, 
        num_classes=10, 
        last_hidden_dim=512, 
        pretrained=False, 
        freeze_backbone=False, 
        type=1, 
        classifier=ClassifierType.linear, 
        classifier_hidden_dim=512,
        enable_sup_con_projection=False):
        super(MultiModalResNet18, self).__init__()
        self.pretrained = pretrained
        self.pretrained_weights = "IMAGENET1K_V1" if pretrained else None
        self.num_classes = num_classes
        self.type = type
        self.enable_sup_con_projection = enable_sup_con_projection
        self.model1 = models.resnet18(weights=self.pretrained_weights)
        self.model2 = models.resnet18(weights=self.pretrained_weights)
        if freeze_backbone:
            for param in self.model1:
                param.requires_grad = False
            for param in self.model2:
                param.requires_grad = False
        # del self.model1.fc
        # del self.model2.fc
        self.model1.fc = nn.Identity()
        self.model2.fc = nn.Identity()

        if enable_sup_con_projection:
            self.projection = CommonSpaceProjection(input_dim=512, hidden_dim=256, out_dim=256)

        if classifier == ClassifierType.linear:
            if type == 1:
                self.fc = ClassifierHead(input_dim=2*last_hidden_dim, num_classes=num_classes, type=self.type)
            if self.type == 2 or self.type == 3:
                self.fc = ClassifierHead(input_dim=last_hidden_dim, num_classes=num_classes, type=self.type)
        elif classifier == ClassifierType.mlp:
            if type == 1:
                self.fc = MLPClassifierHead(input_dim=2*last_hidden_dim, num_classes=num_classes, type=self.type, hidden_dim=classifier_hidden_dim)
            if self.type == 2 or self.type == 3:
                self.fc = MLPClassifierHead(input_dim=last_hidden_dim, num_classes=num_classes, type=self.type, hidden_dim=classifier_hidden_dim)

    def forward(self, eo, sar):
        eo = self.model1(eo)
        sar = self.model2(sar)
        # print(eo.shape, sar.shape)
        # combined = torch.cat((eo, sar), 1)
        out = self.fc(eo, sar)
        if self.enable_sup_con_projection:
            eo, sar = self.projection(eo, sar)
        return out, eo, sar


# classifier can be either "Linear" of "MLP"
class MultiModalResNet18SharedBackbone(nn.Module):
    def __init__(
        self, 
        num_classes=10, 
        last_hidden_dim=512, 
        pretrained=False, 
        freeze_backbone=False, 
        type=1, 
        classifier=ClassifierType.linear, 
        classifier_hidden_dim=512,
        enable_sup_con_projection=False):
        super(MultiModalResNet18SharedBackbone, self).__init__()
        self.num_classes = num_classes
        self.type = type
        self.enable_sup_con_projection = enable_sup_con_projection
        self.pretrained = pretrained
        self.pretrained_weights = "IMAGENET1K_V1" if pretrained else None
        self.model = models.resnet18(weights=self.pretrained_weights)
        if freeze_backbone:
            for param in self.model1:
                param.requires_grad = False
            for param in self.model2:
                param.requires_grad = False
        # del self.model1.fc
        # del self.model2.fc
        self.model.fc = nn.Identity()

        if enable_sup_con_projection:
            self.projection = CommonSpaceProjection(input_dim=512, hidden_dim=256, out_dim=256)

        if classifier == ClassifierType.linear:
            if type == 1:
                self.fc = ClassifierHead(input_dim=2*last_hidden_dim, num_classes=num_classes, type=self.type)
            if self.type == 2 or self.type == 3:
                self.fc = ClassifierHead(input_dim=last_hidden_dim, num_classes=num_classes, type=self.type)
        elif classifier == ClassifierType.mlp:
            if type == 1:
                self.fc = MLPClassifierHead(input_dim=2*last_hidden_dim, num_classes=num_classes, type=self.type, hidden_dim=classifier_hidden_dim)
            if self.type == 2 or self.type == 3:
                self.fc = MLPClassifierHead(input_dim=last_hidden_dim, num_classes=num_classes, type=self.type, hidden_dim=classifier_hidden_dim)

    def forward(self, eo, sar):
        eo = self.model(eo)
        sar = self.model(sar)
        # print(eo.shape, sar.shape)
        # combined = torch.cat((eo, sar), 1)
        out = self.fc(eo, sar)
        if self.enable_sup_con_projection:
            eo, sar = self.projection(eo, sar)
        return out, eo, sar
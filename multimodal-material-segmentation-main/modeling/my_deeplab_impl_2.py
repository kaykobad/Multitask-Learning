'''
These codes are from this Repo: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/resnet.py
'''


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class ResSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ResSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, input_dim=3):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.input_dim = input_dim

        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(self.input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ResNetWithLayerOutputs(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, input_dim=3):
        self.inplanes = 64
        super(ResNetWithLayerOutputs, self).__init__()

        self.input_dim = input_dim

        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(self.input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        layer1_out_feature = x
        x = self.layer2(x)
        layer2_out_feature = x
        x = self.layer3(x)
        layer3_out_feature = x
        x = self.layer4(x)
        layer4_out_feature = x
        return x, low_level_feat, layer1_out_feature, layer2_out_feature, layer3_out_feature, layer4_out_feature

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ResNetWithFeatureFusion(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, input_dim=3):
        self.inplanes = 64
        super(ResNetWithFeatureFusion, self).__init__()

        self.input_dim = input_dim

        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(self.input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input, layer1_out_features=[], layer2_out_features=[], layer3_out_features=[], layer4_out_features=[]):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        for f in layer1_out_features:
            x = x.clone() + f
        low_level_feat = x

        x = self.layer2(x)
        for f in layer2_out_features:
            x = x.clone() + f

        x = self.layer3(x)
        for f in layer3_out_features:
            x = x.clone() + f

        x = self.layer4(x)
        for f in layer4_out_features:
            x = x.clone() + f

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(output_stride, BatchNorm, pretrained=True, input_dim=3):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, input_dim=input_dim)
    return model

def ResNet101WithLayerOutputs(output_stride, BatchNorm, pretrained=True, input_dim=3):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetWithLayerOutputs(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, input_dim=input_dim)
    return model

def ResNet101WithFeatureFusion(output_stride, BatchNorm, pretrained=True, input_dim=3):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetWithFeatureFusion(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, input_dim=input_dim)
    return model


def build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=True):
    if backbone == 'resnet':
        return ResNet101(output_stride, BatchNorm, input_dim=input_dim, pretrained=pretrained)
    else:
        raise NotImplementedError

def build_custom_resnet_backbone(type, output_stride, BatchNorm, input_dim=3, pretrained=True):
    if type == 1:
        return ResNet101WithLayerOutputs(output_stride, BatchNorm, input_dim=input_dim, pretrained=pretrained)
    elif type == 2:
        return ResNet101WithFeatureFusion(output_stride, BatchNorm, input_dim=input_dim, pretrained=pretrained)
    else:
        raise NotImplementedError


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        # TODO: Flip it to 0.5
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, num_modalities=1, enable_se=False):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256 * num_modalities
            last_conv_input = 48 + 256 * num_modalities
        else:
            raise NotImplementedError

        self.enable_se  = enable_se

        if self.enable_se:
            self.se = SELayer(last_conv_input)

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(last_conv_input, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        if self.enable_se:
            x = self.se(x)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, num_modalities=1, enable_se=False):
    return Decoder(num_classes, backbone, BatchNorm, num_modalities, enable_se)


class AdaptiveDecoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, num_modalities=1, enable_se=False):
        super(AdaptiveDecoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256 * num_modalities
            low_level_outplane = 48 * num_modalities
            last_conv_input = low_level_outplane + 256 * num_modalities
        else:
            raise NotImplementedError

        self.enable_se  = enable_se

        if self.enable_se:
            self.se = SELayer(last_conv_input)

        self.conv1 = nn.Conv2d(low_level_inplanes, low_level_outplane, 1, bias=False)
        self.bn1 = BatchNorm(low_level_outplane)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(last_conv_input, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        if self.enable_se:
            x = self.se(x)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_adaptive_decoder(num_classes, backbone, BatchNorm, num_modalities=1, enable_se=False):
    return AdaptiveDecoder(num_classes, backbone, BatchNorm, num_modalities, enable_se)


class AdaptiveDecoder2(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, num_modalities=1, enable_se=False):
        super(AdaptiveDecoder2, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256 * num_modalities
            low_level_outplane = 48 * num_modalities
            last_conv_input = low_level_outplane + 256 * num_modalities
        else:
            raise NotImplementedError

        self.enable_se  = enable_se
        self.num_modalities = num_modalities

        if self.enable_se:
            self.se = SELayer(last_conv_input)

        # self.conv1 = nn.Conv2d(low_level_inplanes, low_level_outplane, 1, bias=False)
        # self.bn1 = BatchNorm(low_level_outplane)
        # self.relu = nn.ReLU()

        # self.cbr11 = [nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
        #                            BatchNorm(low_level_outplane),
        #                            nn.ReLU()) for i in range(self.num_modalities)]

        self.last_conv = nn.Sequential(nn.Conv2d(last_conv_input, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        # low_level_feat = self.conv1(low_level_feat)
        # low_level_feat = self.bn1(low_level_feat)
        # low_level_feat = self.relu(low_level_feat)

        # low_level_feat = []
        # for i in range(self.num_modalities):
        #     low_level_feat.append(self.cbr11[i](low_level_feats[i]))

        # if self.num_modalities == 1:
        #     low_level_feat = low_level_feat[0]
        # else:
        #     low_level_feat = torch.cat(low_level_feat, dim=1) 
 
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        if self.enable_se:
            x = self.se(x)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_adaptive_decoder2(num_classes, backbone, BatchNorm, num_modalities=1, enable_se=False):
    return AdaptiveDecoder2(num_classes, backbone, BatchNorm, num_modalities, enable_se)


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class DeepFuseLab(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_segmap=False,
    ): 
        super(DeepFuseLab, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_segmap = use_segmap

        self.backbones = []
        self.decoders = []

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_custom_resnet_backbone(2, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.backbones.append(self.rgb_backbone)
        self.decoders.append(self.rgb_aspp)
        self.decoders.append(self.decoder)

        if self.use_nir:
            self.nir_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.backbones.append(self.nir_backbone)
            self.decoders.append(self.nir_aspp)
        if self.use_aolp:
            self.aolp_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.backbones.append(self.aolp_backbone)
            self.decoders.append(self.aolp_aspp)
        if self.use_dolp:
            self.dolp_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.backbones.append(self.dolp_backbone)
            self.decoders.append(self.dolp_aspp)
        if self.use_segmap:
            self.segmap_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.segmap_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.backbones.append(self.segmap_backbone)
            self.decoders.append(self.segmap_aspp)

        self.freeze_bn = freeze_bn

    def forward(self, rgb, nir=None, aolp=None, dolp=None, segmap=None):
        layer1_outputs = []
        layer2_outputs = []
        layer3_outputs = []
        layer4_outputs = []
        final_features = []

        if self.use_nir and nir is not None:
            x, low_level_feat, l1, l2, l3, l4 = self.nir_backbone(nir)
            x = self.nir_aspp(x)
            layer1_outputs.append(l1)
            layer2_outputs.append(l2)
            layer3_outputs.append(l3)
            layer4_outputs.append(l4)
            final_features.append(x)
        if self.use_aolp and aolp is not None:
            x, low_level_feat, l1, l2, l3, l4 = self.aolp_backbone(aolp)
            x = self.aolp_aspp(x)
            layer1_outputs.append(l1)
            layer2_outputs.append(l2)
            layer3_outputs.append(l3)
            layer4_outputs.append(l4)
            final_features.append(x)
        if self.use_dolp and dolp is not None:
            x, low_level_feat, l1, l2, l3, l4 = self.dolp_backbone(dolp)
            x = self.dolp_aspp(x)
            layer1_outputs.append(l1)
            layer2_outputs.append(l2)
            layer3_outputs.append(l3)
            layer4_outputs.append(l4)
            final_features.append(x)
        if self.use_segmap and segmap is not None:
            x, low_level_feat, l1, l2, l3, l4 = self.segmap_backbone(segmap)
            x = self.segmap_aspp(x)
            layer1_outputs.append(l1)
            layer2_outputs.append(l2)
            layer3_outputs.append(l3)
            layer4_outputs.append(l4)
            final_features.append(x)

        x, low_level_feat = self.rgb_backbone(rgb, layer1_outputs, layer2_outputs, layer3_outputs, layer4_outputs)
        x = self.rgb_aspp(x)
        for f in final_features:
            x += f
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class MMDeepLab(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_segmap=False,
        enable_se= False,
    ): 
        super(MMDeepLab, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_segmap = use_segmap
        self.enable_se = enable_se
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.backbones.append(self.rgb_backbone)
        self.decoders.append(self.rgb_aspp)

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.backbones.append(self.nir_backbone)
            self.decoders.append(self.nir_aspp)
        if self.use_aolp:
            self.num_modalities += 1
            self.aolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.backbones.append(self.aolp_backbone)
            self.decoders.append(self.aolp_aspp)
        if self.use_dolp:
            self.num_modalities += 1
            self.dolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.backbones.append(self.dolp_backbone)
            self.decoders.append(self.dolp_aspp)
        # if self.use_segmap:
        #     self.segmap_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
        #     self.segmap_aspp = build_aspp(backbone, output_stride, BatchNorm)
        #     self.backbones.append(self.segmap_backbone)
        #     self.decoders.append(self.segmap_aspp)

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities, enable_se=self.enable_se)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, aolp=None, dolp=None, segmap=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x = [x1]
        low_level_feat = [low_level_feat1]

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x.append(x2)
            low_level_feat.append(low_level_feat2)
        if self.use_aolp and aolp is not None:
            x3, low_level_feat3 = self.aolp_backbone(aolp)
            x3 = self.aolp_aspp(x3)
            x.append(x3)
            low_level_feat.append(low_level_feat3)
        if self.use_dolp and dolp is not None:
            x4, low_level_feat4 = self.dolp_backbone(dolp)
            x4 = self.dolp_aspp(x4)
            x.append(x4)
            low_level_feat.append(low_level_feat4)

        if self.num_modalities == 1:
            x = x1
            low_level_feat = low_level_feat1
        else:
            x = torch.cat(x, dim=1)
            low_level_feat = torch.cat(low_level_feat, dim=1) 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class MMDeepLab2(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_segmap=False,
        enable_se= False,
    ): 
        super(MMDeepLab2, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_segmap = use_segmap
        self.enable_se = enable_se
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.rgb_se_llf = SELayer(256)
        self.rgb_se_hlf = SELayer(256)
        self.backbones.extend([self.rgb_backbone, self.rgb_se_llf, self.rgb_se_hlf])
        self.decoders.append(self.rgb_aspp)

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.nir_se_llf = SELayer(256)
            self.nir_se_hlf = SELayer(256)
            self.backbones.extend([self.nir_backbone, self.nir_se_llf, self.nir_se_hlf])
            self.decoders.append(self.nir_aspp)
        if self.use_aolp:
            self.num_modalities += 1
            self.aolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.aolp_se_llf = SELayer(256)
            self.aolp_se_hlf = SELayer(256)
            self.backbones.extend([self.aolp_backbone, self.aolp_se_llf, self.aolp_se_hlf])
            self.decoders.append(self.aolp_aspp)
        if self.use_dolp:
            self.num_modalities += 1
            self.dolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.dolp_se_llf = SELayer(256)
            self.dolp_se_hlf = SELayer(256)
            self.backbones.extend([self.dolp_backbone, self.dolp_se_llf, self.dolp_se_hlf])
            self.decoders.append(self.dolp_aspp)
        # if self.use_segmap:
        #     self.segmap_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
        #     self.segmap_aspp = build_aspp(backbone, output_stride, BatchNorm)
        #     self.backbones.append(self.segmap_backbone)
        #     self.decoders.append(self.segmap_aspp)

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities, enable_se=self.enable_se)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, aolp=None, dolp=None, segmap=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x1 = self.rgb_se_hlf(x1)
        low_level_feat1 = self.rgb_se_llf(low_level_feat1)
        # print("---------", x1.shape, low_level_feat1.shape)
        # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
        x = [x1]
        low_level_feat = [low_level_feat1]

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x2 = self.nir_se_hlf(x2)
            low_level_feat2 = self.nir_se_llf(low_level_feat2)
            x.append(x2)
            low_level_feat.append(low_level_feat2)
        if self.use_aolp and aolp is not None:
            x3, low_level_feat3 = self.aolp_backbone(aolp)
            x3 = self.aolp_aspp(x3)
            x3 = self.aolp_se_hlf(x3)
            low_level_feat3 = self.aolp_se_llf(low_level_feat3)
            x.append(x3)
            low_level_feat.append(low_level_feat3)
        if self.use_dolp and dolp is not None:
            x4, low_level_feat4 = self.dolp_backbone(dolp)
            x4 = self.dolp_aspp(x4)
            x4 = self.dolp_se_hlf(x4)
            low_level_feat4 = self.dolp_se_llf(low_level_feat4)
            x.append(x4)
            low_level_feat.append(low_level_feat4)

        if self.num_modalities == 1:
            x = x1
            low_level_feat = low_level_feat1
        else:
            x = torch.cat(x, dim=1)
            low_level_feat = torch.cat(low_level_feat, dim=1) 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class MMDeepLabResSE(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_segmap=False,
        enable_se= False,
    ): 
        super(MMDeepLabResSE, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_segmap = use_segmap
        self.enable_se = enable_se
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.rgb_se_llf = ResSELayer(256)
        self.rgb_se_hlf = ResSELayer(256)
        self.backbones.extend([self.rgb_backbone, self.rgb_se_llf, self.rgb_se_hlf])
        self.decoders.append(self.rgb_aspp)

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.nir_se_llf = ResSELayer(256)
            self.nir_se_hlf = ResSELayer(256)
            self.backbones.extend([self.nir_backbone, self.nir_se_llf, self.nir_se_hlf])
            self.decoders.append(self.nir_aspp)
        if self.use_aolp:
            self.num_modalities += 1
            self.aolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.aolp_se_llf = ResSELayer(256)
            self.aolp_se_hlf = ResSELayer(256)
            self.backbones.extend([self.aolp_backbone, self.aolp_se_llf, self.aolp_se_hlf])
            self.decoders.append(self.aolp_aspp)
        if self.use_dolp:
            self.num_modalities += 1
            self.dolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.dolp_se_llf = ResSELayer(256)
            self.dolp_se_hlf = ResSELayer(256)
            self.backbones.extend([self.dolp_backbone, self.dolp_se_llf, self.dolp_se_hlf])
            self.decoders.append(self.dolp_aspp)
        # if self.use_segmap:
        #     self.segmap_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
        #     self.segmap_aspp = build_aspp(backbone, output_stride, BatchNorm)
        #     self.backbones.append(self.segmap_backbone)
        #     self.decoders.append(self.segmap_aspp)

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities, enable_se=self.enable_se)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, aolp=None, dolp=None, segmap=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x1 = self.rgb_se_hlf(x1)
        low_level_feat1 = self.rgb_se_llf(low_level_feat1)
        # print("---------", x1.shape, low_level_feat1.shape)
        # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
        x = [x1]
        low_level_feat = [low_level_feat1]

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x2 = self.nir_se_hlf(x2)
            low_level_feat2 = self.nir_se_llf(low_level_feat2)
            x.append(x2)
            low_level_feat.append(low_level_feat2)
        if self.use_aolp and aolp is not None:
            x3, low_level_feat3 = self.aolp_backbone(aolp)
            x3 = self.aolp_aspp(x3)
            x3 = self.aolp_se_hlf(x3)
            low_level_feat3 = self.aolp_se_llf(low_level_feat3)
            x.append(x3)
            low_level_feat.append(low_level_feat3)
        if self.use_dolp and dolp is not None:
            x4, low_level_feat4 = self.dolp_backbone(dolp)
            x4 = self.dolp_aspp(x4)
            x4 = self.dolp_se_hlf(x4)
            low_level_feat4 = self.dolp_se_llf(low_level_feat4)
            x.append(x4)
            low_level_feat.append(low_level_feat4)

        if self.num_modalities == 1:
            x = x1
            low_level_feat = low_level_feat1
        else:
            x = torch.cat(x, dim=1)
            low_level_feat = torch.cat(low_level_feat, dim=1) 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class MMDeepLabAD(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_segmap=False,
        enable_se= False,
    ): 
        super(MMDeepLabAD, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_segmap = use_segmap
        self.enable_se = enable_se
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.rgb_se_llf = SELayer(256)
        self.rgb_se_hlf = SELayer(256)
        self.backbones.extend([self.rgb_backbone, self.rgb_se_llf, self.rgb_se_hlf])
        self.decoders.append(self.rgb_aspp)

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.nir_se_llf = SELayer(256)
            self.nir_se_hlf = SELayer(256)
            self.backbones.extend([self.nir_backbone, self.nir_se_llf, self.nir_se_hlf])
            self.decoders.append(self.nir_aspp)
        if self.use_aolp:
            self.num_modalities += 1
            self.aolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.aolp_se_llf = SELayer(256)
            self.aolp_se_hlf = SELayer(256)
            self.backbones.extend([self.aolp_backbone, self.aolp_se_llf, self.aolp_se_hlf])
            self.decoders.append(self.aolp_aspp)
        if self.use_dolp:
            self.num_modalities += 1
            self.dolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.dolp_se_llf = SELayer(256)
            self.dolp_se_hlf = SELayer(256)
            self.backbones.extend([self.dolp_backbone, self.dolp_se_llf, self.dolp_se_hlf])
            self.decoders.append(self.dolp_aspp)
        # if self.use_segmap:
        #     self.segmap_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
        #     self.segmap_aspp = build_aspp(backbone, output_stride, BatchNorm)
        #     self.backbones.append(self.segmap_backbone)
        #     self.decoders.append(self.segmap_aspp)

        self.decoder = build_adaptive_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities, enable_se=self.enable_se)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, aolp=None, dolp=None, segmap=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x1 = self.rgb_se_hlf(x1)
        low_level_feat1 = self.rgb_se_llf(low_level_feat1)
        # print("---------", x1.shape, low_level_feat1.shape)
        # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
        x = [x1]
        low_level_feat = [low_level_feat1]

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x2 = self.nir_se_hlf(x2)
            low_level_feat2 = self.nir_se_llf(low_level_feat2)
            x.append(x2)
            low_level_feat.append(low_level_feat2)
        if self.use_aolp and aolp is not None:
            x3, low_level_feat3 = self.aolp_backbone(aolp)
            x3 = self.aolp_aspp(x3)
            x3 = self.aolp_se_hlf(x3)
            low_level_feat3 = self.aolp_se_llf(low_level_feat3)
            x.append(x3)
            low_level_feat.append(low_level_feat3)
        if self.use_dolp and dolp is not None:
            x4, low_level_feat4 = self.dolp_backbone(dolp)
            x4 = self.dolp_aspp(x4)
            x4 = self.dolp_se_hlf(x4)
            low_level_feat4 = self.dolp_se_llf(low_level_feat4)
            x.append(x4)
            low_level_feat.append(low_level_feat4)

        if self.num_modalities == 1:
            x = x1
            low_level_feat = low_level_feat1
        else:
            x = torch.cat(x, dim=1)
            low_level_feat = torch.cat(low_level_feat, dim=1) 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

class MMDeepLabAD2(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_segmap=False,
        enable_se= False,
    ): 
        super(MMDeepLabAD2, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_segmap = use_segmap
        self.enable_se = enable_se
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.rgb_se_llf = SELayer(256)
        self.rgb_se_hlf = SELayer(256)
        self.rgb_cbr11 = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                       BatchNorm(48),
                                       nn.ReLU())
        self.backbones.extend([self.rgb_backbone, self.rgb_se_llf, self.rgb_se_hlf])
        self.decoders.extend([self.rgb_aspp, self.rgb_cbr11])

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.nir_se_llf = SELayer(256)
            self.nir_se_hlf = SELayer(256)
            self.nir_cbr11 = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                       BatchNorm(48),
                                       nn.ReLU())
            self.backbones.extend([self.nir_backbone, self.nir_se_llf, self.nir_se_hlf])
            self.decoders.extend([self.nir_aspp, self.nir_cbr11])
        if self.use_aolp:
            self.num_modalities += 1
            self.aolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.aolp_se_llf = SELayer(256)
            self.aolp_se_hlf = SELayer(256)
            self.aolp_cbr11 = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                       BatchNorm(48),
                                       nn.ReLU())
            self.backbones.extend([self.aolp_backbone, self.aolp_se_llf, self.aolp_se_hlf])
            self.decoders.extend([self.aolp_aspp, self.aolp_cbr11])
        if self.use_dolp:
            self.num_modalities += 1
            self.dolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.dolp_se_llf = SELayer(256)
            self.dolp_se_hlf = SELayer(256)
            self.dolp_cbr11 = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                       BatchNorm(48),
                                       nn.ReLU())
            self.backbones.extend([self.dolp_backbone, self.dolp_se_llf, self.dolp_se_hlf])
            self.decoders.extend([self.dolp_aspp, self.dolp_cbr11])
        # if self.use_segmap:
        #     self.segmap_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
        #     self.segmap_aspp = build_aspp(backbone, output_stride, BatchNorm)
        #     self.backbones.append(self.segmap_backbone)
        #     self.decoders.append(self.segmap_aspp)

        self.decoder = build_adaptive_decoder2(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities, enable_se=self.enable_se)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, aolp=None, dolp=None, segmap=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x1 = self.rgb_se_hlf(x1)
        low_level_feat1 = self.rgb_se_llf(low_level_feat1)
        low_level_feat1 = self.rgb_cbr11(low_level_feat1)
        # print("---------", x1.shape, low_level_feat1.shape)
        # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
        x = [x1]
        low_level_feat = [low_level_feat1]

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x2 = self.nir_se_hlf(x2)
            low_level_feat2 = self.nir_se_llf(low_level_feat2)
            low_level_feat2 = self.nir_cbr11(low_level_feat2)
            x.append(x2)
            low_level_feat.append(low_level_feat2)
        if self.use_aolp and aolp is not None:
            x3, low_level_feat3 = self.aolp_backbone(aolp)
            x3 = self.aolp_aspp(x3)
            x3 = self.aolp_se_hlf(x3)
            low_level_feat3 = self.aolp_se_llf(low_level_feat3)
            low_level_feat3 = self.aolp_cbr11(low_level_feat3)
            x.append(x3)
            low_level_feat.append(low_level_feat3)
        if self.use_dolp and dolp is not None:
            x4, low_level_feat4 = self.dolp_backbone(dolp)
            x4 = self.dolp_aspp(x4)
            x4 = self.dolp_se_hlf(x4)
            low_level_feat4 = self.dolp_se_llf(low_level_feat4)
            low_level_feat4 = self.dolp_cbr11(low_level_feat4)
            x.append(x4)
            low_level_feat.append(low_level_feat4)

        if self.num_modalities == 1:
            x = x1
            low_level_feat = low_level_feat1
        else:
            x = torch.cat(x, dim=1)
            low_level_feat = torch.cat(low_level_feat, dim=1) 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class MMDeepLabResSEAD(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_segmap=False,
        enable_se= False,
    ): 
        super(MMDeepLabResSEAD, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_segmap = use_segmap
        self.enable_se = enable_se
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.rgb_se_llf = ResSELayer(256)
        self.rgb_se_hlf = ResSELayer(256)
        self.backbones.extend([self.rgb_backbone, self.rgb_se_llf, self.rgb_se_hlf])
        self.decoders.append(self.rgb_aspp)

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.nir_se_llf = ResSELayer(256)
            self.nir_se_hlf = ResSELayer(256)
            self.backbones.extend([self.nir_backbone, self.nir_se_llf, self.nir_se_hlf])
            self.decoders.append(self.nir_aspp)
        if self.use_aolp:
            self.num_modalities += 1
            self.aolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.aolp_se_llf = ResSELayer(256)
            self.aolp_se_hlf = ResSELayer(256)
            self.backbones.extend([self.aolp_backbone, self.aolp_se_llf, self.aolp_se_hlf])
            self.decoders.append(self.aolp_aspp)
        if self.use_dolp:
            self.num_modalities += 1
            self.dolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.dolp_se_llf = ResSELayer(256)
            self.dolp_se_hlf = ResSELayer(256)
            self.backbones.extend([self.dolp_backbone, self.dolp_se_llf, self.dolp_se_hlf])
            self.decoders.append(self.dolp_aspp)
        # if self.use_segmap:
        #     self.segmap_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
        #     self.segmap_aspp = build_aspp(backbone, output_stride, BatchNorm)
        #     self.backbones.append(self.segmap_backbone)
        #     self.decoders.append(self.segmap_aspp)

        self.decoder = build_adaptive_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities, enable_se=self.enable_se)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, aolp=None, dolp=None, segmap=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x1 = self.rgb_se_hlf(x1)
        low_level_feat1 = self.rgb_se_llf(low_level_feat1)
        # print("---------", x1.shape, low_level_feat1.shape)
        # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
        x = [x1]
        low_level_feat = [low_level_feat1]

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x2 = self.nir_se_hlf(x2)
            low_level_feat2 = self.nir_se_llf(low_level_feat2)
            x.append(x2)
            low_level_feat.append(low_level_feat2)
        if self.use_aolp and aolp is not None:
            x3, low_level_feat3 = self.aolp_backbone(aolp)
            x3 = self.aolp_aspp(x3)
            x3 = self.aolp_se_hlf(x3)
            low_level_feat3 = self.aolp_se_llf(low_level_feat3)
            x.append(x3)
            low_level_feat.append(low_level_feat3)
        if self.use_dolp and dolp is not None:
            x4, low_level_feat4 = self.dolp_backbone(dolp)
            x4 = self.dolp_aspp(x4)
            x4 = self.dolp_se_hlf(x4)
            low_level_feat4 = self.dolp_se_llf(low_level_feat4)
            x.append(x4)
            low_level_feat.append(low_level_feat4)

        if self.num_modalities == 1:
            x = x1
            low_level_feat = low_level_feat1
        else:
            x = torch.cat(x, dim=1)
            low_level_feat = torch.cat(low_level_feat, dim=1) 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

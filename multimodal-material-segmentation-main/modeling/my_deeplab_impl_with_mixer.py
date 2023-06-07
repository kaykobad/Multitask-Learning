'''
These codes are from this Repo: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/resnet.py
'''


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import numpy as np
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
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


class MaskBlock(nn.Module):
    def __init__(self, shape=(256, 256, 256)):
        super(MaskBlock, self).__init__()

        self.c = nn.Parameter(torch.ones((1, shape[0])), requires_grad=True)
        self.h = nn.Parameter(torch.ones((1, shape[1])), requires_grad=True)
        self.w = nn.Parameter(torch.ones((1, shape[2])), requires_grad=True)
        self.shape = shape

    def forward(self, x):
        # f = torch.mul(torch.mul(self.c.view(1, -1, 1, 1), self.h.view(1, 1, -1, 1)), self.w.view(1, 1, 1, -1))
        x = torch.mul(x, self.c.view(1, -1, 1, 1))
        x = torch.mul(x, self.h.view(1, 1, -1, 1))
        x = torch.mul(x, self.w.view(1, 1, 1, -1))
        # self.c.retain_grad()
        # self.h.retain_grad()
        # self.w.retain_grad()
        # print(f.shape)
        # x = torch.mul(x, f)
        # print(x)
        return x


class SEMaskBlock(nn.Module):
    def __init__(self, channel, shape):
        super(SEMaskBlock, self).__init__()

        self.se = SEBlock(channel=channel)
        self.mask = MaskBlock(shape=shape)

    def forward(self, x):
        x = self.se(x)
        x = self.mask(x)
        return x


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


def resnet101(output_stride, BatchNorm, pretrained=True, input_dim=3):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, input_dim=input_dim)
    return model


def build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=True):
    if backbone == 'resnet':
        return resnet101(output_stride, BatchNorm, input_dim=input_dim, pretrained=pretrained)
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
    def __init__(self, num_classes, backbone, BatchNorm, num_modalities=1):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
            last_conv_input = 48 + 256
        else:
            raise NotImplementedError

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

def build_decoder(num_classes, backbone, BatchNorm, num_modalities=1):
    return Decoder(num_classes, backbone, BatchNorm, num_modalities)


class DecoderWithSEMask(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, num_modalities=1):
        super(DecoderWithSEMask, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
            last_conv_input = 48 + 256
        else:
            raise NotImplementedError

        self.low_level_feature_shape = (last_conv_input, 128, 128)
        self.semask = SEMaskBlock(last_conv_input, self.low_level_feature_shape)

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

        x = self.semask(x)
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

def build_decoder_with_semask(num_classes, backbone, BatchNorm, num_modalities=1):
    return DecoderWithSEMask(num_classes, backbone, BatchNorm, num_modalities)


class MMDeepLabSEMask(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_pol=False,
        use_segmap=False,
        enable_se= False,
    ): 
        super(MMDeepLabSEMask, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_pol = use_pol
        self.use_segmap = use_segmap
        self.enable_se = enable_se
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1

        self.low_level_feature_channels = 256
        self.low_level_feature_shape = (256, 128, 128)

        self.high_level_feature_channels = 256
        self.high_level_feature_shape = (256, 32, 32)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.rgb_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
        self.rgb_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
        self.backbones.extend([self.rgb_backbone, self.rgb_llf, self.rgb_hlf])
        self.decoders.append(self.rgb_aspp)

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.nir_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.nir_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.nir_backbone, self.nir_llf, self.nir_hlf])
            self.decoders.append(self.nir_aspp)
        if self.use_aolp:
            self.num_modalities += 1
            self.aolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.aolp_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.aolp_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.aolp_backbone, self.aolp_llf, self.aolp_hlf])
            self.decoders.append(self.aolp_aspp)
        if self.use_dolp:
            self.num_modalities += 1
            self.dolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.dolp_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.dolp_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.dolp_backbone, self.dolp_llf, self.dolp_hlf])
            self.decoders.append(self.dolp_aspp)
        if self.use_pol:
            self.num_modalities += 1
            self.pol_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=False)
            self.pol_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.pol_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.pol_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.pol_backbone, self.pol_llf, self.pol_hlf])
            self.decoders.append(self.pol_aspp)
        # if self.use_segmap:
        #     self.segmap_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
        #     self.segmap_aspp = build_aspp(backbone, output_stride, BatchNorm)
        #     self.backbones.append(self.segmap_backbone)
        #     self.decoders.append(self.segmap_aspp)

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, aolp=None, dolp=None, pol=None, segmap=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x1 = self.rgb_hlf(x1)
        low_level_feat1 = self.rgb_llf(low_level_feat1)
        # print("---------", x1.shape, low_level_feat1.shape)
        # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
        x = x1
        low_level_feat = low_level_feat1

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x2 = self.nir_hlf(x2)
            low_level_feat2 = self.nir_llf(low_level_feat2)
            x = torch.add(x, x2)
            low_level_feat = torch.add(low_level_feat, low_level_feat2)
        if self.use_aolp and aolp is not None:
            x3, low_level_feat3 = self.aolp_backbone(aolp)
            x3 = self.aolp_aspp(x3)
            x3 = self.aolp_se_hlf(x3)
            low_level_feat3 = self.aolp_se_llf(low_level_feat3)
            x = torch.add(x, x3)
            low_level_feat = torch.add(low_level_feat, low_level_feat3)
        if self.use_dolp and dolp is not None:
            x4, low_level_feat4 = self.dolp_backbone(dolp)
            x4 = self.dolp_aspp(x4)
            x4 = self.dolp_se_hlf(x4)
            low_level_feat4 = self.dolp_se_llf(low_level_feat4)
            x = torch.add(x, x4)
            low_level_feat = torch.add(low_level_feat, low_level_feat4)
        if self.use_pol and pol is not None:
            x5, low_level_feat5 = self.pol_backbone(pol)
            x5 = self.pol_aspp(x5)
            x5 = self.pol_hlf(x5)
            low_level_feat5 = self.pol_llf(low_level_feat5)
            x = torch.add(x, x5)
            low_level_feat = torch.add(low_level_feat, low_level_feat5)

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
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
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
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class MMDeepLabSEMask2(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_pol=False,
        use_segmap=False,
        enable_se= False,
    ): 
        super(MMDeepLabSEMask2, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_pol = use_pol
        self.use_segmap = use_segmap
        self.enable_se = enable_se
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1

        self.low_level_feature_channels = 256
        self.low_level_feature_shape = (256, 128, 128)

        self.high_level_feature_channels = 256
        self.high_level_feature_shape = (256, 32, 32)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.rgb_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
        self.rgb_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
        self.backbones.extend([self.rgb_backbone, self.rgb_llf, self.rgb_hlf])
        self.decoders.append(self.rgb_aspp)

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.nir_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.nir_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.nir_backbone, self.nir_llf, self.nir_hlf])
            self.decoders.append(self.nir_aspp)
        if self.use_aolp:
            self.num_modalities += 1
            self.aolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.aolp_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.aolp_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.aolp_backbone, self.aolp_llf, self.aolp_hlf])
            self.decoders.append(self.aolp_aspp)
        if self.use_dolp:
            self.num_modalities += 1
            self.dolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.dolp_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.dolp_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.dolp_backbone, self.dolp_llf, self.dolp_hlf])
            self.decoders.append(self.dolp_aspp)
        if self.use_pol:
            self.num_modalities += 1
            self.pol_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=False)
            self.pol_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.pol_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.pol_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.pol_backbone, self.pol_llf, self.pol_hlf])
            self.decoders.append(self.pol_aspp)
        # if self.use_segmap:
        #     self.segmap_backbone = build_custom_resnet_backbone(1, output_stride, BatchNorm, input_dim=1, pretrained=False)
        #     self.segmap_aspp = build_aspp(backbone, output_stride, BatchNorm)
        #     self.backbones.append(self.segmap_backbone)
        #     self.decoders.append(self.segmap_aspp)

        self.decoder = build_decoder_with_semask(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, aolp=None, dolp=None, pol=None, segmap=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x1 = self.rgb_hlf(x1)
        low_level_feat1 = self.rgb_llf(low_level_feat1)
        # print("---------", x1.shape, low_level_feat1.shape)
        # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
        x = x1
        low_level_feat = low_level_feat1

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x2 = self.nir_hlf(x2)
            low_level_feat2 = self.nir_llf(low_level_feat2)
            x = torch.add(x, x2)
            low_level_feat = torch.add(low_level_feat, low_level_feat2)
        if self.use_aolp and aolp is not None:
            x3, low_level_feat3 = self.aolp_backbone(aolp)
            x3 = self.aolp_aspp(x3)
            x3 = self.aolp_se_hlf(x3)
            low_level_feat3 = self.aolp_se_llf(low_level_feat3)
            x = torch.add(x, x3)
            low_level_feat = torch.add(low_level_feat, low_level_feat3)
        if self.use_dolp and dolp is not None:
            x4, low_level_feat4 = self.dolp_backbone(dolp)
            x4 = self.dolp_aspp(x4)
            x4 = self.dolp_se_hlf(x4)
            low_level_feat4 = self.dolp_se_llf(low_level_feat4)
            x = torch.add(x, x4)
            low_level_feat = torch.add(low_level_feat, low_level_feat4)
        if self.use_pol and pol is not None:
            x5, low_level_feat5 = self.pol_backbone(pol)
            x5 = self.pol_aspp(x5)
            x5 = self.pol_hlf(x5)
            low_level_feat5 = self.pol_llf(low_level_feat5)
            x = torch.add(x, x5)
            low_level_feat = torch.add(low_level_feat, low_level_feat5)

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
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
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
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


# norm = 'avg' -> Average / 'bn' -> BatchNorm / 'bnr' -> BatchNorm + ReLU
class MMDeepLabSEMaskWithNorm(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_aolp=False,
        use_dolp=False,
        use_pol=False,
        use_segmap=False,
        enable_se=False,
        norm='avg',
    ): 
        super(MMDeepLabSEMaskWithNorm, self).__init__()
        self.use_nir = use_nir
        self.use_aolp = use_aolp
        self.use_dolp = use_dolp
        self.use_pol = use_pol
        self.use_segmap = use_segmap
        self.enable_se = enable_se
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1
        self.norm = norm

        self.low_level_feature_channels = 256
        self.low_level_feature_shape = (256, 128, 128)

        self.high_level_feature_channels = 256
        self.high_level_feature_shape = (256, 32, 32)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.rgb_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
        self.rgb_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
        self.backbones.extend([self.rgb_backbone, self.rgb_llf, self.rgb_hlf])
        self.decoders.append(self.rgb_aspp)

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.nir_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.nir_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.nir_backbone, self.nir_llf, self.nir_hlf])
            self.decoders.append(self.nir_aspp)
        if self.use_aolp:
            self.num_modalities += 1
            self.aolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=2, pretrained=False)
            self.aolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.aolp_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.aolp_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.aolp_backbone, self.aolp_llf, self.aolp_hlf])
            self.decoders.append(self.aolp_aspp)
        if self.use_dolp:
            self.num_modalities += 1
            self.dolp_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.dolp_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.dolp_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.dolp_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.dolp_backbone, self.dolp_llf, self.dolp_hlf])
            self.decoders.append(self.dolp_aspp)
        if self.use_pol:
            self.num_modalities += 1
            self.pol_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=False)
            self.pol_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.pol_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.pol_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.pol_backbone, self.pol_llf, self.pol_hlf])
            self.decoders.append(self.pol_aspp)

        if self.norm == 'bnr':
            self.hlf_norm = nn.Sequential(
                BatchNorm(self.high_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.llf_norm = nn.Sequential(
                BatchNorm(self.low_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.backbones.extend([self.hlf_norm, self.llf_norm])
        elif self.norm == 'bn':
            self.hlf_norm = BatchNorm(self.high_level_feature_channels)
            self.llf_norm = BatchNorm(self.low_level_feature_channels)
            self.backbones.extend([self.hlf_norm, self.llf_norm])

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, aolp=None, dolp=None, pol=None, segmap=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x1 = self.rgb_hlf(x1)
        low_level_feat1 = self.rgb_llf(low_level_feat1)
        # print("---------", x1.shape, low_level_feat1.shape)
        # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
        x = x1
        low_level_feat = low_level_feat1
        active_modalities = 1

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x2 = self.nir_hlf(x2)
            low_level_feat2 = self.nir_llf(low_level_feat2)
            x = torch.add(x, x2)
            low_level_feat = torch.add(low_level_feat, low_level_feat2)
            active_modalities += 1 
        if self.use_aolp and aolp is not None:
            x3, low_level_feat3 = self.aolp_backbone(aolp)
            x3 = self.aolp_aspp(x3)
            x3 = self.aolp_se_hlf(x3)
            low_level_feat3 = self.aolp_se_llf(low_level_feat3)
            x = torch.add(x, x3)
            low_level_feat = torch.add(low_level_feat, low_level_feat3)
            active_modalities += 1
        if self.use_dolp and dolp is not None:
            x4, low_level_feat4 = self.dolp_backbone(dolp)
            x4 = self.dolp_aspp(x4)
            x4 = self.dolp_se_hlf(x4)
            low_level_feat4 = self.dolp_se_llf(low_level_feat4)
            x = torch.add(x, x4)
            low_level_feat = torch.add(low_level_feat, low_level_feat4)
            active_modalities += 1
        if self.use_pol and pol is not None:
            x5, low_level_feat5 = self.pol_backbone(pol)
            x5 = self.pol_aspp(x5)
            x5 = self.pol_hlf(x5)
            low_level_feat5 = self.pol_llf(low_level_feat5)
            x = torch.add(x, x5)
            low_level_feat = torch.add(low_level_feat, low_level_feat5)
            active_modalities += 1

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        if self.norm == 'avg':
            x = torch.div(x, active_modalities)
            low_level_feat = torch.div(low_level_feat, active_modalities)
        else:
            x = self.hlf_norm(x)
            low_level_feat = self.llf_norm(low_level_feat)

        # print("Is Leaf h:", self.rgb_hlf.mask.h.is_leaf)
        # print("Grad se fc:", self.rgb_hlf.se.fc[0].weight.grad)
        # print("Weigh se fc:", self.rgb_hlf.se.fc[0].weight)
        # print("Perameter h:", self.rgb_hlf.mask.h)
        # print("Grad:", self.rgb_hlf.mask.h.grad)
        # print("Perameter w:", self.rgb_hlf.mask.w)
        # print("Grad:", self.rgb_hlf.mask.w.grad)
        # print("Perameter c:", self.rgb_hlf.mask.c)
        # print("Grad:", self.rgb_hlf.mask.c.grad)
        # print(self.nir_hlf.mask.h)
        # print(self.pol_hlf.mask.h)
        
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
                    # print("Module M: ", m)
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
                        # print("M Comes Inside", m)
                        for p in m[1].parameters():
                            if p.requires_grad:
                                # print("Yeilding M: ", p)
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
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class MMDeepLabSEMaskWithNormForRGBD(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=40,      
        sync_bn=True, 
        freeze_bn=False,
        use_rgb=False,
        use_depth=False,
        norm='avg',
    ): 
        super(MMDeepLabSEMaskWithNormForRGBD, self).__init__()
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 0
        self.norm = norm

        self.low_level_feature_channels = 256
        self.low_level_feature_shape = (256, 120, 160)

        self.high_level_feature_channels = 256
        self.high_level_feature_shape = (256, 30, 40)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if self.use_rgb:
            self.num_modalities += 1
            self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
            self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.rgb_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.rgb_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.rgb_backbone, self.rgb_llf, self.rgb_hlf])
            self.decoders.append(self.rgb_aspp)

        if self.use_depth:
            self.num_modalities += 1
            self.depth_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.depth_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.depth_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            self.depth_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend([self.depth_backbone, self.depth_llf, self.depth_hlf])
            self.decoders.append(self.depth_aspp)

        if self.norm == 'bnr':
            self.hlf_norm = nn.Sequential(
                BatchNorm(self.high_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.llf_norm = nn.Sequential(
                BatchNorm(self.low_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.backbones.extend([self.hlf_norm, self.llf_norm])
        elif self.norm == 'bn':
            self.hlf_norm = BatchNorm(self.high_level_feature_channels)
            self.llf_norm = BatchNorm(self.low_level_feature_channels)
            self.backbones.extend([self.hlf_norm, self.llf_norm])

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities)
        self.decoders.append(self.decoder)

    def forward(self, rgb=None, depth=None):
        active_modalities = 0
        if self.use_rgb and rgb is not None:
            x1, low_level_feat1 = self.rgb_backbone(rgb)
            x1 = self.rgb_aspp(x1)
            x1 = self.rgb_hlf(x1)
            low_level_feat1 = self.rgb_llf(low_level_feat1)
            # print("---------", x1.shape, low_level_feat1.shape)
            # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
            # -------- torch.Size([4, 256, 30, 40]) torch.Size([4, 256, 120, 160])
            x = x1
            low_level_feat = low_level_feat1
            active_modalities += 1

        if self.use_depth and depth is not None:
            x2, low_level_feat2 = self.depth_backbone(depth)
            x2 = self.depth_aspp(x2)
            x2 = self.depth_hlf(x2)
            low_level_feat2 = self.depth_llf(low_level_feat2)
            if self.use_rgb and rgb is not None:
                x = torch.add(x, x2)
                low_level_feat = torch.add(low_level_feat, low_level_feat2)
            else:
                x = x2
                low_level_feat = low_level_feat2
            active_modalities += 1 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        if self.norm == 'avg':
            x = torch.div(x, active_modalities)
            low_level_feat = torch.div(low_level_feat, active_modalities)
        else:
            x = self.hlf_norm(x)
            low_level_feat = self.llf_norm(low_level_feat)
        
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:] if self.use_rgb else depth.size()[2:], mode='bilinear', align_corners=True)

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
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
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
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class MMDeepLabForRGB(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=40,      
        sync_bn=True, 
        freeze_bn=False,
        use_rgb=False,
        use_depth=False,
        norm='avg',
    ): 
        super(MMDeepLabForRGB, self).__init__()
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 0
        self.norm = norm

        self.low_level_feature_channels = 256
        self.low_level_feature_shape = (256, 120, 160)

        self.high_level_feature_channels = 256
        self.high_level_feature_shape = (256, 30, 40)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if self.use_rgb:
            self.num_modalities += 1
            self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
            self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
            # self.rgb_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            # self.rgb_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.append(self.rgb_backbone)
            self.decoders.append(self.rgb_aspp)

        if self.use_depth:
            self.num_modalities += 1
            self.depth_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=1, pretrained=False)
            self.depth_aspp = build_aspp(backbone, output_stride, BatchNorm)
            # self.depth_llf = SEMaskBlock(self.low_level_feature_channels, self.low_level_feature_shape)
            # self.depth_hlf = SEMaskBlock(self.high_level_feature_channels, self.high_level_feature_shape)
            self.backbones.extend(self.depth_backbone)
            self.decoders.append(self.depth_aspp)

        # if self.norm == 'bnr':
        #     self.hlf_norm = nn.Sequential(
        #         BatchNorm(self.high_level_feature_channels),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.llf_norm = nn.Sequential(
        #         BatchNorm(self.low_level_feature_channels),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.backbones.extend([self.hlf_norm, self.llf_norm])
        # elif self.norm == 'bn':
        #     self.hlf_norm = BatchNorm(self.high_level_feature_channels)
        #     self.llf_norm = BatchNorm(self.low_level_feature_channels)
        #     self.backbones.extend([self.hlf_norm, self.llf_norm])

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities)
        self.decoders.append(self.decoder)

    def forward(self, rgb=None, depth=None):
        active_modalities = 0
        if self.use_rgb and rgb is not None:
            x1, low_level_feat1 = self.rgb_backbone(rgb)
            x1 = self.rgb_aspp(x1)
            # x1 = self.rgb_hlf(x1)
            # low_level_feat1 = self.rgb_llf(low_level_feat1)
            # print("---------", x1.shape, low_level_feat1.shape)
            # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
            # -------- torch.Size([4, 256, 30, 40]) torch.Size([4, 256, 120, 160])
            x = x1
            low_level_feat = low_level_feat1
            active_modalities += 1

        if self.use_depth and depth is not None:
            x2, low_level_feat2 = self.depth_backbone(depth)
            x2 = self.depth_aspp(x2)
            # x2 = self.depth_hlf(x2)
            # low_level_feat2 = self.depth_llf(low_level_feat2)
            if self.use_rgb and rgb is not None:
                x = torch.add(x, x2)
                low_level_feat = torch.add(low_level_feat, low_level_feat2)
            else:
                x = x2
                low_level_feat = low_level_feat2
            active_modalities += 1 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        if self.norm == 'avg':
            x = torch.div(x, active_modalities)
            low_level_feat = torch.div(low_level_feat, active_modalities)
        else:
            x = self.hlf_norm(x)
            low_level_feat = self.llf_norm(low_level_feat)
        
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:] if self.use_rgb else depth.size()[2:], mode='bilinear', align_corners=True)

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
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
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
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SEMaskBlock):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
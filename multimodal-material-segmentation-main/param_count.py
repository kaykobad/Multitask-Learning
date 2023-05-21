import torch
import torch.nn as nn
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class CBR256_48(nn.Module):
    def __init__(self):
        super(CBR256_48, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                       SynchronizedBatchNorm2d(48),
                                       nn.ReLU())

    def forward(self, x):
        pass



class CBR1024_192(nn.Module):
    def __init__(self):
        super(CBR1024_192, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(1024, 192, 1, bias=False),
                                       SynchronizedBatchNorm2d(192),
                                       nn.ReLU())

    def forward(self, x):
        pass


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
        x = torch.mul(x, self.c.view(1, -1, 1, 1))
        x = torch.mul(x, self.h.view(1, 1, -1, 1))
        x = torch.mul(x, self.w.view(1, 1, 1, -1))
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


def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# model1 = CBR256_48()
# model2 = CBR1024_192()

# print("CBR 256->48 parameter count: ", sum(p.numel() for p in model1.parameters() if p.requires_grad))
# print("CBR 1024->192 parameter count: ", sum(p.numel() for p in model2.parameters() if p.requires_grad))

# CBR 256->48 parameter count:  12384
# CBR 1024->192 parameter count:  196992

low_level_feature_channels = 256
low_level_feature_shape = (256, 128, 128)

high_level_feature_channels = 256
high_level_feature_shape = (256, 32, 32)

last_conv_input = 48 + 256
decoder_feature_shape = (last_conv_input, 128, 128)

llf_fusion = SEMaskBlock(low_level_feature_channels, low_level_feature_shape)
hlf_fusion = SEMaskBlock(high_level_feature_channels, high_level_feature_shape)
decoder_fusion = SEMaskBlock(last_conv_input, decoder_feature_shape)

print("Parameters for Low-Level Feature Fusion")
llf_param_count = count_param(llf_fusion)
print("1   Modality: ", llf_param_count)
print("2 Modalities: ", 2*llf_param_count)
print("3 Modalities: ", 3*llf_param_count)

print("\nParameters for High-Level Feature Fusion")
hlf_param_count = count_param(hlf_fusion)
print("1   Modality: ", hlf_param_count)
print("2 Modalities: ", 2*hlf_param_count)
print("3 Modalities: ", 3*hlf_param_count)

print("\nParameters for Decoder-Level Feature Fusion")
decoder_param_count = count_param(decoder_fusion)
print("All Modalities: ", decoder_param_count)

print("\nTotal extra prameter count for fusion blocks: Fusion after encoder")
print("1   Modality: ", llf_param_count+hlf_param_count)
print("2 Modalities: ", 2*(llf_param_count+hlf_param_count))
print("3 Modalities: ", 3*(llf_param_count+hlf_param_count))

print("\nTotal extra prameter count for fusion blocks: Fusion after encoder and before decoder")
print("1   Modality: ", llf_param_count+hlf_param_count+decoder_param_count)
print("2 Modalities: ", 2*(llf_param_count+hlf_param_count)+decoder_param_count)
print("3 Modalities: ", 3*(llf_param_count+hlf_param_count)+decoder_param_count)

# Parameters for Low-Level Feature Fusion
# 1   Modality:  8704
# 2 Modalities:  17408
# 3 Modalities:  26112

# Parameters for High-Level Feature Fusion
# 1   Modality:  8512
# 2 Modalities:  17024
# 3 Modalities:  25536

# Parameters for Decoder-Level Feature Fusion
# All Modalities:  12112

# Total extra prameter count for fusion blocks: Fusion after encoder
# 1   Modality:  17216
# 2 Modalities:  34432
# 3 Modalities:  51648

# Total extra prameter count for fusion blocks: Fusion after encoder and before decoder
# 1   Modality:  29328
# 2 Modalities:  46544
# 3 Modalities:  63760
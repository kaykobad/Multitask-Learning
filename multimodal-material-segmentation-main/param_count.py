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



model1 = CBR256_48()
model2 = CBR1024_192()

print("CBR 256->48 parameter count: ", sum(p.numel() for p in model1.parameters() if p.requires_grad))
print("CBR 1024->192 parameter count: ", sum(p.numel() for p in model2.parameters() if p.requires_grad))

# CBR 256->48 parameter count:  12384
# CBR 1024->192 parameter count:  196992
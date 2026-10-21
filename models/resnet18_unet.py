# U-Net with a ResNet-18 encoder. Three inits: random / imagenet / uscl.
# USCL ships a ResNet-18, not a U-Net, so "USCL init" requires this architecture.
# The random arm is the capacity control: it isolates the weights from the backbone swap.
import torch
import torch.nn as nn
from torchvision.models import resnet18
from model.unet import DoubleConv

# USCL stores the backbone as nn.Sequential(*resnet18().children()); 2/3/8 are
# relu/maxpool/avgpool and carry no parameters.
_STAGE = {'0': 'conv1', '1': 'bn1', '4': 'layer1',
          '5': 'layer2', '6': 'layer3', '7': 'layer4'}


def _load_uscl(enc, path):
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('state_dict', sd)
    mapped = {}
    for k, v in sd.items():
        if not k.startswith('features.'):
            continue                      # l1/l2/fc1/fc2: projection + 12-way head
        idx, tail = k[len('features.'):].split('.', 1)
        if idx in _STAGE:
            mapped[f'{_STAGE[idx]}.{tail}'] = v
    missing, unexpected = enc.load_state_dict(mapped, strict=False)
    print(f'USCL: mapped {len(mapped)} tensors, '
          f'{len(missing)} missing, {len(unexpected)} unexpected')
    assert set(missing) <= {'fc.weight', 'fc.bias'}, f'bad remap: {missing}'
    assert not unexpected, f'bad remap: {unexpected}'


class ResNet18Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, init='random',
                 weights='/workspace/AutoLabelTools/checkpoints/uscl_resnet18.pth'):
        super().__init__()
        assert in_ch == 3, 'ResNet-18 stem expects 3 channels'
        if init == 'imagenet':
            try:
                enc = resnet18(weights='IMAGENET1K_V1')
            except TypeError:                       # torchvision < 0.13
                enc = resnet18(pretrained=True)
        else:
            try:
                enc = resnet18(weights=None)
            except TypeError:
                enc = resnet18(pretrained=False)
            if init == 'uscl':
                _load_uscl(enc, weights)
            elif init != 'random':
                raise ValueError(f'unknown init: {init}')
        print(f'resnet18-unet init={init}')

        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # /2    64
        self.pool = enc.maxpool                                   # /4
        self.l1, self.l2 = enc.layer1, enc.layer2                 # /4 64, /8 128
        self.l3, self.l4 = enc.layer3, enc.layer4                 # /16 256, /32 512

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2); self.d4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2); self.d3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128,  64, 2, 2); self.d2 = DoubleConv(128,  64)
        self.up1 = nn.ConvTranspose2d( 64,  64, 2, 2); self.d1 = DoubleConv(128,  64)
        self.up0 = nn.ConvTranspose2d( 64,  32, 2, 2); self.d0 = DoubleConv( 32,  32)
        self.out = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        s  = self.stem(x)                 # 112
        c1 = self.l1(self.pool(s))        # 56
        c2 = self.l2(c1)                  # 28
        c3 = self.l3(c2)                  # 14
        c4 = self.l4(c3)                  # 7
        x = self.d4(torch.cat([self.up4(c4), c3], 1))
        x = self.d3(torch.cat([self.up3(x),  c2], 1))
        x = self.d2(torch.cat([self.up2(x),  c1], 1))
        x = self.d1(torch.cat([self.up1(x),  s ], 1))
        return self.out(self.d0(self.up0(x)))

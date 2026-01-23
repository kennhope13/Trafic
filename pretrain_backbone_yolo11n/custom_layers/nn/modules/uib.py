# custom_layers/nn/modules/uib.py
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UIB(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, e=6.0, start_dw=True, mid_dw=True, end_dw=False):
        super().__init__()
        assert int(s) in (1, 2), f"UIB stride must be 1 or 2, got s={s}"
        assert int(k) in (3, 5, 7)

        self.use_res = (int(s) == 1 and c1 == c2)
        hidden = int(round(c1 * float(e)))
        k = int(k); s = int(s)

        layers = []
        if start_dw:
            layers.append(ConvBNAct(c1, c1, k=k, s=1, p=k // 2, g=c1, act=True))

        if float(e) != 1.0:
            layers.append(ConvBNAct(c1, hidden, k=1, s=1, p=0, g=1, act=True))
        else:
            hidden = c1

        if mid_dw:
            layers.append(ConvBNAct(hidden, hidden, k=k, s=s, p=k // 2, g=hidden, act=True))
        else:
            if s == 2:
                layers.append(ConvBNAct(hidden, hidden, k=k, s=2, p=k // 2, g=hidden, act=True))

        layers.append(ConvBNAct(hidden, c2, k=1, s=1, p=0, g=1, act=False))

        if end_dw:
            layers.append(ConvBNAct(c2, c2, k=k, s=1, p=k // 2, g=c2, act=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class UIBDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, e=6.0):
        super().__init__()
        self.m = UIB(c1, c2, k=k, s=s, e=e, start_dw=True, mid_dw=True, end_dw=False)

    def forward(self, x):
        return self.m(x)

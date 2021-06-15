# Obtained from: https://github.com/meliketoy/wide-resnet.pytorch
# Adapted to match:
# https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

import torch.nn as nn
import torch.nn.functional as F


class WideBasic(nn.Module):
    def __init__(self, in_c, out_c, stride, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        kernel = 3
        padding = 1
        self.conv1 = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)

        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel, 1, padding, bias=False)

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, 1, stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))

        out = self.conv1(out)

        out = F.relu(self.bn2(out))

        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = self.conv2(out)
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(
        self, depth=28, widen_factor=10, num_classes=None, dropout_rate=0.3,
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"

        self.dropout_rate = dropout_rate

        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        strides = [1, 1, 2, 2]

        self.conv1 = nn.Conv2d(3, nStages[0], 3, strides[0], 1, bias=False)
        self.layer1 = self._wide_layer(nStages[0:2], n, strides[1])
        self.layer2 = self._wide_layer(nStages[1:3], n, strides[2])
        self.layer3 = self._wide_layer(nStages[2:4], n, strides[3])

        self.bn1 = nn.BatchNorm2d(nStages[3])

        self.num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(nStages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L17
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L25
                nn.init.uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L21
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        in_c, out_c = channels

        for stride in strides:
            layers.append(WideBasic(in_c, out_c, stride, self.dropout_rate))
            in_c = out_c

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.flatten(1)

        if self.num_classes is not None:
            out = self.linear(out)

        return out

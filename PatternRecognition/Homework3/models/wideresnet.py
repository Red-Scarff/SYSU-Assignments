"""
WideResNet-28-2 implementation for semi-supervised learning
Based on the original WideResNet paper with modifications for SSL
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for WideResNet"""

    def __init__(self, in_channels, out_channels, stride, dropout_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = dropout_rate
        self.equalInOut = in_channels == out_channels
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            or None
        )
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """Network block consisting of multiple BasicBlocks"""

    def __init__(
        self, nb_layers, in_channels, out_channels, block, stride, dropout_rate=0.0, activate_before_residual=False
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_channels, out_channels, nb_layers, stride, dropout_rate, activate_before_residual
        )

    def _make_layer(self, block, in_channels, out_channels, nb_layers, stride, dropout_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_channels or out_channels,
                    out_channels,
                    i == 0 and stride or 1,
                    dropout_rate,
                    activate_before_residual and i == 0,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet-28-2 implementation
    - depth: 28 (total layers)
    - widen_factor: 2 (channel multiplier)
    """

    def __init__(self, num_classes=10, depth=28, widen_factor=2, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) / 6
        block = BasicBlock

        # First convolution
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Three residual blocks
        self.block1 = NetworkBlock(
            n, n_channels[0], n_channels[1], block, 1, dropout_rate, activate_before_residual=True
        )
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, dropout_rate)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, dropout_rate)

        # Final batch norm and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.n_channels)
        return self.fc(out)


def create_wideresnet28_2(num_classes=10, dropout_rate=0.0):
    """Create WideResNet-28-2 model"""
    return WideResNet(num_classes=num_classes, depth=28, widen_factor=2, dropout_rate=dropout_rate)


if __name__ == "__main__":
    # Test the model
    model = create_wideresnet28_2()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

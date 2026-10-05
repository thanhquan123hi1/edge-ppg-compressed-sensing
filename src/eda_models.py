"""Approved EDA decoders; the historical CNN and Linear remain unchanged."""
import torch
from torch import nn

from model import CNN1DDecoder, LinearDecoder


class LiteResidualBlock(nn.Module):
    """Two same-length dilated temporal convolutions with a local skip."""

    def __init__(self, channels=16, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation)
        self.act = nn.LeakyReLU(.1)

    def forward(self, x):
        return self.act(x + self.conv2(self.act(self.conv1(x))))


class ResLinCNNLite(nn.Module):
    """Learned linear reconstruction plus a zero-initialized residual CNN."""

    def __init__(self, M, N=256, channels=16):
        super().__init__()
        if N != 256 or channels != 16:
            raise ValueError('The approved Lite architecture requires N=256 and channels=16')
        if M <= 0:
            raise ValueError('M must be positive')
        self.M, self.N = M, N
        self.linear = nn.Linear(M, N)
        self.stem = nn.Conv1d(1, channels, 5, padding=2)
        self.act = nn.LeakyReLU(.1)
        self.blocks = nn.ModuleList([LiteResidualBlock(channels, d) for d in (1, 2, 4)])
        self.head = nn.Conv1d(channels, 1, 5, padding=2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, y):
        x0 = self.linear(y)
        features = self.act(self.stem(x0.unsqueeze(1)))
        for block in self.blocks:
            features = block(features)
        return x0 + self.head(features).squeeze(1)


def build_eda_decoder(method, M):
    if method == 'cnn':
        return CNN1DDecoder(M)
    if method == 'reslincnn_lite':
        return ResLinCNNLite(M)
    if method == 'linear':
        return LinearDecoder(M)
    raise ValueError(f'Unknown EDA decoder {method}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

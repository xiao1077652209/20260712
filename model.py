"""Neural models used in the JAC-SCM paper experiments."""
from __future__ import annotations
import math
import torch
from torch import nn


class SSCJAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, alpha: float = 0.7):
        super().__init__(); hidden = max(channels // reduction, 1); self.alpha = alpha
        self.channel_mlp = nn.Sequential(nn.Conv1d(channels, hidden, 1, bias=False), nn.ReLU(),
                                         nn.Conv1d(hidden, channels, 1, bias=False))
        self.spatial = nn.Sequential(nn.Conv1d(channels, hidden, 3, padding=1, bias=False),
                                     nn.BatchNorm1d(hidden), nn.ReLU(),
                                     nn.Conv1d(hidden, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        avg, maximum = x.mean(-1, keepdim=True), x.amax(-1, keepdim=True)
        channel = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(maximum))
        return self.alpha * x * channel + (1 - self.alpha) * x * self.spatial(x)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel, regularized=False):
        super().__init__()
        normalization = [nn.BatchNorm1d(cout)] if regularized else []
        self.layers = nn.Sequential(nn.Conv1d(cin, cout, kernel, padding="same"), *normalization,
                                    nn.ReLU(), nn.MaxPool1d(2))
    def forward(self, x): return self.layers(x)


class LeNet5Spectral(nn.Module):
    """Paper ablations: attention=False/True produces LeNet-5/LeNet-5-SSC-JAM."""
    def __init__(self, input_length=114, num_classes=10, attention=False):
        super().__init__()
        layers = [ConvBlock(1, 32, 7)]
        if attention: layers.append(SSCJAM(32))
        layers += [ConvBlock(32, 64, 4), ConvBlock(64, 128, 4)]
        self.conv = nn.Sequential(*layers)
        with torch.no_grad(): width = self.conv(torch.zeros(1, 1, input_length)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(width, num_classes))
    def forward(self, x): return self.fc(self.conv(x.unsqueeze(1) if x.ndim == 2 else x))


class JACSCM(LeNet5Spectral):
    """Tobacco-adapted JAC-SCM with a regularized, length-robust classifier."""
    def __init__(self, input_length=114, num_classes=10, dropout=.3, **_):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(ConvBlock(1, 32, 7, True), SSCJAM(32),
                                  ConvBlock(32, 64, 4, True), ConvBlock(64, 128, 4, True))
        with torch.no_grad(): width = self.conv(torch.zeros(1,1,input_length)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(width, 128), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(128, num_classes))

    def forward(self, x):
        features = self.conv(x.unsqueeze(1) if x.ndim == 2 else x)
        return self.fc(features)


class LSTMClassifier(nn.Module):
    def __init__(self, input_length=114, num_classes=10, hidden=64):
        super().__init__(); self.rnn = nn.LSTM(1, hidden, batch_first=True); self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x): return self.fc(self.rnn(x.unsqueeze(-1))[0][:, -1])


class TemporalBlock(nn.Module):
    def __init__(self, cin, cout, dilation):
        super().__init__(); pad = dilation * 2
        self.net = nn.Sequential(nn.Conv1d(cin, cout, 3, padding=pad, dilation=dilation), nn.ReLU(),
                                 nn.Conv1d(cout, cout, 3, padding=pad, dilation=dilation), nn.ReLU())
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity(); self.pad = pad
    def forward(self, x):
        y = self.net(x)
        if self.pad: y = y[..., :-2 * self.pad]
        return torch.relu(y + self.skip(x))


class TCNClassifier(nn.Module):
    def __init__(self, input_length=114, num_classes=10):
        super().__init__(); self.net = nn.Sequential(TemporalBlock(1, 32, 1), TemporalBlock(32, 64, 2))
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x): return self.fc(self.net(x.unsqueeze(1)).mean(-1))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.net = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels),
                                                     nn.ReLU(), nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels))
    def forward(self, x): return torch.relu(x + self.net(x))


class ResNetClassifier(nn.Module):
    def __init__(self, input_length=114, num_classes=10):
        super().__init__(); self.net = nn.Sequential(nn.Conv1d(1, 64, 7, padding=3), nn.ReLU(),
                                                     ResidualBlock(64), ResidualBlock(64))
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x): return self.fc(self.net(x.unsqueeze(1)).mean(-1))


class PositionalEncoding(nn.Module):
    def __init__(self, dim, length=2048):
        super().__init__(); pos = torch.arange(length).unsqueeze(1); scale = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000) / dim))
        pe = torch.zeros(length, dim); pe[:, 0::2] = torch.sin(pos * scale); pe[:, 1::2] = torch.cos(pos * scale); self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[:x.shape[1]]


class TransformerClassifier(nn.Module):
    def __init__(self, input_length=114, num_classes=10, dim=64):
        super().__init__(); self.embed = nn.Linear(1, dim); self.pos = PositionalEncoding(dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 4, 128, batch_first=True), 2)
        self.fc = nn.Linear(dim, num_classes)
    def forward(self, x): return self.fc(self.encoder(self.pos(self.embed(x.unsqueeze(-1)))).mean(1))


def build_neural_model(name, input_length=114, num_classes=10):
    models = {"lenet5": lambda: LeNet5Spectral(input_length, num_classes),
              "lenet5_ssc_jam": lambda: LeNet5Spectral(input_length, num_classes, True),
              "lenet5_lawd": lambda: LeNet5Spectral(input_length, num_classes),
              "jac_scm": lambda: JACSCM(input_length, num_classes), "lstm": lambda: LSTMClassifier(input_length, num_classes),
              "tcn": lambda: TCNClassifier(input_length, num_classes), "resnet": lambda: ResNetClassifier(input_length, num_classes),
              "transformer": lambda: TransformerClassifier(input_length, num_classes)}
    if name not in models: raise ValueError(f"Unknown neural model: {name}")
    return models[name]()


def make_lawd_adamw(model, base_lr=1e-3):
    attention, convolution, fc = [], [], []
    for name, p in model.named_parameters():
        (attention if "conv.1" in name and isinstance(model, JACSCM) else fc if name.startswith("fc") else convolution).append(p)
    groups = []
    if attention: groups.append({"params": attention, "lr": 1.2*base_lr, "weight_decay": 1e-4})
    if convolution: groups.append({"params": convolution, "lr": base_lr, "weight_decay": 1e-4})
    if fc: groups.append({"params": fc, "lr": .9*base_lr, "weight_decay": 5e-4})
    return torch.optim.AdamW(groups)

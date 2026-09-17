"""A compact 1D-CNN fault classifier and its train / inference helpers.

The network is deliberately small: raw-waveform inputs, three convolutional
blocks with global pooling. It trains in seconds per epoch on CPU and gives a
well-behaved softmax, which is what the conformal layer consumes.
"""

import numpy as np
import torch
import torch.nn as nn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN1D(nn.Module):
    def __init__(self, n_classes, width=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, width, 15, padding=7), nn.BatchNorm1d(width), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(width, width * 2, 9, padding=4), nn.BatchNorm1d(width * 2), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(width * 2, width * 4, 5, padding=2), nn.BatchNorm1d(width * 4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(width * 4, n_classes)

    def forward(self, x):
        z = self.features(x).squeeze(-1)
        return self.head(z)


def train_model(cfg, X, y, seed):
    """Train the classifier on source-domain data and return it in eval mode."""
    torch.manual_seed(seed)
    device = get_device()
    model = CNN1D(cfg.n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, X, batch_size=256):
    """Return softmax probabilities for inputs X."""
    device = get_device()
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)

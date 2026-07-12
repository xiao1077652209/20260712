"""Train JAC-SCM from a CSV whose final column is the class label."""

from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import JACSCM, make_lawd_adamw
from preprocessing import cars_select, msc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = np.genfromtxt(args.csv, delimiter=",", dtype=str, skip_header=1)
    x, labels = data[:, :-1].astype(np.float32), data[:, -1]
    y = LabelEncoder().fit_transform(labels)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.4,
                                            stratify=y, random_state=args.seed)
    x_train, ref = msc(x[train_idx]); x_test, _ = msc(x[test_idx], ref)
    if x.shape[1] != 114:
        selected = cars_select(x_train, y[train_idx], n_features=114, seed=args.seed)
        x_train, x_test = x_train[:, selected], x_test[:, selected]
        np.savetxt("cars_indices.csv", selected, fmt="%d", delimiter=",")
    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y[train_idx]))
    test_x, test_y = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y[test_idx])
    model = JACSCM(x_train.shape[1], len(np.unique(y)))
    optimizer = make_lawd_adamw(model, args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_fn = nn.CrossEntropyLoss(); best, stale, state = -1.0, 0, None
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in DataLoader(train_ds, args.batch_size, shuffle=True):
            optimizer.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad(): accuracy = (model(test_x).argmax(1) == test_y).float().mean().item()
        print(f"epoch={epoch + 1:03d} loss={loss.item():.6f} test_accuracy={accuracy:.4f}")
        if accuracy > best:
            best, stale = accuracy, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    torch.save({"model": state, "accuracy": best}, "jac_scm.pt")


if __name__ == "__main__":
    main()

"""Train RetinaLens:  python train.py"""
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

import config as C
from dataset import get_loaders
from model import build_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device)).squeeze(1)).cpu()
        probs.append(p); ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    loaders = get_loaders()
    model = build_model().to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.EPOCHS)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_auc, bad = 0.0, 0
    C.CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in tqdm(loaders["train"], desc=f"Epoch {epoch}/{C.EPOCHS}", leave=False):
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total += loss.item() * len(y); n += len(y)
        sched.step()

        p, y = predict(model, loaders["validation"], device)
        auc = roc_auc_score(y, p)
        acc = ((p >= 0.5) == y).mean()
        print(f"Epoch {epoch:02d} | train loss {total/n:.4f} | val AUC {auc:.4f} | val acc {acc:.4f}")

        if auc > best_auc:
            best_auc, bad = auc, 0
            torch.save({"model": model.state_dict(),
                        "threshold": youden_threshold(y, p),
                        "val_auc": auc}, C.CKPT_PATH)
            print(f"  saved best model (AUC {auc:.4f})")
        else:
            bad += 1
            if bad >= C.PATIENCE:
                print("Early stopping."); break
    print("Best validation AUC:", round(best_auc, 4))


if __name__ == "__main__":
    main()

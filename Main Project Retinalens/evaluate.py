"""Evaluate on the held-out test set:  python evaluate.py"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score)

import config as C
from dataset import get_loaders
from model import build_model
from train import predict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(C.CKPT_PATH, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    thr = ck["threshold"]

    p, y = predict(model, get_loaders()["test"], device)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    m = {
        "threshold": thr,
        "accuracy": accuracy_score(y, pred),
        "auc": roc_auc_score(y, p),
        "precision": precision_score(y, pred),
        "recall_sensitivity": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    C.OUT_DIR.mkdir(exist_ok=True)
    (C.OUT_DIR / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    print(json.dumps(m, indent=2, default=float))

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC = {m['auc']:.3f}"); ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve"); ax[0].legend()
    cm = np.array([[tn, fp], [fn, tp]])
    ax[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, cm[i, j], ha="center", va="center")
    ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES,
              xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    plt.tight_layout(); plt.savefig(C.OUT_DIR / "eval.png", dpi=150)
    print("Saved outputs/metrics.json and outputs/eval.png")


if __name__ == "__main__":
    main()

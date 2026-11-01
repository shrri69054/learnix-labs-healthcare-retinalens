"""Data loading and augmentation."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

import config as C


def build_transforms():
    train_tf = T.Compose([
        T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(C.MEAN, C.STD),
    ])
    eval_tf = T.Compose([
        T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(C.MEAN, C.STD),
    ])
    return train_tf, eval_tf


def get_loaders():
    train_tf, eval_tf = build_transforms()
    sets = {
        "train": datasets.ImageFolder(C.DATA_DIR / "train", train_tf),
        "validation": datasets.ImageFolder(C.DATA_DIR / "validation", eval_tf),
        "test": datasets.ImageFolder(C.DATA_DIR / "test", eval_tf),
    }
    assert sets["train"].classes == C.CLASS_NAMES, sets["train"].classes
    loaders = {
        k: DataLoader(v, batch_size=C.BATCH_SIZE, shuffle=(k == "train"),
                      num_workers=C.NUM_WORKERS, pin_memory=torch.cuda.is_available())
        for k, v in sets.items()
    }
    return loaders

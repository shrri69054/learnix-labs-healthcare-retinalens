"""Central configuration for RetinaLens."""
from pathlib import Path

ROOT = Path(__file__).parent
# Pick one of: release-crop, release-pad, release-raw (same split, different preprocessing)
DATA_DIR = ROOT / "data" / "release-crop"
CKPT_PATH = ROOT / "checkpoints" / "best.pt"
OUT_DIR = ROOT / "outputs"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4          # early stopping on validation AUC
NUM_WORKERS = 4
SEED = 42

# ImageFolder sorts classes alphabetically: NRG -> 0, RG -> 1
CLASS_NAMES = ["NRG", "RG"]
CLASS_LABELS = {"NRG": "Non-referable glaucoma", "RG": "Referable glaucoma"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

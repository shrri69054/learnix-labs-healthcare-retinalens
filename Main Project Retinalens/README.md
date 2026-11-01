# RetinaLens: Glaucoma Screening from Retinal Fundus Images

Deep-learning pipeline that classifies retinal fundus photos as **referable glaucoma (RG)** or **non-referable glaucoma (NRG)**, with Grad-CAM explanations and a web app for live predictions.

- **Data:** EyePACS-AIROGS-style set, 8,000 train / 770 validation / 770 test images, balanced classes, in three preprocessing variants (`crop`, `pad`, `raw`).
- **Model:** EfficientNet-B0 (ImageNet pretrained), fine-tuned end to end with AdamW, cosine LR, mixed precision, early stopping on validation AUC.
- **Decision threshold:** chosen on the validation set (Youden's J), then applied to the test set.
- **Explainability:** Grad-CAM heatmap overlaid on every prediction.
- **Stack:** Python, PyTorch, torchvision, scikit-learn, OpenCV, Flask.

## Setup
```bash
pip install -r requirements.txt
bash prepare_data.sh /path/to/archive.zip     # creates data/release-crop, release-pad, release-raw
```

## Run
```bash
python train.py        # trains, saves checkpoints/best.pt
python evaluate.py     # test metrics -> outputs/metrics.json, outputs/eval.png
python app.py          # web app at http://127.0.0.1:5000
```

## Project layout
```
config.py        paths and hyperparameters
dataset.py       loaders and augmentation
model.py         EfficientNet-B0 classifier
train.py         training loop
evaluate.py      test metrics, ROC curve, confusion matrix
gradcam.py       Grad-CAM heatmaps
app.py           Flask API
templates/       web UI
```

## Disclaimer
Research and learning project only. Not a medical device.

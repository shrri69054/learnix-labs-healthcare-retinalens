"""Web app:  python app.py  ->  http://127.0.0.1:5000"""
import base64, io
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms as T

import config as C
from model import build_model
from gradcam import GradCAM, overlay

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(C.CKPT_PATH, map_location=device)
model = build_model(pretrained=False).to(device)
model.load_state_dict(ck["model"])
THRESHOLD = ck["threshold"]
cam_engine = GradCAM(model)
tf = T.Compose([T.Resize((C.IMG_SIZE, C.IMG_SIZE)), T.ToTensor(), T.Normalize(C.MEAN, C.STD)])


def to_b64(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="Choose a retinal fundus image to analyze."), 400
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception:
        return jsonify(error="That file isn't a readable image. Upload a JPG or PNG."), 400

    x = tf(img).unsqueeze(0).to(device)
    prob, cam = cam_engine(x)
    rgb = np.array(img.resize((C.IMG_SIZE * 2, C.IMG_SIZE * 2)))
    label = "RG" if prob >= THRESHOLD else "NRG"
    return jsonify(
        label=label,
        label_text=C.CLASS_LABELS[label],
        probability=round(prob, 4),
        threshold=round(THRESHOLD, 4),
        original=to_b64(rgb),
        heatmap=to_b64(overlay(rgb, cam)),
    )


if __name__ == "__main__":
    app.run(debug=False)

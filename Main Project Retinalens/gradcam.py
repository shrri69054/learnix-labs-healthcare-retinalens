"""Grad-CAM heatmaps for EfficientNet-B0."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self.acts = self.grads = None
        layer = model.features[-1]
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x):
        """x: (1,3,H,W). Returns (probability of RG, heatmap in [0,1] sized HxW)."""
        x = x.requires_grad_(True)
        logit = self.model(x).squeeze()
        self.model.zero_grad()
        logit.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return torch.sigmoid(logit).item(), cam.detach().cpu().numpy()


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    """rgb uint8 HxWx3, cam float HxW -> blended uint8 RGB."""
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)

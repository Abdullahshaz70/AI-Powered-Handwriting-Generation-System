"""
Core generation functions: load model, generate single character image.
Sentence layout and PNG export live in generate_samples.py.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model  import CharNet, char_idx_to_cat
from bezier import label_to_curves, add_variation, draw_bezier
from data   import CHAR_TO_IDX, char_idx_to_cat as data_cat, CANVAS_SIZE

_HERE     = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.normpath(os.path.join(_HERE, '..', 'checkpoints', 'style_net.pt'))


def load_model(ckpt_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path  = ckpt_path or CKPT_PATH
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = CharNet(num_chars=62).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt, device


def generate_char(model, char, device, noise_scale=0.018):
    """Predict Bézier curves for char, add variation, render. Returns uint8 (128,128)."""
    idx      = CHAR_TO_IDX[char]
    char_t   = torch.tensor([idx],            device=device)
    cat_t    = torch.tensor([data_cat(idx)],  device=device)

    with torch.no_grad():
        label = model(char_t, cat_t).squeeze().cpu().numpy()

    curves = label_to_curves(label)
    varied = add_variation(curves, noise_scale=noise_scale)
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
    return draw_bezier(canvas, varied, thickness=3)

"""
Bézier curve math, rendering, fitting, and label conversion.
3 cubic curves per character = 24 floats (3 × 4 pts × 2 coords).
"""
import numpy as np
from scipy.special import comb

N_CURVES     = 6
CP_PER_CURVE = 4
LABEL_DIM    = N_CURVES * CP_PER_CURVE * 2   # 48


def bezier_point(t, pts):
    n = len(pts) - 1
    p = np.zeros(2)
    for i, q in enumerate(pts):
        p += comb(n, i, exact=True) * (t**i) * ((1-t)**(n-i)) * np.array(q)
    return p


def draw_bezier(canvas, curves, thickness=3):
    """
    Draw curves (normalised [0,1] coords) onto a uint8 numpy canvas.
    Uses PIL so cv2 is not required.
    """
    from PIL import Image, ImageDraw
    h, w = canvas.shape
    img  = Image.fromarray(canvas, mode='L')
    draw = ImageDraw.Draw(img)
    for pts in curves:
        prev = None
        for t in np.linspace(0, 1, 120):
            p  = bezier_point(t, pts)
            px = int(np.clip(p[0] * w, 0, w - 1))
            py = int(np.clip(p[1] * h, 0, h - 1))
            if prev is not None:
                draw.line([prev, (px, py)], fill=0, width=thickness)
            prev = (px, py)
    return np.array(img)


def add_variation(curves, noise_scale=0.018):
    """Perturb inner control points P1,P2 only — endpoints stay fixed."""
    return [
        [(x + (np.random.normal(0, noise_scale) if j in (1, 2) else 0),
          y + (np.random.normal(0, noise_scale) if j in (1, 2) else 0))
         for j, (x, y) in enumerate(curve)]
        for curve in curves
    ]


def fit_bezier_to_points(points, n_curves=N_CURVES):
    """
    Fit n_curves cubic Bézier curves to an ordered list of (x, y) points.
    Points must already be normalised to [0, 1].
    """
    if len(points) < 4:
        return [[(0.5, 0.5)] * CP_PER_CURVE for _ in range(n_curves)]

    pts      = np.array(points, dtype=float)
    seg_size = max(4, len(pts) // n_curves)
    curves   = []
    for i in range(n_curves):
        start   = i * seg_size
        end     = (i + 1) * seg_size if i < n_curves - 1 else len(pts)
        seg     = pts[start:end]
        if len(seg) < 4:
            seg = pts[max(0, end - 4):end]
        indices = np.linspace(0, len(seg) - 1, CP_PER_CURVE, dtype=int)
        curves.append([tuple(seg[j]) for j in indices])
    return curves


def curves_to_label(curves):
    flat = []
    for curve in curves[:N_CURVES]:
        for x, y in curve:
            flat.extend([float(x), float(y)])
    while len(flat) < LABEL_DIM:
        flat.append(0.0)
    return np.array(flat[:LABEL_DIM], dtype=np.float32)


def label_to_curves(label):
    curves = []
    for i in range(N_CURVES):
        curve = []
        for j in range(CP_PER_CURVE):
            idx = (i * CP_PER_CURVE + j) * 2
            curve.append((float(label[idx]), float(label[idx + 1])))
        curves.append(curve)
    return curves

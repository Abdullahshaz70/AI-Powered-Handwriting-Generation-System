"""Bézier curve math, rendering utilities and constants."""
import numpy as np
from scipy.special import comb

MAX_CURVES    = 5
CP_PER_CURVE  = 4
LABEL_DIM     = MAX_CURVES * CP_PER_CURVE * 2   # 40
NUM_CHARS     = 62                                # A-Z a-z 0-9


def bezier_point(t, pts):
    """Evaluate one cubic Bézier curve at parameter t ∈ [0,1]."""
    n = len(pts) - 1
    p = np.zeros(2)
    for i, q in enumerate(pts):
        p += comb(n, i, exact=True) * (t ** i) * ((1 - t) ** (n - i)) * np.array(q)
    return p


def draw_bezier(canvas, curves, thickness=3):
    """
    Draw Bézier curves (normalised coords 0-1) onto a uint8 numpy canvas.
    Uses PIL for drawing so cv2 is not required.
    """
    from PIL import Image, ImageDraw
    h, w = canvas.shape

    img  = Image.fromarray(canvas, mode='L')
    draw = ImageDraw.Draw(img)

    for pts in curves:
        prev = None
        for t in np.linspace(0, 1, 100):
            p  = bezier_point(t, pts)
            px = int(np.clip(p[0] * w, 0, w - 1))
            py = int(np.clip(p[1] * h, 0, h - 1))
            if prev is not None:
                for d in range(thickness):
                    draw.line([prev, (px, py)], fill=0, width=max(1, thickness))
            prev = (px, py)

    return np.array(img)


def fit_bezier_to_points(points, n_curves=MAX_CURVES):
    """
    Fit n_curves cubic Bézier curves to an ordered list of (x, y) pixel coords.
    Uses 4 evenly-spaced points per segment as control point estimates.
    """
    if len(points) < 4:
        return [[(0.5, 0.5)] * CP_PER_CURVE for _ in range(n_curves)]

    pts      = np.array(points, dtype=float)
    seg_size = max(4, len(pts) // n_curves)
    curves   = []

    for i in range(n_curves):
        start = i * seg_size
        end   = (i + 1) * seg_size if i < n_curves - 1 else len(pts)
        seg   = pts[start:end]
        if len(seg) < 4:
            seg = pts[max(0, end - 4):end]
        indices = np.linspace(0, len(seg) - 1, CP_PER_CURVE, dtype=int)
        ctrl    = seg[indices]
        curves.append([(float(x / 128.0), float(y / 128.0)) for x, y in ctrl])

    return curves


def curves_to_label(curves):
    """Flatten a list of curves → LABEL_DIM-float numpy array."""
    flat = []
    for curve in curves[:MAX_CURVES]:
        for x, y in curve:
            flat.extend([x, y])
    while len(flat) < LABEL_DIM:
        flat.append(0.0)
    return np.array(flat[:LABEL_DIM], dtype=np.float32)


def label_to_curves(label):
    """Reconstruct list of MAX_CURVES curves from a flat label vector."""
    curves = []
    for i in range(MAX_CURVES):
        curve = []
        for j in range(CP_PER_CURVE):
            idx = (i * CP_PER_CURVE + j) * 2
            curve.append((float(label[idx]), float(label[idx + 1])))
        curves.append(curve)
    return curves

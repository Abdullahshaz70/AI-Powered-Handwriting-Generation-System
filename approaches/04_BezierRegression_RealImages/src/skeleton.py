"""
Skeletonization: thin a binary handwriting image to a 1-pixel skeleton,
then extract an ordered list of (x, y) points for Bézier fitting.
"""
import numpy as np


def skeletonize_image(img_array):
    """
    Thin a uint8 image (0=ink, 255=background) to a 1-pixel skeleton.
    Returns a boolean array: True = skeleton pixel.
    """
    try:
        import cv2
        import cv2.ximgproc as ximg
        ink  = np.where(img_array < 128, 255, 0).astype(np.uint8)
        skel = ximg.thinning(ink, thinningType=ximg.THINNING_ZHANGSUEN)
        return skel > 0
    except Exception:
        pass

    try:
        from skimage.morphology import skeletonize
        return skeletonize(img_array < 128)
    except Exception:
        pass

    # Fallback: ridge detection via distance transform
    from scipy.ndimage import distance_transform_edt, maximum_filter
    ink      = img_array < 128
    dist     = distance_transform_edt(ink)
    local_max = (dist == maximum_filter(dist, size=3)) & ink
    return local_max


def extract_ordered_points(skeleton):
    """
    Extract (x, y) skeleton pixels in approximate stroke order via greedy
    nearest-neighbour. Returns list of (x_norm, y_norm) tuples in [0, 1].
    """
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return []

    h, w  = skeleton.shape
    pts   = np.array(list(zip(xs, ys)), dtype=float)

    if len(pts) == 1:
        return [(float(pts[0, 0]) / w, float(pts[0, 1]) / h)]

    visited = np.zeros(len(pts), dtype=bool)
    ordered = []
    # Start from topmost-leftmost pixel
    current = int(np.argmin(pts[:, 1] * w + pts[:, 0]))

    for _ in range(len(pts)):
        visited[current] = True
        ordered.append(current)
        dists = np.sum((pts - pts[current]) ** 2, axis=1)
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        if np.isinf(dists[nxt]):
            break
        current = nxt

    return [(float(pts[i, 0]) / w, float(pts[i, 1]) / h) for i in ordered]

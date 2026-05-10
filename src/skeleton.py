"""
Skeletonization: reduces thick ink strokes to a 1-pixel skeleton path,
then extracts an ordered sequence of (x, y) points from it.

Preference order: opencv-contrib ximgproc → scikit-image → scipy fallback.
cv2 itself is only used for ximgproc; all other operations use numpy/scipy.
"""
import numpy as np

try:
    import cv2
    try:
        _cv2_thin  = cv2.ximgproc.thinning
        _THINTYPE  = cv2.ximgproc.THINNING_ZHANGSUEN
        HAS_XIMGPROC = True
    except AttributeError:
        HAS_XIMGPROC = False
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = HAS_XIMGPROC = False

try:
    from skimage.morphology import skeletonize as _sk_skel
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def _binarise(arr):
    """Return uint8 binary: 255 where ink, 0 elsewhere (inverted convention)."""
    return (arr < 128).astype(np.uint8) * 255


def skeletonize_image(img_uint8):
    """
    img_uint8 : grayscale numpy array, white background (255), black ink (0).
    Returns   : skeleton image – white path (255) on black background (0).
    """
    inv = 255 - img_uint8
    binary = (inv > 127).astype(np.uint8) * 255     # ink = 255

    if HAS_XIMGPROC:
        return _cv2_thin(binary, thinningType=_THINTYPE)

    if HAS_SKIMAGE:
        bool_arr  = binary > 0
        skel_bool = _sk_skel(bool_arr)
        return skel_bool.astype(np.uint8) * 255

    # Pure scipy morphological fallback
    from scipy.ndimage import binary_erosion, binary_dilation
    skeleton = np.zeros_like(binary, dtype=bool)
    img_bool = binary > 0
    for _ in range(50):
        eroded = binary_erosion(img_bool)
        temp   = binary_dilation(eroded)
        skeleton |= (img_bool & ~temp)
        img_bool = eroded
        if not img_bool.any():
            break
    return skeleton.astype(np.uint8) * 255


def extract_ordered_points(skeleton):
    """
    Return an ordered list of (x, y) pixel coordinates from the skeleton image.
    Uses a nearest-neighbour walk starting from the topmost-leftmost pixel.
    """
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return []

    points = list(zip(xs.tolist(), ys.tolist()))

    # Nearest-neighbour walk
    visited  = np.zeros(len(points), dtype=bool)
    ordered  = []
    pt_arr   = np.array(points, dtype=float)

    # Start from topmost-leftmost pixel
    start_idx = int(np.argmin(ys * 10000 + xs))   # top-left bias
    current   = start_idx

    for _ in range(len(points)):
        visited[current] = True
        ordered.append(points[current])
        dists           = np.sum((pt_arr - pt_arr[current]) ** 2, axis=1)
        dists[visited]  = np.inf
        nxt = int(np.argmin(dists))
        if dists[nxt] == np.inf:
            break
        current = nxt

    return ordered

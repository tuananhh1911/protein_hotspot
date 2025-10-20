import cv2
import numpy as np

def to_grayscale(img):
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")

def to_8bit(img):
    if img.dtype == np.uint8:
        return img
    im = img.astype(np.float32)
    im -= im.min()
    denom = im.max() - im.min() + 1e-8
    im = (255.0 * (im / denom)).clip(0, 255).astype(np.uint8)
    return im

def ensure_odd(n):
    return n if n % 2 == 1 else max(1, n - 1)

def gaussian_blur(gray, ksize, sigma):
    if ksize <= 1:
        return gray
    k = ensure_odd(int(ksize))
    return cv2.GaussianBlur(gray, (k, k), sigmaX=float(sigma))

def apply_clahe(gray, clip_limit=2.0, tile_grid_size=8, enabled=True):
    if not enabled:
        return gray
    if gray.dtype != np.uint8:
        g8 = to_8bit(gray)
    else:
        g8 = gray
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    return clahe.apply(g8)

def safe_read(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return im

def normalize_float(im):
    im = im.astype(np.float32)
    if im.max() > 0:
        im = im / im.max()
    return im

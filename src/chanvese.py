# src/chanvese.py
import numpy as np
import cv2
import os, time, yaml
from skimage.segmentation import morphological_chan_vese
from skimage.filters import threshold_otsu

# Lưu thời gian sửa cuối cùng của file YAML để theo dõi thay đổi
_CONFIG_CACHE = {"path": None, "mtime": None, "cfg": None}

def _load_yaml_if_changed(yaml_path: str):
    """Tự động reload file YAML nếu nó thay đổi"""
    global _CONFIG_CACHE
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {yaml_path}")

    mtime = os.path.getmtime(yaml_path)
    if (_CONFIG_CACHE["path"] != yaml_path) or (_CONFIG_CACHE["mtime"] != mtime):
        with open(yaml_path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE["cfg"] = yaml.safe_load(f)
        _CONFIG_CACHE["path"] = yaml_path
        _CONFIG_CACHE["mtime"] = mtime
        print(f"[ChanVese] Reloaded config.yaml (modified {time.ctime(mtime)})")

    return _CONFIG_CACHE["cfg"]

def _normalize01(img):
    img = img.astype(np.float32)
    m, M = np.min(img), np.max(img)
    return (img - m) / (M - m) if M > m else np.zeros_like(img, dtype=np.float32)

def _remove_small_components(bin_img, min_region):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (bin_img > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(bin_img, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_region):
            out[labels == i] = 255
    return out

def _fill_small_holes(bin_img, min_hole):
    inv = (bin_img == 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep = np.zeros_like(inv, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_hole):
            keep[labels == i] = 255
    return ((keep == 0).astype(np.uint8) * 255)

def _keep_largest_component(bin_img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (bin_img > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return bin_img
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(bin_img, dtype=np.uint8)
    out[labels == largest_idx] = 255
    return out


def chan_vese_mask(gray, cfg=None, config_path="config.yaml"):
    """
    Giai đoạn 1 – Tách foreground (protein-positive region)
    - Tự động reload config.yaml nếu thay đổi trong khi chạy
    - Dùng morphological Chan–Vese
    - Trả về mask_foreground và ảnh foreground
    """

    # ----- nếu không truyền cfg, tự load từ file -----
    if cfg is None:
        cfg = _load_yaml_if_changed(config_path)
    elif isinstance(cfg, str) and os.path.exists(cfg):
        cfg = _load_yaml_if_changed(cfg)

    # ----- lấy tham số từ YAML -----
    params = cfg.get("chanvese", {})
    iterations   = int(params.get("iterations", 300))
    smoothing    = int(params.get("smoothing", 2))
    lambda1      = float(params.get("lambda1", 1.0))
    lambda2      = float(params.get("lambda2", 2.0))
    init_mode    = str(params.get("init", "otsu"))
    disk_radius  = int(params.get("disk_radius", 30))
    min_region   = int(params.get("min_region_area", 500))
    min_hole     = int(params.get("min_hole_area", 100))
    keep_biggest = bool(params.get("keep_largest", False))

    # ----- in tham số đang dùng -----
    print(f"[ChanVese] iter={iterations}, λ1={lambda1}, λ2={lambda2}, init={init_mode}, "
          f"disk={disk_radius}, keep_biggest={keep_biggest}")

    # ----- chuẩn hoá ảnh -----
    I = _normalize01(gray)

    # ----- khởi tạo level set -----
    if init_mode == "otsu":
        t = threshold_otsu((I * 255).astype(np.uint8))
        init_ls = (I > t / 255.0).astype(np.uint8)
    elif init_mode == "disk":
        h, w = I.shape
        Y, X = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        init_ls = ((X - cx) ** 2 + (Y - cy) ** 2) <= (disk_radius ** 2)
        init_ls = init_ls.astype(np.uint8)
    else:
        init_ls = "checkerboard"

    # ----- chạy morphological Chan–Vese -----
    ls = morphological_chan_vese(
        I, num_iter=iterations, init_level_set=init_ls,
        smoothing=smoothing, lambda1=lambda1, lambda2=lambda2
    )
    mask = (ls.astype(np.uint8) * 255)

    # ----- hậu xử lý -----
    if min_region > 0:
        mask = _remove_small_components(mask, min_region)
    if min_hole > 0:
        mask = _fill_small_holes(mask, min_hole)
    if keep_biggest:
        mask = _keep_largest_component(mask)

    # ----- tạo ảnh foreground -----
    foreground = (gray.astype(np.float32) * (mask > 0)).astype(np.uint8)
    return mask, foreground

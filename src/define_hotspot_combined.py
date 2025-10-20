import numpy as np
import cv2
import yaml
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from src.chanvese import _remove_small_components, _fill_small_holes, _keep_largest_component


# ================== HÀM TIỆN ÍCH ==================
def _normalize01(img):
    """Chuẩn hóa ảnh về khoảng [0,1]."""
    img = img.astype(np.float32)
    m, M = np.min(img), np.max(img)
    return (img - m) / (M - m) if M > m else np.zeros_like(img)


def _load_yaml(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ================== PHÁT HIỆN HOTSPOT ==================
def define_hotspot_combined(foreground, baseline=None, cfg=None, config_path="config.yaml"):
    """
    Phiên bản cải tiến: K-means + adaptive Otsu + local contrast.
    Hoạt động ổn định hơn với ảnh nhiễu và hotspot mờ.
    """
    if cfg is None:
        cfg = _load_yaml(config_path)
    params = cfg.get("hotspot", {})

    min_region = int(params.get("min_region_area", 100))
    min_hole = int(params.get("min_hole_area", 50))
    keep_biggest = bool(params.get("keep_largest", True))
    k_clusters = int(params.get("k_clusters", 3))

    # --- Chuẩn hóa ---
    I = _normalize01(foreground)

    # --- K-means ---
    valid = I[I > 0].reshape(-1, 1)
    if len(valid) < 10:
        raise ValueError("Không đủ pixel foreground hợp lệ để phân cụm.")

    kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
    kmeans.fit(valid)
    centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.predict(I.reshape(-1, 1)).reshape(I.shape)

    # --- Lấy 2 cụm sáng nhất làm vùng tiềm năng ---
    bright_idx = np.argsort(centers)[-2:]
    region_mask = np.isin(labels, bright_idx).astype(np.uint8)

    # --- Local contrast (nâng biên vùng sáng) ---
    blur = cv2.medianBlur((I * 255).astype(np.uint8), 15)
    local_contrast = cv2.normalize(I - blur / 255.0, None, 0, 1, cv2.NORM_MINMAX)

    # --- Weighted image: cân bằng sáng và tương phản ---
    weighted_img = 0.6 * I + 0.4 * local_contrast
    weighted_img = cv2.GaussianBlur(weighted_img, (5, 5), 0)

    # --- Otsu threshold ---
    otsu_thr, _ = cv2.threshold((weighted_img * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_val = otsu_thr / 255.0

    mask_hot = ((weighted_img > thr_val) & (region_mask > 0)).astype(np.uint8) * 255

    # --- Adaptive correction nếu vùng quá nhỏ ---
    ratio = np.sum(mask_hot > 0) / mask_hot.size
    if ratio < 0.03:
        adj_thr = max(0, thr_val * 0.9)
        mask_hot = ((weighted_img > adj_thr) & (region_mask > 0)).astype(np.uint8) * 255

    # --- Hậu xử lý ---
    mask_hot = _remove_small_components(mask_hot, min_region)
    mask_hot = _fill_small_holes(mask_hot, min_hole)
    if keep_biggest:
        mask_hot = _keep_largest_component(mask_hot)

    # --- Tính thông số ---
    labeled = label(mask_hot > 0)
    props = regionprops(labeled)
    area = sum(p.area for p in props)
    num = len(props)

    hotspot_pixels = I[mask_hot > 0]
    mean_val = float(np.mean(hotspot_pixels)) if len(hotspot_pixels) else 0
    std_val = float(np.std(hotspot_pixels)) if len(hotspot_pixels) else 0

    metrics = {
        "mean_intensity": mean_val,
        "std_intensity": std_val,
        "threshold": float(thr_val),
        "num_hotspots": num,
        "total_area": int(area),
        "cluster_centers": centers.tolist()
    }

    return mask_hot, metrics



# ================== OVERLAY ==================
def overlay_hotspot(gray, mask_hotspot, alpha=0.5):
    """
    Tạo overlay vùng hotspot (đỏ, fill kín toàn bộ vùng).
    """
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Đảm bảo mask đã fill kín (nếu chưa)
    mask_filled = np.zeros_like(mask_hotspot)
    contours, _ = cv2.findContours(mask_hotspot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(mask_filled, contours, -1, 255, -1)

    # Tạo lớp màu đỏ
    color = np.zeros_like(overlay)
    color[:, :, 2] = 255  # đỏ

    # Áp chồng (alpha blending)
    m = mask_filled > 0
    if np.count_nonzero(m) > 0:
        blended = cv2.addWeighted(overlay[m], 1 - alpha, color[m], alpha, 0)
        overlay[m] = blended

    return overlay

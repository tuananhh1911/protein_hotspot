import numpy as np
import cv2
import yaml
from skimage.filters import threshold_multiotsu
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
    Phát hiện vùng tập trung protein (hotspot) trong ảnh huỳnh quang đơn kênh.

    Quy trình:
    1. Chuẩn hóa ảnh và trừ nền (nếu có)
    2. Phân cụm sáng bằng K-Means (k=3)
    3. Chọn 2 cụm sáng nhất làm vùng protein
    4. Ngưỡng hóa tinh chỉnh và hậu xử lý (lọc nhiễu, lấp lỗ, nối kín biên)
    5. Fill kín toàn bộ vùng bên trong
    """
    # --- Đọc config ---
    if cfg is None:
        cfg = _load_yaml(config_path)
    params = cfg.get("hotspot", {})

    k_factor = float(params.get("k_factor", 0.3))               # ⭐ giảm ngưỡng để không cắt mất vùng sáng
    min_region = int(params.get("min_region_area", 100))
    min_hole = int(params.get("min_hole_area", 50))
    keep_biggest = bool(params.get("keep_largest", True))
    k_clusters = int(params.get("k_clusters", 3))

    # --- Chuẩn hóa & trừ nền ---
    I = _normalize01(foreground)
    #if baseline is not None:
     #  base = _normalize01(baseline)
     #   I = np.clip(I - base, 0, 1)

    # --- Kiểm tra hợp lệ ---
    valid = I[I > 0].reshape(-1, 1)
    if len(valid) < 10:
        raise ValueError("Không đủ pixel foreground hợp lệ để phân cụm.")

    # --- Phân cụm K-Means ---
    kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
    kmeans.fit(valid)
    centers = sorted(kmeans.cluster_centers_.flatten())
    labels = kmeans.predict(I.reshape(-1, 1)).reshape(I.shape)

    # --- ⭐ Chọn 2 cụm sáng nhất thay vì 1 ---
    if k_clusters >= 3:
        centers_sorted = np.argsort(kmeans.cluster_centers_.flatten())
        bright_labels = centers_sorted[-2:]  # 2 cụm sáng nhất
    else:
        bright_labels = [np.argmax(kmeans.cluster_centers_.flatten())]

    # --- Lọc tinh cụm sáng ---
    cluster_pixels = I[np.isin(labels, bright_labels)]
    thr_values = threshold_multiotsu(cluster_pixels, classes=3)
    thr = thr_values[-1] * 0.95

    # ⭐ Giữ lại vùng sáng trong 2 cụm sáng nhất
    mask_hot = ((I > thr) & np.isin(labels, bright_labels)).astype(np.uint8) * 255


    # --- Hậu xử lý mask ---
    if min_region > 0:
        mask_hot = _remove_small_components(mask_hot, min_region)
    if min_hole > 0:
        mask_hot = _fill_small_holes(mask_hot, min_hole)
    if keep_biggest:
        mask_hot = _keep_largest_component(mask_hot)

    # --- Tính thông số vùng hotspot ---
    labeled = label(mask_hot > 0)
    props = regionprops(labeled)
    area = sum(p.area for p in props)
    num = len(props)

    # Calculate intensity statistics within hotspot regions
    hotspot_pixels = I[mask_hot > 0]
    mean_val = np.mean(hotspot_pixels) if len(hotspot_pixels) > 0 else 0
    std_val = np.std(hotspot_pixels) if len(hotspot_pixels) > 0 else 0

    metrics = {
        "mean_intensity": float(mean_val),
        "std_intensity": float(std_val),
        "threshold": float(thr),
        "num_hotspots": num,
        "total_area": int(area),
        "cluster_centers": centers
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

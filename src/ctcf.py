import numpy as np
from .utils import to_grayscale

def compute_ctcf(img, mask, cfg_bg, cfg_clip):
    gray = to_grayscale(img).astype(np.float32)

    # Optional clipping
    cmin = cfg_clip.get("clip_min", None)
    cmax = cfg_clip.get("clip_max", None)
    if cmin is not None:
        gray = np.maximum(gray, float(cmin))
    if cmax is not None and cmax > 0:
        gray = np.minimum(gray, float(cmax))

    m = (mask > 0)
    area = float(m.sum())
    integrated_density = float(gray[m].sum()) if area > 0 else 0.0

    # Background estimation
    method = cfg_bg.get("method", "outside_mask")
    if method == "percentile":
        p = float(cfg_bg.get("percentile", 20))
        bg_mean = float(np.percentile(gray.flatten(), p))
    else:
        outside = gray[~m]
        bg_mean = float(outside.mean()) if outside.size > 0 else 0.0

    ctcf = integrated_density - bg_mean * area

    return {
        "area": area,
        "integrated_density": integrated_density,
        "bg_mean": bg_mean,
        "ctcf": ctcf,
    }

from .utils import to_grayscale, gaussian_blur, apply_clahe

def preprocess_image(img, cfg):
    gray = to_grayscale(img)

    denoise = cfg.get("denoise", {})
    gray = gaussian_blur(gray, denoise.get("gaussian_ksize", 5), denoise.get("gaussian_sigma", 0.8))

    clahe_cfg = cfg.get("clahe", {})
    gray = apply_clahe(
        gray,
        clip_limit=clahe_cfg.get("clip_limit", 1.5),
        tile_grid_size=clahe_cfg.get("tile_grid_size", 8),
        enabled=clahe_cfg.get("enabled", True),
    )
    return gray

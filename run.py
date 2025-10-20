import os, argparse, json, csv, time, yaml, cv2, numpy as np, tkinter as tk
from datetime import datetime
from tkinter import filedialog

from src.utils import safe_read, to_8bit
from src.preprocess import preprocess_image
from src.chanvese import chan_vese_mask
from src.ctcf import compute_ctcf
from src.define_hotspot_combined import define_hotspot_combined, overlay_hotspot


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _convert_numpy(obj):
    """Chuyển đổi giá trị numpy sang kiểu Python để lưu JSON."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    return obj

def save_json(path, obj):
    obj_clean = _convert_numpy(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj_clean, f, indent=2, ensure_ascii=False)

def save_image(path, im):
    cv2.imwrite(path, im)

def overlay_mask(gray8, mask):
    out = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    alpha = 0.5
    color = np.zeros_like(out)
    color[:, :, 1] = 255  # overlay xanh lá
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * color[m]).astype(np.uint8)
    return out

def select_input_images():
    root = tk.Tk()
    root.withdraw()
    return list(filedialog.askopenfilenames(
        title="Chọn ảnh huỳnh quang",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")]
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None,
                    help="Ảnh baseline (Kenh truoc khi ap dien).")
    args = ap.parse_args()

    # --- Đọc YAML ---
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seg_mode = (cfg.get("segmentation", {}).get("mode", "chanvese")).lower()
    save_mode = cfg.get("io", {}).get("save_mode", "report").lower()

    print(f"🧭 Segmentation mode: {seg_mode.upper()}")
    print(f"💾 Save mode: {save_mode}")

    base_out = args.out or cfg["io"]["out_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_out, timestamp)
    ensure_dir(out_dir)

    # --- Danh sách ảnh ---
    if args.input is None:
        imgs = select_input_images()
    else:
        in_dir = args.input or cfg["io"]["input_dir"]
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        imgs = [os.path.join(in_dir, f) for f in os.listdir(in_dir)
                if f.lower().endswith(exts)]
    if not imgs:
        print("❌ Không tìm thấy ảnh input!")
        return

    # --- Ảnh baseline ---
    baseline_path = args.baseline or cfg.get("io", {}).get("baseline_path", None)
    baseline_img = None
    if baseline_path and os.path.exists(baseline_path):
        baseline_img = cv2.imread(baseline_path, cv2.IMREAD_GRAYSCALE)
        print(f"📎 Baseline loaded: {os.path.basename(baseline_path)}")

    csv_path = os.path.join(out_dir, "measure_summary.csv")
    fieldnames = ["image", "bg_mean", "area",
                  "integrated_density", "ctcf"]

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for i, path in enumerate(imgs, 1):
            t0 = time.time()
            stem = os.path.splitext(os.path.basename(path))[0]
            print(f"[{i}/{len(imgs)}] ▶ Xử lý: {stem}")

            odir = os.path.join(out_dir, stem)
            mask_dir = os.path.join(odir, "masks")
            overlay_dir = os.path.join(odir, "overlays")
            measure_dir = os.path.join(odir, "measures")
            for d in (mask_dir, overlay_dir, measure_dir):
                ensure_dir(d)

            # --- Tiền xử lý ---
            img = safe_read(path)
            gray_pre = preprocess_image(img, cfg.get("preprocess", {}))
            gray8 = to_8bit(gray_pre)

            # ========== 🧠 CHẾ ĐỘ LƯU DỮ LIỆU U-NET ==========
            if save_mode == "unet":
                print("🧩 Đang tạo tập dữ liệu huấn luyện cho mô hình U-Net...")

                # Thư mục đích: data/raw/
                base_raw = os.path.join("data", "raw")
                orig_dir = os.path.join(base_raw, "images_original")
                fore_dir = os.path.join(base_raw, "images_foreground")
                mask_dir_unet = os.path.join(base_raw, "masks_protein")
                for d in (orig_dir, fore_dir, mask_dir_unet):
                    ensure_dir(d)

                # --- 1️⃣ Ảnh gốc ---
                save_image(os.path.join(orig_dir, f"{stem}.png"), gray8)

                # --- 2️⃣ Tách foreground bằng Chan-Vese ---
                mask_fore, fore = chan_vese_mask(gray_pre, cfg)
                save_image(os.path.join(fore_dir, f"{stem}_foreground.png"), fore)

                # --- 3️⃣ Tạo mask hotspot thật (ground truth) ---
                mask_hot, metrics = define_hotspot_combined(fore, baseline=baseline_img, cfg=cfg)
                save_image(os.path.join(mask_dir_unet, f"{stem}_mask.png"), mask_hot)

                # --- 4️⃣ Ghi thông tin thống kê (tùy chọn) ---
                save_json(os.path.join(measure_dir, "hotspot_metrics.json"), metrics)

                print(f"✅ Đã lưu cho U-Net: {stem}.png | {stem}_foreground.png | {stem}_mask.png\n")
                continue  # bỏ qua các bước đo lường còn lại
            # ==================================================

            if seg_mode == "chanvese":
                # Giai đoạn 1: foreground
                mask_fore, fore = chan_vese_mask(gray_pre, cfg)
                save_image(os.path.join(mask_dir, f"{stem}_mask_foreground.png"), mask_fore)
                save_image(os.path.join(overlay_dir, f"{stem}_overlay_fore.png"),
                           overlay_mask(gray8, mask_fore))

                # Giai đoạn 2: hotspot (ΔI + K-Means + mean + k×std)
                mask_hot, metrics = define_hotspot_combined(fore, baseline=baseline_img, cfg=cfg)
                overlay = overlay_hotspot(gray8, mask_hot)
                save_image(os.path.join(mask_dir, f"{stem}_mask_hotspot.png"), mask_hot)
                save_image(os.path.join(overlay_dir, f"{stem}_overlay_hotspot.png"), overlay)
                save_json(os.path.join(measure_dir, "hotspot_metrics.json"), metrics)

                print(f"🔥 Hotspot: mean={metrics['mean_intensity']:.3f}, "
                      f"std={metrics['std_intensity']:.3f}, thr={metrics['threshold']:.3f}, "
                      f"regions={metrics['num_hotspots']}")

            elif seg_mode == "unet":
                # Inference thông thường (nếu không phải chế độ save_mode unet)
                from src.unet.evaluate_unet import predict_mask_unet
                uc = cfg.get("unet", {})
                mask = predict_mask_unet(img,
                                         model_path=uc.get("model_path", "checkpoints/unet_best.pt"),
                                         size=int(uc.get("input_size", 256)),
                                         threshold=float(uc.get("threshold", 0.5)))
                save_image(os.path.join(mask_dir, f"{stem}_mask.png"), mask)
                save_image(os.path.join(overlay_dir, f"{stem}_overlay.png"),
                           overlay_mask(gray8, mask))
            else:
                raise ValueError(f"❌ segmentation.mode không hợp lệ: {seg_mode}")

            # --- CTCF ---
            if "background" in cfg and "ctcf" in cfg:
                mask_used = mask_hot if seg_mode == "chanvese" else mask
                ctcf = compute_ctcf(img, mask_used,
                                    cfg["background"], cfg["ctcf"])
                save_json(os.path.join(measure_dir, "ctcf.json"), ctcf)
                writer.writerow({
                    "image": os.path.basename(path),
                    "bg_mean": ctcf["bg_mean"],
                    "area": ctcf["area"],
                    "integrated_density": ctcf["integrated_density"],
                    "ctcf": ctcf["ctcf"]
                })

            print(f"⏱️  {stem} done in {time.time() - t0:.2f}s\n")

    print(f"\n✅ Hoàn tất xử lý {len(imgs)} ảnh → {out_dir}")


if __name__ == "__main__":
    main()

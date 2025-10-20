import os
import cv2
import torch
import numpy as np
from src.unet.model_unet import UNet

def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image_for_unet(img, size=256):
    """Chuyển ảnh OpenCV sang tensor chuẩn U-Net"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (size, size))
    gray = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor

def predict_mask_unet(img, model_path="checkpoints/unet_best.pt", size=256, threshold=0.5):
    """Dự đoán mask vùng protein bằng U-Net"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    tensor = preprocess_image_for_unet(img, size).to(device)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    mask = (pred > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # trả về kích thước gốc
    return mask

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

class ProteinDataset(Dataset):
    """
    Dataset đọc ảnh huỳnh quang và mask vùng protein cho mô hình U-Net.
    """

    def __init__(self, image_dir, mask_dir, image_size=256, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        # Lấy danh sách ảnh gốc
        self.images = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # Đọc ảnh grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Không thể đọc ảnh hoặc mask: {fname}")

        # Resize về cùng kích thước
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Normalize về [0,1]
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        if mask.sum() == 0:
            print(f"⚠️ Warning: mask rỗng - {fname}")


        # Chuyển sang tensor (C,H,W)
        image = torch.from_numpy(image).unsqueeze(0)  # [1,H,W]
        mask = torch.from_numpy(mask).unsqueeze(0)

        # --- Augmentation (nếu bật) ---
        if self.augment:
            if torch.rand(1) < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if torch.rand(1) < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if torch.rand(1) < 0.3:
                angle = torch.randint(-15, 15, (1,)).item()
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        return image, mask


def get_dataloaders(base_dir="data/dataset", image_size=256, batch_size=4, augment=True):
    """
    Trả về DataLoader cho train, val, test.
    """
    paths = {
        "train": (os.path.join(base_dir, "train/images"), os.path.join(base_dir, "train/masks")),
        "val":   (os.path.join(base_dir, "val/images"), os.path.join(base_dir, "val/masks")),
        "test":  (os.path.join(base_dir, "test/images"), os.path.join(base_dir, "test/masks")),
    }

    train_ds = ProteinDataset(*paths["train"], image_size=image_size, augment=augment)
    val_ds   = ProteinDataset(*paths["val"], image_size=image_size, augment=False)
    test_ds  = ProteinDataset(*paths["test"], image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

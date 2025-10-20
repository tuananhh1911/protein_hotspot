import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src.unet.model_unet import UNet
from src.unet.dataset_loader import get_dataloaders


# --- Dice coefficient (for metric only) ---
def dice_coeff(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return dice.mean()


# --- Combo Loss: Dice + BCE (for training) ---
class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred = torch.sigmoid(pred)

        smooth = 1e-6
        inter = (pred * target).sum(dim=(2, 3))
        dice = (2.0 * inter + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        dice_loss = 1 - dice.mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# --- Train one epoch ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)


# --- Evaluate on validation set ---
def evaluate(model, loader, device):
    model.eval()
    dice_score = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validating", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.sigmoid(outputs)
            dice_score += dice_coeff(preds, masks).item()
    return dice_score / len(loader)


# --- Main ---
def main():
    # Config
    base_dir = "data/dataset"
    image_size = 256
    batch_size = 4
    epochs = 50
    lr = 1e-4
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("⚙️  Training on:", device)

    # Dataset
    train_loader, val_loader, _ = get_dataloaders(base_dir, image_size, batch_size, augment=True)

    # Model
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Loss + Optimizer
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)


    # TensorBoard
    writer = SummaryWriter(log_dir="runs/unet_training")

    best_dice = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\n🟢 Epoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_dice = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "unet_best.pt"))
            print(f"✅ Model improved (Dice={val_dice:.4f}) → saved!")

        # Save last model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "unet_last.pt"))

    print(f"\n🏁 Training completed! Best Dice = {best_dice:.4f}")

    writer.close()


if __name__ == "__main__":
    main()

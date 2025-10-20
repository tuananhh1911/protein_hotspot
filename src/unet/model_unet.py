import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------
# 🔹 Khối tích chập kép (DoubleConv)
# ---------------------------------------------------
class DoubleConv(nn.Module):
    """
    Một khối gồm 2 lớp Conv2d liên tiếp, mỗi lớp kèm BatchNorm và ReLU.
    Giúp mạng học đặc trưng chi tiết hơn mà vẫn ổn định gradient.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# ---------------------------------------------------
# 🔹 Kiến trúc chính của mô hình U-Net
# ---------------------------------------------------
class UNet(nn.Module):
    """
    Mô hình U-Net cơ bản cho bài toán segmentation:
    Encoder (downsampling) + Bottleneck + Decoder (upsampling) + Skip connections.
    """

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        # --- Encoder ---
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(ch, f))
            ch = f

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # --- Decoder ---
        self.decoder = nn.ModuleList()
        for f in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)   # upsample ×2
            )
            self.decoder.append(DoubleConv(f*2, f))                  # concat skip + conv

        # --- Output layer ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # ----- Encoder -----
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # ----- Bottleneck -----
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # đảo ngược để dễ dùng cho decoder

        # ----- Decoder -----
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # upsample
            skip = skip_connections[idx // 2]

            # cắt cho khớp kích thước (tránh sai lệch do pooling)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat((skip, x), dim=1)  # ghép kênh encoder + decoder
            x = self.decoder[idx + 1](x)

        # ----- Output -----
        return self.final_conv(x)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 22:57:55 2025

@author: shakib
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:47:25 2025

@author: shakib
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.classification import F1Score, Precision, JaccardIndex
from torchmetrics import Dice
# from PIL import Image  # Only needed if you do PIL-based transforms

# ===================== #
#   CBAM ATTENTION
# ===================== #
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # You could also add a max pool path if you want to replicate the full CBAM module
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        out = self.mlp(avg_out).view(b, c, 1, 1)
        return torch.sigmoid(out)


# (Optional) Spatial Attention – classical CBAM typically has both channel + spatial
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv2d = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # Channel-wise pooling: average + max (optional approach)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)  # if you want to combine max + avg
        # x_out = torch.cat([avg_out, max_out], dim=1)
        # For simplicity, let's just use avg_out
        x_out = avg_out
        x_out = self.conv2d(x_out)
        return torch.sigmoid(x_out)


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        # self.spatial_attention = SpatialAttention()  # If you want the full CBAM version

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # If using spatial attention, you'd do:
        # sa = self.spatial_attention(x)
        # x = x * sa

        return x


# ===================== #
#   SIMPLE CBAM UNET
# ===================== #
class SimpleCBAMUNet(nn.Module):
    """
    Simple 2D U-Net with CBAM blocks. Note that the classical CBAM has both
    channel and spatial attention; we added only channel attention by default.
    """
    def __init__(self, in_channels=4, out_channels=3):
        """
        :param in_channels: Number of input channels (for BraTS, typically 4: Flair, T1, T1ce, T2)
        :param out_channels: Number of classes (for your code, 3 classes: 0,1,2)
        """
        super(SimpleCBAMUNet, self).__init__()

        # --- Encoder ---
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(64, 128)

        # --- Decoder ---
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(64 + 64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(32 + 32, 32)
        self.conv_final = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Basic block: (Conv -> ReLU -> Conv -> ReLU -> CBAM).
        You can expand or reduce as needed.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(out_channels)
        )

    def forward(self, x):
        # --- Encoder path ---
        e1 = self.enc1(x)
        e2 = self.pool1(e1)
        e2 = self.enc2(e2)
        e3 = self.pool2(e2)
        e3 = self.enc3(e3)

        # --- Decoder path ---
        d3 = self.up3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out = self.conv_final(d2)
        return out


# ===================== #
#   BraTS DATASET
# ===================== #
class BraTSDataset(Dataset):
    def __init__(self, data_dir, patient_ids):
        """
        :param data_dir: Folder containing all the BraTS subjects
        :param patient_ids: List of subject IDs to use in this dataset
        """
        self.data_dir = data_dir
        self.patient_ids = patient_ids

    def __len__(self):
        # Each subject has 155 slices.
        return len(self.patient_ids) * 155

    def __getitem__(self, idx):
        """
        Returns a single slice from a given subject.
        """
        patient_idx = idx // 155
        slice_idx = idx % 155
        patient_id = self.patient_ids[patient_idx]

        # --- Paths to each modality + segmentation ---
        flair_path = os.path.join(self.data_dir, patient_id, f"{patient_id}_flair.nii.gz")
        t1_path = os.path.join(self.data_dir, patient_id, f"{patient_id}_t1.nii.gz")
        t2_path = os.path.join(self.data_dir, patient_id, f"{patient_id}_t2.nii.gz")
        t1ce_path = os.path.join(self.data_dir, patient_id, f"{patient_id}_t1ce.nii.gz")
        seg_path = os.path.join(self.data_dir, patient_id, f"{patient_id}_seg.nii.gz")

        # --- Load volumes and extract the slice ---
        flair_img = nib.load(flair_path).get_fdata()
        t1_img = nib.load(t1_path).get_fdata()
        t2_img = nib.load(t2_path).get_fdata()
        t1ce_img = nib.load(t1ce_path).get_fdata()
        seg_img = nib.load(seg_path).get_fdata()

        flair_slice = flair_img[:, :, slice_idx]
        t1_slice = t1_img[:, :, slice_idx]
        t2_slice = t2_img[:, :, slice_idx]
        t1ce_slice = t1ce_img[:, :, slice_idx]
        seg_slice = seg_img[:, :, slice_idx]

        # --- Combine BraTS segmentation labels into 0,1,2 ---
        # Standard BraTS classes are {0,1,2,4}. You choose to merge 2 & 4 => 2.
        seg_slice[seg_slice > 1] = 2  

        # --- Normalization per-slice (avoid division by zero) ---
        def normalize(img):
            return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)

        flair_slice = normalize(flair_slice)
        t1_slice = normalize(t1_slice)
        t2_slice = normalize(t2_slice)
        t1ce_slice = normalize(t1ce_slice)

        # Stack into 4 channels => shape: (4, H, W)
        input_img = torch.tensor(np.stack([flair_slice, t1_slice, t1ce_slice, t2_slice], axis=0), dtype=torch.float32)
        seg_slice = torch.tensor(seg_slice, dtype=torch.long)

        return input_img, seg_slice


# ===================== #
#   SAVE METRICS
# ===================== #
def save_metrics_to_excel(metrics, filename="training_metrics5.xlsx"):
    df = pd.DataFrame(metrics)
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"✅ Les métriques ont été sauvegardées dans {filename}")

# ===================== #
#   MAIN TRAINING LOOP
# ===================== #
def main():
    data_dir = '/home/shakib/Desktop/both/Transformer/brats/bratsomar'
    patient_ids = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    train_dataset = BraTSDataset(data_dir, train_ids)
    val_dataset = BraTSDataset(data_dir, val_ids)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IMPORTANT: We have 4 input channels => pass in_channels=4
    model = SimpleCBAMUNet(in_channels=4, out_channels=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    metrics = []

    for epoch in range(num_epochs):
        print(f"\n🔵 Epoch [{epoch+1}/{num_epochs}] - Entraînement en cours...")
        model.train()
        running_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"  🟢 Step [{i+1}/{len(train_loader)}] - Loss: {running_loss/10:.4f}")
                running_loss = 0.0

        # ===================== #
        #    VALIDATION LOOP
        # ===================== #
        print(f"🔵 Epoch [{epoch+1}/{num_epochs}] - Validation en cours...")
        model.eval()
        val_loss = 0.0

        # Torchmetrics objects for 3-class segmentation (macro-average).
        f1_metric = F1Score(task="multiclass", num_classes=3, average="macro").to(device)
        precision_metric = Precision(task="multiclass", num_classes=3, average="macro").to(device)
        dice_metric = Dice(num_classes=3, average="macro").to(device)
        iou_metric = JaccardIndex(task="multiclass", num_classes=3, average="macro").to(device)

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

                preds = torch.argmax(outputs, dim=1)
                f1_metric.update(preds, masks)
                precision_metric.update(preds, masks)
                dice_metric.update(preds, masks)
                iou_metric.update(preds, masks)

        val_loss /= len(val_loader)
        f1_value = f1_metric.compute().item()
        precision_value = precision_metric.compute().item()
        dice_value = dice_metric.compute().item()
        iou_value = iou_metric.compute().item()

        print(f"  🔴 Validation Loss: {val_loss:.4f}")
        print(f"  📊 F1 Score: {f1_value:.4f}")
        print(f"  📊 Precision: {precision_value:.4f}")
        print(f"  📊 Dice: {dice_value:.4f}")
        print(f"  📊 IoU: {iou_value:.4f}")

        metrics.append({
            "Epoch": epoch+1,
            "Validation Loss": val_loss,
            "F1 Score": f1_value,
            "Precision": precision_value,   
            "Dice": dice_value,
            "IoU": iou_value
        })

    save_metrics_to_excel(metrics, filename="training_metrics6.xlsx")
    torch.save(model.state_dict(), "brats_segmentation_model6.pth")


if __name__ == "__main__":
    main()

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
from PIL import Image

# === CBAM Attention === #
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels)
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        out = self.mlp(avg_out).view(x.size(0), x.size(1), 1, 1)
        return torch.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x):
        return x * self.channel_attention(x)

# === UNet avec CBAM === #
class SimpleCBAMUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super(SimpleCBAMUNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(64 + 64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(32 + 32, 32)
        self.conv_final = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(out_channels)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.pool1(e1)
        e2 = self.enc2(e2)
        e3 = self.pool2(e2)
        e3 = self.enc3(e3)

        d3 = self.up3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out = self.conv_final(d2)
        return out

# === Dataset BraTS === #
class BraTSDataset(Dataset):
    def __init__(self, data_dir, patient_ids):
        self.data_dir = data_dir
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids) * 155

    def __getitem__(self, idx):
        patient_idx = idx // 155
        slice_idx = idx % 155
        patient_id = self.patient_ids[patient_idx]

        t2_path = os.path.join(self.data_dir, patient_id, f"{patient_id}_t2.nii.gz")
        t1ce_path = os.path.join(self.data_dir, patient_id, f"{patient_id}_t1ce.nii.gz")
        seg_path = os.path.join(self.data_dir, patient_id, f"{patient_id}_seg.nii.gz")

        t2_img = nib.load(t2_path).get_fdata()
        t1ce_img = nib.load(t1ce_path).get_fdata()
        seg_img = nib.load(seg_path).get_fdata()

        t2_slice = t2_img[:, :, slice_idx]
        t1ce_slice = t1ce_img[:, :, slice_idx]
        seg_slice = seg_img[:, :, slice_idx]

        seg_slice[seg_slice > 1] = 2  

        t2_slice = (t2_slice - np.min(t2_slice)) / (np.max(t2_slice) - np.min(t2_slice) + 1e-6)
        t1ce_slice = (t1ce_slice - np.min(t1ce_slice)) / (np.max(t1ce_slice) - np.min(t1ce_slice) + 1e-6)

        input_img = torch.tensor(np.stack([t2_slice, t1ce_slice], axis=0), dtype=torch.float32)
        seg_slice = torch.tensor(seg_slice, dtype=torch.long)

        return input_img, seg_slice

# === Fonction pour sauvegarder les métriques === #
def save_metrics_to_excel(metrics, filename="training_metrics3.xlsx"):
    df = pd.DataFrame(metrics)
    df.to_excel(filename, index=False, engine="openpyxl")  # Ajout du moteur OpenPyXL
    print(f"✅ Les métriques ont été sauvegardées dans {filename}")

# === Entraînement === #
def main():
    data_dir = '/home/shakib/Desktop/both/Transformer/brats/minibrats'
    patient_ids = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    train_dataset = BraTSDataset(data_dir, train_ids)
    val_dataset = BraTSDataset(data_dir, val_ids)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCBAMUNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
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

        # Validation
        print(f"🔵 Epoch [{epoch+1}/{num_epochs}] - Validation en cours...")
        model.eval()
        val_loss = 0.0
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

        print(f"  🔴 Validation Loss: {val_loss/len(val_loader):.4f}")
        print(f"  📊 F1 Score: {f1_metric.compute()}")
        print(f"  📊 Precision: {precision_metric.compute()}")
        print(f"  📊 Dice: {dice_metric.compute()}")
        print(f"  📊 IoU: {iou_metric.compute()}")
        metrics.append({"Epoch": epoch+1, "F1 Score": f1_metric.compute().item(), "IoU": iou_metric.compute().item(), "Precision": precision_metric.compute().item(), "Dice": dice_metric.compute().item()})

    save_metrics_to_excel(metrics)
    torch.save(model.state_dict(), "brats_segmentation_model3.pth")

if __name__ == "__main__":
    main()

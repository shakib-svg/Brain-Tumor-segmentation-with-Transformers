import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import scipy

class BratsDataset3D(Dataset):
    def __init__(self, patient_folders, transform=None):
        self.patient_folders = patient_folders
        self.transform = transform
    
    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_folder = self.patient_folders[idx]

    
        flair = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_flair.nii.gz")).get_fdata()
        t1 = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t1.nii.gz")).get_fdata()
        t1ce = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t1ce.nii.gz")).get_fdata()
        t2 = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t2.nii.gz")).get_fdata()

        
        mask = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_seg.nii.gz")).get_fdata()

        
        flair = np.clip(flair, 0, np.percentile(flair, 99)) / np.percentile(flair, 99)
        t1 = np.clip(t1, 0, np.percentile(t1, 99)) / np.percentile(t1, 99)
        t1ce = np.clip(t1ce, 0, np.percentile(t1ce, 99)) / np.percentile(t1ce, 99)
        t2 = np.clip(t2, 0, np.percentile(t2, 99)) / np.percentile(t2, 99)

        mask = (mask > 0).astype(np.float32)

        
        image = torch.tensor(np.stack([flair, t1, t1ce, t2], axis=0), dtype=torch.float32)  # 4 canaux
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # 1 canal

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

class CBAMBlock3D(nn.Module):
    def __init__(self, channels):
        super(CBAMBlock3D, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv3d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv3d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x) * x
        sa = self.spatial_att(torch.cat([torch.max(ca, 1, keepdim=True)[0], torch.mean(ca, 1, keepdim=True)], dim=1)) * ca
        return sa


class CBAM_TransUNET_3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(CBAM_TransUNET_3D, self).__init__()

      
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.cbam = CBAMBlock3D(64)

        self.decoder = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.cbam(x)
        x = self.decoder(x)
        
        return x

BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/BraTS2021_Training_Data"
patient_folders = sorted(glob(os.path.join(BRATS_PATH, "BraTS2021_*")))

train_patients, test_patients = train_test_split(patient_folders, test_size=0.2, random_state=42)
val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=42)
train_dataset = BratsDataset3D(train_patients)
val_dataset = BratsDataset3D(val_patients)
test_dataset = BratsDataset3D(test_patients)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = CBAM_TransUNET_3D()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path="cbam_transunet_3d_weights.pth"):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}")

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

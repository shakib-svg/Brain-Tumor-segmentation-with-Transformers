import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob

# Classe Dataset pour BRATS 2021
class BratsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.patient_folders = sorted(glob(os.path.join(data_dir, "BraTS2021_*")))
    
    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_folder = self.patient_folders[idx]

        # Charger les modalités
        flair = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_flair.nii.gz")).get_fdata()
        t1 = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t1.nii.gz")).get_fdata()
        t1ce = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t1ce.nii.gz")).get_fdata()
        t2 = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t2.nii.gz")).get_fdata()

        # Charger le masque
        mask = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_seg.nii.gz")).get_fdata()

        # Normalisation des modalités
        flair = np.clip(flair, 0, np.percentile(flair, 99)) / np.percentile(flair, 99)
        t1 = np.clip(t1, 0, np.percentile(t1, 99)) / np.percentile(t1, 99)
        t1ce = np.clip(t1ce, 0, np.percentile(t1ce, 99)) / np.percentile(t1ce, 99)
        t2 = np.clip(t2, 0, np.percentile(t2, 99)) / np.percentile(t2, 99)

        # Binarisation du masque
        mask = (mask > 0).astype(np.float32)

        # Convertir en tenseurs PyTorch
        image = torch.tensor(np.stack([flair, t1, t1ce, t2], axis=0), dtype=torch.float32)  # 4 canaux
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # 1 canal

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Définir le CBAM-TransUNET
class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super(CBAMBlock, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x) * x
        sa = self.spatial_att(torch.cat([torch.max(ca, 1, keepdim=True)[0], torch.mean(ca, 1, keepdim=True)], dim=1)) * ca
        return sa

class CBAM_TransUNET(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(CBAM_TransUNET, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            CBAMBlock(64),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Calcul des métriques (Dice Score, IoU)
def dice_score(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def iou_score(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

# Entraînement du modèle
def train_model(model, loader, criterion, optimizer, num_epochs=10, save_path="cbam_transunet_weights.pth"):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        dice_scores = []
        iou_scores = []

        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calcul des métriques
            with torch.no_grad():
                dice = dice_score(outputs, masks)
                iou = iou_score(outputs, masks)
                dice_scores.append(dice)
                iou_scores.append(iou)

        avg_loss = epoch_loss / len(loader)
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

    # Sauvegarder les poids
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}")

# Initialisation
BRATS_PATH = "/chemin/vers/brats2021_Training_Data"
batch_size = 2
num_epochs = 10
learning_rate = 1e-4

# Chargement des données
train_dataset = BratsDataset(BRATS_PATH)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Configuration du modèle
model = CBAM_TransUNET()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lancer l'entraînement
train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

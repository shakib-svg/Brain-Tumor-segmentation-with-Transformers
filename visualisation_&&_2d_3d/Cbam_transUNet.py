import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob

# Classe Dataset pour BRATS 2021 (3D volumétrique)
class BratsDataset3D(Dataset):
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

# CBAM Block (3D)
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

# CBAM-TransUNET 3D
class CBAM_TransUNET_3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(CBAM_TransUNET_3D, self).__init__()

        # Initial Conv Block (3D)
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # CBAM Block (3D)
        self.cbam = CBAMBlock3D(64)

        # Decoder (UNet style, 3D)
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Initial Conv
        x = self.initial_conv(x)

        # CBAM Attention
        x = self.cbam(x)

        # Decoder
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
def train_model(model, loader, criterion, optimizer, num_epochs=10, save_path="cbam_transunet_3d_weights.pth"):
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
BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/BraTS2021_Training_Data"
batch_size = 1  # Taille réduite pour les volumes 3D
num_epochs = 10
learning_rate = 1e-4

# Chargement des données
train_dataset = BratsDataset3D(BRATS_PATH)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Configuration du modèle
model = CBAM_TransUNET_3D()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lancer l'entraînement
train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)









#%%
BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/train_datasetBraTS2021_Training_Data"  # Assurez-vous que ce chemin est correct

# Vérifier si les sous-dossiers sont bien trouvés
patient_folders = sorted(glob(os.path.join(BRATS_PATH, "BraTS2021_*")))

print(f"Nombre total de patients trouvés : {len(patient_folders)}")
if len(patient_folders) == 0:
    raise ValueError("🚨 Aucun dossier patient trouvé ! Vérifiez le chemin BRATS_PATH.")

# Afficher quelques exemples de dossiers trouvés
print("Exemples de dossiers patients :", patient_folders[:5])

#%%
class BratsDataset3D(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.patient_folders = sorted(glob(os.path.join(data_dir, "BraTS2021_*")))

        # Vérification des fichiers dans les 5 premiers dossiers
        for patient_folder in self.patient_folders[:5]:
            required_files = [
                f"{os.path.basename(patient_folder)}_flair.nii.gz",
                f"{os.path.basename(patient_folder)}_t1.nii.gz",
                f"{os.path.basename(patient_folder)}_t1ce.nii.gz",
                f"{os.path.basename(patient_folder)}_t2.nii.gz",
                f"{os.path.basename(patient_folder)}_seg.nii.gz"
            ]
            for file in required_files:
                file_path = os.path.join(patient_folder, file)
                if not os.path.exists(file_path):
                    print(f"⚠️ Fichier manquant : {file_path}")
        
        if len(self.patient_folders) == 0:
            raise ValueError("🚨 Aucun patient trouvé ! Vérifiez la structure du dataset.")

    def __len__(self):
        return len(self.patient_folders)


#%%
if len(train_dataset) == 0:
    raise ValueError("🚨 Le dataset est vide ! Vérifiez les fichiers et le chemin BRATS_PATH.")




















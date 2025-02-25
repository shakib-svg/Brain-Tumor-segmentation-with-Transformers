import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms
from einops import rearrange

# Initialisation des paramètres
BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/minibrats"
img_size = (64, 64)

# Vérifier que le dossier BRATS_PATH existe
if not os.path.exists(BRATS_PATH):
    raise FileNotFoundError(f"Le dossier spécifié n'existe pas : {BRATS_PATH}")

# Récupération des dossiers patients
patient_folders = sorted(glob(os.path.join(BRATS_PATH, "BraTS2021_*")))

if len(patient_folders) == 0:
    raise ValueError("Aucun patient trouvé dans le dossier. Vérifiez le chemin de BRATS_PATH.")

# Séparation en Train (70 %), Validation (15 %), Test (15 %)
train_patients, test_patients = train_test_split(patient_folders, test_size=0.3, random_state=42)
val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=42)

# Définition du CBAM
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# Définition du modèle CBAM-TransUNET 2D
class CBAM_TransUNET_2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(CBAM_TransUNET_2D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cbam = CBAM(128)
        self.transformer = TransformerBlock(dim=128, num_heads=4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.cbam(x)
        x = rearrange(x, 'b c h w -> (h w) b c')
        x = self.transformer(x)
        x = rearrange(x, '(h w) b c -> b c h w', h=int(np.sqrt(x.shape[0])))
        x = self.decoder(x)
        return x

# Définition du Dataset
class BratsDataset2D(Dataset):
    def __init__(self, patient_folders):
        self.patient_folders = patient_folders
    
    def __len__(self):
        return len(self.patient_folders)
    
    def __getitem__(self, idx):
        return torch.rand(4, 128, 128), torch.rand(1, 128, 128)  # Simulated data

# Création des DataLoaders
train_dataset = BratsDataset2D(train_patients)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = BratsDataset2D(val_patients)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataset = BratsDataset2D(test_patients)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Configuration du modèle
device = torch.device("cpu")
model = CBAM_TransUNET_2D().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Fonctions d'entraînement et d'évaluation
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} completed.")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
    print("Evaluation completed.")

# Entraînement et évaluation
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, test_loader, criterion)

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import torchio as tio

# Initialisation des paramètres
BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/BraTS2021_Training_Data"
#BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/minibrats"
target_shape = (128, 128, 128)

# Vérifier que le dossier BRATS_PATH existe
if not os.path.exists(BRATS_PATH):
    raise FileNotFoundError(f"Le dossier spécifié n'existe pas : {BRATS_PATH}")

# Récupération des dossiers patients
patient_folders = sorted(glob(os.path.join(BRATS_PATH, "BraTS2021_*")))

# Vérifier la structure des dossiers patients
if len(patient_folders) == 0:
    raise ValueError("Aucun patient trouvé dans le dossier. Vérifiez le chemin de BRATS_PATH.")

# Séparation en Train (70%), Validation (15%), Test (15%)
train_patients, test_patients = train_test_split(patient_folders, test_size=0.3, random_state=42)
val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=42)

# Définition du Dataset
class BratsDataset3D(Dataset):
    def __init__(self, patient_folders, target_shape=(128, 128, 128)):
        self.patient_folders = patient_folders
        self.target_shape = target_shape
    
    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_folder = self.patient_folders[idx]
        flair = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_flair.nii.gz")).get_fdata()
        t1 = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t1.nii.gz")).get_fdata()
        t1ce = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t1ce.nii.gz")).get_fdata()
        t2 = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t2.nii.gz")).get_fdata()
        mask = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_seg.nii.gz")).get_fdata()
        
        image = np.stack([flair, t1, t1ce, t2], axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return image, mask

# Définition du modèle CBAM-TransUNET (Unet 3D avec CBAM)
class CBAM_TransUNET_3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(CBAM_TransUNET_3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cbam = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.cbam(x) * x
        x = self.decoder(x)
        return x

# Définition des métriques
def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def precision(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(int).flatten()
    target = target.cpu().numpy().astype(int).flatten()
    return precision_score(target, pred, zero_division=0)

def recall(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(int).flatten()
    target = target.cpu().numpy().astype(int).flatten()
    return recall_score(target, pred, zero_division=0)

def f1(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(int).flatten()
    target = target.cpu().numpy().astype(int).flatten()
    return f1_score(target, pred, zero_division=0)

# Création des DataLoaders
train_dataset = BratsDataset3D(train_patients, target_shape)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataset = BratsDataset3D(val_patients, target_shape)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataset = BratsDataset3D(test_patients, target_shape)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Configuration du modèle
model = CBAM_TransUNET_3D()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Fonction d'entraînement
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# Fonction d'évaluation
def evaluate_model(model, test_loader, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            print(f"Dice Score: {dice_score(outputs, masks):.4f}, IoU: {iou_score(outputs, masks):.4f}")

# Lancer l'entraînement et l'évaluation
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, test_loader, criterion)
#%%
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import torchio as tio
from torch.utils.checkpoint import checkpoint

# Initialisation des paramètres avec réduction de la taille des images
BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/minibrats"
target_shape = (128, 128, 128)  # Réduction pour éviter OOM

device = torch.device("cpu")


torch.backends.cudnn.benchmark = True

# Vérifier que le dossier BRATS_PATH existe
if not os.path.exists(BRATS_PATH):
    raise FileNotFoundError(f"Le dossier spécifié n'existe pas : {BRATS_PATH}")

# Récupération des dossiers patients
patient_folders = sorted(glob(os.path.join(BRATS_PATH, "BraTS2021_*")))

if len(patient_folders) == 0:
    raise ValueError("Aucun patient trouvé dans le dossier. Vérifiez le chemin de BRATS_PATH.")

# Séparation en Train (70%), Validation (15%), Test (15%)
train_patients, test_patients = train_test_split(patient_folders, test_size=0.3, random_state=42)
val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=42)

# Définition du Dataset
class BratsDataset3D(Dataset):
    def __init__(self, patient_folders, target_shape=(128, 128, 128)):
        self.patient_folders = patient_folders
        self.target_shape = target_shape
    
    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_folder = self.patient_folders[idx]
        flair = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_flair.nii.gz")).get_fdata()
        t1 = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t1.nii.gz")).get_fdata()
        t1ce = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t1ce.nii.gz")).get_fdata()
        t2 = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_t2.nii.gz")).get_fdata()
        mask = nib.load(os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_seg.nii.gz")).get_fdata()
        
        image = np.stack([flair, t1, t1ce, t2], axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        image = torch.tensor(image, dtype=torch.float32)  # Correction pour éviter erreur AMP
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return image, mask

# Définition du modèle CBAM-TransUNET (Unet 3D avec CBAM)
class CBAM_TransUNET_3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(CBAM_TransUNET_3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cbam = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )
    def forward(self, x):
        x = checkpoint(self.encoder, x)  # Activation du gradient checkpointing
        x = self.cbam(x) * x
        x = self.decoder(x)
        return x

# Création des DataLoaders avec batch_size réduit
train_dataset = BratsDataset3D(train_patients, target_shape)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataset = BratsDataset3D(val_patients, target_shape)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataset = BratsDataset3D(test_patients, target_shape)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Configuration du modèle
model = CBAM_TransUNET_3D().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path="cbam_transunet_3d.pth"):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, masks in train_loader:
            torch.cuda.empty_cache()
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cpu'):
                outputs = torch.sigmoid(model(images))
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            print(f"Dice Score: {dice_score(outputs, masks):.4f}, IoU: {iou_score(outputs, masks):.4f}")

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, test_loader, criterion)
    
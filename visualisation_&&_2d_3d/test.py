#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 23:20:42 2025

@author: shakib
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from dataset_loader import BrainTumorDataset  # Assurez-vous d'avoir un DataLoader adapté
from model import MySegmentationModel  # Importer le bon modèle utilisé

# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/shakib/Desktop/both/Transformer/brats/brats_segmentation_model.pth"
model = MySegmentationModel()  # Remplace par ton architecture
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Charger le dataset
data_dir = '/home/shakib/Desktop/both/Transformer/brats/minibrats'
dataset = BrainTumorDataset(data_dir)  # Assurez-vous que le dataset est bien défini
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Fonction pour calculer les métriques
def compute_metrics(pred, target):
    pred = pred > 0.5  # Seuil de segmentation binaire
    target = target > 0.5
    
    TP = (pred & target).sum()
    FP = (pred & ~target).sum()
    FN = (~pred & target).sum()
    TN = (~pred & ~target).sum()
    
    intersection = TP
    union = (pred | target).sum()
    
    iou = intersection / (union + 1e-6)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)  # Précision = TP / (TP + FP)
    
    return iou, dice, precision

# Boucle sur les patients et évaluation
iou_scores = []
dice_scores = []
precision_scores = []

for i, (image, mask) in enumerate(dataloader):
    image, mask = image.to(device), mask.to(device)

    # Prédiction du modèle
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)  # Activation pour segmentation binaire

    # Convertir en numpy
    image_np = image.cpu().squeeze().numpy()
    mask_np = mask.cpu().squeeze().numpy()
    pred_np = output.cpu().squeeze().numpy()

    # Calculer les métriques
    iou, dice, precision = compute_metrics(pred_np, mask_np)
    iou_scores.append(iou)
    dice_scores.append(dice)
    precision_scores.append(precision)

    # Afficher les résultats
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_np, cmap="gray")
    axs[0].set_title("Image originale")

    axs[1].imshow(mask_np, cmap="jet", alpha=0.5)
    axs[1].set_title("Masque réel")

    axs[2].imshow(pred_np, cmap="jet", alpha=0.5)
    axs[2].set_title("Prédiction du modèle")

    plt.show()

# Affichage des métriques moyennes
print(f"Score IoU moyen: {np.mean(iou_scores):.4f}")
print(f"Score Dice moyen: {np.mean(dice_scores):.4f}")
print(f"Précision moyenne: {np.mean(precision_scores):.4f}")

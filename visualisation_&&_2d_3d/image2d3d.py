import os
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob

# 🔹 Chemin du dataset BRATS 2021
BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/BraTS2021_Training_Data"

# 🔹 Vérifier que le dataset existe
if not os.path.exists(BRATS_PATH):
    raise FileNotFoundError(f"Le dossier {BRATS_PATH} n'existe pas ! Vérifie le chemin.")

# 🔹 Récupérer la liste des dossiers patients
patient_folders = sorted(glob(os.path.join(BRATS_PATH, "BraTS2021_*")))

if len(patient_folders) == 0:
    raise ValueError("Aucun patient trouvé dans le dossier. Vérifie ton dataset.")

# 🔹 Sélectionner un patient aléatoire
random_patient = random.choice(patient_folders)
print(f"Patient sélectionné : {os.path.basename(random_patient)}")

# 🔹 Chemins des fichiers IRM et segmentation
flair_path = os.path.join(random_patient, f"{os.path.basename(random_patient)}_t1.nii.gz")
mask_path = os.path.join(random_patient, f"{os.path.basename(random_patient)}_seg.nii.gz")

# 🔹 Charger les images FLAIR et le masque
flair_img = nib.load(flair_path).get_fdata()
mask_img = nib.load(mask_path).get_fdata()

# 🔹 Sélectionner une tranche médiane
slice_idx = flair_img.shape[2] // 2
flair_slice = flair_img[:, :, slice_idx]
mask_slice = mask_img[:, :, slice_idx]

# 🔹 Afficher les images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 🔹 Image IRM FLAIR seule
ax[0].imshow(flair_slice, cmap="gray")
ax[0].set_title("IRM FLAIR (Coupe 2D)")
ax[0].axis("off")

# 🔹 Image IRM avec superposition du masque
ax[1].imshow(flair_slice, cmap="gray")
ax[1].imshow(mask_slice, cmap="jet", alpha=0.5)  # Superposer le masque en transparence
ax[1].set_title("IRM + Masque de Segmentation")
ax[1].axis("off")

# 🔹 Sauvegarde de l'image
plt.tight_layout()
image_path = "brats_patient_sample.png"
plt.savefig(image_path, dpi=300)
plt.show()

print(f"✅ Image sauvegardée sous : {image_path}")
#%%

import os
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob

# 🔹 Chemin du dataset BRATS 2021
BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/BraTS2021_Training_Data"

# 🔹 Vérifier que le dataset existe
if not os.path.exists(BRATS_PATH):
    raise FileNotFoundError(f"Le dossier {BRATS_PATH} n'existe pas ! Vérifie le chemin.")

# 🔹 Récupérer la liste des dossiers patients
patient_folders = sorted(glob(os.path.join(BRATS_PATH, "BraTS2021_*")))

if len(patient_folders) == 0:
    raise ValueError("Aucun patient trouvé dans le dossier. Vérifie ton dataset.")

# 🔹 Sélectionner un patient aléatoire
random_patient = random.choice(patient_folders)
print(f"Patient sélectionné : {os.path.basename(random_patient)}")

# 🔹 Chemins des fichiers IRM et segmentation
flair_path = os.path.join(random_patient, f"{os.path.basename(random_patient)}_t1.nii.gz")
mask_path = os.path.join(random_patient, f"{os.path.basename(random_patient)}_seg.nii.gz")

# 🔹 Charger les images FLAIR et le masque
flair_img = nib.load(flair_path).get_fdata()
mask_img = nib.load(mask_path).get_fdata()

# 🔹 Dimensions du volume 3D
depth = flair_img.shape[2]

# 🔹 Sélectionner 9 tranches régulièrement espacées
num_slices = 9
slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

# 🔹 Afficher les images sur 3x3 sous-plots
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for i, ax in enumerate(axes.flat):
    slice_idx = slice_indices[i]
    
    # 🔹 Afficher l'IRM avec le masque superposé
    ax.imshow(flair_img[:, :, slice_idx], cmap="gray")
    ax.imshow(mask_img[:, :, slice_idx], cmap="jet", alpha=0.5)  # Superposer le masque
    ax.set_title(f"Coupe Z = {slice_idx}")
    ax.axis("off")

# 🔹 Sauvegarde et affichage
plt.tight_layout()
image_path = "brats_3d_slices.png"
plt.savefig(image_path, dpi=300)
plt.show()

print(f"✅ Image sauvegardée sous : {image_path}")
#%% code affichage dans un browser
import os
import random
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import plotly.io as pio
from glob import glob

# 🔹 Définir le renderer de Plotly
pio.renderers.default = "browser"  # "notebook" si tu es dans Jupyter

# 🔹 Chemin du dataset BRATS 2021
BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/BraTS2021_Training_Data"

# 🔹 Vérifier que le dataset existe
if not os.path.exists(BRATS_PATH):
    raise FileNotFoundError(f"Le dossier {BRATS_PATH} n'existe pas ! Vérifie le chemin.")

# 🔹 Récupérer la liste des dossiers patients
patient_folders = sorted(glob(os.path.join(BRATS_PATH, "BraTS2021_*")))

if len(patient_folders) == 0:
    raise ValueError("Aucun patient trouvé dans le dossier. Vérifie ton dataset.")

# 🔹 Sélectionner un patient aléatoire
random_patient = random.choice(patient_folders)
print(f"Patient sélectionné : {os.path.basename(random_patient)}")

# 🔹 Chemins des fichiers IRM et segmentation
flair_path = os.path.join(random_patient, f"{os.path.basename(random_patient)}_flair.nii.gz")
mask_path = os.path.join(random_patient, f"{os.path.basename(random_patient)}_seg.nii.gz")

# 🔹 Charger les images FLAIR et le masque
flair_img = nib.load(flair_path).get_fdata()
mask_img = nib.load(mask_path).get_fdata()

# 🔹 Normaliser l'IRM pour l'affichage
flair_img = (flair_img - np.min(flair_img)) / (np.max(flair_img) - np.min(flair_img))

# 🔹 Extraire les coordonnées des voxels tumoraux
x, y, z = np.where(mask_img > 0)

# 🔹 Création de la visualisation 3D avec Plotly
fig = go.Figure()

# 🔹 Ajouter les points IRM (seulement un échantillon pour performance)
num_points = 50000  # Limiter le nombre de points pour l'affichage
indices = np.random.choice(len(x), min(len(x), num_points), replace=False)

fig.add_trace(go.Scatter3d(
    x=x[indices], y=y[indices], z=z[indices],
    mode='markers',
    marker=dict(
        size=3,
        color=z[indices],  # Dégradé de couleurs basé sur l'axe Z
        colorscale='Jet', 
        opacity=0.5
    ),
    name="Tumeur (Segmentation)"
))

# 🔹 Paramètres de mise en page
fig.update_layout(
    title=f"Visualisation 3D du Patient {os.path.basename(random_patient)}",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

# 🔹 Essayer plusieurs méthodes d'affichage
try:
    fig.show()
except:
    print("⚠️ `fig.show()` ne fonctionne pas, on enregistre l'image en HTML.")
    fig.write_html("visualisation_3d.html")
    print("✅ Visualisation enregistrée sous 'visualisation_3d.html'. Ouvre ce fichier dans ton navigateur.")


#%%spyder
import os
import random
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import plotly.io as pio
from glob import glob
from IPython.display import display

# 🔹 Activer un renderer compatible avec Spyder
pio.renderers.default = "iframe_connected"

# 🔹 Chemin du dataset BRATS 2021
BRATS_PATH = "/home/shakib/Desktop/both/Transformer/brats/BraTS2021_Training_Data"

# 🔹 Vérifier que le dataset existe
if not os.path.exists(BRATS_PATH):
    raise FileNotFoundError(f"Le dossier {BRATS_PATH} n'existe pas ! Vérifie le chemin.")

# 🔹 Récupérer la liste des dossiers patients
patient_folders = sorted(glob(os.path.join(BRATS_PATH, "BraTS2021_*")))

if len(patient_folders) == 0:
    raise ValueError("Aucun patient trouvé dans le dossier. Vérifie ton dataset.")

# 🔹 Sélectionner un patient aléatoire
random_patient = random.choice(patient_folders)
print(f"Patient sélectionné : {os.path.basename(random_patient)}")

# 🔹 Chemins des fichiers IRM et segmentation
flair_path = os.path.join(random_patient, f"{os.path.basename(random_patient)}_flair.nii.gz")
mask_path = os.path.join(random_patient, f"{os.path.basename(random_patient)}_seg.nii.gz")

# 🔹 Charger les images FLAIR et le masque
flair_img = nib.load(flair_path).get_fdata()
mask_img = nib.load(mask_path).get_fdata()

# 🔹 Normaliser l'IRM pour l'affichage
flair_img = (flair_img - np.min(flair_img)) / (np.max(flair_img) - np.min(flair_img))

# 🔹 Extraire les coordonnées des voxels tumoraux
x, y, z = np.where(mask_img > 0)

# 🔹 Création de la visualisation 3D avec Plotly
fig = go.Figure()

# 🔹 Ajouter les points IRM (seulement un échantillon pour performance)
num_points = 50000  # Limiter le nombre de points pour l'affichage
indices = np.random.choice(len(x), min(len(x), num_points), replace=False)

fig.add_trace(go.Scatter3d(
    x=x[indices], y=y[indices], z=z[indices],
    mode='markers',
    marker=dict(
        size=3,
        color=z[indices],  # Dégradé de couleurs basé sur l'axe Z
        colorscale='Jet', 
        opacity=0.5
    ),
    name="Tumeur (Segmentation)"
))

# 🔹 Paramètres de mise en page
fig.update_layout(
    title=f"Visualisation 3D du Patient {os.path.basename(random_patient)}",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

# 🔹 Méthodes pour afficher dans Spyder
try:
    fig.show()  # Essayer l'affichage normal
except:
    print("⚠️ `fig.show()` ne fonctionne pas, utilisation de `display(fig)`")
    display(fig)  # Alternative pour Spyder

# 🔹 Alternative : Sauvegarder et ouvrir un fichier HTML si nécessaire
fig.write_html("visualisation_3d.html")
print("✅ Ouvre le fichier 'visualisation_3d.html' dans ton navigateur si l'affichage ne fonctionne pas.")


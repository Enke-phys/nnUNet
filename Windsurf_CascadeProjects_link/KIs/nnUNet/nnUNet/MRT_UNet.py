import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

# 1. Bild und Label laden
img = nib.load("image.nii.gz").get_fdata()
lbl = nib.load("label.nii.gz").get_fdata()

# 2. Normalisieren und Tensor bauen
img = (img - np.mean(img)) / np.std(img)
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
lbl = torch.tensor(lbl, dtype=torch.long).unsqueeze(0)  # (1, D, H, W)

# 3. Einfaches 3D-UNet-Modell
class Simple3DUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(4, 8, 3, padding=1),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Conv3d(8, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(4, 3, 1)  # 3 Klassen (z. B. Hintergrund, Muskel, Fett)
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

# 4. Modell, Loss und Optimizer
model = Simple3DUNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Ein Trainingsschritt
output = model(img)  # (1, 3, D, H, W)
loss = criterion(output, lbl)
loss.backward()
optimizer.step()

print("Trainingsverlust:", loss.item())

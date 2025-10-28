from test2 import criterion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#Schritt 1: Bilder vorbereiten
transform = transforms.Compose([
    transforms.Grayscale(),           # In Graustufen umwandeln
    transforms.Resize((128, 128)),    # Einheitliche Größe
    transforms.ToTensor(),            # In Tensor umwandeln
])

#Schritt 2: Dataset integrieren
dataset = datasets.ImageFolder("/Users/enke/Documents/Brain_Cancer", transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#Schritt 3: Einfaches neuronales Netz definieren
class BrainTumorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*128, 64)
        self.fc2 = nn.Linear(64, 3)  # Drei Klassen: Glioma, Menin, Tumor

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = BrainTumorNet()

#Schritt 4: Verlustfunktion und Optimierer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#Schritt 5: Training
epochs = 10
for epoch in range(epochs):
    for batch in train_loader:
        images, labels = batch
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
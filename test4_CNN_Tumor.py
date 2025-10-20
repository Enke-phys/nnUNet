import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 1. CNN-Modell
class TumorCNN(nn.Module):
    def __init__(self):
        super(TumorCNN, self).__init__()
        #Convolution Layer 1, 1=1 Farbkanal, 16=16 neue Bildversionen (Kanten, Helligkeit, Muster), kernel_size 3= bei jedem falten 3x3 Pixel, padding 1= Bildgröße beibehalten)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        #Merkt sich nur wichtigste infos aus kernels und verkleinert die Bildgröße
        self.pool = nn.MaxPool2d(2, 2)
        #Convolution Layer 2, 16 Bildmerkmale als Input -> 32 neue Bildmerkmale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Bildgröße: 224 -> 112 -> 56 -> Output: [32, 56, 56]
        #Fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)  # 3 Klassen (logits)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [B, 16, 112, 112]
        x = self.pool(torch.relu(self.conv2(x)))  # [B, 32, 56, 56]
        x = x.view(-1, 32 * 56 * 56)              # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Kein Softmax – wird im Loss berechnet

# 2. Bildtransformationen
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Falls Bilder RGB sind
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3. Datensatz laden (z. B. data/glioma/, data/meningioma/, data/pituitary/)
full_dataset = datasets.ImageFolder(root="/Users/enke/Documents/Brain_Cancer", transform=transform)

# 4. Train/Validation-Split (z. B. 80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)  # Reproduzierbar
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

# 5. Dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 6. Setup: Gerät, Modell, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TumorCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Trainings- und Validierungsschleife
for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    # Validierung
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Validation Acc: {acc:.2f}%")
# import os
# import yaml
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms
# from PIL import Image, UnidentifiedImageError
# import kaggle

# os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath(".")

# # -----------------------------
# # Load config
# # -----------------------------
# with open("config.yaml", "r") as f:
#     CONFIG = yaml.safe_load(f)


# # -----------------------------
# # Download Kaggle dataset
# # -----------------------------
# def download_kaggle_dataset():
#     dataset = "mahekmorzaria/face-emotion"
#     data_dir = "data"
#     output_path = os.path.join(data_dir, "downloaded_faces")

#     if not os.path.exists(output_path):
#         os.makedirs(data_dir, exist_ok=True)
#         print("⬇️ Downloading dataset from Kaggle...")
#         kaggle.api.dataset_download_files(dataset, path=data_dir, unzip=True)
#         print("✅ Dataset downloaded and extracted")
#     else:
#         print("✅ Dataset already exists, skipping download")

#     return output_path


# # -----------------------------
# # Safe ImageFolder to skip corrupt images
# # -----------------------------
# class SafeImageFolder(datasets.ImageFolder):
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         try:
#             sample = self.loader(path)
#         except (UnidentifiedImageError, OSError):
#             # If unreadable, return dummy tensor with label -1
#             sample = Image.new("RGB", (224, 224))
#             target = -1
#         if self.transform is not None:
#             sample = self.transform(sample)
#         return sample, target


# # -----------------------------
# # Simple CNN Model
# # -----------------------------
# class Net(nn.Module):
#     def __init__(self, num_classes):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 56 * 56, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# # -----------------------------
# # Helpers for FL
# # -----------------------------
# def get_weights(model):
#     return [val.cpu().numpy() for _, val in model.state_dict().items()]

# def set_weights(model, weights):
#     state_dict = model.state_dict()
#     for k, (key, _) in enumerate(state_dict.items()):
#         state_dict[key] = torch.tensor(weights[k])
#     model.load_state_dict(state_dict)


# # -----------------------------
# # Training function
# # -----------------------------
# def train(model, trainloader, epochs=1):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=CONFIG["training"]["lr"])
#     criterion = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for images, labels in trainloader:
#             # Skip invalid (-1) labels
#             mask = labels != -1
#             if mask.sum() == 0:
#                 continue
#             images, labels = images[mask], labels[mask]

#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")
#     return model


# # -----------------------------
# # Evaluation
# # -----------------------------
# def test(model, loader):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()

#     model.eval()
#     correct, total, loss_total = 0, 0, 0.0
#     with torch.no_grad():
#         for images, labels in loader:
#             # Skip invalid (-1) labels
#             mask = labels != -1
#             if mask.sum() == 0:
#                 continue
#             images, labels = images[mask], labels[mask]

#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss_total += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)

#     return loss_total / len(loader), correct / total


# # -----------------------------
# # Dataloaders + class detection
# # -----------------------------
# def get_dataloaders():
#     dataset_path = download_kaggle_dataset()

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     full_dataset = SafeImageFolder(root=dataset_path, transform=transform)

#     # ✅ Auto-detect number of classes
#     num_classes = len(full_dataset.classes)
#     print(f"✅ Detected {num_classes} emotion classes: {full_dataset.classes}")

#     train_size = int(CONFIG["data"]["train_split"] * len(full_dataset))
#     val_size = int(CONFIG["data"]["val_split"] * len(full_dataset))
#     test_size = len(full_dataset) - train_size - val_size

#     train_dataset, val_dataset, test_dataset = random_split(
#         full_dataset, [train_size, val_size, test_size]
#     )

#     train_loader = DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=False)

#     return train_loader, val_loader, test_loader, num_classes


import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image, UnidentifiedImageError
import kaggle

os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath(".")

# -----------------------------
# Load config
# -----------------------------
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


# -----------------------------
# Download Kaggle dataset
# -----------------------------
def download_kaggle_dataset():
    dataset = "mahekmorzaria/face-emotion"
    data_dir = "data"
    output_path = os.path.join(data_dir, "downloaded_faces")

    if not os.path.exists(output_path):
        os.makedirs(data_dir, exist_ok=True)
        print("⬇️ Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(dataset, path=data_dir, unzip=True)
        print("✅ Dataset downloaded and extracted")
    else:
        print("✅ Dataset already exists, skipping download")

    return output_path


# -----------------------------
# Safe ImageFolder to skip corrupt images
# -----------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError):
            # If unreadable, return dummy tensor with label -1
            sample = Image.new("RGB", (224, 224))
            target = -1
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


# -----------------------------
# Simple CNN Model
# -----------------------------
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Helpers for FL
# -----------------------------
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    state_dict = model.state_dict()
    for k, (key, _) in enumerate(state_dict.items()):
        state_dict[key] = torch.tensor(weights[k])
    model.load_state_dict(state_dict)


# -----------------------------
# Training function
# -----------------------------
def train(model, trainloader, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["training"]["lr"])
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            # Skip invalid (-1) labels
            mask = labels != -1
            if mask.sum() == 0:
                continue
            images, labels = images[mask], labels[mask]

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")
    return model


# -----------------------------
# Evaluation
# -----------------------------
def test(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    correct, total, loss_total = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            # Skip invalid (-1) labels
            mask = labels != -1
            if mask.sum() == 0:
                continue
            images, labels = images[mask], labels[mask]

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return loss_total / len(loader), correct / total


# -----------------------------
# Dataloaders + class detection
# -----------------------------
def get_dataloaders():
    dataset_path = download_kaggle_dataset()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = SafeImageFolder(root=dataset_path, transform=transform)

    # ✅ Auto-detect number of classes
    num_classes = len(full_dataset.classes)
    print(f"✅ Detected {num_classes} emotion classes: {full_dataset.classes}")

    train_size = int(CONFIG["data"]["train_split"] * len(full_dataset))
    val_size = int(CONFIG["data"]["val_split"] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader, num_classes


# -----------------------------
# Save & Load Model Helpers
# -----------------------------
def save_model(model, path="saved_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")

def load_model(path, num_classes):
    model = Net(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    print(f"✅ Model loaded from {path}")
    return model

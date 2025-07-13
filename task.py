import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import requests
from io import BytesIO
import os
import yaml

# Load config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Label encoder (string to int)
label_to_index = {}
index_to_label = {}

class URLDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.data = self.load_json(json_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((CONFIG["training"]["image_size"], CONFIG["training"]["image_size"])),
            transforms.ToTensor()
        ])
        self.build_label_index()

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def build_label_index(self):
        labels = sorted(set(d[CONFIG["data"]["label_column"]] for d in self.data))
        global label_to_index, index_to_label
        label_to_index = {label: i for i, label in enumerate(labels)}
        index_to_label = {i: label for label, i in label_to_index.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        url = entry[CONFIG["data"]["url_column"]]
        label = label_to_index[entry[CONFIG["data"]["label_column"]]]

        try:
            response = requests.get(url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return self.transform(image), label
        except:
            # Return dummy tensor and invalid label if image fails
            return torch.zeros(3, CONFIG["training"]["image_size"], CONFIG["training"]["image_size"]), -1

def get_dataloaders():
    train_dataset = URLDataset(CONFIG["data"]["train_file"])
    val_dataset = URLDataset(CONFIG["data"]["val_file"])
    return (
        DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True, num_workers=CONFIG["training"]["num_workers"]),
        DataLoader(val_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=False, num_workers=CONFIG["training"]["num_workers"]),
    )

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 53 * 53, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), weights):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=False)

def train(model, trainloader, epochs=1):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for images, labels in trainloader:
            mask = labels != -1
            if mask.sum() == 0: continue
            images, labels = images[mask], labels[mask]
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
    return model

def test(model, valloader):
    model.eval()
    correct, total, loss_total = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in valloader:
            mask = labels != -1
            if mask.sum() == 0: continue
            images, labels = images[mask], labels[mask]
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_total += loss.item()
    return loss_total / len(valloader), correct / total

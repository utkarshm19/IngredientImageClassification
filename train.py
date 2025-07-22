import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms


def get_loaders(data_dir: Path, batch_size: int = 32):
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    val_ds = torchvision.datasets.ImageFolder(val_dir, transform=transform)
    test_ds = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes: int):
    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def train(args):
    data_dir = Path(args.data)
    train_loader, val_loader, test_loader, classes = get_loaders(data_dir, args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        train_acc = accuracy(model, train_loader, device)
        val_acc = accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    test_acc = accuracy(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.3f}")

    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'classes': classes}, save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ingredient classifier')
    parser.add_argument('--data', required=True, help='Path to dataset split directory')
    parser.add_argument('--output', default='model.pt', help='Where to save the model file')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)

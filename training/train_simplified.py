#!/usr/bin/env python3
"""
Train a simplified species classifier focused on deer detection.

Categories:
- Deer (priority)
- Empty
- Turkey
- Other Mammal (Raccoon, Rabbit, Squirrel, Coyote, Bobcat, Opossum)
- Other (Person, Vehicle, Quail, misc)
"""
import os
import sys
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import models, transforms
except ImportError:
    print("Missing deps. Install: pip install torch torchvision pillow numpy")
    sys.exit(1)

# Category mapping
CATEGORY_MAP = {
    "Deer": "Deer",
    "Empty": "Empty",
    "Turkey": "Turkey",
    # Other Mammal
    "Raccoon": "Other Mammal",
    "Rabbit": "Other Mammal",
    "Squirrel": "Other Mammal",
    "Coyote": "Other Mammal",
    "Bobcat": "Other Mammal",
    "Opossum": "Other Mammal",
    # Other
    "Person": "Other",
    "Vehicle": "Other",
    "Quail": "Other",
    "Other": "Other",
    "Other Bird": "Other",
}

LABELS = ["Deer", "Empty", "Other", "Other Mammal", "Turkey"]


def load_data(db_path: str) -> List[Tuple[str, str]]:
    """Load photos and map to simplified categories."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT p.file_path, t.tag_name
        FROM photos p
        JOIN tags t ON t.photo_id = p.id
        WHERE p.file_path != '' AND t.tag_name != ''
        AND t.tag_name NOT IN ('Buck', 'Doe')
    """)

    data = []
    for row in cur.fetchall():
        path = row["file_path"]
        tag = row["tag_name"].strip()
        if not tag or not os.path.exists(path):
            continue
        # Map to simplified category
        category = CATEGORY_MAP.get(tag, "Other")
        data.append((path, category))

    conn.close()
    return data


class SimpleDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], img_size: int = 224, augment: bool = True):
        self.items = items
        self.img_size = img_size
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label_idx = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img, label_idx
        except Exception:
            # Return a blank image on error
            return torch.zeros(3, self.img_size, self.img_size), label_idx


def train_model(db_path: str, output_path: str, epochs: int = 15, batch_size: int = 32):
    print(f"[train] Loading data from {db_path}")
    raw_data = load_data(db_path)

    # Convert to label indices
    label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
    data = [(path, label_to_idx[cat]) for path, cat in raw_data]

    # Print distribution
    print(f"\n[train] Category distribution:")
    for lbl in LABELS:
        count = sum(1 for _, cat in raw_data if cat == lbl)
        print(f"  {lbl}: {count}")
    print(f"  Total: {len(data)}")

    # Split train/val (90/10)
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    split = int(0.9 * len(data))
    train_idx, val_idx = indices[:split], indices[split:]
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

    print(f"\n[train] Train: {len(train_data)}, Val: {len(val_data)}")

    # Compute class weights for balanced sampling
    label_counts = [0] * len(LABELS)
    for _, lbl_idx in train_data:
        label_counts[lbl_idx] += 1

    total = sum(label_counts)
    class_weights = torch.tensor([total / (len(LABELS) * c) if c > 0 else 0 for c in label_counts], dtype=torch.float32)
    sample_weights = [class_weights[lbl_idx].item() for _, lbl_idx in train_data]

    # Create datasets and loaders
    train_ds = SimpleDataset(train_data, augment=True)
    val_ds = SimpleDataset(val_data, augment=False)

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n[train] Using device: {device}")

    # Create model - ResNet18 with custom head
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze early layers, train last block + head
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam([
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")], "lr": 1e-4}
    ])

    # Training loop
    best_val_loss = float("inf")
    best_state = None

    print(f"\n[train] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_count += imgs.size(0)

        train_loss /= train_count

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_count += imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss /= val_count
        val_acc = correct / val_count * 100

        print(f"[train] Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Export to ONNX
    print(f"\n[train] Exporting to ONNX: {output_path}")
    model = model.to("cpu")
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )

    # Write labels file
    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w") as f:
        for lbl in LABELS:
            f.write(lbl + "\n")

    print(f"[train] Labels saved to: {labels_path}")
    print(f"\n[train] Done! Model saved to {output_path}")

    return model


if __name__ == "__main__":
    db_path = os.path.expanduser("~/.trailcam/trailcam.db")
    output_path = "models/species.onnx"
    train_model(db_path, output_path, epochs=15)

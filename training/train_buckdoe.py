#!/usr/bin/env python3
"""
Train a buck/doe classifier with strong class balancing.

Uses photos tagged as Buck or Doe directly from the database.
Applies heavy class weighting and oversampling to handle imbalance.
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

LABELS = ["buck", "doe"]


def load_data(db_path: str) -> List[Tuple[str, int]]:
    """Load photos tagged as Buck or Doe."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT p.file_path, LOWER(t.tag_name) as sex
        FROM photos p
        JOIN tags t ON t.photo_id = p.id
        WHERE LOWER(t.tag_name) IN ('buck', 'doe')
        AND p.file_path IS NOT NULL
        AND p.file_path != ''
    """)

    data = []
    for row in cur.fetchall():
        path = row["file_path"]
        sex = row["sex"]
        if not os.path.exists(path):
            continue
        label_idx = 0 if sex == "buck" else 1
        data.append((path, label_idx))

    conn.close()
    return data


class BuckDoeDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], img_size: int = 224, augment: bool = True):
        self.items = items
        self.img_size = img_size
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
                transforms.RandomGrayscale(p=0.1),
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
            return torch.zeros(3, self.img_size, self.img_size), label_idx


def train_model(db_path: str, output_path: str, epochs: int = 20, batch_size: int = 32):
    print(f"[buckdoe] Loading data from {db_path}")
    data = load_data(db_path)

    # Count classes
    buck_count = sum(1 for _, lbl in data if lbl == 0)
    doe_count = sum(1 for _, lbl in data if lbl == 1)

    print(f"\n[buckdoe] Class distribution:")
    print(f"  Buck: {buck_count}")
    print(f"  Doe: {doe_count}")
    print(f"  Total: {len(data)}")
    print(f"  Imbalance ratio: {buck_count/doe_count:.1f}:1")

    # Split train/val (85/15)
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    split = int(0.85 * len(data))
    train_idx, val_idx = indices[:split], indices[split:]
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

    print(f"\n[buckdoe] Train: {len(train_data)}, Val: {len(val_data)}")

    # Compute STRONG class weights to handle imbalance
    # Use inverse frequency with extra boost for minority class
    train_buck = sum(1 for _, lbl in train_data if lbl == 0)
    train_doe = sum(1 for _, lbl in train_data if lbl == 1)

    # Extra strong weighting for minority class
    total = train_buck + train_doe
    buck_weight = total / (2 * train_buck) if train_buck > 0 else 1.0
    doe_weight = (total / (2 * train_doe)) * 1.5 if train_doe > 0 else 1.0  # 1.5x boost for minority

    class_weights = torch.tensor([buck_weight, doe_weight], dtype=torch.float32)
    print(f"\n[buckdoe] Class weights: Buck={buck_weight:.2f}, Doe={doe_weight:.2f}")

    # Sample weights for WeightedRandomSampler - oversample minority class
    sample_weights = [doe_weight * 2 if lbl == 1 else buck_weight for _, lbl in train_data]

    # Create datasets
    train_ds = BuckDoeDataset(train_data, augment=True)
    val_ds = BuckDoeDataset(val_data, augment=False)

    # Use weighted sampler to oversample minority class
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights) * 2, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n[buckdoe] Using device: {device}")

    # Create model - ResNet18 with unfrozen last layers
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Unfreeze more layers for better learning
    for name, param in model.named_parameters():
        if not (name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Loss with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW([
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")], "lr": 5e-5}
    ], weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0.0
    best_state = None

    print(f"\n[buckdoe] Starting training for {epochs} epochs...")
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
        scheduler.step()

        # Validation with per-class accuracy
        model.eval()
        val_loss = 0.0
        val_count = 0
        buck_correct, buck_total = 0, 0
        doe_correct, doe_total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_count += imgs.size(0)
                preds = outputs.argmax(dim=1)

                # Per-class accuracy
                for pred, lbl in zip(preds, labels):
                    if lbl == 0:  # Buck
                        buck_total += 1
                        if pred == 0:
                            buck_correct += 1
                    else:  # Doe
                        doe_total += 1
                        if pred == 1:
                            doe_correct += 1

        val_loss /= val_count
        buck_acc = buck_correct / buck_total * 100 if buck_total > 0 else 0
        doe_acc = doe_correct / doe_total * 100 if doe_total > 0 else 0
        overall_acc = (buck_correct + doe_correct) / (buck_total + doe_total) * 100

        print(f"[buckdoe] Epoch {epoch}/{epochs} - Loss: {train_loss:.4f}/{val_loss:.4f}, "
              f"Acc: {overall_acc:.1f}% (Buck: {buck_acc:.0f}%, Doe: {doe_acc:.0f}%)")

        # Save best model based on balanced accuracy (average of both classes)
        balanced_acc = (buck_acc + doe_acc) / 2
        if balanced_acc > best_val_acc:
            best_val_acc = balanced_acc
            best_state = model.state_dict().copy()

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n[buckdoe] Best balanced accuracy: {best_val_acc:.1f}%")

    # Export to ONNX
    print(f"\n[buckdoe] Exporting to ONNX: {output_path}")
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
    labels_path = output_dir / "buckdoe_labels.txt"
    with open(labels_path, "w") as f:
        for lbl in LABELS:
            f.write(lbl + "\n")

    print(f"[buckdoe] Labels saved to: {labels_path}")
    print(f"\n[buckdoe] Done! Model saved to {output_path}")


if __name__ == "__main__":
    db_path = os.path.expanduser("~/.trailcam/trailcam.db")
    output_path = "models/buckdoe.onnx"
    train_model(db_path, output_path, epochs=20)

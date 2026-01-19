"""
Buck/Doe Classifier v2.0 - Per-Subject Training

Improvements over v1:
- Trains on individual subject crops (not full photos)
- Only uses area within bounding box
- Excludes overlapping boxes with different sex labels
- Better data balancing with weighted sampling

Input: Cropped subject from detection box
Output: Buck or Doe classification
"""

import os
import sys
import random
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def boxes_overlap(box1, box2, iou_threshold=0.1):
    """Check if two boxes overlap significantly.

    Args:
        box1, box2: dicts with x1, y1, x2, y2 keys (0-1 normalized)
        iou_threshold: minimum IoU to consider as overlapping

    Returns:
        True if boxes overlap more than threshold
    """
    # Calculate intersection
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    if x2 <= x1 or y2 <= y1:
        return False  # No intersection

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection

    if union <= 0:
        return False

    iou = intersection / union
    return iou >= iou_threshold


def load_training_data(db_path: str):
    """Load buck/doe training data from database.

    Returns list of dicts with:
    - file_path: path to image
    - box: x1, y1, x2, y2 coordinates (0-1 normalized)
    - label: 'Buck' or 'Doe'
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all photos with Buck or Doe tags
    cursor.execute("""
        SELECT DISTINCT p.id, p.file_path
        FROM photos p
        JOIN tags t ON p.id = t.photo_id
        WHERE t.tag_name IN ('Buck', 'Doe')
        AND p.archived = 0
    """)

    photos = {row[0]: row[1] for row in cursor.fetchall()}
    print(f"Found {len(photos)} photos with Buck/Doe tags")

    # Get all annotation boxes for these photos
    cursor.execute("""
        SELECT ab.id, ab.photo_id, ab.x1, ab.y1, ab.x2, ab.y2, ab.sex
        FROM annotation_boxes ab
        WHERE ab.photo_id IN ({})
        AND ab.label IN ('subject', 'ai_animal')
        AND ab.x1 IS NOT NULL
    """.format(','.join(map(str, photos.keys()))))

    boxes_by_photo = {}
    for row in cursor.fetchall():
        box_id, photo_id, x1, y1, x2, y2, sex = row
        if photo_id not in boxes_by_photo:
            boxes_by_photo[photo_id] = []
        boxes_by_photo[photo_id].append({
            'id': box_id,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'sex': sex
        })

    # Get photo-level tags to assign to boxes without explicit sex
    cursor.execute("""
        SELECT photo_id, tag_name
        FROM tags
        WHERE tag_name IN ('Buck', 'Doe')
    """)

    photo_tags = {}
    for photo_id, tag_name in cursor.fetchall():
        if photo_id not in photo_tags:
            photo_tags[photo_id] = set()
        photo_tags[photo_id].add(tag_name)

    conn.close()

    # Process boxes and filter out overlapping different-sex boxes
    training_samples = []
    excluded_overlap = 0
    excluded_ambiguous = 0
    excluded_missing = 0

    for photo_id, file_path in photos.items():
        if not os.path.exists(file_path):
            excluded_missing += 1
            continue

        boxes = boxes_by_photo.get(photo_id, [])
        tags = photo_tags.get(photo_id, set())

        # If photo has BOTH Buck and Doe tags, it's ambiguous at photo level
        has_both_tags = 'Buck' in tags and 'Doe' in tags

        for i, box in enumerate(boxes):
            # Determine box label
            if box['sex']:
                # Box has explicit sex label
                label = 'Buck' if box['sex'].lower() == 'buck' else 'Doe'
            elif has_both_tags:
                # Photo has both tags but box has no explicit sex - skip
                excluded_ambiguous += 1
                continue
            elif len(tags) == 1:
                # Photo has single tag, assign to box
                label = list(tags)[0]
            else:
                # No clear label
                excluded_ambiguous += 1
                continue

            # Check for overlapping boxes with different sex
            has_conflict = False
            for j, other_box in enumerate(boxes):
                if i == j:
                    continue

                # Determine other box's label
                if other_box['sex']:
                    other_label = 'Buck' if other_box['sex'].lower() == 'buck' else 'Doe'
                elif len(tags) == 1:
                    other_label = list(tags)[0]
                else:
                    continue  # Can't determine other box's label

                # If different labels and overlapping, exclude
                if other_label != label and boxes_overlap(box, other_box):
                    has_conflict = True
                    break

            if has_conflict:
                excluded_overlap += 1
                continue

            training_samples.append({
                'file_path': file_path,
                'box': {'x1': box['x1'], 'y1': box['y1'], 'x2': box['x2'], 'y2': box['y2']},
                'label': label
            })

    print(f"Training samples: {len(training_samples)}")
    print(f"Excluded - overlapping different sex: {excluded_overlap}")
    print(f"Excluded - ambiguous label: {excluded_ambiguous}")
    print(f"Excluded - missing file: {excluded_missing}")

    # Count labels
    label_counts = Counter(s['label'] for s in training_samples)
    print(f"Label distribution: {dict(label_counts)}")

    return training_samples


class BuckDoeDataset(Dataset):
    """Dataset of cropped subjects for buck/doe classification."""

    def __init__(self, samples, transform=None, augment=False):
        self.samples = samples
        self.transform = transform
        self.augment = augment
        self.label_to_idx = {'Buck': 0, 'Doe': 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and crop image
        img = Image.open(sample['file_path']).convert('RGB')
        img_w, img_h = img.size

        box = sample['box']
        x1 = int(box['x1'] * img_w)
        y1 = int(box['y1'] * img_h)
        x2 = int(box['x2'] * img_w)
        y2 = int(box['y2'] * img_h)

        # Ensure valid box
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)

        # Crop to subject
        crop = img.crop((x1, y1, x2, y2))

        # Augmentation
        if self.augment:
            # Horizontal flip (50%)
            if random.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

            # Grayscale (30% - for night photos)
            if random.random() < 0.3:
                crop = crop.convert('L').convert('RGB')

            # Brightness/contrast
            if random.random() < 0.5:
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(crop)
                crop = enhancer.enhance(random.uniform(0.7, 1.3))
                enhancer = ImageEnhance.Contrast(crop)
                crop = enhancer.enhance(random.uniform(0.8, 1.2))

        if self.transform:
            crop = self.transform(crop)

        label = self.label_to_idx[sample['label']]
        return crop, label


class BuckDoeModel(nn.Module):
    """CNN for buck/doe classification."""

    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()

        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Per-class tracking
    class_correct = [0, 0]  # Buck, Doe
    class_total = [0, 0]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    acc = 100.0 * correct / total
    buck_acc = 100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    doe_acc = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0

    return total_loss / len(loader), acc, buck_acc, doe_acc


def main():
    config = {
        'db_path': os.path.expanduser("~/.trailcam/trailcam.db"),
        'output_dir': Path(__file__).parent.parent / "outputs" / "buckdoe_v2",
        'img_size': 256,
        'backbone': 'resnet18',
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'seed': 42,
    }

    config['output_dir'].mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: img_size={config['img_size']}, backbone={config['backbone']}")
    print()

    # Load data
    print("Loading training data...")
    samples = load_training_data(config['db_path'])

    if len(samples) < 20:
        print("Not enough samples for training!")
        return

    # Split by label to ensure balance in val set
    random.seed(config['seed'])
    bucks = [s for s in samples if s['label'] == 'Buck']
    does = [s for s in samples if s['label'] == 'Doe']

    random.shuffle(bucks)
    random.shuffle(does)

    # 80/20 split
    buck_split = int(len(bucks) * 0.8)
    doe_split = int(len(does) * 0.8)

    train_samples = bucks[:buck_split] + does[:doe_split]
    val_samples = bucks[buck_split:] + does[doe_split:]

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"\nTrain: {len(train_samples)} ({len(bucks[:buck_split])} Buck, {len(does[:doe_split])} Doe)")
    print(f"Val: {len(val_samples)} ({len(bucks[buck_split:])} Buck, {len(does[doe_split:])} Doe)")

    # Calculate class weights for balanced sampling
    train_labels = [0 if s['label'] == 'Buck' else 1 for s in train_samples]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = BuckDoeDataset(train_samples, train_transform, augment=True)
    val_dataset = BuckDoeDataset(val_samples, val_transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # Model
    print(f"\nCreating model with {config['backbone']} backbone...")
    model = BuckDoeModel(backbone=config['backbone'], pretrained=True)
    model = model.to(device)

    # Training setup - use class weights in loss too
    class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], device=device)
    class_weights = class_weights / class_weights.sum() * 2  # Normalize
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    # Training loop
    best_val_acc = 0
    best_epoch = 0

    print(f"\nTraining for {config['num_epochs']} epochs...")
    print("-" * 90)

    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, buck_acc, doe_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | "
                  f"Val Acc={val_acc:.1f}% (Buck={buck_acc:.1f}%, Doe={doe_acc:.1f}%)")

        # Save best model based on balanced accuracy (average of buck and doe acc)
        balanced_acc = (buck_acc + doe_acc) / 2
        if balanced_acc > best_val_acc:
            best_val_acc = balanced_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'buck_acc': buck_acc,
                'doe_acc': doe_acc,
                'balanced_acc': balanced_acc,
                'config': config,
            }, config['output_dir'] / "best_model.pt")

    print("-" * 90)
    print(f"Best balanced accuracy: {best_val_acc:.1f}% at epoch {best_epoch}")

    # Load best model for final evaluation
    checkpoint = torch.load(config['output_dir'] / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_loss, val_acc, buck_acc, doe_acc = validate(model, val_loader, criterion, device)

    # Export to ONNX
    print("\nExporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, config['img_size'], config['img_size']).to(device)
    onnx_path = config['output_dir'] / "buckdoe_v2.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Saved ONNX model to: {onnx_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - Buck/Doe Classifier v2")
    print(f"{'='*60}")
    print(f"Model saved to: {config['output_dir'] / 'best_model.pt'}")
    print(f"ONNX model: {onnx_path}")
    print()
    print(f"Results (best epoch {best_epoch}):")
    print(f"  Overall accuracy: {val_acc:.1f}%")
    print(f"  Buck accuracy:    {buck_acc:.1f}%")
    print(f"  Doe accuracy:     {doe_acc:.1f}%")
    print(f"  Balanced accuracy: {(buck_acc + doe_acc)/2:.1f}%")
    print()
    print(f"Training data: {len(train_samples)} samples")
    print(f"Improvements over v1:")
    print(f"  - Per-subject crops (not full photos)")
    print(f"  - Excluded overlapping different-sex boxes")
    print(f"  - Weighted sampling for class balance")
    print(f"  - Class-weighted loss function")

    # Save labels file
    with open(config['output_dir'] / "labels.txt", 'w') as f:
        f.write("Buck\nDoe\n")
    print(f"  Labels: {config['output_dir'] / 'labels.txt'}")


if __name__ == "__main__":
    main()

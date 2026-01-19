"""
Head Keypoint Model v2.0 - Improved Training

Improvements over v1:
- Include annotations with notes that have valid coordinates
- Horizontal flip augmentation (with keypoint adjustment)
- Grayscale augmentation (for night/IR photos)
- Higher resolution option (256x256)
- Better train/val split with stratification by quality

Input: Deer crop (from detection box)
Output: 4 values (skull_x, skull_y, nose_x, nose_y) as 0-1 relative coords
"""

import os
import sys
import random
import sqlite3
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class HeadKeypointDataset(Dataset):
    """Dataset of deer crops with head keypoint annotations."""

    def __init__(self, annotations, transform=None, img_size=256, augment=False):
        """
        Args:
            annotations: List of dicts with file_path, box coords, head coords
            transform: Image transforms (applied after augmentation)
            img_size: Target image size
            augment: Whether to apply augmentation
        """
        self.annotations = annotations
        self.transform = transform
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # Load and crop image
        img = Image.open(ann['file_path']).convert('RGB')
        img_w, img_h = img.size

        # Get box in pixels
        x1 = int(ann['box_x1'] * img_w)
        y1 = int(ann['box_y1'] * img_h)
        x2 = int(ann['box_x2'] * img_w)
        y2 = int(ann['box_y2'] * img_h)

        # Ensure valid box
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)

        # Crop
        crop = img.crop((x1, y1, x2, y2))

        # Convert head coords from image-relative to crop-relative (0-1)
        box_w = x2 - x1
        box_h = y2 - y1

        skull_x = (ann['head_x1'] * img_w - x1) / box_w
        skull_y = (ann['head_y1'] * img_h - y1) / box_h
        nose_x = (ann['head_x2'] * img_w - x1) / box_w
        nose_y = (ann['head_y2'] * img_h - y1) / box_h

        # Apply augmentation
        if self.augment:
            # Horizontal flip (50% chance)
            if random.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                # Flip x coordinates
                skull_x = 1.0 - skull_x
                nose_x = 1.0 - nose_x

            # Grayscale (30% chance - helps with night photos)
            if random.random() < 0.3:
                crop = crop.convert('L').convert('RGB')

            # Random brightness/contrast
            if random.random() < 0.5:
                from PIL import ImageEnhance
                # Brightness
                enhancer = ImageEnhance.Brightness(crop)
                crop = enhancer.enhance(random.uniform(0.7, 1.3))
                # Contrast
                enhancer = ImageEnhance.Contrast(crop)
                crop = enhancer.enhance(random.uniform(0.8, 1.2))

        # Clamp to 0-1
        skull_x = max(0, min(1, skull_x))
        skull_y = max(0, min(1, skull_y))
        nose_x = max(0, min(1, nose_x))
        nose_y = max(0, min(1, nose_y))

        if self.transform:
            crop = self.transform(crop)

        keypoints = torch.tensor([skull_x, skull_y, nose_x, nose_y], dtype=torch.float32)

        return crop, keypoints


class HeadKeypointModel(nn.Module):
    """CNN for head keypoint regression with improved head."""

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

        # Improved regression head with more capacity
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Output 0-1 range
        )

    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.head(features)
        return keypoints


def load_annotations(db_path: str, include_noted: bool = True):
    """Load head annotations from database.

    Args:
        db_path: Path to SQLite database
        include_noted: If True, include annotations with notes that have valid coordinates
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Base query - must have head coordinates
    query = """
        SELECT p.file_path,
               ab.x1, ab.y1, ab.x2, ab.y2,
               ab.head_x1, ab.head_y1, ab.head_x2, ab.head_y2,
               ab.head_notes
        FROM annotation_boxes ab
        JOIN photos p ON ab.photo_id = p.id
        WHERE ab.head_x1 IS NOT NULL
          AND ab.label IN ('subject', 'ai_animal')
    """

    if not include_noted:
        query += " AND (ab.head_notes IS NULL OR ab.head_notes = '')"

    cursor.execute(query)

    annotations = []
    clean_count = 0
    noted_count = 0

    for row in cursor.fetchall():
        # Skip if file doesn't exist
        if not os.path.exists(row[0]):
            continue

        has_notes = row[9] and row[9].strip()

        annotations.append({
            'file_path': row[0],
            'box_x1': row[1], 'box_y1': row[2], 'box_x2': row[3], 'box_y2': row[4],
            'head_x1': row[5], 'head_y1': row[6], 'head_x2': row[7], 'head_y2': row[8],
            'notes': row[9],
            'is_clean': not has_notes
        })

        if has_notes:
            noted_count += 1
        else:
            clean_count += 1

    conn.close()
    print(f"Loaded {len(annotations)} annotations ({clean_count} clean, {noted_count} with notes)")
    return annotations


def calculate_error(pred, target):
    """Calculate average Euclidean distance error for skull and nose."""
    skull_err = torch.sqrt((pred[:, 0] - target[:, 0])**2 + (pred[:, 1] - target[:, 1])**2)
    nose_err = torch.sqrt((pred[:, 2] - target[:, 2])**2 + (pred[:, 3] - target[:, 3])**2)
    return skull_err.mean(), nose_err.mean()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_skull_err = 0
    total_nose_err = 0

    for images, keypoints in loader:
        images = images.to(device)
        keypoints = keypoints.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        skull_err, nose_err = calculate_error(outputs, keypoints)
        total_skull_err += skull_err.item()
        total_nose_err += nose_err.item()

    n = len(loader)
    return total_loss / n, total_skull_err / n, total_nose_err / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_skull_err = 0
    total_nose_err = 0

    with torch.no_grad():
        for images, keypoints in loader:
            images = images.to(device)
            keypoints = keypoints.to(device)

            outputs = model(images)
            loss = criterion(outputs, keypoints)

            total_loss += loss.item()
            skull_err, nose_err = calculate_error(outputs, keypoints)
            total_skull_err += skull_err.item()
            total_nose_err += nose_err.item()

    n = len(loader)
    return total_loss / n, total_skull_err / n, total_nose_err / n


def visualize_predictions(model, dataset, device, output_dir, num_samples=20):
    """Save visualization of predictions vs ground truth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        img_tensor, gt_keypoints = dataset[idx]

        with torch.no_grad():
            pred_keypoints = model(img_tensor.unsqueeze(0).to(device)).cpu().squeeze()

        # Convert tensor to image for display
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)

        h, w = img.shape[:2]

        # Ground truth
        gt_skull = (gt_keypoints[0].item() * w, gt_keypoints[1].item() * h)
        gt_nose = (gt_keypoints[2].item() * w, gt_keypoints[3].item() * h)

        # Prediction
        pred_skull = (pred_keypoints[0].item() * w, pred_keypoints[1].item() * h)
        pred_nose = (pred_keypoints[2].item() * w, pred_keypoints[3].item() * h)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)

        # Ground truth in green
        plt.plot([gt_skull[0], gt_nose[0]], [gt_skull[1], gt_nose[1]], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(gt_skull[0], gt_skull[1], 'go', markersize=10)
        plt.plot(gt_nose[0], gt_nose[1], 'g^', markersize=10)

        # Prediction in red
        plt.plot([pred_skull[0], pred_nose[0]], [pred_skull[1], pred_nose[1]], 'r-', linewidth=2, label='Prediction')
        plt.plot(pred_skull[0], pred_skull[1], 'ro', markersize=10)
        plt.plot(pred_nose[0], pred_nose[1], 'r^', markersize=10)

        plt.legend()
        plt.title(f'Sample {i+1}: Green=GT, Red=Pred')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'prediction_{i+1}.png'), bbox_inches='tight', dpi=100)
        plt.close()

    print(f"Saved {len(indices)} visualization images to {output_dir}")


def main():
    # Configuration
    config = {
        'db_path': os.path.expanduser("~/.trailcam/trailcam.db"),
        'output_dir': Path(__file__).parent.parent / "outputs" / "head_keypoints_v2",
        'img_size': 256,  # Increased from 224
        'backbone': 'resnet18',
        'batch_size': 16,
        'num_epochs': 75,
        'learning_rate': 1e-4,
        'include_noted': True,  # Include annotations with notes
        'seed': 42,
    }

    config['output_dir'].mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: img_size={config['img_size']}, backbone={config['backbone']}, include_noted={config['include_noted']}")

    # Load annotations
    print("\nLoading annotations...")
    annotations = load_annotations(config['db_path'], include_noted=config['include_noted'])

    if len(annotations) < 20:
        print("Not enough annotations for training. Need at least 20.")
        return

    # Shuffle and split (stratified by clean/noted)
    random.seed(config['seed'])
    clean = [a for a in annotations if a['is_clean']]
    noted = [a for a in annotations if not a['is_clean']]

    random.shuffle(clean)
    random.shuffle(noted)

    # 80/20 split for both
    clean_split = int(len(clean) * 0.8)
    noted_split = int(len(noted) * 0.8)

    train_annotations = clean[:clean_split] + noted[:noted_split]
    val_annotations = clean[clean_split:] + noted[noted_split:]

    random.shuffle(train_annotations)
    random.shuffle(val_annotations)

    print(f"Train: {len(train_annotations)} ({len(clean[:clean_split])} clean, {len(noted[:noted_split])} noted)")
    print(f"Val: {len(val_annotations)} ({len(clean[clean_split:])} clean, {len(noted[noted_split:])} noted)")

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

    train_dataset = HeadKeypointDataset(train_annotations, train_transform, config['img_size'], augment=True)
    val_dataset = HeadKeypointDataset(val_annotations, val_transform, config['img_size'], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # Model
    print(f"\nCreating model with {config['backbone']} backbone...")
    model = HeadKeypointModel(backbone=config['backbone'], pretrained=True)
    model = model.to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\nTraining for {config['num_epochs']} epochs...")
    print("-" * 80)

    for epoch in range(config['num_epochs']):
        train_loss, train_skull_err, train_nose_err = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_skull_err, val_nose_err = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Error is in 0-1 range, convert to percentage
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} | "
                  f"Skull={val_skull_err*100:.1f}%, Nose={val_nose_err*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_skull_err': val_skull_err,
                'val_nose_err': val_nose_err,
                'config': config,
            }, config['output_dir'] / "best_model.pt")

    print("-" * 80)
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Load best model and visualize
    print("\nGenerating visualizations...")
    checkpoint = torch.load(config['output_dir'] / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    visualize_predictions(model, val_dataset, device, config['output_dir'] / "visualizations", num_samples=25)

    # Final evaluation
    val_loss, val_skull_err, val_nose_err = validate(model, val_loader, criterion, device)

    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - Head Keypoints v2")
    print(f"{'='*60}")
    print(f"Model saved to: {config['output_dir'] / 'best_model.pt'}")
    print(f"Visualizations: {config['output_dir'] / 'visualizations'}")
    print(f"")
    print(f"Results:")
    print(f"  Skull error: {val_skull_err*100:.1f}% (v1 was ~14%)")
    print(f"  Nose error:  {val_nose_err*100:.1f}% (v1 was ~16%)")
    print(f"  For {config['img_size']}px crop: ~{val_skull_err*config['img_size']:.0f}px skull, ~{val_nose_err*config['img_size']:.0f}px nose")
    print(f"")
    print(f"Training data: {len(train_annotations)} samples")
    print(f"Improvements over v1:")
    print(f"  - {config['img_size']}px resolution (was 224px)")
    print(f"  - Horizontal flip augmentation")
    print(f"  - Grayscale augmentation for night photos")
    print(f"  - Included {len(noted[:noted_split])} 'noted' annotations with valid coords")


if __name__ == "__main__":
    main()

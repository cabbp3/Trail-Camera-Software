"""
Head Keypoint Model v1.0 - Exploratory Training

Trains a simple CNN to predict skull and nose positions from deer crops.
This is an exploratory model to see if we can learn head features from
the current annotations.

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

    def __init__(self, annotations, transform=None):
        """
        Args:
            annotations: List of dicts with file_path, box coords, head coords
            transform: Image transforms
        """
        self.annotations = annotations
        self.transform = transform

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
    """Simple CNN for head keypoint regression."""

    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()

        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Regression head: 4 outputs (skull_x, skull_y, nose_x, nose_y)
        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Output 0-1 range
        )

    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.head(features)
        return keypoints


def load_annotations(db_path: str, clean_only: bool = True):
    """Load head annotations from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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

    if clean_only:
        query += " AND (ab.head_notes IS NULL OR ab.head_notes = '')"

    cursor.execute(query)

    annotations = []
    for row in cursor.fetchall():
        # Skip if file doesn't exist
        if not os.path.exists(row[0]):
            continue

        annotations.append({
            'file_path': row[0],
            'box_x1': row[1], 'box_y1': row[2], 'box_x2': row[3], 'box_y2': row[4],
            'head_x1': row[5], 'head_y1': row[6], 'head_x2': row[7], 'head_y2': row[8],
            'notes': row[9]
        })

    conn.close()
    return annotations


def calculate_error(pred, target):
    """Calculate average Euclidean distance error for skull and nose."""
    # pred and target are [skull_x, skull_y, nose_x, nose_y]
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


def visualize_predictions(model, dataset, device, output_dir, num_samples=10):
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
    db_path = os.path.expanduser("~/.trailcam/trailcam.db")
    output_dir = Path(__file__).parent.parent / "outputs" / "head_keypoints_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(db_path, clean_only=True)
    print(f"Found {len(annotations)} clean annotations")

    if len(annotations) < 20:
        print("Not enough annotations for training. Need at least 20.")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(annotations)

    split_idx = int(len(annotations) * 0.8)
    train_annotations = annotations[:split_idx]
    val_annotations = annotations[split_idx:]

    print(f"Train: {len(train_annotations)}, Val: {len(val_annotations)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.0),  # Don't flip - would need to flip keypoints too
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HeadKeypointDataset(train_annotations, train_transform)
    val_dataset = HeadKeypointDataset(val_annotations, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Model
    print("Creating model...")
    model = HeadKeypointModel(backbone='resnet18', pretrained=True)
    model = model.to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 70)

    for epoch in range(num_epochs):
        train_loss, train_skull_err, train_nose_err = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_skull_err, val_nose_err = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Error is in 0-1 range, convert to percentage of image
        print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} | "
              f"Val Skull Err={val_skull_err*100:.1f}%, Val Nose Err={val_nose_err*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    print("-" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Load best model and visualize
    print("\nGenerating visualizations...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    visualize_predictions(model, val_dataset, device, output_dir / "visualizations", num_samples=min(20, len(val_dataset)))

    # Summary
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print(f"Visualizations saved to: {output_dir / 'visualizations'}")

    # Final evaluation interpretation
    print(f"\nInterpretation:")
    print(f"  Skull error {val_skull_err*100:.1f}% means predictions are off by ~{val_skull_err*100:.0f}% of crop width/height")
    print(f"  Nose error {val_nose_err*100:.1f}% means predictions are off by ~{val_nose_err*100:.0f}% of crop width/height")
    print(f"  For a 200px crop, that's ~{val_skull_err*200:.0f}px skull error, ~{val_nose_err*200:.0f}px nose error")


if __name__ == "__main__":
    main()

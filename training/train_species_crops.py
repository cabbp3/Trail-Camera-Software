"""
Train species classifier using on-the-fly cropping from database boxes.

This script:
1. Queries photos with human species tags AND bounding boxes
2. Pre-loads and caches all crops in memory for fast training
3. Trains a species classifier

Usage:
    python training/train_species_crops.py

Requires: torch, torchvision, timm, Pillow
"""
import os
import sys
import random
from pathlib import Path
from collections import Counter

# Add parent directory to path for database import
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import TrailCamDatabase


def main():
    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
        from torchvision import transforms
        from PIL import Image
        import timm
    except ImportError as e:
        print(f"Missing dependencies. Install: pip install torch torchvision timm Pillow", flush=True)
        print(f"Error: {e}", flush=True)
        sys.exit(1)

    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50  # Increased for better accuracy (overnight run)
    LR = 1e-3
    MODEL_NAME = "resnet18"  # Faster than convnext_tiny
    VAL_SPLIT = 0.15

    # Species mapping - keep each species separate
    # Note: Empty is excluded since crops are only from detected animals
    # If MegaDetector finds nothing, it's Empty (handled at detection stage)
    # Species with very few samples (<5) are EXCLUDED from training (set to None)
    # The AI will suggest "Unknown" for species it can't identify
    SPECIES_MAP = {
        "Deer": "Deer",
        "Turkey": "Turkey",
        "Raccoon": "Raccoon",
        "Rabbit": "Rabbit",
        "Squirrel": "Squirrel",
        "Coyote": "Coyote",
        "Bobcat": "Bobcat",
        "Opossum": "Opossum",
        "Fox": "Fox",
        "Person": "Person",
        "Vehicle": "Vehicle",
        "House Cat": "House Cat",
        "Dog": None,           # Excluded - only 3 samples
        "Quail": None,         # Excluded - only 2 samples
        "Armadillo": None,     # Excluded - only 2 samples
        "Chipmunk": None,      # Excluded - only 2 samples
        "Skunk": None,         # Excluded - only 2 samples
        "Ground Hog": None,    # Excluded - only 4 samples
        "Flicker": None,       # Excluded - only 3 samples
        "Turkey Buzzard": None,  # Excluded - only 11 samples
        "Other Bird": None,    # Excluded - too varied
    }

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[train] Using Apple Metal (MPS) GPU", flush=True)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[train] Using CUDA GPU", flush=True)
    else:
        device = torch.device("cpu")
        print("[train] Using CPU", flush=True)

    # Load data from database
    print("\n[train] Loading data from database...", flush=True)
    db = TrailCamDatabase()

    # Query photos with species tags and boxes
    cursor = db.conn.cursor()
    # Get all species from the mapping
    species_list = list(SPECIES_MAP.keys())
    placeholders = ','.join(['?' for _ in species_list])
    cursor.execute(f"""
        SELECT DISTINCT p.id, p.file_path, t.tag_name
        FROM photos p
        JOIN tags t ON p.id = t.photo_id
        JOIN annotation_boxes b ON p.id = b.photo_id
        WHERE t.tag_name IN ({placeholders})
          AND b.label IN ('subject', 'ai_animal')
    """, species_list)

    # Build dataset entries: (photo_id, file_path, species, box)
    photo_data = {}
    for row in cursor.fetchall():
        photo_id = row[0]
        file_path = row[1]
        species = SPECIES_MAP.get(row[2])
        # Skip species that are excluded from training (mapped to None)
        if species is None:
            continue
        if photo_id not in photo_data:
            photo_data[photo_id] = {"file_path": file_path, "species": species}

    # Get boxes for each photo - use ALL boxes to increase sample size
    samples = []
    for photo_id, data in photo_data.items():
        boxes = db.get_boxes(photo_id)
        # Filter to subject/ai_animal boxes
        boxes = [b for b in boxes if b.get("label") in ("subject", "ai_animal")]
        # Add each box as a separate training sample (helps with flocks/groups)
        for box in boxes:
            samples.append({
                "photo_id": photo_id,
                "file_path": data["file_path"],
                "species": data["species"],
                "box": box
            })

    db.close()

    if not samples:
        print("[train] No samples found with species tags and boxes!", flush=True)
        sys.exit(1)

    # Print distribution
    species_counts = Counter(s["species"] for s in samples)
    print(f"\n[train] Found {len(samples)} samples with boxes:", flush=True)
    for species, count in sorted(species_counts.items(), key=lambda x: -x[1]):
        print(f"  {species}: {count}", flush=True)

    # Build class list
    classes = sorted(species_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"\n[train] Classes: {classes}", flush=True)

    # Split into train/val
    random.seed(42)
    random.shuffle(samples)
    val_size = int(len(samples) * VAL_SPLIT)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    print(f"[train] Train: {len(train_samples)}, Val: {len(val_samples)}", flush=True)

    # Pre-load all crops into memory for fast training
    print("\n[train] Pre-loading and caching all crops in memory...", flush=True)

    def load_crop(sample):
        """Load and crop an image, return as PIL Image."""
        try:
            img = Image.open(sample["file_path"]).convert("RGB")
        except Exception as e:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        # Crop to bounding box (coordinates are relative 0-1)
        box = sample["box"]
        w, h = img.size
        x1 = int(box["x1"] * w)
        y1 = int(box["y1"] * h)
        x2 = int(box["x2"] * w)
        y2 = int(box["y2"] * h)

        # Ensure valid crop
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        return img.crop((x1, y1, x2, y2))

    # Load all crops
    train_crops = []
    train_labels = []
    for i, sample in enumerate(train_samples):
        crop = load_crop(sample)
        train_crops.append(crop)
        train_labels.append(class_to_idx[sample["species"]])
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i+1}/{len(train_samples)} train crops...", flush=True)

    val_crops = []
    val_labels = []
    for sample in val_samples:
        crop = load_crop(sample)
        val_crops.append(crop)
        val_labels.append(class_to_idx[sample["species"]])

    print(f"[train] Cached {len(train_crops)} train + {len(val_crops)} val crops in memory", flush=True)

    # Dataset that uses pre-loaded crops
    class CachedCropDataset(Dataset):
        def __init__(self, crops, labels, transform=None):
            self.crops = crops
            self.labels = labels
            self.transform = transform
            self.targets = labels  # For weighted sampler

        def __len__(self):
            return len(self.crops)

        def __getitem__(self, idx):
            img = self.crops[idx]
            label = self.labels[idx]

            if self.transform:
                img = self.transform(img)

            return img, label

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CachedCropDataset(train_crops, train_labels, train_transform)
    val_ds = CachedCropDataset(val_crops, val_labels, val_transform)

    # No weighted sampling - let the model learn natural distribution
    # (88% Deer is fine - we care most about identifying Deer correctly)
    train_class_counts = Counter(train_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    # Build model
    num_classes = len(classes)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model.to(device)
    print(f"\n[train] Model: {MODEL_NAME} ({num_classes} classes)", flush=True)

    # Standard cross-entropy loss (no class weighting - natural distribution)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Training
    os.makedirs("training/outputs", exist_ok=True)
    best_acc = 0.0

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        per_class_correct = Counter()
        per_class_total = Counter()

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()

                for pred, label in zip(preds.cpu().numpy(), y.cpu().numpy()):
                    per_class_total[label] += 1
                    if pred == label:
                        per_class_correct[label] += 1

        overall = correct / max(total, 1)

        # Per-class accuracy
        per_class = {}
        for cls_idx in range(num_classes):
            if per_class_total[cls_idx] > 0:
                per_class[classes[cls_idx]] = per_class_correct[cls_idx] / per_class_total[cls_idx]
            else:
                per_class[classes[cls_idx]] = 0.0

        return overall, per_class

    print("\n[train] Starting training...", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        acc, per_class = evaluate()

        # Print progress
        per_class_str = " | ".join([f"{c[:4]}:{v:.0%}" for c, v in per_class.items()])
        print(f"[train] Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} val_acc={acc:.1%} [{per_class_str}]", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "training/outputs/species_crops_best.pt")

    print(f"\n[train] Training complete! Best val accuracy: {best_acc:.1%}", flush=True)

    # Export to ONNX
    print("\n[train] Exporting to ONNX...", flush=True)
    model.load_state_dict(torch.load("training/outputs/species_crops_best.pt", map_location=device, weights_only=True))
    model.eval()
    model.to("cpu")  # Export from CPU for compatibility

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    export_path = "training/outputs/species_crops.onnx"
    torch.onnx.export(
        model,
        dummy,
        export_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
    )
    print(f"[train] Exported ONNX -> {export_path}", flush=True)

    # Save labels
    labels_path = "training/outputs/species_crops_labels.txt"
    with open(labels_path, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"[train] Saved labels -> {labels_path}", flush=True)

    print(f"\n[train] Done!", flush=True)
    print(f"[train] To deploy: copy species_crops.onnx to models/species.onnx", flush=True)
    print(f"[train]            copy species_crops_labels.txt to models/labels.txt", flush=True)


if __name__ == "__main__":
    main()

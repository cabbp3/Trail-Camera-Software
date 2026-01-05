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
    #
    # IMPORTANT: Person and Vehicle are EXCLUDED from training!
    # MegaDetector already detects ai_person and ai_vehicle boxes directly.
    # The species classifier should NOT predict these - they are auto-classified
    # based on MegaDetector detection labels, not the species model.
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
        "House Cat": "House Cat",
        "Person": None,        # Excluded - MegaDetector handles this via ai_person
        "Vehicle": None,       # Excluded - MegaDetector handles this via ai_vehicle
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

    # Get boxes for each photo, filtering overlapping boxes to reduce pseudo-replication
    def calc_iou(box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1["x1"], box2["x1"])
        y1 = max(box1["y1"], box2["y1"])
        x2 = min(box1["x2"], box2["x2"])
        y2 = min(box1["y2"], box2["y2"])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def filter_overlapping_boxes(boxes, iou_threshold=0.5):
        """Remove boxes with significant overlap, keeping the larger one."""
        if len(boxes) <= 1:
            return boxes

        # Sort by area (largest first) so we keep bigger boxes
        boxes = sorted(boxes, key=lambda b: (b["x2"]-b["x1"])*(b["y2"]-b["y1"]), reverse=True)

        keep = []
        for box in boxes:
            # Check if this box overlaps significantly with any kept box
            dominated = False
            for kept_box in keep:
                if calc_iou(box, kept_box) > iou_threshold:
                    dominated = True
                    break
            if not dominated:
                keep.append(box)

        return keep

    samples = []
    boxes_before = 0
    boxes_after = 0
    for photo_id, data in photo_data.items():
        boxes = db.get_boxes(photo_id)
        # Filter to subject/ai_animal boxes
        boxes = [b for b in boxes if b.get("label") in ("subject", "ai_animal")]
        boxes_before += len(boxes)

        # Remove overlapping boxes to reduce pseudo-replication
        boxes = filter_overlapping_boxes(boxes, iou_threshold=0.5)
        boxes_after += len(boxes)

        # Add each non-overlapping box as a training sample
        for box in boxes:
            samples.append({
                "photo_id": photo_id,
                "file_path": data["file_path"],
                "species": data["species"],
                "box": box
            })

    print(f"[train] Filtered overlapping boxes: {boxes_before} -> {boxes_after} ({boxes_before - boxes_after} removed)", flush=True)

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

    # Stratified split into train/val/test by species
    # This ensures each species is proportionally represented in all splits
    TEST_SPLIT = 0.10  # 10% for test
    # VAL_SPLIT is 0.15 of remaining = ~13.5% of total

    random.seed(42)

    # Group samples by species
    species_to_samples = {}
    for sample in samples:
        species = sample["species"]
        if species not in species_to_samples:
            species_to_samples[species] = []
        species_to_samples[species].append(sample)

    train_samples = []
    val_samples = []
    test_samples = []

    print(f"\n[train] Stratified split by species:", flush=True)
    for species in sorted(species_to_samples.keys()):
        species_samples = species_to_samples[species]
        random.shuffle(species_samples)

        n = len(species_samples)
        n_test = max(1, int(n * TEST_SPLIT))  # At least 1 for test
        n_val = max(1, int((n - n_test) * VAL_SPLIT))  # At least 1 for val
        n_train = n - n_test - n_val

        # Ensure we have at least 1 in each split for species with few samples
        if n < 3:
            # Too few samples - put in train only with warning
            print(f"  {species}: {n} total (all in train - too few for split)", flush=True)
            train_samples.extend(species_samples)
            continue

        test_samples.extend(species_samples[:n_test])
        val_samples.extend(species_samples[n_test:n_test + n_val])
        train_samples.extend(species_samples[n_test + n_val:])

        print(f"  {species}: {n_train} train / {n_val} val / {n_test} test", flush=True)

    # Shuffle each split
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    print(f"\n[train] Total: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test", flush=True)

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

    test_crops = []
    test_labels = []
    for sample in test_samples:
        crop = load_crop(sample)
        test_crops.append(crop)
        test_labels.append(class_to_idx[sample["species"]])

    print(f"[train] Cached {len(train_crops)} train + {len(val_crops)} val + {len(test_crops)} test crops in memory", flush=True)

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
    test_ds = CachedCropDataset(test_crops, test_labels, val_transform)  # No augmentation for test

    # Square root weighted sampling - balances between natural distribution and uniform
    # This down-weights abundant classes (Deer) while still giving them more weight than rare ones
    # Weight = 1 / sqrt(class_count) - gives rare classes a fighting chance
    train_class_counts = Counter(train_labels)
    print(f"\n[train] Class distribution:", flush=True)
    for cls_idx, cls_name in enumerate(classes):
        count = train_class_counts.get(cls_idx, 0)
        print(f"  {cls_name}: {count}", flush=True)

    # Calculate per-sample weights using square root of inverse frequency
    class_weights = {}
    for cls_idx in range(len(classes)):
        count = train_class_counts.get(cls_idx, 1)
        class_weights[cls_idx] = 1.0 / (count ** 0.5)  # Square root weighting

    # Normalize weights so they sum to num_classes (keeps learning rate stable)
    weight_sum = sum(class_weights.values())
    for cls_idx in class_weights:
        class_weights[cls_idx] = class_weights[cls_idx] * len(classes) / weight_sum

    print(f"\n[train] Square root weights (normalized):", flush=True)
    for cls_idx, cls_name in enumerate(classes):
        print(f"  {cls_name}: {class_weights[cls_idx]:.3f}", flush=True)

    # Create per-sample weights for the sampler
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
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

    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        per_class_correct = Counter()
        per_class_total = Counter()

        with torch.no_grad():
            for x, y in loader:
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
        acc, per_class = evaluate(val_loader)

        # Print progress
        per_class_str = " | ".join([f"{c[:4]}:{v:.0%}" for c, v in per_class.items()])
        print(f"[train] Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} val_acc={acc:.1%} [{per_class_str}]", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "training/outputs/species_crops_best.pt")

    print(f"\n[train] Training complete! Best val accuracy: {best_acc:.1%}", flush=True)

    # Final test set evaluation (held-out data never seen during training)
    print("\n[train] Evaluating on held-out test set...", flush=True)
    model.load_state_dict(torch.load("training/outputs/species_crops_best.pt", map_location=device, weights_only=True))
    test_acc, test_per_class = evaluate(test_loader)
    print(f"[train] TEST SET ACCURACY: {test_acc:.1%}", flush=True)
    print("[train] Per-class test accuracy:", flush=True)
    for cls_name, acc in sorted(test_per_class.items(), key=lambda x: -x[1]):
        print(f"  {cls_name}: {acc:.0%}", flush=True)

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

    # Save training summary for MODEL_HISTORY.md
    from datetime import datetime
    summary_path = "training/outputs/species_crops_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Species Classifier Training Summary\n")
        f.write(f"===================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(f"Total samples: {len(samples)} (after overlap filtering)\n")
        f.write(f"Boxes filtered: {boxes_before} -> {boxes_after} ({boxes_before - boxes_after} removed for overlap)\n")
        f.write(f"Stratified split: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test\n\n")
        f.write(f"Best validation accuracy: {best_acc:.1%}\n")
        f.write(f"TEST SET ACCURACY: {test_acc:.1%}\n\n")
        f.write(f"Per-species sample counts:\n")
        for species, count in sorted(species_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {species}: {count}\n")
        f.write(f"\nPer-class TEST accuracy:\n")
        for cls_name, acc in sorted(test_per_class.items(), key=lambda x: -x[1]):
            f.write(f"  {cls_name}: {acc:.0%}\n")
    print(f"[train] Saved training summary -> {summary_path}", flush=True)

    print(f"\n[train] Done!", flush=True)
    print(f"[train] To deploy: copy species_crops.onnx to models/species.onnx", flush=True)
    print(f"[train]            copy species_crops_labels.txt to models/labels.txt", flush=True)
    print(f"[train] Update models/MODEL_HISTORY.md with info from {summary_path}", flush=True)


if __name__ == "__main__":
    main()

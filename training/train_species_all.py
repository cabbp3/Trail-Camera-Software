"""
Train species classifier with ALL species and sqrt-weighted sampling.

Features:
- Individual species labels (not grouped)
- Birds except Turkey grouped into "Other Bird"
- Square root weighting to reduce overrepresentation
"""
import os
import sys
import random
import math
from pathlib import Path
from collections import Counter

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
        print(f"Missing dependencies. Install: pip install torch torchvision timm Pillow")
        sys.exit(1)

    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    MODEL_NAME = "resnet18"
    VAL_SPLIT = 0.15

    # Species to keep as individual classes
    INDIVIDUAL_SPECIES = {
        "Deer", "Squirrel", "Turkey", "Raccoon", "Opossum", "Rabbit",
        "Person", "Coyote", "Fox", "House Cat", "Bobcat", "Dog",
        "Ground Hog", "Vehicle", "Armadillo", "Skunk"
    }

    # Birds to group (except Turkey)
    BIRD_SPECIES = {"Turkey Buzzard", "Other Bird", "Flicker", "Quail"}

    # Skip these tags (not species)
    SKIP_TAGS = {"Buck", "Doe", "Empty", "Other"}

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

    # Load data
    print("\n[train] Loading data from database...", flush=True)
    db = TrailCamDatabase()
    cursor = db.conn.cursor()

    # Get photos with species tags and visible boxes
    cursor.execute("""
        SELECT DISTINCT p.id, p.file_path, t.tag_name
        FROM photos p
        JOIN tags t ON p.id = t.photo_id
        JOIN annotation_boxes b ON p.id = b.photo_id
        WHERE (b.label IN ('subject', 'ai_animal', 'ai_subject') OR b.label LIKE 'ai_%')
          AND b.y1 < 0.95
    """)

    photo_data = {}
    for row in cursor.fetchall():
        photo_id, file_path, tag = row

        # Skip non-species tags
        if tag in SKIP_TAGS:
            continue

        # Map to class
        if tag in INDIVIDUAL_SPECIES:
            species = tag
        elif tag in BIRD_SPECIES:
            species = "Other Bird"
        else:
            continue  # Skip unknown

        if photo_id not in photo_data:
            photo_data[photo_id] = {"file_path": file_path, "species": species}

    # Get boxes for each photo
    samples = []
    for photo_id, data in photo_data.items():
        boxes = db.get_boxes(photo_id)
        boxes = [b for b in boxes if b.get("label") in ("subject", "ai_animal", "ai_subject")
                 or str(b.get("label", "")).startswith("ai_")]
        boxes = [b for b in boxes if b.get("y1", 0) < 0.95]  # Skip timestamp boxes
        if boxes:
            samples.append({
                "photo_id": photo_id,
                "file_path": data["file_path"],
                "species": data["species"],
                "box": boxes[0]
            })

    db.close()

    if not samples:
        print("[train] No samples found!")
        sys.exit(1)

    # Print distribution
    species_counts = Counter(s["species"] for s in samples)
    print(f"\n[train] Found {len(samples)} samples:", flush=True)
    for species, count in sorted(species_counts.items(), key=lambda x: -x[1]):
        print(f"  {species}: {count}", flush=True)

    # Filter classes with minimum samples
    MIN_SAMPLES = 5
    valid_species = {s for s, c in species_counts.items() if c >= MIN_SAMPLES}
    samples = [s for s in samples if s["species"] in valid_species]
    species_counts = Counter(s["species"] for s in samples)

    print(f"\n[train] After filtering (min {MIN_SAMPLES} samples): {len(samples)} samples", flush=True)

    # Build class list
    classes = sorted(species_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[train] Classes ({len(classes)}): {classes}", flush=True)

    # Split into train/val
    random.seed(42)
    random.shuffle(samples)
    val_size = int(len(samples) * VAL_SPLIT)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    print(f"[train] Train: {len(train_samples)}, Val: {len(val_samples)}", flush=True)

    # Pre-load crops
    print("\n[train] Pre-loading crops...", flush=True)

    def load_crop(sample):
        try:
            img = Image.open(sample["file_path"]).convert("RGB")
        except Exception:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        box = sample["box"]
        w, h = img.size
        x1 = max(0, int(box["x1"] * w))
        y1 = max(0, int(box["y1"] * h))
        x2 = min(w, int(box["x2"] * w))
        y2 = min(h, int(box["y2"] * h))
        return img.crop((x1, y1, x2, y2))

    train_crops = []
    train_labels = []
    for i, sample in enumerate(train_samples):
        train_crops.append(load_crop(sample))
        train_labels.append(class_to_idx[sample["species"]])
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i+1}/{len(train_samples)}...", flush=True)

    val_crops = []
    val_labels = []
    for sample in val_samples:
        val_crops.append(load_crop(sample))
        val_labels.append(class_to_idx[sample["species"]])

    print(f"[train] Cached {len(train_crops)} train + {len(val_crops)} val crops", flush=True)

    # Dataset class
    class CropDataset(Dataset):
        def __init__(self, crops, labels, transform=None):
            self.crops = crops
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.crops)

        def __getitem__(self, idx):
            img = self.crops[idx]
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

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

    train_ds = CropDataset(train_crops, train_labels, train_transform)
    val_ds = CropDataset(val_crops, val_labels, val_transform)

    # Square root weighted sampling
    print("\n[train] Computing sqrt-weighted sampler...", flush=True)
    train_class_counts = Counter(train_labels)

    # Sqrt transform: weight = 1 / sqrt(count)
    class_weights = {}
    for cls_idx, count in train_class_counts.items():
        class_weights[cls_idx] = 1.0 / math.sqrt(count)

    # Normalize weights
    total_weight = sum(class_weights.values())
    for cls_idx in class_weights:
        class_weights[cls_idx] /= total_weight

    # Print effective weights
    print("[train] Sqrt-weighted class distribution:", flush=True)
    for cls_idx, weight in sorted(class_weights.items(), key=lambda x: -train_class_counts[x[0]]):
        cls_name = classes[cls_idx]
        count = train_class_counts[cls_idx]
        effective_pct = weight * 100
        print(f"  {cls_name}: {count} samples -> {effective_pct:.1f}% effective weight", flush=True)

    # Sample weights for each training sample
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    num_classes = len(classes)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model.to(device)
    print(f"\n[train] Model: {MODEL_NAME} ({num_classes} classes)", flush=True)

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
        per_class = {}
        for cls_idx in range(num_classes):
            if per_class_total[cls_idx] > 0:
                per_class[classes[cls_idx]] = per_class_correct[cls_idx] / per_class_total[cls_idx]
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

        # Compact per-class summary (top 5)
        sorted_pc = sorted(per_class.items(), key=lambda x: -species_counts.get(x[0], 0))[:5]
        pc_str = " | ".join([f"{c[:6]}:{v:.0%}" for c, v in sorted_pc])
        print(f"[train] Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} val_acc={acc:.1%} [{pc_str}]", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "training/outputs/species_all_best.pt")

    print(f"\n[train] Training complete! Best val accuracy: {best_acc:.1%}", flush=True)

    # Export to ONNX
    print("\n[train] Exporting to ONNX...", flush=True)
    model.load_state_dict(torch.load("training/outputs/species_all_best.pt", map_location=device, weights_only=True))
    model.eval()
    model.to("cpu")

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    export_path = "training/outputs/species_all.onnx"
    torch.onnx.export(
        model, dummy, export_path,
        input_names=["input"], output_names=["logits"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
    )
    print(f"[train] Exported -> {export_path}", flush=True)

    # Save labels
    labels_path = "training/outputs/species_all_labels.txt"
    with open(labels_path, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"[train] Labels -> {labels_path}", flush=True)

    print(f"\n[train] Done! Deploy with:")
    print(f"  cp training/outputs/species_all.onnx models/species.onnx")
    print(f"  cp training/outputs/species_all_labels.txt models/labels.txt")


if __name__ == "__main__":
    main()

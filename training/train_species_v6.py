"""
Train species classifier v6.0 with EfficientNet-B2.

Improvements over v5.0:
- EfficientNet-B2 instead of ResNet18 (better feature extraction)
- Heavier augmentation (grayscale for IR, random erasing for occlusion)
- Same weighted sampling for class imbalance

Run overnight:
    cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
    source .venv/bin/activate
    python training/train_species_v6.py

Expected runtime: ~2-3 hours on Apple Metal GPU
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

    # Configuration - CHANGED for v6.0
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Smaller for larger model
    EPOCHS = 50
    LR = 5e-4  # Slightly lower for EfficientNet
    MODEL_NAME = "efficientnet_b2"  # UPGRADED from resnet18
    VAL_SPLIT = 0.15

    # Species mapping - same as v5.0
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
        "Person": None,        # Excluded - MegaDetector handles
        "Vehicle": None,       # Excluded - MegaDetector handles
        "Dog": None,
        "Quail": None,
        "Armadillo": None,
        "Chipmunk": None,
        "Skunk": None,
        "Ground Hog": None,
        "Flicker": None,
        "Turkey Buzzard": None,
        "Other Bird": None,
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

    cursor = db.conn.cursor()
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

    photo_data = {}
    for row in cursor.fetchall():
        photo_id = row[0]
        file_path = row[1]
        species = SPECIES_MAP.get(row[2])
        if species is None:
            continue
        if photo_id not in photo_data:
            photo_data[photo_id] = {"file_path": file_path, "species": species}

    # IoU filtering functions
    def calc_iou(box1, box2):
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
        if len(boxes) <= 1:
            return boxes
        boxes = sorted(boxes, key=lambda b: (b["x2"]-b["x1"])*(b["y2"]-b["y1"]), reverse=True)
        keep = []
        for box in boxes:
            dominated = False
            for kept_box in keep:
                if calc_iou(box, kept_box) > iou_threshold:
                    dominated = True
                    break
            if not dominated:
                keep.append(box)
        return keep

    # Build samples
    samples = []
    boxes_before = 0
    boxes_after = 0
    for photo_id, data in photo_data.items():
        boxes = db.get_boxes(photo_id)
        boxes = [b for b in boxes if b.get("label") in ("subject", "ai_animal")]
        boxes_before += len(boxes)
        boxes = filter_overlapping_boxes(boxes, iou_threshold=0.5)
        boxes_after += len(boxes)
        for box in boxes:
            samples.append({
                "photo_id": photo_id,
                "file_path": data["file_path"],
                "species": data["species"],
                "box": box
            })

    print(f"[train] Filtered overlapping boxes: {boxes_before} -> {boxes_after}", flush=True)
    db.close()

    if not samples:
        print("[train] No samples found!", flush=True)
        sys.exit(1)

    # Print distribution
    species_counts = Counter(s["species"] for s in samples)
    print(f"\n[train] Found {len(samples)} samples:", flush=True)
    for species, count in sorted(species_counts.items(), key=lambda x: -x[1]):
        print(f"  {species}: {count}", flush=True)

    classes = sorted(species_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"\n[train] Classes ({len(classes)}): {classes}", flush=True)

    # Stratified split
    TEST_SPLIT = 0.10
    random.seed(42)

    species_to_samples = {}
    for sample in samples:
        species = sample["species"]
        if species not in species_to_samples:
            species_to_samples[species] = []
        species_to_samples[species].append(sample)

    train_samples, val_samples, test_samples = [], [], []

    print(f"\n[train] Stratified split by species:", flush=True)
    for species in sorted(species_to_samples.keys()):
        species_samples = species_to_samples[species]
        random.shuffle(species_samples)
        n = len(species_samples)
        n_test = max(1, int(n * TEST_SPLIT))
        n_val = max(1, int((n - n_test) * VAL_SPLIT))
        n_train = n - n_test - n_val

        if n < 3:
            print(f"  {species}: {n} total (all in train)", flush=True)
            train_samples.extend(species_samples)
            continue

        test_samples.extend(species_samples[:n_test])
        val_samples.extend(species_samples[n_test:n_test + n_val])
        train_samples.extend(species_samples[n_test + n_val:])
        print(f"  {species}: {n_train} train / {n_val} val / {n_test} test", flush=True)

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    print(f"\n[train] Total: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test", flush=True)

    # Pre-load crops
    print("\n[train] Pre-loading crops...", flush=True)

    def load_crop(sample):
        try:
            with Image.open(sample["file_path"]) as img:
                img = img.convert("RGB")
                box = sample["box"]
                w, h = img.size
                x1, y1 = int(box["x1"] * w), int(box["y1"] * h)
                x2, y2 = int(box["x2"] * w), int(box["y2"] * h)
                x1, x2 = max(0, min(x1, w-1)), max(x1+1, min(x2, w))
                y1, y2 = max(0, min(y1, h-1)), max(y1+1, min(y2, h))
                return img.crop((x1, y1, x2, y2)).copy()
        except Exception:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

    train_crops, train_labels = [], []
    for i, sample in enumerate(train_samples):
        train_crops.append(load_crop(sample))
        train_labels.append(class_to_idx[sample["species"]])
        if (i + 1) % 500 == 0:
            print(f"  Loaded {i+1}/{len(train_samples)} train crops...", flush=True)

    val_crops = [load_crop(s) for s in val_samples]
    val_labels = [class_to_idx[s["species"]] for s in val_samples]
    test_crops = [load_crop(s) for s in test_samples]
    test_labels = [class_to_idx[s["species"]] for s in test_samples]

    print(f"[train] Cached {len(train_crops)} train + {len(val_crops)} val + {len(test_crops)} test crops", flush=True)

    # Dataset
    class CachedCropDataset(Dataset):
        def __init__(self, crops, labels, transform=None):
            self.crops = crops
            self.labels = labels
            self.transform = transform
            self.targets = labels

        def __len__(self):
            return len(self.crops)

        def __getitem__(self, idx):
            img = self.crops[idx]
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

    # IMPROVED augmentation for v6.0
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomGrayscale(p=0.25),  # NEW: Simulate IR cameras
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15),  # NEW: Simulate occlusion
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CachedCropDataset(train_crops, train_labels, train_transform)
    val_ds = CachedCropDataset(val_crops, val_labels, val_transform)
    test_ds = CachedCropDataset(test_crops, test_labels, val_transform)

    # Weighted sampling (square root weighting)
    train_class_counts = Counter(train_labels)
    class_weights = {}
    for cls_idx in range(len(classes)):
        count = train_class_counts.get(cls_idx, 1)
        class_weights[cls_idx] = 1.0 / (count ** 0.5)

    weight_sum = sum(class_weights.values())
    for cls_idx in class_weights:
        class_weights[cls_idx] = class_weights[cls_idx] * len(classes) / weight_sum

    print(f"\n[train] Class weights:", flush=True)
    for cls_idx, cls_name in enumerate(classes):
        print(f"  {cls_name}: {class_weights[cls_idx]:.3f} ({train_class_counts.get(cls_idx, 0)} samples)", flush=True)

    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    num_classes = len(classes)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model.to(device)
    print(f"\n[train] Model: {MODEL_NAME} ({num_classes} classes)", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs("training/outputs", exist_ok=True)
    best_acc = 0.0

    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        per_class_correct = Counter()
        per_class_total = Counter()

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
                for pred, label in zip(preds.cpu().numpy(), y.cpu().numpy()):
                    per_class_total[label] += 1
                    if pred == label:
                        per_class_correct[label] += 1

        overall = correct / max(total, 1)
        per_class = {classes[i]: per_class_correct[i] / max(per_class_total[i], 1)
                     for i in range(num_classes)}
        return overall, per_class

    print("\n[train] Starting training...", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = epoch_loss / max(batch_count, 1)
        acc, per_class = evaluate(val_loader)

        # Compact per-class display
        weak_classes = ["Coyote", "Squirrel", "Bobcat", "Turkey"]
        weak_str = " | ".join([f"{c[:3]}:{per_class.get(c, 0):.0%}" for c in weak_classes if c in per_class])
        print(f"[train] Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} val={acc:.1%} [{weak_str}]", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "training/outputs/species_v6_best.pt")
            print(f"  -> New best! Saved.", flush=True)

    print(f"\n[train] Training complete! Best val: {best_acc:.1%}", flush=True)

    # Test evaluation
    print("\n[train] Evaluating on test set...", flush=True)
    model.load_state_dict(torch.load("training/outputs/species_v6_best.pt", map_location=device, weights_only=True))
    test_acc, test_per_class = evaluate(test_loader)

    print(f"\n[train] ========== TEST RESULTS ==========", flush=True)
    print(f"[train] Overall: {test_acc:.1%}", flush=True)
    for cls_name, acc in sorted(test_per_class.items(), key=lambda x: -x[1]):
        print(f"  {cls_name}: {acc:.0%}", flush=True)

    # Export ONNX
    print("\n[train] Exporting to ONNX...", flush=True)
    model.eval()
    model.to("cpu")
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model, dummy, "training/outputs/species_v6.onnx",
                      input_names=["input"], output_names=["logits"], opset_version=12)
    print(f"[train] Exported -> training/outputs/species_v6.onnx", flush=True)

    # Save labels
    with open("training/outputs/species_v6_labels.txt", "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")

    # Save summary
    from datetime import datetime
    with open("training/outputs/species_v6_summary.txt", "w") as f:
        f.write(f"Species Classifier v6.0 Training Summary\n")
        f.write(f"========================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Split: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test\n\n")
        f.write(f"Best validation: {best_acc:.1%}\n")
        f.write(f"TEST ACCURACY: {test_acc:.1%}\n\n")
        f.write(f"Per-class TEST accuracy:\n")
        for cls_name, acc in sorted(test_per_class.items(), key=lambda x: -x[1]):
            f.write(f"  {cls_name}: {acc:.0%}\n")

    print(f"\n[train] Done!", flush=True)
    print(f"\n[train] To deploy:", flush=True)
    print(f"  cp training/outputs/species_v6.onnx models/species.onnx", flush=True)
    print(f"  cp training/outputs/species_v6_labels.txt models/labels.txt", flush=True)


if __name__ == "__main__":
    main()

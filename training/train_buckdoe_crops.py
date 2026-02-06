"""
Train buck/doe classifier using detection box crops from database.

This script:
1. Queries photos with Buck or Doe tags AND bounding boxes
2. Pre-loads and caches all crops in memory for fast training
3. Trains a buck/doe classifier using weighted sampling for class imbalance

Usage:
    python training/train_buckdoe_crops.py

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
    BATCH_SIZE = 16  # Smaller batch for larger model
    EPOCHS = 50
    LR = 5e-4  # Slightly lower for larger model
    MODEL_NAME = "efficientnet_b2"  # Better accuracy, fits in 1-2 hours
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.10

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
    print("\n[train] Loading buck/doe data from database...", flush=True)
    db = TrailCamDatabase()

    cursor = db.conn.cursor()

    # Get boxes with sex labels directly from annotation_boxes
    # This captures both photo-level tagged photos AND box-level labeled boxes
    cursor.execute("""
        SELECT b.id, b.photo_id, p.file_path, b.sex, b.x1, b.y1, b.x2, b.y2, b.species
        FROM annotation_boxes b
        JOIN photos p ON b.photo_id = p.id
        WHERE LOWER(b.sex) IN ('buck', 'doe')
          AND b.label IN ('subject', 'ai_animal')
          AND (LOWER(b.species) = 'deer' OR b.species IS NULL OR b.species = '')
    """)

    # Build dataset entries directly from boxes with sex labels
    box_data = []
    for row in cursor.fetchall():
        box_id, photo_id, file_path, sex, x1, y1, x2, y2, species = row
        box_data.append({
            "box_id": box_id,
            "photo_id": photo_id,
            "file_path": file_path,
            "sex": sex.capitalize(),  # Normalize to Buck or Doe
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "species": species}
        })

    print(f"[train] Found {len(box_data)} boxes with Buck/Doe labels directly", flush=True)

    # IoU calculation for overlap filtering
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

    # Cache image dimensions
    image_dimensions_cache = {}

    def get_image_dimensions(file_path):
        if file_path not in image_dimensions_cache:
            try:
                with Image.open(file_path) as img:
                    image_dimensions_cache[file_path] = img.size
            except Exception:
                image_dimensions_cache[file_path] = (1920, 1080)
        return image_dimensions_cache[file_path]

    def calc_pixel_area(box, img_width, img_height):
        box_w = (box["x2"] - box["x1"]) * img_width
        box_h = (box["y2"] - box["y1"]) * img_height
        return int(box_w * box_h)

    # Build samples directly from box_data
    samples = []

    for data in box_data:
        img_w, img_h = get_image_dimensions(data["file_path"])
        pixel_area = calc_pixel_area(data["box"], img_w, img_h)
        samples.append({
            "box_id": data["box_id"],
            "photo_id": data["photo_id"],
            "file_path": data["file_path"],
            "sex": data["sex"],
            "box": data["box"],
            "pixel_area": pixel_area
        })

    print(f"[train] Built {len(samples)} samples from box-level sex labels", flush=True)

    db.close()

    if not samples:
        print("[train] No samples found with Buck/Doe tags and boxes!", flush=True)
        sys.exit(1)

    # Print distribution
    sex_counts = Counter(s["sex"] for s in samples)
    print(f"\n[train] Found {len(samples)} samples:", flush=True)
    for sex, count in sorted(sex_counts.items()):
        print(f"  {sex}: {count}", flush=True)

    imbalance_ratio = max(sex_counts.values()) / min(sex_counts.values())
    print(f"\n[train] Class imbalance ratio: {imbalance_ratio:.1f}:1", flush=True)

    # Pixel area statistics
    all_pixel_areas = sorted([s["pixel_area"] for s in samples])
    n = len(all_pixel_areas)
    print(f"\n[train] === PIXEL AREA STATISTICS ===", flush=True)
    print(f"  Min:    {all_pixel_areas[0]:,} px", flush=True)
    print(f"  Median: {all_pixel_areas[int(n*0.50)]:,} px", flush=True)
    print(f"  Max:    {all_pixel_areas[-1]:,} px", flush=True)

    # Build class list
    classes = sorted(sex_counts.keys())  # ['Buck', 'Doe']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"\n[train] Classes: {classes}", flush=True)

    # Stratified split
    random.seed(42)

    sex_to_samples = {}
    for sample in samples:
        sex = sample["sex"]
        if sex not in sex_to_samples:
            sex_to_samples[sex] = []
        sex_to_samples[sex].append(sample)

    train_samples = []
    val_samples = []
    test_samples = []

    print(f"\n[train] Stratified split:", flush=True)
    for sex in sorted(sex_to_samples.keys()):
        sex_samples = sex_to_samples[sex]
        random.shuffle(sex_samples)

        n = len(sex_samples)
        n_test = max(1, int(n * TEST_SPLIT))
        n_val = max(1, int((n - n_test) * VAL_SPLIT))
        n_train = n - n_test - n_val

        test_samples.extend(sex_samples[:n_test])
        val_samples.extend(sex_samples[n_test:n_test + n_val])
        train_samples.extend(sex_samples[n_test + n_val:])

        print(f"  {sex}: {n_train} train / {n_val} val / {n_test} test", flush=True)

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    print(f"\n[train] Total: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test", flush=True)

    # Pre-load all crops
    print("\n[train] Pre-loading and caching all crops in memory...", flush=True)

    def load_crop(sample):
        try:
            with Image.open(sample["file_path"]) as img:
                img = img.convert("RGB")
                box = sample["box"]
                w, h = img.size
                x1 = int(box["x1"] * w)
                y1 = int(box["y1"] * h)
                x2 = int(box["x2"] * w)
                y2 = int(box["y2"] * h)

                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))

                return img.crop((x1, y1, x2, y2)).copy()
        except Exception as e:
            print(f"[train] Warning: Failed to load {sample['file_path']}: {e}", flush=True)
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

    train_crops = []
    train_labels = []
    for i, sample in enumerate(train_samples):
        crop = load_crop(sample)
        train_crops.append(crop)
        train_labels.append(class_to_idx[sample["sex"]])
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(train_samples)} train crops...", flush=True)

    val_crops = []
    val_labels = []
    for sample in val_samples:
        val_crops.append(load_crop(sample))
        val_labels.append(class_to_idx[sample["sex"]])

    test_crops = []
    test_labels = []
    for sample in test_samples:
        test_crops.append(load_crop(sample))
        test_labels.append(class_to_idx[sample["sex"]])

    print(f"[train] Cached {len(train_crops)} train + {len(val_crops)} val + {len(test_crops)} test crops", flush=True)

    # Dataset class
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
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    # Transforms - heavier augmentation for small dataset
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomGrayscale(p=0.2),  # Simulate IR cameras
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),  # Random occlusion
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CachedCropDataset(train_crops, train_labels, train_transform)
    val_ds = CachedCropDataset(val_crops, val_labels, val_transform)
    test_ds = CachedCropDataset(test_crops, test_labels, val_transform)

    # Weighted sampling for class imbalance
    train_class_counts = Counter(train_labels)
    print(f"\n[train] Training class counts:", flush=True)
    for cls_idx, cls_name in enumerate(classes):
        count = train_class_counts.get(cls_idx, 0)
        print(f"  {cls_name}: {count}", flush=True)

    # Inverse frequency weighting (aggressive for imbalanced data)
    class_weights = {}
    for cls_idx in range(len(classes)):
        count = train_class_counts.get(cls_idx, 1)
        class_weights[cls_idx] = 1.0 / count

    # Normalize
    weight_sum = sum(class_weights.values())
    for cls_idx in class_weights:
        class_weights[cls_idx] = class_weights[cls_idx] * len(classes) / weight_sum

    print(f"\n[train] Inverse frequency weights (normalized):", flush=True)
    for cls_idx, cls_name in enumerate(classes):
        print(f"  {cls_name}: {class_weights[cls_idx]:.3f}", flush=True)

    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model
    num_classes = len(classes)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model.to(device)
    print(f"\n[train] Model: {MODEL_NAME} ({num_classes} classes)", flush=True)

    # Class-weighted loss as well (belt and suspenders for imbalance)
    weight_tensor = torch.tensor([class_weights[i] for i in range(len(classes))], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    os.makedirs("training/outputs", exist_ok=True)
    best_acc = 0.0
    best_balanced_acc = 0.0

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

        per_class = {}
        for cls_idx in range(num_classes):
            if per_class_total[cls_idx] > 0:
                per_class[classes[cls_idx]] = per_class_correct[cls_idx] / per_class_total[cls_idx]
            else:
                per_class[classes[cls_idx]] = 0.0

        # Balanced accuracy (average of per-class accuracies)
        balanced = sum(per_class.values()) / len(per_class) if per_class else 0.0

        return overall, balanced, per_class

    print("\n[train] Starting training...", flush=True)
    print(f"[train] Target: maximize balanced accuracy (Buck + Doe equally weighted)", flush=True)

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

        scheduler.step()

        avg_loss = epoch_loss / max(batch_count, 1)
        acc, balanced_acc, per_class = evaluate(val_loader)

        per_class_str = " | ".join([f"{c}:{v:.0%}" for c, v in per_class.items()])
        print(f"[train] Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} val={acc:.1%} balanced={balanced_acc:.1%} [{per_class_str}]", flush=True)

        # Save best by balanced accuracy (fairer metric for imbalanced data)
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_acc = acc
            torch.save(model.state_dict(), "training/outputs/buckdoe_best.pt")
            print(f"  -> New best! Saved checkpoint.", flush=True)

    print(f"\n[train] Training complete!", flush=True)
    print(f"[train] Best validation: {best_acc:.1%} overall, {best_balanced_acc:.1%} balanced", flush=True)

    # Test set evaluation
    print("\n[train] Evaluating on held-out test set...", flush=True)
    model.load_state_dict(torch.load("training/outputs/buckdoe_best.pt", map_location=device, weights_only=True))
    test_acc, test_balanced, test_per_class = evaluate(test_loader)

    print(f"\n[train] ========== TEST RESULTS ==========", flush=True)
    print(f"[train] Overall accuracy: {test_acc:.1%}", flush=True)
    print(f"[train] Balanced accuracy: {test_balanced:.1%}", flush=True)
    print(f"[train] Per-class:", flush=True)
    for cls_name, acc in test_per_class.items():
        print(f"  {cls_name}: {acc:.1%}", flush=True)

    # Export to ONNX
    print("\n[train] Exporting to ONNX...", flush=True)
    model.eval()
    model.to("cpu")

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    export_path = "training/outputs/buckdoe_v2.onnx"
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

    # Save labels (lowercase for compatibility with existing code)
    labels_path = "training/outputs/buckdoe_v2_labels.txt"
    with open(labels_path, "w") as f:
        for cls in classes:
            f.write(f"{cls.lower()}\n")
    print(f"[train] Saved labels -> {labels_path}", flush=True)

    # Save training summary
    from datetime import datetime
    summary_path = "training/outputs/buckdoe_v2_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Buck/Doe Classifier Training Summary\n")
        f.write(f"====================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Total samples: {len(samples)} (after overlap filtering)\n")
        f.write(f"Stratified split: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test\n\n")
        f.write(f"Class distribution:\n")
        for sex, count in sorted(sex_counts.items()):
            f.write(f"  {sex}: {count}\n")
        f.write(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1\n")
        f.write(f"Weighted sampling: Yes (inverse frequency)\n")
        f.write(f"Class-weighted loss: Yes\n\n")
        f.write(f"Best validation accuracy: {best_acc:.1%}\n")
        f.write(f"Best balanced validation accuracy: {best_balanced_acc:.1%}\n\n")
        f.write(f"TEST SET RESULTS:\n")
        f.write(f"  Overall accuracy: {test_acc:.1%}\n")
        f.write(f"  Balanced accuracy: {test_balanced:.1%}\n")
        for cls_name, acc in test_per_class.items():
            f.write(f"  {cls_name}: {acc:.1%}\n")
    print(f"[train] Saved training summary -> {summary_path}", flush=True)

    print(f"\n[train] Done!", flush=True)
    print(f"\n[train] To deploy:", flush=True)
    print(f"  cp training/outputs/buckdoe_v2.onnx models/buckdoe.onnx", flush=True)
    print(f"  cp training/outputs/buckdoe_v2_labels.txt models/buckdoe_labels.txt", flush=True)


if __name__ == "__main__":
    main()

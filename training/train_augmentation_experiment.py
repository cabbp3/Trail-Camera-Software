"""
Augmentation Experiment - Evaluate different augmentation techniques with replicates.

This script runs multiple training experiments to determine which augmentations
improve model performance. Each augmentation is tested with 3 different random
seeds for statistical confidence.

Usage:
    python training/train_augmentation_experiment.py

Expected runtime: 7-10 hours (21 training runs)

Results saved to: training/outputs/augmentation_experiment_results.csv
"""

import os
import sys
import csv
import time
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

def run_experiment():
    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
        from torchvision import transforms
        from PIL import Image, ImageFilter
        import numpy as np
        import timm
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        sys.exit(1)

    from database import TrailCamDatabase

    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 30  # Reduced for faster experiments (still enough to see differences)
    LR = 1e-3
    MODEL_NAME = "resnet18"
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.10

    # Experiment configuration
    SEEDS = [42, 123, 456]  # 3 replicates

    AUGMENTATION_CONFIGS = {
        "A0_baseline": {},
        "A1_grayscale": {"grayscale_prob": 0.3},
        "A2_noise": {"noise_std": 0.05},
        "A3_brightness": {"brightness_range": (0.3, 1.7)},
        "A4_blur": {"blur_prob": 0.2},
        "A5_erasing": {"erasing_prob": 0.2},
        "A6_all": {
            "grayscale_prob": 0.3,
            "noise_std": 0.05,
            "brightness_range": (0.3, 1.7),
            "blur_prob": 0.2,
            "erasing_prob": 0.2,
        },
    }

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[experiment] Using Apple Metal (MPS) GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[experiment] Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("[experiment] Using CPU")

    # Species mapping
    SPECIES_MAP = {
        "Deer": "Deer", "Turkey": "Turkey", "Raccoon": "Raccoon",
        "Rabbit": "Rabbit", "Squirrel": "Squirrel", "Coyote": "Coyote",
        "Bobcat": "Bobcat", "Opossum": "Opossum", "Fox": "Fox",
        "House Cat": "House Cat",
    }

    # Load data once (reuse across experiments)
    print("\n[experiment] Loading data from database...")
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
        photo_id, file_path, tag_name = row
        species = SPECIES_MAP.get(tag_name)
        if species and photo_id not in photo_data:
            photo_data[photo_id] = {"file_path": file_path, "species": species}

    # IoU overlap filtering
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
            if not any(calc_iou(box, kept) > iou_threshold for kept in keep):
                keep.append(box)
        return keep

    # Build samples
    samples = []
    for photo_id, data in photo_data.items():
        boxes = db.get_boxes(photo_id)
        boxes = [b for b in boxes if b.get("label") in ("subject", "ai_animal")]
        boxes = filter_overlapping_boxes(boxes, 0.5)
        for box in boxes:
            samples.append({
                "photo_id": photo_id,
                "file_path": data["file_path"],
                "species": data["species"],
                "box": box
            })

    db.close()

    species_counts = Counter(s["species"] for s in samples)
    classes = sorted(species_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"[experiment] Loaded {len(samples)} samples, {len(classes)} classes")

    # Create fixed stratified split (same for all experiments)
    def create_stratified_split(samples, seed):
        random.seed(seed)
        species_to_samples = {}
        for sample in samples:
            species = sample["species"]
            if species not in species_to_samples:
                species_to_samples[species] = []
            species_to_samples[species].append(sample)

        train_samples, val_samples, test_samples = [], [], []
        for species in sorted(species_to_samples.keys()):
            ss = species_to_samples[species][:]
            random.shuffle(ss)
            n = len(ss)
            n_test = max(1, int(n * TEST_SPLIT))
            n_val = max(1, int((n - n_test) * VAL_SPLIT))
            if n < 3:
                train_samples.extend(ss)
                continue
            test_samples.extend(ss[:n_test])
            val_samples.extend(ss[n_test:n_test + n_val])
            train_samples.extend(ss[n_test + n_val:])

        return train_samples, val_samples, test_samples

    # Use seed 42 for the data split (consistent across all augmentation experiments)
    train_samples, val_samples, test_samples = create_stratified_split(samples, seed=42)
    print(f"[experiment] Split: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test")

    # Pre-load crops
    def load_crop(sample):
        try:
            img = Image.open(sample["file_path"]).convert("RGB")
        except:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        box = sample["box"]
        w, h = img.size
        x1, y1 = int(box["x1"] * w), int(box["y1"] * h)
        x2, y2 = int(box["x2"] * w), int(box["y2"] * h)
        x1, x2 = max(0, min(x1, w-1)), max(x1+1, min(x2, w))
        y1, y2 = max(0, min(y1, h-1)), max(y1+1, min(y2, h))
        return img.crop((x1, y1, x2, y2))

    print("[experiment] Pre-loading crops...")
    train_crops = [(load_crop(s), class_to_idx[s["species"]]) for s in train_samples]
    val_crops = [(load_crop(s), class_to_idx[s["species"]]) for s in val_samples]
    test_crops = [(load_crop(s), class_to_idx[s["species"]]) for s in test_samples]
    print(f"[experiment] Loaded {len(train_crops)} train, {len(val_crops)} val, {len(test_crops)} test crops")

    # Custom augmentation transforms
    class GaussianNoise:
        def __init__(self, std=0.05):
            self.std = std
        def __call__(self, tensor):
            return tensor + torch.randn_like(tensor) * self.std

    class RandomGrayscale:
        def __init__(self, p=0.3):
            self.p = p
        def __call__(self, img):
            if random.random() < self.p:
                return img.convert("L").convert("RGB")
            return img

    class RandomBlur:
        def __init__(self, p=0.2):
            self.p = p
        def __call__(self, img):
            if random.random() < self.p:
                return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
            return img

    def build_transform(config, is_train=True):
        transform_list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]

        if is_train:
            # PIL-based augmentations (before ToTensor)
            if config.get("grayscale_prob"):
                transform_list.append(RandomGrayscale(config["grayscale_prob"]))
            if config.get("blur_prob"):
                transform_list.append(RandomBlur(config["blur_prob"]))

            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomRotation(15))

            brightness_range = config.get("brightness_range", (0.8, 1.2))
            transform_list.append(transforms.ColorJitter(
                brightness=(brightness_range[0], brightness_range[1]),
                contrast=0.2, saturation=0.2, hue=0.05
            ))

            if config.get("erasing_prob"):
                # RandomErasing happens after ToTensor
                pass

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ))

        if is_train:
            if config.get("noise_std"):
                transform_list.append(GaussianNoise(config["noise_std"]))
            if config.get("erasing_prob"):
                transform_list.append(transforms.RandomErasing(p=config["erasing_prob"]))

        return transforms.Compose(transform_list)

    # Dataset
    class CropDataset(Dataset):
        def __init__(self, crops_labels, transform):
            self.crops_labels = crops_labels
            self.transform = transform
        def __len__(self):
            return len(self.crops_labels)
        def __getitem__(self, idx):
            crop, label = self.crops_labels[idx]
            return self.transform(crop), label

    # Training function
    def train_model(config_name, config, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_transform = build_transform(config, is_train=True)
        val_transform = build_transform({}, is_train=False)

        train_ds = CropDataset(train_crops, train_transform)
        val_ds = CropDataset(val_crops, val_transform)
        test_ds = CropDataset(test_crops, val_transform)

        # Weighted sampler
        train_labels = [label for _, label in train_crops]
        train_class_counts = Counter(train_labels)
        class_weights = {i: 1.0 / (train_class_counts.get(i, 1) ** 0.5) for i in range(len(classes))}
        weight_sum = sum(class_weights.values())
        class_weights = {k: v * len(classes) / weight_sum for k, v in class_weights.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Model
        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes))
        model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

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
                         for i in range(len(classes))}
            return overall, per_class

        # Training loop
        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            val_acc, _ = evaluate(val_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()

        # Final test evaluation
        model.load_state_dict(best_state)
        test_acc, test_per_class = evaluate(test_loader)
        train_time = time.time() - start_time

        return {
            "config": config_name,
            "seed": seed,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "train_time": train_time,
            **{f"test_{species}": acc for species, acc in test_per_class.items()}
        }

    # Run all experiments
    results = []
    total_runs = len(AUGMENTATION_CONFIGS) * len(SEEDS)
    run_num = 0

    print(f"\n[experiment] Starting {total_runs} training runs...")
    print("=" * 60)

    for config_name, config in AUGMENTATION_CONFIGS.items():
        for seed in SEEDS:
            run_num += 1
            print(f"\n[{run_num}/{total_runs}] {config_name} (seed={seed})")
            result = train_model(config_name, config, seed)
            results.append(result)
            print(f"  Val: {result['best_val_acc']:.1%} | Test: {result['test_acc']:.1%} | Time: {result['train_time']:.0f}s")

    # Save results
    os.makedirs("training/outputs", exist_ok=True)
    output_path = f"training/outputs/augmentation_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    fieldnames = ["config", "seed", "best_val_acc", "test_acc", "train_time"] + [f"test_{c}" for c in classes]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[experiment] Results saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for config_name in AUGMENTATION_CONFIGS.keys():
        config_results = [r for r in results if r["config"] == config_name]
        test_accs = [r["test_acc"] for r in config_results]
        mean_acc = sum(test_accs) / len(test_accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in test_accs) / len(test_accs)) ** 0.5
        print(f"{config_name:20s}: {mean_acc:.1%} ± {std_acc:.1%}")

    print("\nRare species improvement (vs baseline):")
    baseline_results = [r for r in results if r["config"] == "A0_baseline"]
    for species in ["Coyote", "Fox", "Bobcat"]:
        baseline_mean = sum(r.get(f"test_{species}", 0) for r in baseline_results) / len(baseline_results)
        print(f"\n  {species} (baseline: {baseline_mean:.0%}):")
        for config_name in AUGMENTATION_CONFIGS.keys():
            if config_name == "A0_baseline":
                continue
            config_results = [r for r in results if r["config"] == config_name]
            species_mean = sum(r.get(f"test_{species}", 0) for r in config_results) / len(config_results)
            diff = species_mean - baseline_mean
            print(f"    {config_name}: {species_mean:.0%} ({diff:+.0%})")

if __name__ == "__main__":
    run_experiment()

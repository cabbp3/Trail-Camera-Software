"""
Train a lightweight species classifier from the current SQLite DB and export to ONNX.

Usage:
  python training/export_species_onnx.py --db ~/.trailcam/trailcam.db --out models/species.onnx --labels models/labels.txt --epochs 15

Notes:
- Reads species labels from tags on each photo (excluding buck/doe).
- Uses a simple torchvision model (resnet18) with frozen backbone and a small head.
- Adds class weighting + balanced sampling to handle label imbalance.
- Optional subject crops via a pretrained detector (--use_detector) to reduce background bias.
- Requires: torch torchvision pillow numpy onnx onnxruntime
"""
import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import sqlite3
import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import models, transforms
except Exception as exc:
    print("Missing deps. Install: python -m pip install torch torchvision pillow numpy")
    raise
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    DET_AVAILABLE = True
except Exception:
    DET_AVAILABLE = False


def load_species_from_db(db_path: Path, exclude: Tuple[str, ...] = ("buck", "doe")) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return list of (path, species) and label list."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # join photos and tags
    cur.execute("""
        SELECT p.file_path, t.tag_name
        FROM photos p
        JOIN tags t ON t.photo_id = p.id
        WHERE p.file_path != '' AND t.tag_name != ''
    """)
    rows = cur.fetchall()
    species_map = []
    labels = []
    for r in rows:
        tag = (r["tag_name"] or "").strip()
        if not tag:
            continue
        if tag.lower() in exclude:
            continue
        species_map.append((r["file_path"], tag))
        labels.append(tag)
    uniq_labels = sorted({l for l in labels})
    return species_map, uniq_labels


class SpeciesDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], img_size: int = 224, aug: bool = True):
        self.items = items
        if aug:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.25, 0.25, 0.25, 0.1),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


def _compute_sample_weights(idx_items: List[Tuple[str, int]], num_classes: int) -> Tuple[List[float], torch.Tensor]:
    """Return per-sample weights and class weights for loss."""
    counts = [0] * num_classes
    for _, lbl in idx_items:
        counts[lbl] += 1
    class_weights = []
    for c in counts:
        class_weights.append(1.0 / max(c, 1))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
    sample_weights = [class_weights[lbl] for _, lbl in idx_items]
    return sample_weights, class_weights_t


def train_model(items: List[Tuple[str, str]], labels: List[str], epochs: int = 15, batch: int = 32, lr_head: float = 1e-3, lr_backbone: float = 1e-4, img_size: int = 224):
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    idx_items = [(p, label_to_idx[lbl]) for p, lbl in items if lbl in label_to_idx]
    if not idx_items:
        raise RuntimeError("No training items found.")
    # Split train/val (15% val)
    split = int(len(idx_items) * 0.15)
    val_items = idx_items[:split]
    train_items = idx_items[split:] if split > 0 else idx_items
    train_ds = SpeciesDataset(train_items, img_size=img_size, aug=True)
    val_ds = SpeciesDataset(val_items, img_size=img_size, aug=False) if val_items else None
    sample_weights, class_weights = _compute_sample_weights(train_items, len(labels))
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    dl = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2) if val_ds else None
    # Use GPU if available: CUDA (NVIDIA) or MPS (Apple Silicon)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[species] Using device: {device}")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Unfreeze last block + head
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("layer4") or name.startswith("fc")
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt = optim.Adam([
        {"params": model.fc.parameters(), "lr": lr_head},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")], "lr": lr_backbone},
    ])

    best_state = None
    best_val = float("inf")
    for ep in range(epochs):
        total = 0.0
        count = 0
        for imgs, y in dl:
            imgs = imgs.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            total += loss.item() * imgs.size(0)
            count += imgs.size(0)
        train_loss = total / max(count, 1)
        val_loss = None
        if val_dl:
            model.eval()
            vtotal = 0.0
            vcount = 0
            with torch.no_grad():
                for vi, vy in val_dl:
                    vi = vi.to(device)
                    vy = vy.to(device)
                    vo = model(vi)
                    vloss = crit(vo, vy)
                    vtotal += vloss.item() * vi.size(0)
                    vcount += vi.size(0)
            val_loss = vtotal / max(vcount, 1)
            model.train()
        print(f"[species] epoch {ep+1}/{epochs} train_loss={train_loss:.4f}" + (f" val_loss={val_loss:.4f}" if val_loss is not None else ""))
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return model, labels


def _detect_and_crop(paths: List[str], tmp_dir: Path, conf_thresh: float = 0.6, min_size: int = 64) -> List[str]:
    """Run a pretrained detector to crop subjects; returns list of crop paths."""
    if not DET_AVAILABLE:
        print("[species] Detector not available; using full images.")
        return paths
    # Use GPU for detector if available
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    try:
        det = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
        det.eval()
    except Exception as exc:
        print(f"[species] Detector load failed ({exc}); using full images.")
        return paths
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    crops = []
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = tf(img).to(device)
            with torch.no_grad():
                out = det([tensor])[0]
            keep = [i for i, s in enumerate(out["scores"]) if float(s) >= conf_thresh]
            if not keep:
                continue
            # Take top-1 box
            i = keep[0]
            box = out["boxes"][i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            x1 = max(x1, 0); y1 = max(y1, 0); x2 = min(x2, img.width); y2 = min(y2, img.height)
            if x2 - x1 < min_size or y2 - y1 < min_size:
                continue
            crop = img.crop((x1, y1, x2, y2))
            out_path = tmp_dir / f"{Path(p).stem}_crop.jpg"
            crop.save(out_path, "JPEG", quality=90)
            crops.append(str(out_path))
        except Exception:
            continue
    # Fallback: if detector failed a lot, return originals
    if len(crops) < max(1, int(0.1 * len(paths))):
        print("[species] Too few crops detected; using original images.")
        return paths
    print(f"[species] Crops created: {len(crops)} / {len(paths)}")
    return crops


def export_onnx(model, labels: List[str], out_path: Path, labels_path: Path, img_size: int = 224):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8") as f:
        for lbl in labels:
            f.write(lbl + "\n")
    # Export on CPU for best ONNX compatibility
    model = model.to("cpu")
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device="cpu")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
    )
    print(f"[species] Exported ONNX -> {out_path}")
    print(f"[species] Labels -> {labels_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path.home() / ".trailcam" / "trailcam.db"))
    ap.add_argument("--out", default="models/species.onnx")
    ap.add_argument("--labels", default="models/labels.txt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--use_detector", action="store_true", help="Use a pretrained detector to crop subjects before training.")
    ap.add_argument("--det_conf", type=float, default=0.6, help="Detector confidence threshold.")
    args = ap.parse_args()

    items, labels = load_species_from_db(Path(args.db))
    if not labels:
        raise RuntimeError("No species labels found in DB.")
    print(f"[species] Found {len(labels)} labels: {labels}")
    print(f"[species] Training on {len(items)} labeled photos")
    if args.use_detector:
        tmp_dir = Path(tempfile.mkdtemp(prefix="species_crops_"))
        cropped = _detect_and_crop([p for p, _ in items], tmp_dir, conf_thresh=args.det_conf)
        # remap items to crops where available
        crop_map = {Path(c).stem.replace("_crop", ""): c for c in cropped}
        new_items = []
        for p, lbl in items:
            stem = Path(p).stem
            if stem in crop_map:
                new_items.append((crop_map[stem], lbl))
            else:
                new_items.append((p, lbl))
        items = new_items
        print(f"[species] Using detector crops where available.")
    model, labels = train_model(items, labels, epochs=args.epochs, batch=args.batch, img_size=args.img_size)
    export_onnx(model, labels, Path(args.out), Path(args.labels), img_size=args.img_size)


if __name__ == "__main__":
    main()

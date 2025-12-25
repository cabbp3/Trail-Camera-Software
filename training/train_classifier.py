"""
Train an image classifier (species/empty or buck/doe).

Supports both config file and command-line arguments.
Handles class imbalance via weighted sampling and loss weighting.

Requires: torch, torchvision, timm, pyyaml.
"""
import argparse
import os
import sys
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--config", default=None, help="Path to classifier config yaml (optional)")
    parser.add_argument("--train_dir", default=None, help="Training data directory (class subfolders)")
    parser.add_argument("--val_dir", default=None, help="Validation data directory")
    parser.add_argument("--out", default="outputs/classifier.onnx", help="Output ONNX path")
    parser.add_argument("--labels", default="outputs/classifier_labels.txt", help="Output labels path")
    parser.add_argument("--model", default="convnext_tiny", help="timm model name")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--weighted", action="store_true", help="Use class weighting for imbalanced data")
    parser.add_argument("--augment", action="store_true", default=True, help="Use data augmentation")
    args = parser.parse_args()

    # Load config file if provided
    cfg = {}
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    # Command line args override config
    train_dir = args.train_dir or cfg.get("train_dir")
    val_dir = args.val_dir or cfg.get("val_dir")
    if not train_dir or not val_dir:
        print("Error: Must specify --train_dir and --val_dir (or use --config)")
        sys.exit(1)

    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torchvision import datasets, transforms
        import timm
    except Exception as exc:
        print("Missing deps. Install: python3 -m pip install torch torchvision timm pyyaml")
        print(f"Details: {exc}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[classifier] Device: {device}")

    img_size = args.img_size or cfg.get("img_size", 224)
    epochs = args.epochs or cfg.get("epochs", 25)
    batch_size = args.batch or cfg.get("batch", 32)
    lr = args.lr or cfg.get("lr", 1e-3)
    model_name = args.model or cfg.get("model_name", "convnext_tiny")
    use_weighted = args.weighted or cfg.get("weighted", False)

    # Transforms with augmentation
    if args.augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    print(f"[classifier] Classes: {train_ds.classes}")
    print(f"[classifier] Train samples: {len(train_ds)}")
    print(f"[classifier] Val samples: {len(val_ds)}")

    # Count class distribution
    class_counts = Counter(train_ds.targets)
    for cls_idx, count in sorted(class_counts.items()):
        print(f"  {train_ds.classes[cls_idx]}: {count}")

    # Weighted sampling for class imbalance
    if use_weighted:
        print("[classifier] Using weighted sampling for class imbalance")
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[t] for t in train_ds.targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

        # Also weight the loss
        weight_tensor = torch.tensor([1.0 / class_counts[i] for i in range(len(train_ds.classes))], device=device)
        weight_tensor = weight_tensor / weight_tensor.sum() * len(train_ds.classes)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_ds.classes)
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)
    print(f"[classifier] Model: {model_name} ({num_classes} classes)")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 1e-4))
    best_acc = 0.0
    os.makedirs("outputs", exist_ok=True)

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
        return correct / max(total, 1)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        acc = evaluate()
        print(f"[classifier] Epoch {epoch+1}/{epochs} val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "outputs/classifier_best.pt")

    # Export ONNX
    export_path = args.out or cfg.get("export_path", "outputs/classifier.onnx")
    labels_out = args.labels or cfg.get("labels_out", "outputs/classifier_labels.txt")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else "outputs", exist_ok=True)
    os.makedirs(os.path.dirname(labels_out) if os.path.dirname(labels_out) else "outputs", exist_ok=True)

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    model.load_state_dict(torch.load("outputs/classifier_best.pt", map_location=device))
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        export_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
    )
    print(f"[classifier] Exported ONNX -> {export_path}")

    with open(labels_out, "w", encoding="utf-8") as f:
        for cls in train_ds.classes:
            f.write(f"{cls}\n")
    print(f"[classifier] Wrote labels -> {labels_out}")

    print(f"\n[classifier] Done! Best val accuracy: {best_acc:.3f}")
    print(f"[classifier] Model: {export_path}")
    print(f"[classifier] Labels: {labels_out}")


if __name__ == "__main__":
    main()

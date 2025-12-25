"""
Multi-head antler count trainer for per-antler typical/abnormal counts.

Input CSV format:
path,left_typical,right_typical,left_abnormal,right_abnormal,left_typical_uncertain,right_typical_uncertain,left_abnormal_uncertain,right_abnormal_uncertain
(uncertain columns optional; use 0/1 if present)

Model: shared backbone (timm) with four small classification heads (0-20 bins) predicting counts.
Exports ONNX for app integration.
"""
import argparse
import os
import sys
import yaml
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Training CSV with columns path,left_typical,right_typical,left_abnormal,right_abnormal")
    parser.add_argument("--val_csv", required=False, help="Optional validation CSV")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_name", default="resnet34")
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--export_path", default="outputs/antler_heads.onnx")
    args = parser.parse_args()

    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        import timm
    except Exception as exc:
        print("Missing deps. Install: python3 -m pip install torch torchvision timm pandas")
        print(f"Details: {exc}")
        sys.exit(1)

    class AntlerDataset(Dataset):
        def __init__(self, csv_path, img_size):
            df = pd.read_csv(csv_path)
            self.paths = df["path"].tolist()
            self.lt = df["left_typical"].fillna(-1).astype(int).tolist()
            self.rt = df["right_typical"].fillna(-1).astype(int).tolist()
            self.la = df["left_abnormal"].fillna(-1).astype(int).tolist()
            self.ra = df["right_abnormal"].fillna(-1).astype(int).tolist()
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
            ])

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            from PIL import Image
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.tf(img), (
                self.lt[idx],
                self.rt[idx],
                self.la[idx],
                self.ra[idx],
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = AntlerDataset(args.csv, args.img_size)
    val_ds = AntlerDataset(args.val_csv, args.img_size) if args.val_csv else None
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True) if val_ds else None

    num_classes = 21  # predict 0-20 points; adjust if needed
    backbone = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    feats = backbone.num_features
    heads = nn.ModuleDict({
        "lt": nn.Linear(feats, num_classes),
        "rt": nn.Linear(feats, num_classes),
        "la": nn.Linear(feats, num_classes),
        "ra": nn.Linear(feats, num_classes),
    })
    model = nn.Sequential(backbone, heads).to(device)

    def forward(x):
        features = backbone(x)
        return {
            "lt": heads["lt"](features),
            "rt": heads["rt"](features),
            "la": heads["la"](features),
            "ra": heads["ra"](features),
        }

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(list(backbone.parameters()) + list(heads.parameters()), lr=args.lr, weight_decay=1e-4)

    def run_epoch(loader, train=True):
        if not loader:
            return 0.0
        total = 0
        count = 0
        if train:
            model.train()
        else:
            model.eval()
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = [t.to(device) for t in targets]
            if train:
                optimizer.zero_grad()
            outputs = forward(imgs)
            loss = 0
            for key, tgt in zip(["lt", "rt", "la", "ra"], targets):
                loss += criterion(outputs[key], tgt)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item() * imgs.size(0)
            count += imgs.size(0)
        return total / max(count, 1)

    os.makedirs("outputs", exist_ok=True)
    best_val = 1e9
    for epoch in range(args.epochs):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False) if val_loader else 0.0
        print(f"[antler_heads] Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.3f} val_loss={val_loss:.3f}")
        if val_loader and val_loss < best_val:
            best_val = val_loss
            torch.save({"backbone": backbone.state_dict(), "heads": heads.state_dict()}, "outputs/antler_heads_best.pt")

    if args.export_onnx:
        # load best if available
        if os.path.exists("outputs/antler_heads_best.pt"):
            ckpt = torch.load("outputs/antler_heads_best.pt", map_location=device)
            backbone.load_state_dict(ckpt["backbone"])
            heads.load_state_dict(ckpt["heads"])
        backbone.eval()
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        class Wrapper(nn.Module):
            def __init__(self, backbone, heads):
                super().__init__()
                self.backbone = backbone
                self.heads = heads
            def forward(self, x):
                feats = self.backbone(x)
                return (
                    self.heads["lt"](feats),
                    self.heads["rt"](feats),
                    self.heads["la"](feats),
                    self.heads["ra"](feats),
                )
        wrapper = Wrapper(backbone, heads).to(device)
        torch.onnx.export(
            wrapper,
            dummy,
            args.export_path,
            input_names=["input"],
            output_names=["lt", "rt", "la", "ra"],
            opset_version=12,
        )
        print(f"[antler_heads] Exported ONNX -> {args.export_path}")


if __name__ == "__main__":
    main()

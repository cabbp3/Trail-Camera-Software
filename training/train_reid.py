"""
Train a deer re-ID embedding model (metric learning).
Requires: torch, torchvision, timm, pytorch-metric-learning, pyyaml, pandas.
"""
import argparse
import os
import sys
import yaml
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reid.yaml", help="Path to re-id config yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        import timm
        from pytorch_metric_learning import losses
    except Exception as exc:
        print("Missing deps. Install: python3 -m pip install torch torchvision timm pytorch-metric-learning pyyaml pandas")
        print(f"Details: {exc}")
        sys.exit(1)

    class ReidDataset(Dataset):
        def __init__(self, csv_path, img_size):
            df = pd.read_csv(csv_path)
            self.paths = df["path"].tolist()
            self.ids = df["deer_id"].astype("category")
            self.labels = torch.tensor(self.ids.cat.codes.values, dtype=torch.long)
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
            return self.tf(img), self.labels[idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = cfg.get("img_size", 224)
    train_ds = ReidDataset(cfg["csv_path"], img_size)
    val_ds = ReidDataset(cfg["val_csv"], img_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch", 64), shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch", 64), shuffle=False, num_workers=4, pin_memory=True)

    num_ids = len(train_ds.ids.cat.categories)
    backbone = timm.create_model(cfg.get("model_name", "resnet50"), pretrained=True, num_classes=0)
    embed_dim = cfg.get("embedding_dim", 256)
    projector = nn.Linear(backbone.num_features, embed_dim)
    model = nn.Sequential(backbone, projector).to(device)

    criterion = losses.TripletMarginLoss(margin=cfg.get("margin", 0.2))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4))

    def evaluate():
        # Simple validation: average positive vs negative distance
        model.eval()
        import torch.nn.functional as F
        with torch.no_grad():
            embs = []
            labels = []
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                z = model(x)
                z = F.normalize(z, dim=1)
                embs.append(z.cpu())
                labels.append(y.cpu())
            if not embs:
                return 0.0
            embs = torch.cat(embs, dim=0)
            labels = torch.cat(labels, dim=0)
            sims = embs @ embs.T
            same = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos = sims[same & (~torch.eye(len(sims), dtype=bool))]
            neg = sims[~same]
            return float(pos.mean() - neg.mean()) if len(pos) and len(neg) else 0.0

    epochs = cfg.get("epochs", 30)
    best_metric = -1e9
    os.makedirs("outputs", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            embeds = model(x)
            loss = criterion(embeds, y)
            loss.backward()
            optimizer.step()
        metric = evaluate()
        print(f"[reid] Epoch {epoch+1}/{epochs} metric(pos-neg mean sim)={metric:.4f}")
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), "outputs/reid_best.pt")

    if cfg.get("export_onnx", True):
        model.load_state_dict(torch.load("outputs/reid_best.pt", map_location=device))
        model.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        export_path = cfg.get("export_path", "outputs/reid.onnx")
        torch.onnx.export(
            model,
            dummy,
            export_path,
            input_names=["input"],
            output_names=["embedding"],
            opset_version=12,
        )
        print(f"[reid] Exported ONNX -> {export_path}")

    print("[reid] Done. outputs/ has artifacts.")


if __name__ == "__main__":
    main()

"""
Train a YOLO detector for species + empty (and optionally person/vehicle).
Requires: ultralytics, torch, pyyaml.
"""
import argparse
import os
import shutil
import sys
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detector.yaml", help="Path to detector config yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("Ultralytics not installed. Install with: python3 -m pip install ultralytics")
        print(f"Details: {exc}")
        sys.exit(1)

    classes = cfg["classes"]
    data_yaml = {
        "path": ".",  # unused; we point directly to abs paths
        "train": cfg["train_images"],
        "val": cfg["val_images"],
        "names": {i: name for i, name in enumerate(classes)},
    }

    # Persist a temporary data yaml
    os.makedirs("outputs", exist_ok=True)
    data_path = "outputs/detector_data.yaml"
    with open(data_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f)

    model = YOLO(cfg.get("model_name", "yolov8n.pt"))
    results = model.train(
        data=data_path,
        epochs=cfg.get("epochs", 50),
        imgsz=cfg.get("img_size", 640),
        batch=cfg.get("batch", 16),
        project="outputs",
        name="detector_runs",
        exist_ok=True,
        pretrained=True,
    )

    # Take the last/best model for export
    last_weights = os.path.join("outputs", "detector_runs", "weights", "last.pt")
    best_weights = os.path.join("outputs", "detector_runs", "weights", "best.pt")
    chosen = best_weights if os.path.exists(best_weights) else last_weights

    if cfg.get("export_onnx", True):
        onnx_out = cfg.get("export_path", "outputs/detector.onnx")
        model = YOLO(chosen)
        model.export(format="onnx", imgsz=cfg.get("img_size", 640), opset=12, dynamic=False, simplify=True, filename=onnx_out)
        print(f"[detector] Exported ONNX -> {onnx_out}")

        labels_out = cfg.get("labels_out", "outputs/detector_labels.txt")
        with open(labels_out, "w", encoding="utf-8") as f:
            for name in classes:
                f.write(f"{name}\n")
        print(f"[detector] Wrote labels -> {labels_out}")

    print("[detector] Done. Training logs are under outputs/detector_runs")


if __name__ == "__main__":
    main()

# Training Companion

This folder is a standalone, code-only companion to train and export models that plug into the main app (no downloads or training are run automatically).

## Goals
- Species/empty detector or classifier (better suggested tags).
- Buck vs doe classifier.
- Deer re-ID embeddings for individual identification across seasons.

## Layout
- `configs/` – YAML configs to point at your data and set hyperparameters.
- `train_detector.py` – YOLO-based detector for species + empty (+ optional buck/doe attribute).
- `train_classifier.py` – Image classifier for species/empty or buck/doe on crops.
- `train_reid.py` – Metric-learning embeddings for individual deer ID.
- `export_to_app.py` – Copy/rename ONNX exports into `../models/` for the main app.
- `train_antler_heads.py` – Multi-head model to predict per-antler typical/abnormal counts (future AI tagging).

## Data expectations
- Detector: YOLO-format dataset (images + `labels/*.txt`). Classes in config `classes: ["deer","turkey","coyote","raccoon","empty"]` etc.
- Classifier: Folder-per-class (`data/train/deer`, `data/train/empty`, etc.) or buck/doe crops.
- Re-ID: A CSV with `path,deer_id` (head/shoulder crops are best). Optional `season` column for reporting.

## Dependencies (install when you’re ready to train)
```bash
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics torch torchvision timm pytorch-metric-learning pyyaml onnx onnxruntime
```
> If you only train detector/classifier, you can skip `pytorch-metric-learning`.

## Workflow
1) Prepare data and edit the relevant config in `configs/`.
2) Run a trainer, e.g. `python3 train_detector.py --config configs/detector.yaml`.
3) ONNX exports land in `training/outputs/…`.
4) Copy into the app: `python3 export_to_app.py --detector outputs/detector.onnx --labels outputs/detector_labels.txt` (and similarly for classifier/reid).
5) Launch the app; `ai_suggester` will prefer your ONNX classifier when `models/species.onnx` exists.

## Notes
- Keep a held-out val split; watch per-class precision/recall (detector) and top-1/top-5 (classifier), rank-1/mAP (re-ID).
- Include night IR, motion blur, false triggers, and off-season examples.
- Re-ID is the hardest: more views per deer across seasons means better generalization. Add new seasons and periodically retrain. Keep an audit log of model versions used.

"""
Run species classifier on unlabeled photos with bounding boxes.

Stores predictions in the ai_suggestions table for review.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from database import TrailCamDatabase
from PIL import Image
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)


def main():
    # Load model
    model_path = "models/species.onnx"
    labels_path = "models/labels.txt"

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]

    print(f"[suggest] Loaded model with labels: {labels}")

    # Connect to database
    db = TrailCamDatabase()
    cursor = db.conn.cursor()

    # Get unlabeled photos with boxes
    cursor.execute("""
        SELECT DISTINCT p.id, p.file_path
        FROM photos p
        JOIN annotation_boxes b ON p.id = b.photo_id
        WHERE p.id NOT IN (SELECT DISTINCT photo_id FROM tags)
          AND b.label IN ('subject', 'ai_animal')
        ORDER BY p.id
    """)

    photos = cursor.fetchall()
    print(f"[suggest] Found {len(photos)} unlabeled photos with boxes")

    # Create ai_suggestions table if needed
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_suggestions (
            id INTEGER PRIMARY KEY,
            photo_id INTEGER NOT NULL,
            suggestion_type TEXT NOT NULL,
            suggested_value TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed INTEGER DEFAULT 0,
            FOREIGN KEY (photo_id) REFERENCES photos(id)
        )
    """)
    db.conn.commit()

    # Clear old species suggestions for these photos
    photo_ids = [p[0] for p in photos]
    if photo_ids:
        placeholders = ",".join("?" * len(photo_ids))
        cursor.execute(f"""
            DELETE FROM ai_suggestions
            WHERE photo_id IN ({placeholders}) AND suggestion_type = 'species'
        """, photo_ids)
        db.conn.commit()

    # Process each photo
    results = {"Deer": 0, "Turkey": 0, "Other_Mammal": 0, "Other": 0}

    for i, (photo_id, file_path) in enumerate(photos):
        if not Path(file_path).exists():
            print(f"  Skipping missing file: {file_path}")
            continue

        # Get first box
        boxes = db.get_boxes(photo_id)
        boxes = [b for b in boxes if b.get("label") in ("subject", "ai_animal")]
        if not boxes:
            continue

        box = boxes[0]

        # Load and crop image
        try:
            img = Image.open(file_path).convert("RGB")
            w, h = img.size
            x1 = int(box["x1"] * w)
            y1 = int(box["y1"] * h)
            x2 = int(box["x2"] * w)
            y2 = int(box["y2"] * h)

            # Ensure valid crop
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))

            crop = img.crop((x1, y1, x2, y2))
            crop = crop.resize((224, 224))

            # Preprocess
            arr = np.array(crop).astype("float32") / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype="float32")
            std = np.array([0.229, 0.224, 0.225], dtype="float32")
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)
            arr = np.expand_dims(arr, 0)

            # Run inference
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: arr})
            logits = outputs[0][0]

            # Softmax
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()

            idx = int(np.argmax(probs))
            label = labels[idx]
            conf = float(probs[idx])

            # Store suggestion
            cursor.execute("""
                INSERT INTO ai_suggestions (photo_id, suggestion_type, suggested_value, confidence, model_version)
                VALUES (?, 'species', ?, ?, 'species_v2.0')
            """, (photo_id, label, conf))

            results[label] = results.get(label, 0) + 1

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(photos)}...")
                db.conn.commit()

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue

    db.conn.commit()
    db.close()

    print(f"\n[suggest] Done! Predictions:")
    for label, count in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    total = sum(results.values())
    print(f"\n[suggest] Total: {total} photos with suggestions")
    print("[suggest] Review suggestions in the app or query ai_suggestions table")


if __name__ == "__main__":
    main()

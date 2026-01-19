#!/usr/bin/env python3
"""
Run AI species suggestions on Tightwad House photos using cropped detection boxes.
This gives accurate predictions by classifying animal crops, not whole images.
"""
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import TrailCamDatabase
from ai_suggester import CombinedSuggester
from PIL import Image
import numpy as np

def main():
    db = TrailCamDatabase()
    suggester = CombinedSuggester()

    if not suggester.ready:
        print("ERROR: AI model not ready. Check that models/species.onnx exists.")
        return

    print("AI model loaded successfully")

    # Get all Tightwad House photos with boxes
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT p.id, p.file_path, ab.id as box_id, ab.x1, ab.y1, ab.x2, ab.y2, ab.label,
               GROUP_CONCAT(t.tag_name, ', ') as existing_tags
        FROM photos p
        JOIN annotation_boxes ab ON p.id = ab.photo_id
        LEFT JOIN tags t ON p.id = t.photo_id
        WHERE p.collection = 'Tightwad House'
        AND p.archived = 0
        GROUP BY p.id, ab.id
        ORDER BY p.id
    ''')
    rows = cursor.fetchall()

    print(f"Found {len(rows)} boxes across Tightwad House photos to process")

    # Track predictions per photo for review queue
    photo_predictions = {}  # photo_id -> {existing_label, ai_suggestions: [(species, conf), ...]}

    for i, row in enumerate(rows):
        photo_id, file_path, box_id, x1, y1, x2, y2, box_label, existing_tags = row

        if i % 100 == 0:
            print(f"Processing box {i+1}/{len(rows)}...")

        if not os.path.exists(file_path):
            continue

        try:
            # Load and crop image to box region
            img = Image.open(file_path).convert("RGB")
            w, h = img.size

            # Convert relative coords (0-1) to pixels
            px1 = int(x1 * w)
            py1 = int(y1 * h)
            px2 = int(x2 * w)
            py2 = int(y2 * h)

            # Add small padding (5%)
            pad_w = int((px2 - px1) * 0.05)
            pad_h = int((py2 - py1) * 0.05)
            px1 = max(0, px1 - pad_w)
            py1 = max(0, py1 - pad_h)
            px2 = min(w, px2 + pad_w)
            py2 = min(h, py2 + pad_h)

            # Crop
            crop = img.crop((px1, py1, px2, py2))

            # Save temp file for classifier (it expects a path)
            temp_path = "/tmp/tightwad_crop.jpg"
            crop.save(temp_path, "JPEG", quality=90)

            # Run classifier on crop
            result = suggester.predict(temp_path)
            if result is None:
                continue

            ai_species, confidence = result

            # Update box label in database
            cursor.execute(
                "UPDATE annotation_boxes SET species = ?, confidence = ? WHERE id = ?",
                (ai_species, confidence, box_id)
            )

            # Track for review queue
            if photo_id not in photo_predictions:
                photo_predictions[photo_id] = {
                    'file_path': file_path,
                    'existing_label': existing_tags or '',
                    'ai_suggestions': []
                }
            photo_predictions[photo_id]['ai_suggestions'].append((ai_species, confidence))

        except Exception as e:
            print(f"  Error processing box {box_id}: {e}")
            continue

    db.conn.commit()
    print(f"\nProcessed {len(rows)} boxes")

    # Build review queue - photos where AI disagrees with existing label
    review_queue = []

    for photo_id, data in photo_predictions.items():
        existing = data['existing_label']
        suggestions = data['ai_suggestions']

        if not suggestions:
            continue

        # Get most common AI suggestion for this photo
        from collections import Counter
        species_counts = Counter(s[0] for s in suggestions)
        top_species, count = species_counts.most_common(1)[0]

        # Average confidence for top species
        top_confs = [c for s, c in suggestions if s == top_species]
        avg_conf = sum(top_confs) / len(top_confs)

        # Update photo's suggested_tag
        cursor.execute(
            "UPDATE photos SET suggested_tag = ?, suggested_confidence = ? WHERE id = ?",
            (top_species, avg_conf, photo_id)
        )

        # Add to review queue if AI disagrees with existing label
        if existing and top_species not in existing:
            review_queue.append({
                'photo_id': photo_id,
                'file_path': data['file_path'],
                'existing_label': existing,
                'ai_suggestion': top_species,
                'ai_confidence': avg_conf,
                'box_count': len(suggestions)
            })

    db.conn.commit()

    print(f"\nPhotos where AI disagrees: {len(review_queue)}")

    # Save review queue to JSON
    queue_path = os.path.expanduser("~/.trailcam/tightwad_review_queue.json")
    os.makedirs(os.path.dirname(queue_path), exist_ok=True)

    with open(queue_path, 'w') as f:
        json.dump(review_queue, f, indent=2)

    print(f"Review queue saved to: {queue_path}")

    # Print summary
    if review_queue:
        print("\nSample disagreements:")
        for item in review_queue[:10]:
            print(f"  Photo {item['photo_id']}: existing='{item['existing_label']}' vs AI='{item['ai_suggestion']}' ({item['ai_confidence']:.1%}) [{item['box_count']} boxes]")

    # Clean up temp file
    if os.path.exists("/tmp/tightwad_crop.jpg"):
        os.remove("/tmp/tightwad_crop.jpg")

if __name__ == '__main__':
    main()

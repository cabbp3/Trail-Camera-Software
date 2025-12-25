#!/usr/bin/env python3
"""
Run MegaDetector on photos without detection boxes.
Saves results to database and marks for review.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from database import TrailCamDatabase

def run_megadetector_on_unboxed():
    """Run MegaDetector on all photos without boxes."""

    # Load megadetector
    print("[MegaDetector] Loading model (will download on first run)...")
    from megadetector.detection import run_detector
    from megadetector.visualization import visualization_utils as vis_utils

    model = run_detector.load_detector('MDV5A')
    print("[MegaDetector] Model loaded!")

    # Connect to database
    db = TrailCamDatabase()

    # Get photos without boxes
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT id, file_path FROM photos
        WHERE id NOT IN (SELECT DISTINCT photo_id FROM annotation_boxes)
    """)
    photos = cursor.fetchall()

    print(f"[MegaDetector] Processing {len(photos)} photos without boxes...")

    processed = 0
    detections_found = 0
    errors = 0

    for i, (photo_id, file_path) in enumerate(photos):
        if not file_path or not os.path.exists(file_path):
            errors += 1
            continue

        try:
            # Load image
            image = vis_utils.load_image(file_path)

            # Run detection
            result = model.generate_detections_one_image(image)

            # Filter detections above threshold
            CONFIDENCE_THRESHOLD = 0.2
            detections = [d for d in result.get('detections', []) if d['conf'] >= CONFIDENCE_THRESHOLD]

            if detections:
                boxes = []
                for det in detections:
                    # MegaDetector format: [x, y, width, height] normalized 0-1
                    # category: 1=animal, 2=person, 3=vehicle
                    x, y, w, h = det['bbox']
                    category = det['category']
                    conf = det['conf']

                    # Map category to label
                    label_map = {1: 'ai_animal', 2: 'ai_person', 3: 'ai_vehicle'}
                    label = label_map.get(int(category), 'ai_unknown')

                    boxes.append({
                        'label': label,
                        'x1': x,
                        'y1': y,
                        'x2': x + w,
                        'y2': y + h,
                        'confidence': conf
                    })

                # Save boxes to database
                db.set_boxes(photo_id, boxes)
                detections_found += len(boxes)

            processed += 1

            # Progress update every 50 photos
            if (i + 1) % 50 == 0:
                print(f"[MegaDetector] Processed {i + 1}/{len(photos)} ({detections_found} detections so far)")

        except Exception as e:
            print(f"[MegaDetector] Error on {file_path}: {e}")
            errors += 1

    print(f"\n[MegaDetector] Complete!")
    print(f"  Processed: {processed}")
    print(f"  Detections found: {detections_found}")
    print(f"  Errors: {errors}")

    db.close()
    return processed, detections_found, errors


if __name__ == "__main__":
    run_megadetector_on_unboxed()

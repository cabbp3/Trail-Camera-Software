#!/usr/bin/env python3
"""
Run AI species suggestions on Tightwad House photos and create a review queue
for photos where AI disagrees with the existing label.
"""
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import TrailCamDatabase
from ai_suggester import CombinedSuggester

def main():
    db = TrailCamDatabase()
    suggester = CombinedSuggester()

    if not suggester.ready:
        print("ERROR: AI model not ready. Check that models/species.onnx exists.")
        return

    print("AI model loaded successfully")

    # Get all Tightwad House photos with their tags
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT p.id, p.file_path, GROUP_CONCAT(t.tag_name, ', ')
        FROM photos p
        LEFT JOIN tags t ON p.id = t.photo_id
        WHERE p.collection = 'Tightwad House'
        AND p.archived = 0
        GROUP BY p.id
    ''')
    photos = cursor.fetchall()

    print(f"Found {len(photos)} Tightwad House photos to process")

    # Review queue - photos where AI disagrees with existing label
    review_queue = []

    for i, (photo_id, file_path, existing_label) in enumerate(photos):
        if i % 50 == 0:
            print(f"Processing {i+1}/{len(photos)}...")

        if not os.path.exists(file_path):
            print(f"  Skipping {photo_id}: file not found")
            continue

        # Run AI prediction
        result = suggester.predict(file_path)
        if result is None:
            continue

        ai_species, confidence = result

        # Store suggestion in database
        db.set_suggested_tag(photo_id, ai_species, confidence)

        # Check if AI disagrees with existing label
        if existing_label and ai_species != existing_label:
            review_queue.append({
                'photo_id': photo_id,
                'file_path': file_path,
                'existing_label': existing_label,
                'ai_suggestion': ai_species,
                'ai_confidence': confidence
            })

    print(f"\nProcessing complete!")
    print(f"AI suggestions saved to database")
    print(f"Photos where AI disagrees: {len(review_queue)}")

    # Save review queue to JSON
    queue_path = os.path.expanduser("~/.trailcam/tightwad_review_queue.json")
    os.makedirs(os.path.dirname(queue_path), exist_ok=True)

    with open(queue_path, 'w') as f:
        json.dump(review_queue, f, indent=2)

    print(f"Review queue saved to: {queue_path}")

    # Print summary of disagreements
    if review_queue:
        print("\nSample disagreements:")
        for item in review_queue[:10]:
            print(f"  Photo {item['photo_id']}: existing='{item['existing_label']}' vs AI='{item['ai_suggestion']}' ({item['ai_confidence']:.1%})")

if __name__ == '__main__':
    main()

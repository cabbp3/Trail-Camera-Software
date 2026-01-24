#!/usr/bin/env python3
"""
Compare MegaDetector v5 vs v6 detection performance.

Runs both models on a random sample of photos and compares:
- Detection counts
- Confidence scores
- Inference time
- Agreement rate (both detect same objects)

Usage:
    python tools/compare_megadetector.py [--samples 50] [--conf 0.2]
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_detection import MegaDetectorV5, MegaDetectorV6


def get_sample_photos(photo_dir: Path, n_samples: int = 50) -> list:
    """Get a random sample of photos from the library."""
    photos = []
    for root, dirs, files in os.walk(photo_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                photos.append(os.path.join(root, f))

    if len(photos) <= n_samples:
        return photos
    return random.sample(photos, n_samples)


def iou(box1, box2) -> float:
    """Calculate Intersection over Union for two boxes in (x, y, w, h) format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x1, y1, x2, y2)
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    # Calculate intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compare_detections(v5_dets, v6_dets, iou_threshold=0.5):
    """Compare detections between v5 and v6."""
    matched = 0
    v5_only = 0
    v6_only = 0

    v6_matched = set()

    for d5 in v5_dets:
        best_iou = 0
        best_idx = -1
        for i, d6 in enumerate(v6_dets):
            if i in v6_matched:
                continue
            if d5.category == d6.category:
                overlap = iou(d5.bbox, d6.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = i

        if best_iou >= iou_threshold:
            matched += 1
            v6_matched.add(best_idx)
        else:
            v5_only += 1

    v6_only = len(v6_dets) - len(v6_matched)

    return matched, v5_only, v6_only


def main():
    parser = argparse.ArgumentParser(description='Compare MegaDetector v5 vs v6')
    parser.add_argument('--samples', type=int, default=50, help='Number of photos to test')
    parser.add_argument('--conf', type=float, default=0.2, help='Confidence threshold')
    parser.add_argument('--photo-dir', type=str, default=None, help='Photo directory')
    args = parser.parse_args()

    photo_dir = Path(args.photo_dir) if args.photo_dir else Path.home() / "TrailCamLibrary"

    print("=" * 60)
    print("MegaDetector v5 vs v6 Comparison")
    print("=" * 60)
    print(f"Sample size: {args.samples} photos")
    print(f"Confidence threshold: {args.conf}")
    print(f"Photo directory: {photo_dir}")
    print()

    # Load models
    print("Loading MegaDetector v5...")
    v5 = MegaDetectorV5(confidence_threshold=args.conf)
    if not v5.is_available:
        print(f"  ERROR: {v5.error_message}")
        return

    print("\nLoading MegaDetector v6...")
    v6 = MegaDetectorV6(confidence_threshold=args.conf)
    if not v6.is_available:
        print(f"  ERROR: {v6.error_message}")
        return

    # Get sample photos
    print(f"\nSelecting {args.samples} random photos...")
    photos = get_sample_photos(photo_dir, args.samples)
    print(f"Found {len(photos)} photos to test")

    # Run comparison
    results = {
        'v5_time': 0,
        'v6_time': 0,
        'v5_detections': 0,
        'v6_detections': 0,
        'v5_animals': 0,
        'v6_animals': 0,
        'matched': 0,
        'v5_only': 0,
        'v6_only': 0,
        'v5_conf_sum': 0,
        'v6_conf_sum': 0,
    }

    print("\nRunning detections...")
    for i, photo in enumerate(photos):
        print(f"\r  Processing {i+1}/{len(photos)}: {Path(photo).name[:40]:<40}", end="", flush=True)

        # Run v5
        t0 = time.time()
        v5_dets = v5.detect(photo)
        results['v5_time'] += time.time() - t0

        # Run v6
        t0 = time.time()
        v6_dets = v6.detect(photo)
        results['v6_time'] += time.time() - t0

        # Count detections
        results['v5_detections'] += len(v5_dets)
        results['v6_detections'] += len(v6_dets)

        v5_animals = [d for d in v5_dets if d.category == 'animal']
        v6_animals = [d for d in v6_dets if d.category == 'animal']
        results['v5_animals'] += len(v5_animals)
        results['v6_animals'] += len(v6_animals)

        # Sum confidences
        results['v5_conf_sum'] += sum(d.confidence for d in v5_dets)
        results['v6_conf_sum'] += sum(d.confidence for d in v6_dets)

        # Compare detections
        matched, v5_only, v6_only = compare_detections(v5_dets, v6_dets)
        results['matched'] += matched
        results['v5_only'] += v5_only
        results['v6_only'] += v6_only

    print("\n")

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- Timing ---")
    print(f"  v5 total time: {results['v5_time']:.1f}s ({results['v5_time']/len(photos)*1000:.0f}ms/photo)")
    print(f"  v6 total time: {results['v6_time']:.1f}s ({results['v6_time']/len(photos)*1000:.0f}ms/photo)")
    speedup = results['v5_time'] / results['v6_time'] if results['v6_time'] > 0 else 0
    print(f"  v6 speedup: {speedup:.1f}x faster")

    print("\n--- Detection Counts ---")
    print(f"  v5 total detections: {results['v5_detections']} ({results['v5_animals']} animals)")
    print(f"  v6 total detections: {results['v6_detections']} ({results['v6_animals']} animals)")

    print("\n--- Average Confidence ---")
    v5_avg_conf = results['v5_conf_sum'] / results['v5_detections'] if results['v5_detections'] > 0 else 0
    v6_avg_conf = results['v6_conf_sum'] / results['v6_detections'] if results['v6_detections'] > 0 else 0
    print(f"  v5 average confidence: {v5_avg_conf:.1%}")
    print(f"  v6 average confidence: {v6_avg_conf:.1%}")

    print("\n--- Agreement (IoU >= 0.5) ---")
    total_compared = results['matched'] + results['v5_only'] + results['v6_only']
    if total_compared > 0:
        agreement = results['matched'] / total_compared * 100
        print(f"  Both agree: {results['matched']} ({agreement:.1f}%)")
        print(f"  v5 only (missed by v6): {results['v5_only']}")
        print(f"  v6 only (missed by v5): {results['v6_only']}")

    print("\n--- Model Sizes ---")
    v5_path = Path.home() / '.trailcam' / 'md_v5a.0.1.pt'
    v6_path = Path.home() / '.trailcam' / 'MDV6-yolov9-c.pt'
    v5_size = v5_path.stat().st_size / (1024*1024) if v5_path.exists() else 0
    v6_size = v6_path.stat().st_size / (1024*1024) if v6_path.exists() else 0
    print(f"  v5: {v5_size:.1f} MB")
    print(f"  v6: {v6_size:.1f} MB ({v5_size/v6_size:.1f}x smaller)")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if speedup > 2 and results['v6_animals'] >= results['v5_animals'] * 0.9:
        print("  v6 is recommended: faster and similar detection rate")
    elif results['v6_animals'] > results['v5_animals'] * 1.1:
        print("  v6 is recommended: better detection rate")
    elif results['v5_animals'] > results['v6_animals'] * 1.1:
        print("  v5 is recommended: better detection rate")
    else:
        print("  Both models perform similarly - v6 recommended for smaller size")


if __name__ == '__main__':
    main()

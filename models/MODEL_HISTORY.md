# Model Version History

This file tracks the version history of AI models used in the Trail Camera Software.

---

## Current Versions

| Model | Version | File | Last Updated |
|-------|---------|------|--------------|
| Species Classifier | 6.0 | species.onnx | Jan 27, 2026 |
| Buck/Doe Classifier | 2.0 | buckdoe.onnx | Jan 26, 2026 |
| Object Detector | 1.0 | detector.onnx | Dec 2024 |

---

## Version History

### Species Classifier (species.onnx)

#### v6.0 (January 27, 2026)
- **Training data**: 4,595 detection box crops (after overlap filtering)
  - Deer: 2,745, Turkey: 1,098, Squirrel: 355, Opossum: 133, Raccoon: 131
  - Rabbit: 71, Coyote: 28, Bobcat: 20, Fox: 11, House Cat: 3
- **Split**: 3,521 train / 617 val / 457 test (stratified)
- **Architecture**: EfficientNet-B2 (upgraded from ResNet18)
- **Training**: 50 epochs, LR 5e-4, cosine schedule, square-root class weighting
- **Augmentation**: Horizontal flip, rotation 20°, color jitter, grayscale 25%, random erasing 15%
- **Accuracy**:
  - **Test overall: 97.2%** (up from 92.8%)
  - Deer: 98%, Turkey: 99%, Squirrel: 94%, Raccoon: 100%, Rabbit: 100%
  - Opossum: 92%, Coyote: 50%, Bobcat: 50%
  - Fox: 0%, House Cat: 0% (only 1 test sample each - not meaningful)
- **Key improvements over v5.0**:
  - Squirrel: 43% → 94% (+51%)
  - Turkey: 79% → 99% (+20%)
  - Coyote: 25% → 50% (+25%)
- **Notes**: Backed up v5.0 as species.onnx.backup_v5

#### v5.0 (January 22, 2026)
- **Training data**: 5,315 samples (4,073 train / 714 val / 528 test) with bounding boxes, crops extracted on-the-fly
- **Boxes filtered**: 5,324 → 5,315 (9 removed for IoU > 0.5 overlap)
- **Classes**: 8 species - Bobcat, Coyote, Deer, Opossum, Rabbit, Raccoon, Squirrel, Turkey
- **Per-species counts**:
  - Deer: 4,265
  - Turkey: 786
  - Squirrel: 77
  - Raccoon: 69
  - Rabbit: 44
  - Coyote: 42
  - Bobcat: 21
  - Opossum: 11
- **Architecture**: ResNet18 pretrained, fine-tuned, 50 epochs, square-root class weighting
- **Accuracy**:
  - Validation: 95.0%
  - **Test: 92.8%** (held-out data never seen during training)
- **Per-class TEST accuracy**:
  - Opossum: 100% (1 test sample)
  - Rabbit: 100% (4 test samples)
  - Raccoon: 100% (6 test samples)
  - Deer: 97%
  - Turkey: 79%
  - Bobcat: 50% (2 test samples)
  - Squirrel: 43%
  - Coyote: 25% (4 test samples)
- **Changes from v4.0**:
  - Removed Fox and House Cat (merged into rare species, not enough samples)
  - Added pixel area logging during training (median box: 106k px, range: 784 - 15.5M px)
  - Added pixel area confidence scaling infrastructure to ai_suggester.py (not yet wired up in app)
  - Stratified train/val/test split ensures each species represented proportionally
- **Notes**: Rare species (Coyote, Bobcat) have poor accuracy due to limited training data. Consider sourcing additional data from LILA datasets.

#### v4.0 (January 1, 2026)
- **Training data**: 4,100 samples (3,485 train + 615 val) with bounding boxes, crops extracted on-the-fly
- **Classes**: 10 species - Bobcat, Coyote, Deer, Fox, House Cat, Opossum, Rabbit, Raccoon, Squirrel, Turkey
- **Per-species counts** (not recorded - see v5.0+ for per-species tracking)
- **Architecture**: ResNet18 pretrained, fine-tuned, 50 epochs
- **Accuracy**: 97.2% overall (Deer 99%, Turkey 97%, Squirrel 96%, Raccoon 90%, Opossum 91%, Rabbit 92%, Coyote 56%)
- **Changes from v3.0**:
  - **Removed Person and Vehicle classes** - MegaDetector handles these via ai_person/ai_vehicle detection boxes
  - Species model should now NEVER suggest "Person" - if it does, it's converted to "Unknown"
  - This fixes the issue where subjects were incorrectly classified as "Person" by the species model
- **Notes**: Backed up v3.0 as species.onnx.backup_v3. Person/Vehicle now handled by MegaDetector auto-classification.
- **Known issue**: No overlap filtering - overlapping boxes may have caused pseudo-replication (fixed in v5.0+)

#### v3.0 (December 28, 2024)
- **Training data**: 3,155 samples (2,682 train + 473 val) with bounding boxes, crops extracted on-the-fly
- **Classes**: 12 individual species - Bobcat, Coyote, Deer, Fox, House Cat, Opossum, Person, Rabbit, Raccoon, Squirrel, Turkey, Vehicle
- **Architecture**: ResNet18 pretrained, fine-tuned, 50 epochs
- **Accuracy**: 97.0% overall (Deer 99%, Turkey 99%, Squirrel 96%, Raccoon 100%, Opossum 94%, Rabbit 100%, Person 100%)
- **Training**: `train_species_crops.py` - crops from bounding boxes, no saved crop files
- **Changes from v2.0**:
  - Kept all 12 species separate (no grouping into Other_Mammal/Other)
  - Excluded rare species with <5 samples (Dog, Quail, Armadillo, Chipmunk, Skunk, Ground Hog, Flicker, Turkey Buzzard, Other Bird)
  - Increased training epochs from 15 to 50
  - No "Other" class - AI suggests "Unknown" for uncertain predictions
- **Notes**: Backed up v2.0 as species.onnx.backup_v2

#### v2.0 (December 21, 2024)
- **Training data**: 1,634 photos with bounding boxes, crops extracted on-the-fly
- **Classes**: Deer, Turkey, Other_Mammal (Raccoon/Rabbit/Squirrel/Coyote/Bobcat/Opossum), Other (Person/Vehicle/Quail)
- **Architecture**: ResNet18 pretrained, fine-tuned
- **Accuracy**: 96.7% overall (Deer 99%, Other 100%, Other_Mammal 88%, Turkey 50%)
- **Training**: `train_species_crops.py` - crops from bounding boxes, no saved crop files
- **Notes**: Simplified to 4 classes for better accuracy. Uses subject/ai_animal boxes from database.

#### v1.0 (December 2024)
- **Training data**: ~900 labeled photos from trail cam library
- **Classes**: Bobcat, Coyote, Deer, Empty, Opossum, Person, Quail, Rabbit, Raccoon, Squirrel, Turkey, Vehicle
- **Architecture**: ResNet18 with fine-tuned last layer
- **Notes**: Initial production model (backed up as species.onnx.backup)

---

### Buck/Doe Classifier (buckdoe.onnx)

#### v2.0 (January 26, 2026)
- **Training data**: 1,611 detection box crops (after overlap filtering)
  - Buck: 1,143 samples
  - Doe: 468 samples
  - Class imbalance ratio: 2.4:1
- **Split**: 1,234 train / 217 val / 160 test (stratified)
- **Architecture**: EfficientNet-B2 (pretrained, fine-tuned)
- **Class balancing**:
  - Inverse frequency weighted sampling
  - Class-weighted cross-entropy loss
- **Training**: 50 epochs, cosine LR schedule, heavy augmentation (grayscale 20%, random erasing 10%)
- **Accuracy**:
  - **Test overall: 91.9%**
  - **Test balanced: 92.4%**
  - Buck: 91.2%
  - Doe: 93.5%
- **Improvement over v1.0**: Fixed severe class bias (v1 predicted Buck 97% of time). Now balanced predictions.
- **Key change**: Trained on MegaDetector detection crops instead of whole photos/head crops
- **Notes**: Backed up v1.0 as buckdoe.onnx.backup_v1

#### v1.0 (December 2024)
- **Training data**: Deer head crops from labeled photos
- **Classes**: Buck, Doe
- **Architecture**: ResNet18
- **Accuracy**: 84% overall, heavily biased (97% Buck predictions)
- **Notes**: Initial production model - replaced due to class imbalance issues

---

### Object Detector (detector.onnx)

#### v1.0 (December 2024)
- **Training data**: Labeled bounding boxes from trail cam library
- **Classes**: subject, deer_head
- **Architecture**: YOLOv8
- **Notes**: Initial production model

---

## How to Update

When training a new model:

1. Train the model using the appropriate script in `training/`
2. Export to ONNX and copy to `models/`
3. Update `models/version.txt` with new version number
4. Copy per-species counts from `training/outputs/species_crops_summary.txt`
5. Add a new version entry to this file with:
   - Version number
   - Date
   - Training data summary with **per-species counts**
   - Any architecture changes
   - Performance notes (overall and per-class accuracy)
   - Known issues or improvements

### Training Protocol Notes

**Overlap filtering (added Jan 4, 2026):** Training script now filters boxes with >50% IoU overlap to reduce pseudo-replication. When multiple boxes on the same photo significantly overlap, only the largest box is kept for training. This prevents the same animal from being counted multiple times.

## Rejection Data

Rejected AI suggestions are logged in the database table `ai_rejections` for use in future training. Query with:

```sql
-- View all rejections
SELECT * FROM ai_rejections ORDER BY rejected_at DESC;

-- Rejections by model version
SELECT model_version, COUNT(*) FROM ai_rejections GROUP BY model_version;

-- Species confusion matrix data
SELECT ai_suggested, correct_label, COUNT(*)
FROM ai_rejections
WHERE suggestion_type = 'species' AND correct_label IS NOT NULL
GROUP BY ai_suggested, correct_label;
```

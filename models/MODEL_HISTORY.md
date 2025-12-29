# Model Version History

This file tracks the version history of AI models used in the Trail Camera Software.

---

## Current Versions

| Model | Version | File | Last Updated |
|-------|---------|------|--------------|
| Species Classifier | 3.0 | species.onnx | Dec 28, 2024 |
| Buck/Doe Classifier | 1.0 | buckdoe.onnx | Dec 2024 |
| Object Detector | 1.0 | detector.onnx | Dec 2024 |

---

## Version History

### Species Classifier (species.onnx)

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

#### v1.0 (December 2024)
- **Training data**: Deer head crops from labeled photos
- **Classes**: Buck, Doe
- **Architecture**: ResNet18
- **Notes**: Initial production model

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
4. Add a new version entry to this file with:
   - Version number
   - Date
   - Training data summary
   - Any architecture changes
   - Performance notes
   - Known issues or improvements

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

# AI Refinement Plan

## Overview

Progressive improvement of AI models in 5 stages, where each stage builds on the previous.

---

## AI Pipeline Order of Operations

Each photo flows through the AI pipeline in this order:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. SUBJECT DETECTION                                                   │
│     - Run MegaDetector/detector on photo                                │
│     - Output: Bounding boxes around animals                             │
│     - If NO boxes found → Species = "Empty" (skip remaining steps)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  2. SPECIES IDENTIFICATION                                              │
│     - Run species classifier on each subject crop                       │
│     - Output: Species tag (Deer, Turkey, Other_Mammal, Other)           │
│     - Human labels override AI suggestions                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        (Only if Species = Deer)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  3. DEER HEAD DETECTION                                                 │
│     - Run deer head detector on Deer photos                             │
│     - Output: Bounding box around deer head                             │
│     - Prerequisite for buck/doe classification                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        (Only if head box exists)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  4. BUCK/DOE CLASSIFICATION                                             │
│     - Run buck/doe classifier on deer head crop                         │
│     - Output: Buck, Doe, or Unknown                                     │
│     - Uses head crop for better antler visibility                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                          (Only if Buck)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  5. ANTLER ANALYSIS (Future - Bucks only)                               │
│     - Antler point counting (left/right, typical/abnormal)              │
│     - Antler measurements (spread, beam length, tine lengths)           │
│     - B&C score estimation (rough category or keypoint-based)           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  6. BODY CHARACTERISTICS (Future - TBD placement)                       │
│     - May run on ALL deer photos (not just bucks)                       │
│     - Age estimation from body proportions                              │
│     - Body condition scoring                                            │
│     - Individual deer re-identification                                 │
│     - Placement in pipeline TBD - could branch after Step 3 or 4        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Rules
- **No boxes = Empty**: If subject detection finds nothing, mark as Empty
- **ai_person = Person**: MegaDetector person detection auto-suggests Person (no classifier needed)
- **ai_vehicle = Vehicle**: MegaDetector vehicle detection auto-suggests Vehicle (no classifier needed)
- **ai_animal = Run classifier**: Only animal detections go through species classifier
- **Boxes + uncertain = Unknown**: If boxes exist but classifier can't identify species
- **"Other" is manual-only**: AI never suggests "Other" (converts to Unknown)
- **Train only on photos WITH boxes**: Photos without boxes excluded from training
- **Deer triggers head detection**: Any photo with Deer (AI or human label) is eligible
- **Head crops for buck/doe**: Classification accuracy improves with head-only crops
- **Human labels always override**: AI suggestions are just suggestions

### Current Status
| Step | Model | Status |
|------|-------|--------|
| 1. Subject Detection | MegaDetector v5 | ✅ Working |
| 2. Species ID | species.onnx v2.0 | ✅ Working (96.7%) |
| 3. Deer Head Detection | - | ⚠️ Need 500+ labeled heads |
| 4. Buck/Doe | buckdoe.onnx v1.0 | ✅ Working (93.8%), needs head crops |
| 5. Antler Analysis | - | 🔜 Future |
| 6. Body Characteristics | - | 🔜 Future |

---

## Stage 1: Object Detection Boxes

**Goal:** Reliably detect animals in photos (bounding boxes around subjects)

**Current State:**
- 2,444 boxes total (1,947 original + 497 from MegaDetector)
- Using MegaDetector v5a.0.1 for detection (MIT licensed, commercial OK)
- All photos now have boxes processed

**Tasks:**
1. [X] Run detection on all unprocessed photos (MegaDetector: 492 photos → 497 detections)
2. [ ] Review/correct boxes on a sample (100-200 photos)
3. [ ] Export training crops from corrected boxes
4. [ ] Retrain detector with new data
5. [ ] Measure improvement (precision/recall)

**Success Metric:** >95% of photos with animals have correct boxes

---

## Stage 2: Species Identification from Boxes

**Goal:** Accurately classify species using subject crops (not full photos)

**Status: COMPLETE** (Dec 21, 2024)

**Results:**
- Trained species classifier on 1,634 crops (on-the-fly from bounding boxes)
- 96.7% overall accuracy
- Deer: 99%, Other: 100%, Other_Mammal: 88%, Turkey: 50%
- Simplified to 4 classes: Deer, Turkey, Other_Mammal, Other
- Model deployed to `models/species.onnx` (v2.0)

**Training Script:** `training/train_species_crops.py`
- Queries photos with species tags AND bounding boxes
- Pre-loads all crops in memory for fast training
- Uses ResNet18 pretrained model
- No weighted sampling (natural class distribution)

**Success Metric:** >90% accuracy on species - ACHIEVED (96.7%)

**Future Goal:** Retrain with all individual species once more photos are collected:
- Current: 4 classes (Deer, Turkey, Other_Mammal, Other)
- Goal: 12+ classes (Deer, Turkey, Raccoon, Rabbit, Squirrel, Coyote, Bobcat, Opossum, Person, Vehicle, Quail, Empty)
- Need ~100+ samples per species for good accuracy

---

## Stage 3: Deer Head Detection

**Goal:** Detect deer heads specifically (for antler analysis)

**Current State:**
- Only 63 deer_head boxes (manually drawn)
- Need ~500+ for reliable training

**Tasks:**
1. [ ] Build UI to quickly draw head boxes on Deer photos
2. [ ] Label 500+ deer head boxes (batch workflow)
3. [ ] Train deer head detector (YOLO)
4. [ ] Run on all Deer-tagged photos
5. [ ] Verify head crops are usable for buck/doe

**Success Metric:** >85% of Deer photos have accurate head box

---

## Stage 4: Buck/Doe Classification

**Goal:** Classify deer as Buck or Doe using head crops

**Current State:**
- buckdoe.onnx exists (45MB)
- 1,025 Buck-tagged, 300 Doe-tagged photos
- Only 63 have head boxes to crop from

**Tasks:**
1. [ ] After Stage 3, export head crops from Buck/Doe photos
2. [ ] Retrain buck/doe classifier on head crops only
3. [ ] Run on all Deer photos with head boxes
4. [ ] Measure accuracy (target: >90%)

**Key Insight:** Antlers are on the head. Full-photo classification is harder because:
- Multiple deer in frame
- Body obscured by vegetation
- Head crops isolate the distinguishing features

**Success Metric:** >90% accuracy on Buck vs Doe

---

## Stage 5: Buck Head Analysis

**Goal:** Detailed antler analysis for buck identification

**Tasks:**
1. [ ] Antler point counting (left/right)
2. [ ] Antler characteristics (spread, mass, abnormalities)
3. [ ] Individual buck re-identification
4. [ ] Track bucks across seasons

**Prerequisites:**
- Reliable deer head boxes (Stage 3)
- Good buck/doe split (Stage 4)
- Sufficient labeled examples per buck

---

## Execution Order

```
Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► Stage 5
(Boxes)    (Species)   (Heads)    (Buck/Doe)  (Analysis)
   │           │           │           │
   └───────────┴───────────┴───────────┘
         Each stage feeds the next
```

---

## Quick Wins (Start Here)

1. ~~**Run AI boxes on all photos**~~ ✓ DONE - MegaDetector processed 492 photos
2. **Run AI species on all photos with boxes** - Use crops for better accuracy ← NEXT
3. **Label 200+ deer heads** - Manual but enables Stages 3-5

---

## Tools Needed

- [X] Box drawing UI (exists)
- [X] Crop export script (exists: training/export_crops.py)
- [X] Species classifier training (exists: training/train_classifier.py)
- [ ] Batch head-labeling workflow (could be faster)
- [ ] Deer head detector training config
- [ ] Buck/doe retrain on head crops

---

## Estimated Effort

| Stage | Effort | Bottleneck |
|-------|--------|------------|
| 1. Object boxes | 2-3 hours | Compute time |
| 2. Species classifier | 4-6 hours | Training + eval |
| 3. Deer heads | 4-8 hours | Manual labeling |
| 4. Buck/doe | 2-3 hours | Depends on Stage 3 |
| 5. Buck analysis | Ongoing | Needs more data |

---

## Next Action

**Stage 1 COMPLETE** - MegaDetector v5a.0.1 integrated and run on all photos.

**Stage 2 COMPLETE** (Dec 21, 2024) - Species classifier trained on crops from bounding boxes.
- 96.7% accuracy achieved (target was >90%)
- Model deployed to `models/species.onnx`
- Training script: `training/train_species_crops.py`

**Start Stage 3:** Deer Head Detection
1. Build batch head-labeling workflow (faster than one-by-one)
2. Label 500+ deer head boxes
3. Train deer head detector (YOLO)
4. Run on all Deer-tagged photos

**Key insight:** Stages 1-2 complete. Head detection (Stage 3) is the next bottleneck - need more head box training data.

---

## Session Notes (Dec 21, 2024)

- Completed Stage 2: Species classifier on crops
- Created `training/train_species_crops.py` - crops on-the-fly from database boxes
- Pre-loads all crops in memory for fast training (~3 min total)
- Simplified species to 4 classes: Deer, Turkey, Other_Mammal, Other
- Achieved 96.7% accuracy (target was >90%)
- Model deployed to `models/species.onnx` (v2.0)
- Old model backed up as `models/species.onnx.backup`

---

## Session Notes (Dec 28, 2024)

**Species Model v3.0 Trained:**
- 50 epochs (increased from 15)
- 97.0% accuracy (up from 96.7%)
- 12 classes: Bobcat, Coyote, Deer, Fox, House Cat, Opossum, Person, Rabbit, Raccoon, Squirrel, Turkey, Vehicle
- Rare species excluded from training (< 5 samples)
- Note: Person and Vehicle can be removed in future training since MegaDetector auto-classifies them

**AI Pipeline Updated:**
- ai_person boxes → auto-suggest "Person" (no classifier needed)
- ai_vehicle boxes → auto-suggest "Vehicle" (no classifier needed)
- ai_animal boxes → run species classifier
- No boxes → suggest "Empty"
- Fixed 40 photos incorrectly marked Empty that had ai_person/ai_vehicle boxes

---

## Session Notes (Dec 27, 2024)

**AI Pipeline Rules Refined:**
- Fixed species classifier suggesting "Other" - now converts to "Unknown"
- "Other" reserved for manual entry of custom species
- Rare species (< 5 samples) excluded from training entirely, not lumped
- Updated training script to skip species mapped to None
- Added "Unknown" to SPECIES_OPTIONS and VALID_SPECIES
- When MegaDetector finds boxes but classifier says "Empty", use "Unknown" with 50% confidence

**Training Changes (train_species_crops.py):**
- Rare species now mapped to `None` (excluded) instead of "Other"
- Dog, Quail, Armadillo, Chipmunk, Skunk, Ground Hog, Flicker, Turkey Buzzard, Other Bird all excluded
- Model will have 12 classes instead of 13 (no "Other" class)
- Need to retrain model with these changes

---

## Session Notes (Dec 20, 2024)

- Installed `megadetector` pip package
- Created `run_megadetector.py` for batch processing
- MegaDetector uses MPS (Metal) on Mac for GPU acceleration
- Model downloads automatically on first run (~280MB to temp folder)
- Detection threshold set to 0.2 confidence
- Labels: ai_animal (category 1), ai_person (category 2), ai_vehicle (category 3)
- Review queue green highlighting added but may need refinement

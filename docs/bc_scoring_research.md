# B&C Antler Scoring - AI Research & Options

**Status:** Research complete, ready for implementation when prioritized
**Last Updated:** December 2024
**Decision:** Build in-house (commercial options ruled out as competitors)

## Summary

This document evaluates options for adding AI-powered Boone & Crockett score estimation to Trail Camera Software.

### Quick Resume
- No viable API partners (Rackline.ai is a competitor)
- **Phase 1 MVP:** Train rough category classifier (Small/Medium/Good/Trophy) using buck crops
- **Phase 2:** Keypoint detection for actual measurements
- **Future UI:** Buck profiles will have per-year antler measurement entry (all tines + spread)
- **Future feature:** Video support coming to app (starting with photos first)
  - Video frame extraction can provide more scoring angles per buck
  - Same AI scoring pipeline applies to extracted frames
- **Schema ready:** See `buck_antler_measurements` and `buck_score_categories` tables below

---

## Commercial Landscape (Competitors)

### Rackline.ai (COMPETITOR - DO NOT PARTNER)
- **Website:** https://rackline.ai
- **Platform:** iOS app (launched late 2024)
- **Features:** B&C scoring from photos, age estimation, growth projections
- **Threat level:** Direct competitor to our planned features

### HuntelligenceX
- **Website:** https://www.huntelligencex.com/
- **Features:** AI deer analysis (details limited)

### Spartan Forge
- **Website:** https://spartanforge.ai/pages/deer-prediction
- **Focus:** Deer movement prediction, not antler scoring

---

## Academic Research

### 1. Neural Network Antler Prediction (2019)
- **Paper:** [PMC6386314](https://pmc.ncbi.nlm.nih.gov/articles/PMC6386314/)
- **Method:** Multilayer Perceptron ANN vs linear models
- **Dataset:** White-tailed deer harvest data from BCWMA
- **Finding:** MLPANN with Bayesian Regularization outperformed linear models for predicting beam diameter and length
- **Limitation:** Uses harvest measurements, not photos

### 2. Age Classification via Deep Learning (2025)
- **Paper:** [bioRxiv 2025.07.01.662304](https://www.biorxiv.org/content/10.1101/2025.07.01.662304v1.full)
- **Method:** Computer vision on trail cam imagery
- **Accuracy:** 76.7% (vs 60.6% human expert, 63% morphometric)
- **Features:** Body proportions, antler development, facial features, muscle definition
- **Relevance:** Could combine age estimation with scoring

### 3. Photogrammetric Antler Measurement (2016)
- **Paper:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1537511015303421)
- **Method:** 3D CAD model from 2 photographs
- **Species:** Iberian red deer (n=29)
- **Key insight:** Only 2 photos needed for 3D reconstruction
- **Relevance:** Core technique for photo-based measurement

---

## Build-Your-Own Approach

### Available Datasets

1. **Roboflow Antler Dataset**
   - URL: https://universe.roboflow.com/nimbleappgenie/antler-yq3hj
   - Size: ~210 images
   - Type: Object detection (bounding boxes)
   - Limitation: Detection only, no keypoints

2. **Roboflow Deer Datasets**
   - URL: https://universe.roboflow.com/search?q=class:deer
   - Various deer detection datasets
   - Would need custom keypoint annotation

### Keypoint Detection Frameworks

1. **YOLOv8 Pose** - Fast, easy to train custom keypoints
2. **DeepLabCut** - Designed for animal pose estimation
3. **OpenPifPaf** - Plugin architecture for custom animals
4. **X-Pose** - Multi-modal keypoint detection (UniKPT dataset)

### Required Keypoints for B&C Scoring

```
Per Antler (mirror for both sides):
- Burr (base) center
- Main beam tip
- G1 (brow tine) tip
- G2 tip
- G3 tip
- G4 tip (if present)
- G5+ tips (if present)
- H1 circumference point (between burr and G1)
- H2 circumference point (between G1 and G2)
- H3 circumference point (between G2 and G3)
- H4 circumference point (between G3 and G4)

Reference points:
- Eye center (left/right) - for scale calibration
- Nose tip - for scale calibration
- Ear tips - for orientation
```

### Scale Calibration

Whitetail deer reference measurements:
- Eye-to-nose distance: ~7.5 inches
- Ear length: ~6 inches
- Eye-to-eye (inside): ~7 inches

### Minimum Viable Implementation

**Phase 1: Antler Detection + Basic Scoring**
1. Detect deer with antlers (existing MegaDetector)
2. Classify as buck (existing species model)
3. Crop antler region
4. Estimate rough score category (small/medium/large/trophy)

**Phase 2: Keypoint-Based Measurement**
1. Create training dataset (500-1000 annotated images)
2. Train keypoint detection model (YOLOv8-pose or similar)
3. Calculate measurements from keypoints using scale reference
4. Compute B&C gross score

**Phase 3: Multi-View Scoring**
1. Match same deer across multiple photos
2. Combine measurements from different angles
3. Improve accuracy with temporal data

---

## Recommended Path Forward

### Phase 1: Rough Score Classifier (MVP)
- Train 4-5 class model: "Small" (<100"), "Medium" (100-130"), "Good" (130-150"), "Trophy" (150"+)
- Use existing buck crops from trail cam database
- Requires: User-provided score estimates for training labels
- Accuracy goal: Distinguish major categories, not precise scores
- Timeline: Can start immediately with ~200-500 labeled images

### Phase 2: Keypoint Detection Model
- Annotate antler keypoints on 500-1000 buck images
- Train YOLOv8-pose or similar keypoint model
- Detect: beam tips, tine tips, base points, reference points (eyes/nose)
- Use eye-to-nose distance (~7.5") for scale calibration
- Calculate approximate measurements from 2D projections

### Phase 3: Multi-Angle Scoring
- Match same buck across multiple trail cam photos
- Combine measurements from different angles/times
- Build 3D understanding from 2D observations
- Improve accuracy through temporal averaging

### Phase 4: Harvest Validation
- Collect actual B&C scores from harvested bucks
- Match to trail cam predictions for validation
- Continuous model improvement with ground truth data

---

## Data Collection for Future Training

### Taxidermy Mount Collection (Primary Source)
- Dad is a taxidermist - access to many mounts with measurable ground truth
- Get verbal consent from customers or use personal mounts
- **Video approach:** Walk around mount recording video, extract frames later
  - Faster than individual photos
  - 30-60 second walk-around captures all angles
  - Extract 1 frame/second, skip blurry frames automatically
  - OpenCV `cv2.Laplacian` variance for blur detection
- Record all physical measurements per mount (beams, tines, spread, circumferences)
- Calculate actual B&C gross/net score for ground truth labels

### Other Sources
1. Photos of bucks with known B&C scores (harvested deer)
2. Multiple angles of same buck from trail cams
3. Reference photos with measuring tape visible
4. User-submitted scores for validation

Database schema addition needed:
```sql
-- Detailed antler measurements per buck per season
CREATE TABLE buck_antler_measurements (
    id INTEGER PRIMARY KEY,
    buck_profile_id INTEGER REFERENCES buck_profiles(id),
    season_year INTEGER,  -- e.g., 2024

    -- Main beams (inches)
    left_main_beam REAL,
    right_main_beam REAL,

    -- Inside spread
    inside_spread REAL,

    -- Tine lengths (G1 = brow tine, G2, G3, etc.)
    left_g1 REAL,
    left_g2 REAL,
    left_g3 REAL,
    left_g4 REAL,
    left_g5 REAL,
    right_g1 REAL,
    right_g2 REAL,
    right_g3 REAL,
    right_g4 REAL,
    right_g5 REAL,

    -- Circumferences (H1-H4)
    left_h1 REAL,
    left_h2 REAL,
    left_h3 REAL,
    left_h4 REAL,
    right_h1 REAL,
    right_h2 REAL,
    right_h3 REAL,
    right_h4 REAL,

    -- Computed scores
    gross_score REAL,
    net_score REAL,

    -- Metadata
    measurement_source TEXT,  -- 'manual', 'ai_estimate', 'official'
    point_count_left INTEGER,
    point_count_right INTEGER,
    is_typical INTEGER DEFAULT 1,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- For AI training: rough category labels
CREATE TABLE buck_score_categories (
    id INTEGER PRIMARY KEY,
    buck_profile_id INTEGER REFERENCES buck_profiles(id),
    season_year INTEGER,
    category TEXT,  -- 'small', 'medium', 'good', 'trophy'
    user_estimate_low INTEGER,  -- estimated range low (e.g., 130)
    user_estimate_high INTEGER, -- estimated range high (e.g., 145)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## Sources

- [MSU Deer Lab - B&C Estimation](https://www.msudeer.msstate.edu/estimate-boone-and-crockett-score.php)
- [Boone & Crockett Field Judging Guide](https://www.boone-crockett.org/field-judging-whitetail-deer)
- [Keras Keypoint Detection Tutorial](https://keras.io/examples/vision/keypoint_detection/)
- [X-Pose Paper (arXiv)](https://arxiv.org/html/2310.08530v2)

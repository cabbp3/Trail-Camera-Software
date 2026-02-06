# Individual Buck Identification System

## Overview

A multi-feature discrimination approach to identify and track individual whitetail bucks across trail camera photos. Rather than relying on a single "fingerprint" feature, this system uses multiple stable and semi-stable characteristics to rule out non-matches and score likely matches.

## Core Concept

**Key Insight:** You don't need to prove "this IS Buck #7." You need to prove "this definitely ISN'T Buck #7" for most candidates, leaving few possibilities.

This mirrors how experienced hunters identify deer - combining multiple visual cues rather than relying on any single feature.

---

## Feature Taxonomy

### Tier 1: Stable Discriminators (High Confidence Filters)

These features don't change significantly over a deer's life and can definitively rule out matches.

| Feature | Description | Stability | Detection Difficulty |
|---------|-------------|-----------|---------------------|
| **Tail coloring** | White vs brown ratio on tail | Very High | Medium |
| **Facial mask pattern** | Dark/light zones around eyes and muzzle | High | Medium |
| **Ear shape/notches** | Natural shape, damage, distinctive marks | Very High | Medium |
| **White throat patch** | Shape and boundaries of throat marking | High | Medium |
| **Metatarsal gland** | Coloring of tuft on lower hind leg | High | Hard (angle dependent) |

### Tier 2: Semi-Stable Features (Supporting Evidence)

These features change slowly over time but provide useful matching signals.

| Feature | Description | Stability | Notes |
|---------|-------------|-----------|-------|
| **Body proportions** | Height/length ratios, overall build | Medium | More stable as ratios than absolutes |
| **Antler base position** | Where antlers sprout from skull | Medium | Consistent year to year |
| **Brow tine angle** | Basic geometry of brow tines | Medium | General pattern persists |
| **Nose profile** | Roman nose vs straight profile | High | Bone structure doesn't change |
| **Neck thickness** | Relative neck size | Medium | Changes with rut/season |

### Tier 3: Seasonal Features (Same-Season Only)

These features are highly distinctive but only valid within a single season.

| Feature | Description | Validity |
|---------|-------------|----------|
| **Antler point count** | Number of tines | Current season |
| **Antler spread** | Width classification | Current season |
| **Main beam shape** | Curve, length, direction | Current season |
| **Tine configuration** | Specific point arrangement | Current season |
| **Coat color** | Summer red vs winter gray | Current season |

---

## Matching Algorithm

### Phase 1: Hard Filtering

```
Input: New photo with detected buck
Candidate pool: All known bucks (N)

For each Tier 1 feature detected in photo:
    Filter candidates where feature definitely doesn't match

Result: Reduced candidate pool (typically 10-30% of N)
```

### Phase 2: Soft Scoring

```
For each remaining candidate:
    Score = 0

    For each Tier 2 feature:
        Score += similarity_weight * feature_match_score

    If same season:
        For each Tier 3 feature:
            Score += antler_weight * feature_match_score

    Add embedding_similarity * embedding_weight

Result: Ranked list of candidates with confidence scores
```

### Phase 3: Human Confirmation

```
Present top 3 candidates to user:
    - Side-by-side photo comparison
    - Feature breakdown showing matches/mismatches
    - Confidence percentage

User actions:
    - Confirm match → strengthen model
    - Reject all → create new buck profile
    - Correct match → retrain on error
```

---

## Data Model

### Buck Profile Schema

```python
buck_profile = {
    "id": "uuid",
    "name": "User-assigned name",
    "created_at": "timestamp",
    "last_seen": "timestamp",
    "photo_count": 0,

    # Tier 1 - Stable discriminators
    "features": {
        "tail_white_ratio": 0.85,           # 0.0-1.0 (brown to white)
        "tail_confidence": 0.9,             # How certain we are

        "facial_mask": "dark",              # dark | medium | light | none
        "facial_mask_confidence": 0.85,

        "throat_patch_shape": "large_angular",  # shape classification
        "throat_patch_confidence": 0.8,

        "ear_left": "normal",               # normal | notched | torn | missing_tip
        "ear_right": "notched_tip",
        "ear_confidence": 0.95,
    },

    # Tier 2 - Semi-stable
    "body": {
        "size_class": "large",              # small | medium | large
        "nose_profile": "roman",            # roman | straight
        "build": "heavy",                   # lean | average | heavy
    },

    # Tier 3 - Seasonal (keyed by year)
    "antlers": {
        "2024": {
            "points": 10,
            "spread_class": "wide",         # narrow | medium | wide
            "main_beam_curve": "high",      # low | medium | high
            "brow_tines": "long",           # short | medium | long
            "notable_features": ["split_g2", "kicker_point"],
        },
        "2023": {
            "points": 8,
            # ... previous year data for cross-year tracking
        }
    },

    # ML embeddings
    "embeddings": {
        "face_embedding": [/* 512-dim vector */],
        "body_embedding": [/* 512-dim vector */],
        "updated_at": "timestamp",
    },

    # Activity patterns (from existing tracking)
    "patterns": {
        "primary_cameras": ["camera_id_1", "camera_id_2"],
        "peak_hours": [6, 7, 18, 19],
        "first_seen": "2023-10-15",
    }
}
```

### Photo Annotation Schema

**Current annotations (in database):**
```python
# Existing deer_metadata table
{
    "photo_id": "uuid",

    # Head annotation (IMPLEMENTED)
    "head_x": 150,              # Head bounding box
    "head_y": 200,
    "head_width": 80,
    "head_height": 60,
    "head_direction": "left",   # left | right | toward | away

    # Body annotation (PLANNED)
    "body_x": None,
    "body_y": None,
    "body_width": None,
    "body_height": None,
    "body_direction": None,     # broadside_left | broadside_right | front | rear | quartering_to | quartering_away
}
```

**Future feature annotations:**
```python
photo_features = {
    "photo_id": "uuid",

    # Head features (after head model trained)
    "facial_mask": "dark",              # dark | medium | light | none
    "facial_mask_confidence": 0.85,
    "nose_profile": "roman",            # roman | straight
    "ear_left_shape": "normal",         # normal | notched | torn
    "ear_right_shape": "notched_tip",

    # Body features (after body model trained)
    "tail_white_ratio": 0.85,
    "tail_confidence": 0.9,
    "throat_patch_shape": "large_angular",
    "body_size_class": "large",         # small | medium | large

    # Quality indicators
    "head_visible": True,
    "body_visible": True,
    "tail_visible": False,              # Not visible in this angle
    "lighting": "daylight",             # daylight | flash | ir
    "occlusion": "none",                # none | partial | heavy
}
```

---

## Technical Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Photo Ingestion                          │
│  (MegaDetector → Deer Crop → Quality Filter)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Extraction                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Segmentation │  │ Classifiers │  │  Embedding  │         │
│  │   Model      │  │  (tail,     │  │   Model     │         │
│  │ (body parts) │  │   face...)  │  │ (similarity)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Matching Engine                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Hard     │  │    Soft     │  │   Ranking   │         │
│  │   Filters   │──▶│   Scoring   │──▶│  & Output   │         │
│  │  (Tier 1)   │  │ (Tier 2+3)  │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Human-in-the-Loop                           │
│  ┌─────────────────────────────────────────────────┐       │
│  │  "Is this Buck #12?"                            │       │
│  │  [Yes] [No - New Buck] [No - It's Buck #X]      │       │
│  └─────────────────────────────────────────────────┘       │
│                         │                                   │
│                         ▼                                   │
│              Feedback → Model Improvement                   │
└─────────────────────────────────────────────────────────────┘
```

### ML Models Required

1. **Deer Part Segmentation**
   - Input: Cropped deer image
   - Output: Masks for body, head, tail, antlers
   - Architecture: U-Net or Mask R-CNN fine-tuned on deer

2. **Feature Classifiers**
   - Tail classifier: Predicts white ratio (regression)
   - Face classifier: Predicts mask type (classification)
   - Body classifier: Predicts size/build (classification)
   - Architecture: ResNet or EfficientNet heads

3. **Similarity Embedding**
   - Input: Deer crop
   - Output: 512-dim embedding vector
   - Architecture: Fine-tuned CLIP, DINOv2, or custom triplet network
   - Training: Contrastive learning on confirmed same-buck pairs

---

## Implementation Phases

### Current State (January 2026)

**Foundation: Species Detection**
- [x] MegaDetector integration for animal detection
- [x] Species classification (buck, doe, turkey, etc.)
- [x] Confidence scoring and filtering

**In Progress: Head Annotation**
- [x] Head bounding box annotation in label tool
- [x] Head direction annotation (which way deer is looking)
- [ ] Train head detection model from annotations
- [ ] Train head direction classifier

### Phase 1: Head Feature Extraction

**Goal:** Extract identifying features from head region

Building on head annotations:
- [ ] Facial mask pattern detection (dark/medium/light zones)
- [ ] Ear shape classification
- [ ] Nose profile detection (roman vs straight)
- [ ] Antler base position (for bucks)
- [ ] Head embedding model for similarity

**Dependency:** Sufficient head annotations with direction labels

**Current annotation status (Jan 2026):**
- ~100 head annotations completed
- Several thousand deer photos available for annotation
- Strategy: Bootstrap with semi-supervised loop (see Annotation Strategy below)

### Phase 2: Body Annotation & Direction

**Goal:** Annotate and detect body orientation

- [ ] Add body bounding box annotation to label tool
- [ ] Add body direction annotation (broadside, quartering, front, rear)
- [ ] Train body detection model
- [ ] Train body direction classifier

**Why direction matters:** Features only extractable from certain angles:
- Tail → rear or broadside
- Throat patch → front or quartering-to
- Body proportions → broadside

### Phase 3: Body Feature Extraction

**Goal:** Extract identifying features from body region

- [ ] Tail white/brown ratio detection
- [ ] Throat patch shape classification
- [ ] Body size/build classification
- [ ] Metatarsal gland detection (when visible)
- [ ] Body embedding model for similarity

### Phase 4: Feature Fusion & Matching

**Goal:** Combine head + body features for individual ID

- [ ] Multi-feature buck profile schema
- [ ] Hard filtering on stable features (tail, face mask)
- [ ] Soft scoring with embeddings
- [ ] Same-season matching pipeline
- [ ] "Suggested Match" UI with confirmation

**Success Metric:** 80%+ of suggestions correct for top-3 candidates

### Phase 5: Cross-Season Tracking

**Goal:** Link bucks across years using stable features only

- [ ] Tier 1 feature focus (ignore antlers for cross-year)
- [ ] Cross-year linking UI
- [ ] Antler development history tracking
- [ ] "Buck Biography" generation

### Phase 6: Continuous Learning

**Goal:** Model improves with every user confirmation

- [ ] Active learning pipeline
- [ ] Retrain on confirmed matches
- [ ] A/B test model versions
- [ ] Edge case handling (injuries, similar-looking deer)

---

## Progression Logic

```
Species Detection (DONE)
       │
       ▼
Head Detection + Direction (IN PROGRESS)
       │
       ▼
Head Features (facial mask, ears, nose)
       │
       ▼
Body Detection + Direction
       │
       ▼
Body Features (tail, throat, build)
       │
       ▼
Feature Fusion → Individual ID
       │
       ▼
Cross-Season Tracking
```

Each layer provides training data and validation for the next. No skipping ahead.

**Why this order:**
1. **Species first** → Know what you're looking at before extracting features
2. **Head detection** → Smallest, most consistent region. Visible in most angles.
3. **Head direction** → Required to know which features are extractable (can't see face from behind)
4. **Head features** → Facial mask, ears, nose - high-value stable discriminators
5. **Body detection** → Larger region, more variation in poses
6. **Body direction** → Critical for tail/throat visibility
7. **Body features** → Tail coloring, throat patch, build
8. **Fusion** → Combine all features only after each is reliable standalone

---

## Annotation Strategy

### Bootstrap Loop (Semi-Supervised)

Manual annotation is slow. Use trained models to accelerate:

```
┌─────────────────────────────────────────────────────────┐
│  100 annotations                                        │
│       │                                                 │
│       ▼                                                 │
│  Train v0.1 model (expect ~70-80% accuracy)            │
│       │                                                 │
│       ▼                                                 │
│  Auto-annotate next 500-1000 photos                    │
│       │                                                 │
│       ▼                                                 │
│  Human reviews & corrects (3x faster than from scratch)│
│       │                                                 │
│       ▼                                                 │
│  ~600-1100 annotations                                  │
│       │                                                 │
│       ▼                                                 │
│  Train v0.2 model (expect ~85-90% accuracy)            │
│       │                                                 │
│       ▼                                                 │
│  Repeat until diminishing returns                       │
└─────────────────────────────────────────────────────────┘
```

### Annotation Priorities

1. **Balance head directions** - Need roughly equal left/right/toward/away
2. **Variety of lighting** - Daylight, flash, IR night
3. **Range of distances** - Close, medium, far
4. **Multiple angles** - Don't over-train on one camera's typical view

### Available Data Sources

- Own trail cam photos (primary - matches deployment conditions)
- Public datasets (supplementary - LILA, Snapshot Serengeti)
- Thousands of photos available, annotation is the bottleneck

### Model Recommendations for Small Datasets

- **YOLOv8** - Fine-tunes well with hundreds of examples
- **RT-DETR** - Transformer-based, good with limited data
- **Transfer learning** - Start from COCO-pretrained weights

---

## Research & References

### Existing Work

- **DeerLab:** Manual buck profiling with pattern analysis. No auto-ID.
  - https://deerlab.com/blog/pattern-buck-movements-with-trail-camera-photos

- **Sika Deer Facial Recognition (2023):** Vision Transformer for individual ID
  - https://www.sciencedirect.com/science/article/pii/S1574954123003631

- **Wildbook:** Open-source individual animal ID platform
  - Works well for species with permanent markings (zebras, whale sharks)

- **Wildlife Datasets Toolkit:** Curated animal re-ID datasets
  - https://github.com/DariaKern/IndividualAnimalRe-IDDatasets
  - Note: No whitetail deer datasets currently exist

### Why This Approach is Novel

Most attempts at individual animal ID use pure end-to-end deep learning, which requires massive datasets. Our hybrid approach:

1. Uses domain knowledge (hunter intuition) to define meaningful features
2. Combines hard rules (tail color) with soft matching (embeddings)
3. Keeps human in the loop for confirmation and training
4. Works with smaller datasets by focusing on discriminative features

---

## Open Questions

1. **Minimum viable dataset size?** How many annotated bucks needed for useful matching?

2. **Feature extraction accuracy?** Can we reliably detect tail in IR night photos?

3. **Cross-property matching?** How to handle bucks seen on multiple properties?

4. **Privacy/sharing model?** Should buck profiles be shareable between users?

5. **Handling uncertainty?** UI for "Maybe same buck, not sure" cases?

---

## Success Metrics

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| Annotated bucks | 50+ | 200+ | 500+ |
| Photos per buck | 10+ | 20+ | 30+ |
| Top-3 accuracy | N/A | 80% | 90% |
| User confirmation rate | N/A | 70% | 85% |
| Cross-year link accuracy | N/A | N/A | 75% |

---

*Document created: January 2026*
*Last updated: January 2026*

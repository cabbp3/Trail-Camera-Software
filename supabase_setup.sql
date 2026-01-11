-- Supabase Tables for Trail Camera Software
-- Run this in your Supabase SQL Editor (SQL Editor in left sidebar)

-- Photos sync table (identifiers only, not file paths)
CREATE TABLE photos_sync (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_key TEXT UNIQUE NOT NULL,  -- original_name|date_taken|camera_model
    file_hash TEXT,                  -- MD5 hash for cross-computer matching
    original_name TEXT,
    date_taken TEXT,
    camera_model TEXT,
    camera_location TEXT,
    collection TEXT,                 -- Club/collection name (Brooke Farm, etc.)
    season_year INTEGER,
    favorite BOOLEAN DEFAULT FALSE,
    notes TEXT,
    r2_photo_id TEXT,               -- Local photo ID for R2 URL mapping
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tags (species labels)
CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_key TEXT NOT NULL,
    tag_name TEXT NOT NULL,
    file_hash TEXT,                  -- MD5 hash for cross-computer matching
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(photo_key, tag_name)
);

-- Deer metadata (primary deer per photo)
CREATE TABLE deer_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_key TEXT UNIQUE NOT NULL,
    deer_id TEXT,
    age_class TEXT,
    left_points_min INTEGER,
    left_points_max INTEGER,
    right_points_min INTEGER,
    right_points_max INTEGER,
    left_points_uncertain BOOLEAN DEFAULT FALSE,
    right_points_uncertain BOOLEAN DEFAULT FALSE,
    left_ab_points_min INTEGER,
    left_ab_points_max INTEGER,
    right_ab_points_min INTEGER,
    right_ab_points_max INTEGER,
    abnormal_points_min INTEGER,
    abnormal_points_max INTEGER,
    broken_antler_side TEXT,
    broken_antler_note TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Additional deer in same photo
CREATE TABLE deer_additional (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_key TEXT NOT NULL,
    deer_id TEXT NOT NULL,
    age_class TEXT,
    left_points_min INTEGER,
    left_points_max INTEGER,
    right_points_min INTEGER,
    right_points_max INTEGER,
    left_points_uncertain BOOLEAN DEFAULT FALSE,
    right_points_uncertain BOOLEAN DEFAULT FALSE,
    left_ab_points_min INTEGER,
    left_ab_points_max INTEGER,
    right_ab_points_min INTEGER,
    right_ab_points_max INTEGER,
    abnormal_points_min INTEGER,
    abnormal_points_max INTEGER,
    broken_antler_side TEXT,
    broken_antler_note TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(photo_key, deer_id)
);

-- Buck profiles
CREATE TABLE buck_profiles (
    deer_id TEXT PRIMARY KEY,
    display_name TEXT,
    notes TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Buck profile seasons
CREATE TABLE buck_profile_seasons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deer_id TEXT NOT NULL,
    season_year INTEGER NOT NULL,
    camera_locations TEXT,
    key_characteristics TEXT,
    left_points_min INTEGER,
    left_points_max INTEGER,
    right_points_min INTEGER,
    right_points_max INTEGER,
    left_ab_points_min INTEGER,
    left_ab_points_max INTEGER,
    right_ab_points_min INTEGER,
    right_ab_points_max INTEGER,
    abnormal_points_min INTEGER,
    abnormal_points_max INTEGER,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(deer_id, season_year)
);

-- Annotation boxes
CREATE TABLE annotation_boxes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_key TEXT NOT NULL,
    label TEXT,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,
    confidence REAL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster lookups
CREATE INDEX idx_tags_photo_key ON tags(photo_key);
CREATE INDEX idx_deer_metadata_photo_key ON deer_metadata(photo_key);
CREATE INDEX idx_deer_additional_photo_key ON deer_additional(photo_key);
CREATE INDEX idx_annotation_boxes_photo_key ON annotation_boxes(photo_key);
CREATE INDEX idx_buck_profile_seasons_deer_id ON buck_profile_seasons(deer_id);

-- Enable Row Level Security (required for Supabase)
ALTER TABLE photos_sync ENABLE ROW LEVEL SECURITY;
ALTER TABLE tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE deer_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE deer_additional ENABLE ROW LEVEL SECURITY;
ALTER TABLE buck_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE buck_profile_seasons ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotation_boxes ENABLE ROW LEVEL SECURITY;

-- Create policies to allow access with anon key
CREATE POLICY "Allow all access" ON photos_sync FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access" ON tags FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access" ON deer_metadata FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access" ON deer_additional FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access" ON buck_profiles FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access" ON buck_profile_seasons FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access" ON annotation_boxes FOR ALL USING (true) WITH CHECK (true);

-- ============================================================
-- MIGRATION: Run this if tables already exist
-- ============================================================
-- Add new columns to photos_sync (for mobile app R2 integration)
ALTER TABLE photos_sync ADD COLUMN IF NOT EXISTS file_hash TEXT;
ALTER TABLE photos_sync ADD COLUMN IF NOT EXISTS collection TEXT;
ALTER TABLE photos_sync ADD COLUMN IF NOT EXISTS r2_photo_id TEXT;

-- Add file_hash to tags
ALTER TABLE tags ADD COLUMN IF NOT EXISTS file_hash TEXT;

-- Create index on r2_photo_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_photos_sync_r2_photo_id ON photos_sync(r2_photo_id);
CREATE INDEX IF NOT EXISTS idx_photos_sync_collection ON photos_sync(collection);

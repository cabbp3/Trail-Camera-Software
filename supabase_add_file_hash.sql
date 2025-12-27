-- Add file_hash column for cross-computer sync
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard
-- (Go to your project -> SQL Editor -> paste this -> Run)

-- Add file_hash to all tables that need it
ALTER TABLE photos_sync ADD COLUMN IF NOT EXISTS file_hash TEXT;
ALTER TABLE tags ADD COLUMN IF NOT EXISTS file_hash TEXT;
ALTER TABLE deer_metadata ADD COLUMN IF NOT EXISTS file_hash TEXT;
ALTER TABLE deer_additional ADD COLUMN IF NOT EXISTS file_hash TEXT;
ALTER TABLE annotation_boxes ADD COLUMN IF NOT EXISTS file_hash TEXT;

-- Create indexes for faster hash lookups
CREATE INDEX IF NOT EXISTS idx_photos_sync_file_hash ON photos_sync(file_hash);
CREATE INDEX IF NOT EXISTS idx_tags_file_hash ON tags(file_hash);
CREATE INDEX IF NOT EXISTS idx_deer_metadata_file_hash ON deer_metadata(file_hash);
CREATE INDEX IF NOT EXISTS idx_deer_additional_file_hash ON deer_additional(file_hash);
CREATE INDEX IF NOT EXISTS idx_annotation_boxes_file_hash ON annotation_boxes(file_hash);

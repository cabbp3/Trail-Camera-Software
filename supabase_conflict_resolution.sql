-- ============================================================
-- Multi-Device Conflict Resolution Migration
-- Run this in Supabase SQL Editor BEFORE deploying code changes
-- ============================================================

-- Step 1: Clean up any existing tag duplicates before adding constraint
DELETE FROM tags a USING tags b
WHERE a.ctid < b.ctid AND a.file_hash = b.file_hash AND a.tag_name = b.tag_name;

-- Step 2: Fix tags unique constraint
-- Current upsert uses file_hash,tag_name but UNIQUE is only on (photo_key, tag_name)
CREATE UNIQUE INDEX IF NOT EXISTS idx_tags_file_hash_tag_name ON tags(file_hash, tag_name);

-- Step 3: Tags tombstone support
ALTER TABLE tags ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;
CREATE INDEX IF NOT EXISTS idx_tags_deleted_at ON tags(deleted_at);

-- Step 4: Annotation boxes - stable sync IDs + tombstones
ALTER TABLE annotation_boxes ADD COLUMN IF NOT EXISTS sync_id TEXT;
ALTER TABLE annotation_boxes ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;

-- Step 5: Backfill sync_ids for existing boxes (MUST run before adding UNIQUE constraint)
UPDATE annotation_boxes SET sync_id = gen_random_uuid()::text WHERE sync_id IS NULL;

-- Step 6: Add UNIQUE constraint on sync_id (the new upsert conflict key for boxes)
ALTER TABLE annotation_boxes ADD CONSTRAINT annotation_boxes_sync_id_unique UNIQUE (sync_id);

-- Step 7: Add file_hash column to annotation_boxes if missing
ALTER TABLE annotation_boxes ADD COLUMN IF NOT EXISTS file_hash TEXT;

-- Step 8: Indexes
CREATE INDEX IF NOT EXISTS idx_annotation_boxes_sync_id ON annotation_boxes(sync_id);
CREATE INDEX IF NOT EXISTS idx_annotation_boxes_deleted_at ON annotation_boxes(deleted_at);
CREATE INDEX IF NOT EXISTS idx_annotation_boxes_file_hash ON annotation_boxes(file_hash);

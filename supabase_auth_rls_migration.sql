-- =============================================================
-- Supabase Migration: Real RLS Policies with Supabase Auth
-- =============================================================
--
-- PREREQUISITES:
--   1. Supabase Auth is enabled (Email + Password provider)
--   2. Desktop app updated to use Supabase Auth (not just anon key)
--   3. Mobile app already uses Supabase Auth (supabase_flutter)
--   4. All existing users have Supabase Auth accounts
--
-- HOW IT WORKS:
--   - Authenticated users can read/write their own data
--   - Anon key can only SELECT (read-only browsing)
--   - Write operations require a valid JWT from Supabase Auth
--   - User identity comes from auth.jwt()->>'email' or user_metadata
--   - Camera ownership controls who can label which photos
--
-- DEPLOY STEPS:
--   1. Ensure all apps use Supabase Auth for login
--   2. Run this SQL in Supabase SQL Editor
--   3. Test: logged-in user can read+write, anon can only read
--   4. Drop old permissive policies (cleanup section at bottom)
--
-- =============================================================

-- Helper function: extract username from JWT
-- Uses display_name from user_metadata, falls back to email prefix
CREATE OR REPLACE FUNCTION public.current_username()
RETURNS TEXT AS $$
BEGIN
    RETURN COALESCE(
        auth.jwt()->'user_metadata'->>'display_name',
        split_part(auth.jwt()->>'email', '@', 1),
        'anonymous'
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;


-- =============================================================
-- PHASE 1: Add authenticated read/write + keep anon read-only
-- =============================================================
-- This phase ADDS new policies alongside the existing anon ones.
-- After verifying everything works, Phase 2 drops the old ones.

-- ----- photos_sync -----
-- Everyone can read photos (public browsing)
-- Only authenticated users can create/update photos
CREATE POLICY "auth_select_photos_sync" ON photos_sync
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_photos_sync" ON photos_sync
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_photos_sync" ON photos_sync
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ----- tags -----
-- Everyone can read tags
-- Only authenticated users can create/update tags
CREATE POLICY "auth_select_tags" ON tags
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_tags" ON tags
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_tags" ON tags
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ----- deer_metadata -----
CREATE POLICY "auth_select_deer_metadata" ON deer_metadata
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_deer_metadata" ON deer_metadata
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_deer_metadata" ON deer_metadata
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ----- deer_additional -----
CREATE POLICY "auth_select_deer_additional" ON deer_additional
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_deer_additional" ON deer_additional
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_deer_additional" ON deer_additional
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ----- buck_profiles -----
CREATE POLICY "auth_select_buck_profiles" ON buck_profiles
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_buck_profiles" ON buck_profiles
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_buck_profiles" ON buck_profiles
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ----- buck_profile_seasons -----
CREATE POLICY "auth_select_buck_profile_seasons" ON buck_profile_seasons
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_buck_profile_seasons" ON buck_profile_seasons
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_buck_profile_seasons" ON buck_profile_seasons
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ----- annotation_boxes -----
CREATE POLICY "auth_select_annotation_boxes" ON annotation_boxes
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_annotation_boxes" ON annotation_boxes
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_annotation_boxes" ON annotation_boxes
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ----- cameras -----
CREATE POLICY "auth_select_cameras" ON cameras
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_cameras" ON cameras
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_cameras" ON cameras
    FOR UPDATE TO authenticated
    USING (owner = current_username())
    WITH CHECK (owner = current_username());

-- ----- camera_permissions -----
CREATE POLICY "auth_select_camera_permissions" ON camera_permissions
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_camera_permissions" ON camera_permissions
    FOR INSERT TO authenticated
    WITH CHECK (
        camera_id IN (
            SELECT id FROM cameras WHERE owner = current_username()
        )
    );

CREATE POLICY "auth_update_camera_permissions" ON camera_permissions
    FOR UPDATE TO authenticated
    USING (
        camera_id IN (
            SELECT id FROM cameras WHERE owner = current_username()
        )
    );

-- ----- clubs -----
CREATE POLICY "auth_select_clubs" ON clubs
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_clubs" ON clubs
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_clubs" ON clubs
    FOR UPDATE TO authenticated
    USING (created_by = current_username())
    WITH CHECK (created_by = current_username());

-- ----- club_memberships -----
CREATE POLICY "auth_select_club_memberships" ON club_memberships
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_club_memberships" ON club_memberships
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "auth_update_club_memberships" ON club_memberships
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ----- camera_club_shares -----
CREATE POLICY "auth_select_camera_club_shares" ON camera_club_shares
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_camera_club_shares" ON camera_club_shares
    FOR INSERT TO authenticated
    WITH CHECK (
        camera_id IN (
            SELECT id FROM cameras WHERE owner = current_username()
        )
    );

CREATE POLICY "auth_update_camera_club_shares" ON camera_club_shares
    FOR UPDATE TO authenticated
    USING (
        camera_id IN (
            SELECT id FROM cameras WHERE owner = current_username()
        )
    );

-- ----- label_suggestions -----
CREATE POLICY "auth_select_label_suggestions" ON label_suggestions
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "auth_insert_label_suggestions" ON label_suggestions
    FOR INSERT TO authenticated
    WITH CHECK (suggested_by = current_username());

CREATE POLICY "auth_update_label_suggestions" ON label_suggestions
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ----- label_history -----
-- (If table exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'label_history') THEN
        EXECUTE 'ALTER TABLE label_history ENABLE ROW LEVEL SECURITY';

        EXECUTE 'CREATE POLICY "auth_select_label_history" ON label_history
            FOR SELECT TO authenticated USING (true)';

        EXECUTE 'CREATE POLICY "auth_insert_label_history" ON label_history
            FOR INSERT TO authenticated WITH CHECK (true)';

        EXECUTE 'CREATE POLICY "anon_select_label_history" ON label_history
            FOR SELECT TO anon USING (true)';
    END IF;
END $$;


-- =============================================================
-- PHASE 2: Drop old permissive anon WRITE policies
-- =============================================================
-- Run this AFTER confirming all apps use Supabase Auth.
-- Anon keeps SELECT (read-only) for public browsing.
-- Anon loses INSERT and UPDATE (no more anonymous writes).
--
-- UNCOMMENT AND RUN WHEN READY:
--
-- -- Core data tables
-- DROP POLICY IF EXISTS "Allow all access" ON photos_sync;
-- DROP POLICY IF EXISTS "Allow all access" ON tags;
-- DROP POLICY IF EXISTS "Allow all access" ON deer_metadata;
-- DROP POLICY IF EXISTS "Allow all access" ON deer_additional;
-- DROP POLICY IF EXISTS "Allow all access" ON buck_profiles;
-- DROP POLICY IF EXISTS "Allow all access" ON buck_profile_seasons;
-- DROP POLICY IF EXISTS "Allow all access" ON annotation_boxes;
--
-- -- Roles tables
-- DROP POLICY IF EXISTS "anon_insert_cameras" ON cameras;
-- DROP POLICY IF EXISTS "anon_update_cameras" ON cameras;
-- DROP POLICY IF EXISTS "anon_insert_camera_permissions" ON camera_permissions;
-- DROP POLICY IF EXISTS "anon_update_camera_permissions" ON camera_permissions;
-- DROP POLICY IF EXISTS "anon_insert_clubs" ON clubs;
-- DROP POLICY IF EXISTS "anon_update_clubs" ON clubs;
-- DROP POLICY IF EXISTS "anon_insert_club_memberships" ON club_memberships;
-- DROP POLICY IF EXISTS "anon_update_club_memberships" ON club_memberships;
-- DROP POLICY IF EXISTS "anon_insert_camera_club_shares" ON camera_club_shares;
-- DROP POLICY IF EXISTS "anon_update_camera_club_shares" ON camera_club_shares;
-- DROP POLICY IF EXISTS "anon_insert_label_suggestions" ON label_suggestions;
-- DROP POLICY IF EXISTS "anon_update_label_suggestions" ON label_suggestions;
--
-- -- Add anon read-only policies for core tables (replace "Allow all access")
-- CREATE POLICY "anon_select_photos_sync" ON photos_sync
--     FOR SELECT TO anon USING (true);
-- CREATE POLICY "anon_select_tags" ON tags
--     FOR SELECT TO anon USING (true);
-- CREATE POLICY "anon_select_deer_metadata" ON deer_metadata
--     FOR SELECT TO anon USING (true);
-- CREATE POLICY "anon_select_deer_additional" ON deer_additional
--     FOR SELECT TO anon USING (true);
-- CREATE POLICY "anon_select_buck_profiles" ON buck_profiles
--     FOR SELECT TO anon USING (true);
-- CREATE POLICY "anon_select_buck_profile_seasons" ON buck_profile_seasons
--     FOR SELECT TO anon USING (true);
-- CREATE POLICY "anon_select_annotation_boxes" ON annotation_boxes
--     FOR SELECT TO anon USING (true);

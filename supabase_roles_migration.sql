-- =============================================================
-- Supabase Migration: Camera-Owner Roles & Label Suggestions
-- Run this in Supabase SQL Editor (one-time)
-- =============================================================

-- 1. Cameras table
CREATE TABLE IF NOT EXISTS cameras (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    owner TEXT NOT NULL,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 2. Camera permissions (owner/member per camera)
CREATE TABLE IF NOT EXISTS camera_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id UUID NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    username TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'member')),
    granted_by TEXT,
    granted_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(camera_id, username)
);

-- 3. Clubs
CREATE TABLE IF NOT EXISTS clubs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 4. Club memberships
CREATE TABLE IF NOT EXISTS club_memberships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    club_id UUID NOT NULL REFERENCES clubs(id) ON DELETE CASCADE,
    username TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(club_id, username)
);

-- 5. Camera-to-club shares
CREATE TABLE IF NOT EXISTS camera_club_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id UUID NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    club_id UUID NOT NULL REFERENCES clubs(id) ON DELETE CASCADE,
    shared_by TEXT,
    visibility TEXT DEFAULT 'full' CHECK (visibility IN ('full', 'limited')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(camera_id, club_id)
);

-- 6. Label suggestions (pending labels from members)
CREATE TABLE IF NOT EXISTS label_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id BIGINT,
    file_hash TEXT,
    tag_name TEXT NOT NULL,
    suggested_by TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,
    camera_id UUID REFERENCES cameras(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    UNIQUE(file_hash, tag_name, suggested_by)
);

-- 7. Add camera_id to photos_sync (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'photos_sync' AND column_name = 'camera_id'
    ) THEN
        ALTER TABLE photos_sync ADD COLUMN camera_id UUID;
    END IF;
END $$;

-- 8. Add created_by to tags (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tags' AND column_name = 'created_by'
    ) THEN
        ALTER TABLE tags ADD COLUMN created_by TEXT;
    END IF;
END $$;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_cameras_owner ON cameras(owner);
CREATE INDEX IF NOT EXISTS idx_camera_permissions_camera ON camera_permissions(camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_permissions_user ON camera_permissions(username);
CREATE INDEX IF NOT EXISTS idx_club_memberships_club ON club_memberships(club_id);
CREATE INDEX IF NOT EXISTS idx_club_memberships_user ON club_memberships(username);
CREATE INDEX IF NOT EXISTS idx_camera_club_shares_camera ON camera_club_shares(camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_club_shares_club ON camera_club_shares(club_id);
CREATE INDEX IF NOT EXISTS idx_label_suggestions_photo ON label_suggestions(photo_id);
CREATE INDEX IF NOT EXISTS idx_label_suggestions_file_hash ON label_suggestions(file_hash);
CREATE INDEX IF NOT EXISTS idx_label_suggestions_camera ON label_suggestions(camera_id);
CREATE INDEX IF NOT EXISTS idx_label_suggestions_status ON label_suggestions(status);
CREATE INDEX IF NOT EXISTS idx_label_suggestions_deleted ON label_suggestions(deleted_at);
CREATE INDEX IF NOT EXISTS idx_photos_sync_camera_id ON photos_sync(camera_id);

-- Auto-update updated_at triggers
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY['cameras', 'camera_permissions', 'clubs', 'club_memberships', 'camera_club_shares', 'label_suggestions']
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I', 'trg_' || t || '_updated_at', t);
        EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE ON %I FOR EACH ROW EXECUTE FUNCTION update_updated_at()', 'trg_' || t || '_updated_at', t);
    END LOOP;
END $$;

-- =============================================================
-- Row Level Security (RLS) Policies
-- =============================================================
--
-- MVP PHASE (current): Username-based auth, no Supabase Auth yet.
-- All operations go through the anon key, so we enable RLS but
-- allow full SELECT/INSERT/UPDATE for the anon role. DELETE is
-- intentionally omitted -- use soft-deletes (deleted_at) instead.
--
-- FUTURE PHASE: Once Supabase Auth is integrated, replace these
-- permissive anon policies with proper per-user policies that
-- reference auth.uid() and auth.jwt(). See the commented-out
-- examples below each table's MVP policy.
-- =============================================================

-- ----- cameras -----
ALTER TABLE cameras ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_select_cameras" ON cameras
    FOR SELECT TO anon USING (true);

CREATE POLICY "anon_insert_cameras" ON cameras
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_update_cameras" ON cameras
    FOR UPDATE TO anon USING (true) WITH CHECK (true);

-- FUTURE: Replace the above with authenticated policies:
--
-- CREATE POLICY "owner_select_cameras" ON cameras
--     FOR SELECT TO authenticated
--     USING (
--         owner = auth.jwt()->>'username'
--         OR id IN (
--             SELECT camera_id FROM camera_permissions
--             WHERE username = auth.jwt()->>'username'
--         )
--     );
--
-- CREATE POLICY "owner_insert_cameras" ON cameras
--     FOR INSERT TO authenticated
--     WITH CHECK (owner = auth.jwt()->>'username');
--
-- CREATE POLICY "owner_update_cameras" ON cameras
--     FOR UPDATE TO authenticated
--     USING (owner = auth.jwt()->>'username')
--     WITH CHECK (owner = auth.jwt()->>'username');

-- ----- camera_permissions -----
ALTER TABLE camera_permissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_select_camera_permissions" ON camera_permissions
    FOR SELECT TO anon USING (true);

CREATE POLICY "anon_insert_camera_permissions" ON camera_permissions
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_update_camera_permissions" ON camera_permissions
    FOR UPDATE TO anon USING (true) WITH CHECK (true);

-- FUTURE: Only camera owners can manage permissions:
--
-- CREATE POLICY "owner_manage_camera_permissions" ON camera_permissions
--     FOR ALL TO authenticated
--     USING (
--         camera_id IN (
--             SELECT id FROM cameras WHERE owner = auth.jwt()->>'username'
--         )
--     )
--     WITH CHECK (
--         camera_id IN (
--             SELECT id FROM cameras WHERE owner = auth.jwt()->>'username'
--         )
--     );
--
-- CREATE POLICY "member_view_own_permission" ON camera_permissions
--     FOR SELECT TO authenticated
--     USING (username = auth.jwt()->>'username');

-- ----- clubs -----
ALTER TABLE clubs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_select_clubs" ON clubs
    FOR SELECT TO anon USING (true);

CREATE POLICY "anon_insert_clubs" ON clubs
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_update_clubs" ON clubs
    FOR UPDATE TO anon USING (true) WITH CHECK (true);

-- FUTURE: Club members can view, creator can update:
--
-- CREATE POLICY "member_select_clubs" ON clubs
--     FOR SELECT TO authenticated
--     USING (
--         id IN (
--             SELECT club_id FROM club_memberships
--             WHERE username = auth.jwt()->>'username'
--         )
--     );
--
-- CREATE POLICY "creator_insert_clubs" ON clubs
--     FOR INSERT TO authenticated
--     WITH CHECK (created_by = auth.jwt()->>'username');
--
-- CREATE POLICY "creator_update_clubs" ON clubs
--     FOR UPDATE TO authenticated
--     USING (created_by = auth.jwt()->>'username')
--     WITH CHECK (created_by = auth.jwt()->>'username');

-- ----- club_memberships -----
ALTER TABLE club_memberships ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_select_club_memberships" ON club_memberships
    FOR SELECT TO anon USING (true);

CREATE POLICY "anon_insert_club_memberships" ON club_memberships
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_update_club_memberships" ON club_memberships
    FOR UPDATE TO anon USING (true) WITH CHECK (true);

-- FUTURE: Club creator manages members, members see their own:
--
-- CREATE POLICY "creator_manage_club_memberships" ON club_memberships
--     FOR ALL TO authenticated
--     USING (
--         club_id IN (
--             SELECT id FROM clubs WHERE created_by = auth.jwt()->>'username'
--         )
--     )
--     WITH CHECK (
--         club_id IN (
--             SELECT id FROM clubs WHERE created_by = auth.jwt()->>'username'
--         )
--     );
--
-- CREATE POLICY "member_view_own_membership" ON club_memberships
--     FOR SELECT TO authenticated
--     USING (username = auth.jwt()->>'username');

-- ----- camera_club_shares -----
ALTER TABLE camera_club_shares ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_select_camera_club_shares" ON camera_club_shares
    FOR SELECT TO anon USING (true);

CREATE POLICY "anon_insert_camera_club_shares" ON camera_club_shares
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_update_camera_club_shares" ON camera_club_shares
    FOR UPDATE TO anon USING (true) WITH CHECK (true);

-- FUTURE: Camera owners share, club members can view:
--
-- CREATE POLICY "owner_manage_shares" ON camera_club_shares
--     FOR ALL TO authenticated
--     USING (
--         camera_id IN (
--             SELECT id FROM cameras WHERE owner = auth.jwt()->>'username'
--         )
--     )
--     WITH CHECK (
--         camera_id IN (
--             SELECT id FROM cameras WHERE owner = auth.jwt()->>'username'
--         )
--     );
--
-- CREATE POLICY "club_member_view_shares" ON camera_club_shares
--     FOR SELECT TO authenticated
--     USING (
--         club_id IN (
--             SELECT club_id FROM club_memberships
--             WHERE username = auth.jwt()->>'username'
--         )
--     );

-- ----- label_suggestions -----
ALTER TABLE label_suggestions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_select_label_suggestions" ON label_suggestions
    FOR SELECT TO anon USING (true);

CREATE POLICY "anon_insert_label_suggestions" ON label_suggestions
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_update_label_suggestions" ON label_suggestions
    FOR UPDATE TO anon USING (true) WITH CHECK (true);

-- FUTURE: Anyone with camera access can suggest, owners can review:
--
-- CREATE POLICY "member_insert_label_suggestions" ON label_suggestions
--     FOR INSERT TO authenticated
--     WITH CHECK (
--         suggested_by = auth.jwt()->>'username'
--         AND camera_id IN (
--             SELECT camera_id FROM camera_permissions
--             WHERE username = auth.jwt()->>'username'
--         )
--     );
--
-- CREATE POLICY "member_select_label_suggestions" ON label_suggestions
--     FOR SELECT TO authenticated
--     USING (
--         camera_id IN (
--             SELECT camera_id FROM camera_permissions
--             WHERE username = auth.jwt()->>'username'
--         )
--     );
--
-- CREATE POLICY "owner_review_label_suggestions" ON label_suggestions
--     FOR UPDATE TO authenticated
--     USING (
--         camera_id IN (
--             SELECT id FROM cameras WHERE owner = auth.jwt()->>'username'
--         )
--     )
--     WITH CHECK (
--         reviewed_by = auth.jwt()->>'username'
--     );

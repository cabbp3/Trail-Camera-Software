import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://iwvehmthbjcvdqjqxtty.supabase.co'
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml3dmVobXRoYmpjdmRxanF4dHR5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjYyMDI0NDQsImV4cCI6MjA4MTc3ODQ0NH0._z6WAfUBP_Qda0IcjTS_LEI_J7r147BrmSib3dyneLE'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// R2 config for generating thumbnail URLs
export const r2Config = {
  endpoint: 'https://856273fcd044ac1fac11116e7d92ba0f.r2.cloudflarestorage.com',
  bucket: 'trailcam-photos',
  publicUrl: 'https://pub-trailcam.r2.dev' // Public bucket URL if configured, otherwise use signed URLs
}

// R2 public base URL (from mobile app)
const R2_PUBLIC_URL = 'https://pub-0a6b925ce6bd4d4fb5821bab00479d33.r2.dev'

// Helper to get thumbnail URL from file_hash
export function getThumbnailUrl(fileHash) {
  if (!fileHash) return null
  // Thumbnails are stored as thumbnails/{hash}_thumb.jpg
  return `${R2_PUBLIC_URL}/thumbnails/${fileHash}_thumb.jpg`
}

// Helper to get full photo URL from file_hash
export function getPhotoUrl(fileHash) {
  if (!fileHash) return null
  // Full photos are stored as photos/{hash}.jpg
  return `${R2_PUBLIC_URL}/photos/${fileHash}.jpg`
}

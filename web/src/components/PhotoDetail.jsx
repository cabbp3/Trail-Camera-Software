import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { supabase, getPhotoUrl, getThumbnailUrl } from '../lib/supabase'

function PhotoDetail() {
  const { fileHash } = useParams()
  const navigate = useNavigate()
  const [photo, setPhoto] = useState(null)
  const [tags, setTags] = useState([])
  const [deerMeta, setDeerMeta] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [newTag, setNewTag] = useState('')

  // Zoom state
  const [zoom, setZoom] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const containerRef = useRef(null)

  // Common species tags
  const commonTags = ['Buck', 'Doe', 'Deer', 'Turkey', 'Coyote', 'Raccoon', 'Squirrel', 'Bird', 'Empty']

  useEffect(() => {
    async function fetchPhotoDetails() {
      setLoading(true)

      // Fetch photo
      const { data: photoData } = await supabase
        .from('photos_sync')
        .select('*')
        .eq('file_hash', fileHash)
        .single()

      // Fetch tags
      const { data: tagsData } = await supabase
        .from('tags')
        .select('tag_name')
        .eq('file_hash', fileHash)

      // Fetch deer metadata
      const { data: deerData } = await supabase
        .from('deer_metadata')
        .select('*')
        .eq('file_hash', fileHash)
        .single()

      setPhoto(photoData)
      setTags(tagsData?.map(t => t.tag_name) || [])
      setDeerMeta(deerData)
      setLoading(false)
    }

    if (fileHash) {
      fetchPhotoDetails()
    }
  }, [fileHash])

  const addTag = async (tagName) => {
    if (!tagName || tags.includes(tagName)) return
    setSaving(true)

    const { error } = await supabase
      .from('tags')
      .insert({ file_hash: fileHash, tag_name: tagName })

    if (!error) {
      setTags([...tags, tagName])
    }
    setNewTag('')
    setSaving(false)
  }

  const removeTag = async (tagName) => {
    setSaving(true)

    const { error } = await supabase
      .from('tags')
      .delete()
      .eq('file_hash', fileHash)
      .eq('tag_name', tagName)

    if (!error) {
      setTags(tags.filter(t => t !== tagName))
    }
    setSaving(false)
  }

  const toggleFavorite = async () => {
    if (!photo) return
    setSaving(true)

    const { error } = await supabase
      .from('photos_sync')
      .update({ favorite: !photo.favorite })
      .eq('file_hash', fileHash)

    if (!error) {
      setPhoto({ ...photo, favorite: !photo.favorite })
    }
    setSaving(false)
  }

  const updateNotes = async (notes) => {
    setSaving(true)

    const { error } = await supabase
      .from('photos_sync')
      .update({ notes })
      .eq('file_hash', fileHash)

    if (!error) {
      setPhoto({ ...photo, notes })
    }
    setSaving(false)
  }

  const updateDeerMeta = async (field, value) => {
    setSaving(true)

    if (deerMeta) {
      // Update existing
      const { error } = await supabase
        .from('deer_metadata')
        .update({ [field]: value })
        .eq('file_hash', fileHash)

      if (!error) {
        setDeerMeta({ ...deerMeta, [field]: value })
      }
    } else {
      // Insert new
      const { data, error } = await supabase
        .from('deer_metadata')
        .insert({ file_hash: fileHash, [field]: value })
        .select()
        .single()

      if (!error) {
        setDeerMeta(data)
      }
    }
    setSaving(false)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  if (!photo) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        <p>Photo not found</p>
      </div>
    )
  }

  const photoUrl = getPhotoUrl(fileHash)
  const dateTaken = photo.date_taken
    ? new Date(photo.date_taken).toLocaleString()
    : 'Unknown'

  return (
    <div className="h-full flex flex-col lg:flex-row gap-4">
      {/* Photo */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-center gap-4 mb-4">
          <button
            onClick={() => navigate(-1)}
            className="text-gray-400 hover:text-white"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <h2 className="text-lg font-medium text-white truncate flex-1">
            {photo.original_name || 'Photo'}
          </h2>
          <button
            onClick={toggleFavorite}
            className={`p-2 rounded-lg ${photo.favorite ? 'text-yellow-400' : 'text-gray-500 hover:text-yellow-400'}`}
          >
            <svg className="w-6 h-6 fill-current" viewBox="0 0 20 20">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
          </button>
        </div>
        <div
          ref={containerRef}
          className="flex-1 bg-black rounded-lg overflow-hidden flex items-center justify-center relative cursor-grab active:cursor-grabbing"
          onWheel={(e) => {
            e.preventDefault()
            const delta = e.deltaY > 0 ? -0.25 : 0.25
            setZoom(z => Math.min(Math.max(z + delta, 1), 5))
            if (zoom + delta <= 1) setPosition({ x: 0, y: 0 })
          }}
          onMouseDown={(e) => {
            if (zoom > 1) {
              setIsDragging(true)
              setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y })
            }
          }}
          onMouseMove={(e) => {
            if (isDragging && zoom > 1) {
              setPosition({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y })
            }
          }}
          onMouseUp={() => setIsDragging(false)}
          onMouseLeave={() => setIsDragging(false)}
          onDoubleClick={() => {
            if (zoom === 1) {
              setZoom(2)
            } else {
              setZoom(1)
              setPosition({ x: 0, y: 0 })
            }
          }}
        >
          <img
            src={photoUrl}
            alt={photo.original_name}
            className="max-w-full max-h-full object-contain select-none"
            style={{
              transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
              transition: isDragging ? 'none' : 'transform 0.2s ease'
            }}
            draggable={false}
            crossOrigin="anonymous"
            onError={(e) => {
              e.target.src = getThumbnailUrl(fileHash)
            }}
          />
          {/* Zoom controls */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-gray-900/80 rounded-lg px-3 py-2">
            <button
              onClick={() => { setZoom(z => Math.max(z - 0.5, 1)); if (zoom <= 1.5) setPosition({ x: 0, y: 0 }) }}
              className="text-white hover:text-blue-400 p-1 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Zoom out"
              disabled={zoom <= 1}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
              </svg>
            </button>
            <span className="text-white text-sm min-w-[3rem] text-center">
              {zoom === 1 ? 'Fit' : `${Math.round(zoom * 100)}%`}
            </span>
            <button
              onClick={() => setZoom(z => Math.min(z + 0.5, 5))}
              className="text-white hover:text-blue-400 p-1"
              title="Zoom in"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
            <button
              onClick={() => { setZoom(1); setPosition({ x: 0, y: 0 }) }}
              className="text-white hover:text-blue-400 p-1 ml-2"
              title="Fit to screen"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Details Panel */}
      <div className="w-full lg:w-80 bg-gray-800 rounded-lg p-4 overflow-y-auto">
        <h3 className="text-lg font-medium text-white mb-4">Photo Details</h3>

        {/* Info */}
        <div className="space-y-2 text-sm mb-6">
          <div className="flex justify-between">
            <span className="text-gray-400">Date</span>
            <span className="text-white">{dateTaken}</span>
          </div>
          {photo.camera_location && (
            <div className="flex justify-between">
              <span className="text-gray-400">Camera</span>
              <span className="text-white">{photo.camera_location}</span>
            </div>
          )}
          {photo.collection && (
            <div className="flex justify-between">
              <span className="text-gray-400">Collection</span>
              <span className="text-white">{photo.collection}</span>
            </div>
          )}
        </div>

        {/* Tags */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Species Tags</h4>
          <div className="flex flex-wrap gap-2 mb-3">
            {tags.map(tag => (
              <span
                key={tag}
                className="inline-flex items-center gap-1 px-3 py-1 bg-blue-600 text-white text-sm rounded-full"
              >
                {tag}
                <button
                  onClick={() => removeTag(tag)}
                  className="hover:text-red-300"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </span>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {commonTags.filter(t => !tags.includes(t)).map(tag => (
              <button
                key={tag}
                onClick={() => addTag(tag)}
                className="px-3 py-1 bg-gray-700 text-gray-300 text-sm rounded-full hover:bg-gray-600"
              >
                + {tag}
              </button>
            ))}
          </div>
        </div>

        {/* Deer ID */}
        {(tags.includes('Buck') || tags.includes('Deer') || deerMeta?.deer_id) && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-400 mb-2">Deer ID</h4>
            <input
              type="text"
              value={deerMeta?.deer_id || ''}
              onChange={(e) => updateDeerMeta('deer_id', e.target.value)}
              placeholder="Enter deer name/ID..."
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
          </div>
        )}

        {/* Notes */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Notes</h4>
          <textarea
            value={photo.notes || ''}
            onChange={(e) => updateNotes(e.target.value)}
            placeholder="Add notes..."
            rows={3}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 resize-none"
          />
        </div>

        {/* Saving indicator */}
        {saving && (
          <div className="text-center text-gray-400 text-sm">
            Saving...
          </div>
        )}
      </div>
    </div>
  )
}

export default PhotoDetail

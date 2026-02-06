import { Link } from 'react-router-dom'
import { getThumbnailUrl } from '../lib/supabase'

function PhotoGrid({ photos, loading }) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  if (photos.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        <p>No photos found. Try adjusting your filters.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-wrap gap-3">
      {photos.map((photo) => (
        <PhotoCard key={photo.file_hash} photo={photo} />
      ))}
    </div>
  )
}

function PhotoCard({ photo }) {
  const thumbnailUrl = getThumbnailUrl(photo.file_hash)
  const dateTaken = photo.date_taken
    ? new Date(photo.date_taken).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit'
      })
    : 'Unknown date'

  return (
    <Link
      to={`/photo/${photo.file_hash}`}
      className="photo-card block bg-gray-800 rounded-lg overflow-hidden hover:ring-2 hover:ring-blue-500 transition-all"
      style={{ width: '200px' }}
    >
      <div className="relative">
        {thumbnailUrl ? (
          <img
            src={thumbnailUrl}
            alt={photo.original_name || 'Trail cam photo'}
            className="w-full h-auto"
            loading="lazy"
            crossOrigin="anonymous"
            onError={(e) => {
              e.target.style.display = 'none'
              e.target.nextSibling.style.display = 'flex'
            }}
          />
        ) : null}
        <div
          className="bg-gray-700 items-center justify-center text-gray-500 hidden aspect-video"
          style={{ display: thumbnailUrl ? 'none' : 'flex' }}
        >
          <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        {photo.favorite && (
          <div className="absolute top-2 right-2 text-yellow-400">
            <svg className="w-5 h-5 fill-current" viewBox="0 0 20 20">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
          </div>
        )}
      </div>
      <div className="p-2">
        <p className="text-xs text-gray-400 truncate">{dateTaken}</p>
        {photo.camera_location && (
          <p className="text-xs text-gray-500 truncate">{photo.camera_location}</p>
        )}
      </div>
    </Link>
  )
}

export default PhotoGrid

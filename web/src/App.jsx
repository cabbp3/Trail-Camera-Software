import { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import { supabase } from './lib/supabase'
import PhotoGrid from './components/PhotoGrid'
import PhotoDetail from './components/PhotoDetail'
import FilterBar from './components/FilterBar'
import Sidebar from './components/Sidebar'

const PHOTOS_PER_PAGE = 50

function App() {
  const [photos, setPhotos] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [totalCount, setTotalCount] = useState(0)
  const [filters, setFilters] = useState({
    species: '',
    collection: '',
    camera: '',
    dateFrom: '',
    dateTo: '',
    showArchived: false,
    favoritesOnly: false,
    deerId: ''
  })
  const [filterOptions, setFilterOptions] = useState({
    species: [],
    collections: [],
    cameras: [],
    deerIds: []
  })

  // Reset to page 1 when filters change
  useEffect(() => {
    setPage(1)
  }, [filters])

  // Fetch filter options on mount
  useEffect(() => {
    async function fetchFilterOptions() {
      // Helper to paginate through all rows (Supabase has 1000 row limit per request)
      async function fetchAllRows(table, column) {
        const allValues = new Set()
        let offset = 0
        const batchSize = 1000

        while (true) {
          const { data, error } = await supabase
            .from(table)
            .select(column)
            .range(offset, offset + batchSize - 1)

          if (error || !data || data.length === 0) break

          data.forEach(row => {
            if (row[column]) allValues.add(row[column])
          })

          if (data.length < batchSize) break
          offset += batchSize
        }

        return [...allValues].sort()
      }

      // Fetch all filter options in parallel
      const [species, collections, cameras, deerIds] = await Promise.all([
        fetchAllRows('tags', 'tag_name'),
        fetchAllRows('photos_sync', 'collection'),
        fetchAllRows('photos_sync', 'camera_location'),
        fetchAllRows('deer_metadata', 'deer_id')
      ])

      setFilterOptions({
        species,
        collections,
        cameras,
        deerIds
      })
    }
    fetchFilterOptions()
  }, [])

  // Fetch photos when filters or page change
  useEffect(() => {
    async function fetchPhotos() {
      setLoading(true)

      // Calculate range for pagination
      const from = (page - 1) * PHOTOS_PER_PAGE
      const to = from + PHOTOS_PER_PAGE - 1

      // First get hashes to filter by (species/deer) if needed
      let hashFilter = null

      if (filters.species) {
        const { data: taggedHashes } = await supabase
          .from('tags')
          .select('file_hash')
          .eq('tag_name', filters.species)
        hashFilter = new Set(taggedHashes?.map(t => t.file_hash) || [])
      }

      if (filters.deerId) {
        const { data: deerPhotos } = await supabase
          .from('deer_metadata')
          .select('file_hash')
          .eq('deer_id', filters.deerId)
        const deerHashes = new Set(deerPhotos?.map(d => d.file_hash) || [])
        if (hashFilter) {
          // Intersection
          hashFilter = new Set([...hashFilter].filter(h => deerHashes.has(h)))
        } else {
          hashFilter = deerHashes
        }
      }

      // Build count query
      let countQuery = supabase
        .from('photos_sync')
        .select('*', { count: 'exact', head: true })

      // Build data query
      let query = supabase
        .from('photos_sync')
        .select('*')
        .order('date_taken', { ascending: false })
        .range(from, to)

      // Apply filters to both queries
      if (!filters.showArchived) {
        countQuery = countQuery.eq('archived', false)
        query = query.eq('archived', false)
      }
      if (filters.favoritesOnly) {
        countQuery = countQuery.eq('favorite', true)
        query = query.eq('favorite', true)
      }
      if (filters.collection) {
        countQuery = countQuery.eq('collection', filters.collection)
        query = query.eq('collection', filters.collection)
      }
      if (filters.camera) {
        countQuery = countQuery.eq('camera_location', filters.camera)
        query = query.eq('camera_location', filters.camera)
      }
      if (filters.dateFrom) {
        countQuery = countQuery.gte('date_taken', filters.dateFrom)
        query = query.gte('date_taken', filters.dateFrom)
      }
      if (filters.dateTo) {
        countQuery = countQuery.lte('date_taken', filters.dateTo + 'T23:59:59')
        query = query.lte('date_taken', filters.dateTo + 'T23:59:59')
      }

      // If filtering by species/deer, we need to filter by hash
      if (hashFilter) {
        const hashArray = [...hashFilter]
        countQuery = countQuery.in('file_hash', hashArray)
        query = query.in('file_hash', hashArray)
      }

      // Execute queries
      const [countResult, dataResult] = await Promise.all([
        countQuery,
        query
      ])

      if (dataResult.error) {
        console.error('Error fetching photos:', dataResult.error)
        setLoading(false)
        return
      }

      setTotalCount(countResult.count || 0)
      setPhotos(dataResult.data || [])
      setLoading(false)
    }

    fetchPhotos()
  }, [filters, page])

  return (
    <div className="flex h-screen bg-gray-900">
      <Sidebar
        filters={filters}
        setFilters={setFilters}
        filterOptions={filterOptions}
      />
      <main className="flex-1 overflow-hidden flex flex-col">
        <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">Trail Camera Organizer</h1>
              <p className="text-gray-400 text-sm mt-1">
                {loading ? 'Loading...' : `${totalCount} photos total`}
              </p>
            </div>
            {/* Pagination */}
            {totalCount > PHOTOS_PER_PAGE && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1 || loading}
                  className="px-3 py-1 bg-gray-700 text-white rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Prev
                </button>
                <span className="text-white text-sm">
                  Page {page} of {Math.ceil(totalCount / PHOTOS_PER_PAGE)}
                </span>
                <button
                  onClick={() => setPage(p => Math.min(Math.ceil(totalCount / PHOTOS_PER_PAGE), p + 1))}
                  disabled={page >= Math.ceil(totalCount / PHOTOS_PER_PAGE) || loading}
                  className="px-3 py-1 bg-gray-700 text-white rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        </header>
        <div className="flex-1 overflow-auto p-4">
          <Routes>
            <Route
              path="/"
              element={<PhotoGrid photos={photos} loading={loading} />}
            />
            <Route
              path="/photo/:fileHash"
              element={<PhotoDetail />}
            />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App

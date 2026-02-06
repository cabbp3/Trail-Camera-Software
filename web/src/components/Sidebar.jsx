import { useState } from 'react'

function Sidebar({ filters, setFilters, filterOptions }) {
  const [collapsed, setCollapsed] = useState(false)

  const updateFilter = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  const clearFilters = () => {
    setFilters({
      species: '',
      collection: '',
      camera: '',
      dateFrom: '',
      dateTo: '',
      showArchived: false,
      favoritesOnly: false,
      deerId: ''
    })
  }

  const hasActiveFilters = filters.species || filters.collection || filters.camera ||
    filters.dateFrom || filters.dateTo || filters.showArchived || filters.favoritesOnly || filters.deerId

  if (collapsed) {
    return (
      <div className="w-12 bg-gray-800 border-r border-gray-700 flex flex-col items-center py-4">
        <button
          onClick={() => setCollapsed(false)}
          className="p-2 text-gray-400 hover:text-white"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
    )
  }

  return (
    <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Filters</h2>
        <button
          onClick={() => setCollapsed(true)}
          className="p-1 text-gray-400 hover:text-white"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </button>
      </div>

      {/* Filters */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Quick Filters */}
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.favoritesOnly}
              onChange={(e) => updateFilter('favoritesOnly', e.target.checked)}
              className="rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
            />
            Favorites only
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showArchived}
              onChange={(e) => updateFilter('showArchived', e.target.checked)}
              className="rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
            />
            Show archived
          </label>
        </div>

        {/* Species */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">Species</label>
          <select
            value={filters.species}
            onChange={(e) => updateFilter('species', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="">All species</option>
            {filterOptions.species.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        {/* Deer ID */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">Deer ID</label>
          <select
            value={filters.deerId}
            onChange={(e) => updateFilter('deerId', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="">All deer</option>
            {filterOptions.deerIds.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </div>

        {/* Collection */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">Collection</label>
          <select
            value={filters.collection}
            onChange={(e) => updateFilter('collection', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="">All collections</option>
            {filterOptions.collections.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        {/* Camera */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">Camera</label>
          <select
            value={filters.camera}
            onChange={(e) => updateFilter('camera', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="">All cameras</option>
            {filterOptions.cameras.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        {/* Date Range */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">Date From</label>
          <input
            type="date"
            value={filters.dateFrom}
            onChange={(e) => updateFilter('dateFrom', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">Date To</label>
          <input
            type="date"
            value={filters.dateTo}
            onChange={(e) => updateFilter('dateTo', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>

      {/* Clear Filters */}
      {hasActiveFilters && (
        <div className="p-4 border-t border-gray-700">
          <button
            onClick={clearFilters}
            className="w-full px-4 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
          >
            Clear Filters
          </button>
        </div>
      )}
    </div>
  )
}

export default Sidebar

// TrailCam Trainer - Web Frontend
// Matches desktop app functionality

// ============================================
// Data Store
// ============================================

const Store = {
    photos: [],
    locations: [],
    deer: [],
    recentBucks: [],
    loaded: false,

    async init() {
        try {
            const [photosRes, locationsRes, deerRes] = await Promise.all([
                fetch('photos.json'),
                fetch('locations.json'),
                fetch('deer.json').catch(() => ({ json: () => [] }))
            ]);

            const photosData = await photosRes.json();
            const locationsData = await locationsRes.json();
            const deerData = await deerRes.json().catch(() => []);

            this.photos = photosData.map(p => ({
                id: p.id,
                filename: p.file_path.split('/').pop(),
                filePath: p.file_path,
                date: p.date_taken,
                location: p.camera_location || '',
                species: p.tags ? p.tags.split(',')[0] : null,
                allTags: p.tags ? p.tags.split(',') : [],
                thumbnail: p.thumbnail_path,
                notes: p.notes || '',
                reviewed: p.tags !== null,
                cameraModel: p.camera_model,
                sex: null,
                deerId: null,
                age: null,
                leftPoints: null,
                rightPoints: null,
                characteristics: []
            }));

            this.locations = locationsData.map(l => l.camera_location).filter(Boolean);

            this.deer = deerData.map(d => ({
                id: d.deer_id || d.id,
                name: d.name || d.deer_id
            }));

            // Load local modifications
            this.loadModifications();

            this.loaded = true;
            console.log(`Loaded ${this.photos.length} photos, ${this.locations.length} locations`);
        } catch (error) {
            console.error('Failed to load data:', error);
        }
    },

    loadModifications() {
        try {
            const mods = JSON.parse(localStorage.getItem('photo_mods') || '{}');
            Object.entries(mods).forEach(([id, data]) => {
                const photo = this.photos.find(p => p.id === parseInt(id));
                if (photo) Object.assign(photo, data);
            });

            this.recentBucks = JSON.parse(localStorage.getItem('recent_bucks') || '[]');
        } catch (e) {
            console.error('Failed to load modifications:', e);
        }
    },

    savePhoto(id, updates) {
        const photo = this.photos.find(p => p.id === id);
        if (!photo) return;

        Object.assign(photo, updates);

        // Save to localStorage
        const mods = JSON.parse(localStorage.getItem('photo_mods') || '{}');
        mods[id] = updates;
        localStorage.setItem('photo_mods', JSON.stringify(mods));

        // Track recent bucks
        if (updates.deerId && !this.recentBucks.includes(updates.deerId)) {
            this.recentBucks.unshift(updates.deerId);
            this.recentBucks = this.recentBucks.slice(0, 9);
            localStorage.setItem('recent_bucks', JSON.stringify(this.recentBucks));
        }
    },

    getPhoto(id) {
        return this.photos.find(p => p.id === id);
    },

    getFilteredPhotos(filters = {}) {
        let results = [...this.photos];

        if (filters.location) {
            results = results.filter(p => p.location === filters.location);
        }

        if (filters.species) {
            if (filters.species === 'untagged') {
                results = results.filter(p => !p.species);
            } else {
                results = results.filter(p => p.species?.toLowerCase() === filters.species.toLowerCase());
            }
        }

        if (filters.reviewed === 'unreviewed') {
            results = results.filter(p => !p.reviewed);
        } else if (filters.reviewed === 'reviewed') {
            results = results.filter(p => p.reviewed);
        }

        results.sort((a, b) => new Date(b.date) - new Date(a.date));
        return results;
    }
};


// ============================================
// UI Controller
// ============================================

const UI = {
    currentIndex: -1,
    filteredPhotos: [],
    selectedIds: new Set(),
    zoom: 100,
    panX: 0,
    panY: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    lastPanX: 0,
    lastPanY: 0,
    bulkSpecies: null,

    async init() {
        await Store.init();
        this.bindEvents();
        this.applyFilters();
        this.populateLocationFilter();
        this.populateLocationButtons();
        this.populateDeerSelect();
        this.updateRecentBucks();
    },

    bindEvents() {
        // Filters
        document.getElementById('location-filter').addEventListener('change', () => this.applyFilters());
        document.getElementById('species-filter').addEventListener('change', () => this.applyFilters());
        document.getElementById('review-filter').addEventListener('change', () => this.applyFilters());

        // CuddeLink
        document.getElementById('cuddelink-btn').addEventListener('click', () => this.openCuddelinkModal());
        document.getElementById('close-cuddelink').addEventListener('click', () => this.closeCuddelinkModal());
        document.getElementById('start-cuddelink').addEventListener('click', () => this.startCuddelinkDownload());
        document.getElementById('cuddelink-done').addEventListener('click', () => this.closeCuddelinkModal());
        document.getElementById('cuddelink-retry').addEventListener('click', () => this.resetCuddelinkModal());

        // Navigation
        document.getElementById('prev-btn').addEventListener('click', () => this.navigate(-1));
        document.getElementById('next-btn').addEventListener('click', () => this.navigate(1));

        // Zoom controls
        document.getElementById('zoom-slider').addEventListener('input', (e) => this.setZoom(parseInt(e.target.value)));
        document.getElementById('zoom-in').addEventListener('click', () => this.setZoom(this.zoom + 25));
        document.getElementById('zoom-out').addEventListener('click', () => this.setZoom(this.zoom - 25));
        document.getElementById('zoom-reset').addEventListener('click', () => this.resetZoom());

        // Scroll wheel zoom
        const container = document.getElementById('image-container');
        container.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });

        // Pan with mouse drag
        container.addEventListener('mousedown', (e) => this.startDrag(e));
        container.addEventListener('mousemove', (e) => this.onDrag(e));
        container.addEventListener('mouseup', () => this.endDrag());
        container.addEventListener('mouseleave', () => this.endDrag());

        // Double-click to reset
        container.addEventListener('dblclick', () => this.resetZoom());

        // Species buttons
        document.querySelectorAll('.species-btn').forEach(btn => {
            btn.addEventListener('click', () => this.selectSpecies(btn.dataset.species));
        });

        // Sex buttons
        document.querySelectorAll('.sex-btn').forEach(btn => {
            btn.addEventListener('click', () => this.selectSex(btn.dataset.sex));
        });

        // Save buttons
        document.getElementById('save-btn').addEventListener('click', () => this.saveCurrentPhoto());
        document.getElementById('save-next-btn').addEventListener('click', () => this.saveAndNext());

        // Collapsible sections
        document.getElementById('antler-toggle').addEventListener('click', () => this.toggleCollapse('antler'));
        document.getElementById('char-toggle').addEventListener('click', () => this.toggleCollapse('char'));

        // Characteristics
        document.getElementById('add-char-btn').addEventListener('click', () => this.addCharacteristic());
        document.getElementById('char-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addCharacteristic();
        });

        // Bulk actions
        document.getElementById('select-all-btn').addEventListener('click', () => this.selectAll());
        document.getElementById('clear-selection-btn').addEventListener('click', () => this.clearSelection());
        document.getElementById('compare-btn').addEventListener('click', () => this.openCompare());
        document.getElementById('bulk-species-btn').addEventListener('click', () => this.openBulkModal());

        // Modals
        document.getElementById('close-compare').addEventListener('click', () => this.closeModal('compare-modal'));
        document.getElementById('close-bulk').addEventListener('click', () => this.closeModal('bulk-modal'));
        document.getElementById('cancel-bulk').addEventListener('click', () => this.closeModal('bulk-modal'));
        document.getElementById('apply-bulk').addEventListener('click', () => this.applyBulkSpecies());

        document.querySelectorAll('.modal-backdrop').forEach(el => {
            el.addEventListener('click', () => {
                this.closeModal('compare-modal');
                this.closeModal('bulk-modal');
            });
        });

        // Bulk modal species buttons
        document.querySelectorAll('#bulk-modal .species-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('#bulk-modal .species-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.bulkSpecies = btn.dataset.species;
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    },

    // ============================================
    // Photo List
    // ============================================

    applyFilters() {
        const filters = {
            location: document.getElementById('location-filter').value,
            species: document.getElementById('species-filter').value,
            reviewed: document.getElementById('review-filter').value
        };

        this.filteredPhotos = Store.getFilteredPhotos(filters);
        this.renderPhotoList();

        document.getElementById('photo-count').textContent = `${this.filteredPhotos.length} photos`;

        // Select first photo if none selected
        if (this.currentIndex === -1 && this.filteredPhotos.length > 0) {
            this.selectPhoto(0);
        } else if (this.filteredPhotos.length === 0) {
            this.currentIndex = -1;
            this.clearPreview();
        }
    },

    renderPhotoList() {
        const container = document.getElementById('photo-list');

        container.innerHTML = this.filteredPhotos.map((photo, index) => {
            const date = new Date(photo.date);
            const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            const timeStr = date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
            const isActive = index === this.currentIndex;
            const isSelected = this.selectedIds.has(photo.id);

            let tagsHtml = '';
            if (photo.species) {
                tagsHtml += `<span class="tag-badge ${photo.species.toLowerCase()}">${photo.species}</span>`;
            }
            if (!photo.reviewed) {
                tagsHtml += `<span class="tag-badge unreviewed">New</span>`;
            }

            return `
                <div class="photo-item ${isActive ? 'active' : ''} ${isSelected ? 'selected' : ''}"
                     data-index="${index}" data-id="${photo.id}">
                    <input type="checkbox" class="photo-item-checkbox"
                           ${isSelected ? 'checked' : ''} data-id="${photo.id}">
                    <img class="photo-thumb" src="${this.getImageUrl(photo.thumbnail)}" alt=""
                         onerror="this.style.display='none'">
                    <div class="photo-item-info">
                        <div class="photo-item-date">${dateStr} ${timeStr}</div>
                        <div class="photo-item-tags">${tagsHtml}</div>
                    </div>
                </div>
            `;
        }).join('');

        // Bind click events
        container.querySelectorAll('.photo-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (e.target.type !== 'checkbox') {
                    this.selectPhoto(parseInt(item.dataset.index));
                }
            });
        });

        container.querySelectorAll('.photo-item-checkbox').forEach(cb => {
            cb.addEventListener('change', (e) => {
                const id = parseInt(e.target.dataset.id);
                if (e.target.checked) {
                    this.selectedIds.add(id);
                } else {
                    this.selectedIds.delete(id);
                }
                this.updateSelectionCount();
                e.target.closest('.photo-item').classList.toggle('selected', e.target.checked);
            });
        });
    },

    selectPhoto(index) {
        if (index < 0 || index >= this.filteredPhotos.length) return;

        this.currentIndex = index;
        const photo = this.filteredPhotos[index];

        // Update list highlighting
        document.querySelectorAll('.photo-item').forEach((item, i) => {
            item.classList.toggle('active', i === index);
        });

        // Scroll into view
        const activeItem = document.querySelector('.photo-item.active');
        if (activeItem) {
            activeItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }

        // Load image and form
        this.loadPhoto(photo);
    },

    loadPhoto(photo) {
        const img = document.getElementById('main-image');
        const noImage = document.getElementById('no-image');

        // Reset zoom when loading new photo
        this.zoom = 100;
        this.panX = 0;
        this.panY = 0;
        document.getElementById('zoom-slider').value = 100;
        document.getElementById('zoom-level').textContent = '100%';
        document.getElementById('image-wrapper').style.transform = '';
        document.getElementById('image-container').classList.remove('zoomed');
        document.getElementById('image-container').classList.add('zoom-fit');

        img.src = this.getImageUrl(photo.filePath);
        img.classList.add('visible');
        noImage.classList.add('hidden');

        // Apply proper sizing when image loads
        img.onload = () => {
            this.setZoom(100);
        };

        // Update header
        document.getElementById('current-filename').textContent = photo.filename;
        const date = new Date(photo.date);
        document.getElementById('current-datetime').textContent =
            date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) +
            ' ' + date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });

        const locationBadge = document.getElementById('current-location-badge');
        if (photo.location) {
            locationBadge.textContent = photo.location;
            locationBadge.style.display = 'inline-block';
        } else {
            locationBadge.style.display = 'none';
        }

        // Update nav position
        document.getElementById('nav-position').textContent =
            `${this.currentIndex + 1} / ${this.filteredPhotos.length}`;

        // Load form values
        this.loadFormValues(photo);
    },

    loadFormValues(photo) {
        // Species
        document.querySelectorAll('.species-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.species === photo.species?.toLowerCase());
        });

        // Sex
        document.querySelectorAll('.sex-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.sex === photo.sex);
        });

        // Deer ID
        document.getElementById('deer-id-select').value = photo.deerId || '';

        // Age & Points
        document.getElementById('age-select').value = photo.age || '';
        document.getElementById('left-points').value = photo.leftPoints || '';
        document.getElementById('right-points').value = photo.rightPoints || '';

        // Location
        document.getElementById('camera-location-select').value = photo.location || '';
        this.updateLocationButtons(photo.location);

        // Characteristics
        this.renderCharacteristics(photo.characteristics || []);

        // Notes
        document.getElementById('notes-input').value = photo.notes || '';

        // Reviewed
        document.getElementById('reviewed-checkbox').checked = photo.reviewed;
    },

    clearPreview() {
        document.getElementById('main-image').classList.remove('visible');
        document.getElementById('no-image').classList.remove('hidden');
        document.getElementById('current-filename').textContent = 'No photo selected';
        document.getElementById('current-datetime').textContent = '';
        document.getElementById('current-location-badge').style.display = 'none';
        document.getElementById('nav-position').textContent = '0 / 0';
    },

    // ============================================
    // Navigation
    // ============================================

    navigate(direction) {
        const newIndex = this.currentIndex + direction;
        if (newIndex >= 0 && newIndex < this.filteredPhotos.length) {
            this.selectPhoto(newIndex);
        }
    },

    // ============================================
    // Form Actions
    // ============================================

    selectSpecies(species) {
        document.querySelectorAll('#species-buttons .species-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.species === species);
        });
    },

    selectSex(sex) {
        document.querySelectorAll('.sex-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.sex === sex);
        });
    },

    saveCurrentPhoto() {
        if (this.currentIndex === -1) return;

        const photo = this.filteredPhotos[this.currentIndex];
        const activeSpecies = document.querySelector('#species-buttons .species-btn.active');
        const activeSex = document.querySelector('.sex-btn.active');

        const updates = {
            species: activeSpecies ? activeSpecies.dataset.species : null,
            sex: activeSex ? activeSex.dataset.sex : null,
            deerId: document.getElementById('deer-id-select').value || null,
            age: document.getElementById('age-select').value || null,
            leftPoints: parseInt(document.getElementById('left-points').value) || null,
            rightPoints: parseInt(document.getElementById('right-points').value) || null,
            location: document.getElementById('camera-location-select').value || '',
            notes: document.getElementById('notes-input').value,
            reviewed: document.getElementById('reviewed-checkbox').checked,
            characteristics: this.getCurrentCharacteristics()
        };

        Store.savePhoto(photo.id, updates);
        Object.assign(photo, updates);

        this.showToast('Photo saved', 'success');
        this.renderPhotoList();
        this.updateRecentBucks();
    },

    saveAndNext() {
        this.saveCurrentPhoto();
        this.navigate(1);
    },

    // ============================================
    // Characteristics
    // ============================================

    addCharacteristic() {
        const input = document.getElementById('char-input');
        const value = input.value.trim();
        if (!value) return;

        const container = document.getElementById('char-tags');
        const tag = document.createElement('div');
        tag.className = 'char-tag';
        tag.innerHTML = `
            <span>${value}</span>
            <button class="char-tag-remove">&times;</button>
        `;
        tag.querySelector('.char-tag-remove').addEventListener('click', () => tag.remove());
        container.appendChild(tag);
        input.value = '';
    },

    renderCharacteristics(chars) {
        const container = document.getElementById('char-tags');
        container.innerHTML = chars.map(c => `
            <div class="char-tag">
                <span>${c}</span>
                <button class="char-tag-remove">&times;</button>
            </div>
        `).join('');

        container.querySelectorAll('.char-tag-remove').forEach(btn => {
            btn.addEventListener('click', () => btn.closest('.char-tag').remove());
        });
    },

    getCurrentCharacteristics() {
        return Array.from(document.querySelectorAll('#char-tags .char-tag span'))
            .map(el => el.textContent);
    },

    // ============================================
    // Collapsible Sections
    // ============================================

    toggleCollapse(section) {
        const toggle = document.getElementById(`${section}-toggle`);
        const content = document.getElementById(`${section}-content`);
        toggle.classList.toggle('open');
        content.classList.toggle('open');
    },

    // ============================================
    // Zoom & Pan
    // ============================================

    setZoom(value, centerX = null, centerY = null) {
        const oldZoom = this.zoom;
        // Minimum zoom is 100 (fit to screen), max is 500
        this.zoom = Math.max(100, Math.min(500, parseInt(value)));

        document.getElementById('zoom-slider').value = this.zoom;
        document.getElementById('zoom-level').textContent = `${this.zoom}%`;

        const container = document.getElementById('image-container');
        const wrapper = document.getElementById('image-wrapper');
        const img = document.getElementById('main-image');

        if (!img.naturalWidth) return;

        // Calculate base size to fit container
        const containerRect = container.getBoundingClientRect();
        const imgAspect = img.naturalWidth / img.naturalHeight;
        const containerAspect = containerRect.width / containerRect.height;

        let baseWidth, baseHeight;
        if (imgAspect > containerAspect) {
            baseWidth = containerRect.width;
            baseHeight = containerRect.width / imgAspect;
        } else {
            baseHeight = containerRect.height;
            baseWidth = containerRect.height * imgAspect;
        }

        // Apply zoom
        const scale = this.zoom / 100;
        const newWidth = baseWidth * scale;
        const newHeight = baseHeight * scale;

        img.style.width = `${newWidth}px`;
        img.style.height = `${newHeight}px`;

        // Adjust pan to zoom toward mouse position
        if (centerX !== null && centerY !== null && oldZoom !== this.zoom) {
            const zoomFactor = this.zoom / oldZoom;
            this.panX = centerX - (centerX - this.panX) * zoomFactor;
            this.panY = centerY - (centerY - this.panY) * zoomFactor;
        }

        // Constrain pan to keep image edges at container edges
        this.constrainPan(newWidth, newHeight, containerRect);

        // Apply transform
        this.applyTransform();

        // Update container state
        const isZoomed = this.zoom > 100;
        container.classList.toggle('zoomed', isZoomed);
        container.classList.toggle('zoom-fit', !isZoomed);
    },

    resetZoom() {
        this.zoom = 100;
        this.panX = 0;
        this.panY = 0;
        this.setZoom(100);
    },

    handleWheel(e) {
        if (!document.getElementById('main-image').classList.contains('visible')) return;

        e.preventDefault();

        const container = document.getElementById('image-container');
        const rect = container.getBoundingClientRect();
        const centerX = e.clientX - rect.left - rect.width / 2;
        const centerY = e.clientY - rect.top - rect.height / 2;

        const delta = e.deltaY > 0 ? -15 : 15;
        this.setZoom(this.zoom + delta, centerX, centerY);
    },

    startDrag(e) {
        // Only allow drag when zoomed in
        if (this.zoom <= 100) return;
        if (e.target.closest('.no-image')) return;
        if (e.button !== 0) return; // Left mouse button only

        this.isDragging = true;
        this.dragStartX = e.clientX;
        this.dragStartY = e.clientY;
        this.lastPanX = this.panX;
        this.lastPanY = this.panY;

        document.getElementById('image-container').classList.add('dragging');
        e.preventDefault();
    },

    onDrag(e) {
        if (!this.isDragging) return;

        const deltaX = e.clientX - this.dragStartX;
        const deltaY = e.clientY - this.dragStartY;

        this.panX = this.lastPanX + deltaX;
        this.panY = this.lastPanY + deltaY;

        const container = document.getElementById('image-container');
        const img = document.getElementById('main-image');
        const containerRect = container.getBoundingClientRect();

        this.constrainPan(img.offsetWidth, img.offsetHeight, containerRect);
        this.applyTransform();
    },

    endDrag() {
        if (!this.isDragging) return;
        this.isDragging = false;
        document.getElementById('image-container').classList.remove('dragging');
    },

    constrainPan(imgWidth, imgHeight, containerRect) {
        // Calculate how far the image can move
        // When zoomed, allow panning until image edge reaches container edge
        const maxPanX = Math.max(0, (imgWidth - containerRect.width) / 2);
        const maxPanY = Math.max(0, (imgHeight - containerRect.height) / 2);

        // Clamp pan values
        this.panX = Math.max(-maxPanX, Math.min(maxPanX, this.panX));
        this.panY = Math.max(-maxPanY, Math.min(maxPanY, this.panY));
    },

    applyTransform() {
        const wrapper = document.getElementById('image-wrapper');
        wrapper.style.transform = `translate(${this.panX}px, ${this.panY}px)`;
    },

    // ============================================
    // Selection
    // ============================================

    selectAll() {
        this.filteredPhotos.forEach(p => this.selectedIds.add(p.id));
        this.renderPhotoList();
        this.updateSelectionCount();
    },

    clearSelection() {
        this.selectedIds.clear();
        this.renderPhotoList();
        this.updateSelectionCount();
    },

    updateSelectionCount() {
        document.getElementById('selection-count').textContent = `${this.selectedIds.size} selected`;
    },

    // ============================================
    // Compare
    // ============================================

    openCompare() {
        if (this.selectedIds.size === 0) {
            this.showToast('Select photos to compare', 'warning');
            return;
        }

        const photos = Array.from(this.selectedIds)
            .map(id => Store.getPhoto(id))
            .filter(Boolean)
            .slice(0, 4);

        const grid = document.getElementById('compare-grid');
        grid.innerHTML = photos.map(p => {
            const date = new Date(p.date);
            return `
                <div class="compare-item">
                    <img src="${this.getImageUrl(p.filePath)}" alt="">
                    <div class="compare-item-info">
                        <strong>${p.filename}</strong><br>
                        ${date.toLocaleDateString()} ${date.toLocaleTimeString()}<br>
                        ${p.location || 'Unknown location'}
                        ${p.species ? ` - ${p.species}` : ''}
                    </div>
                </div>
            `;
        }).join('');

        document.getElementById('compare-modal').classList.add('active');
    },

    // ============================================
    // Bulk Operations
    // ============================================

    openBulkModal() {
        if (this.selectedIds.size === 0) {
            this.showToast('Select photos first', 'warning');
            return;
        }

        document.getElementById('bulk-count').textContent = `${this.selectedIds.size} photos selected`;
        document.querySelectorAll('#bulk-modal .species-btn').forEach(b => b.classList.remove('active'));
        this.bulkSpecies = null;
        document.getElementById('bulk-modal').classList.add('active');
    },

    applyBulkSpecies() {
        if (!this.bulkSpecies) {
            this.showToast('Select a species first', 'warning');
            return;
        }

        this.selectedIds.forEach(id => {
            Store.savePhoto(id, { species: this.bulkSpecies, reviewed: true });
            const photo = Store.getPhoto(id);
            if (photo) {
                photo.species = this.bulkSpecies;
                photo.reviewed = true;
            }
        });

        this.showToast(`Applied "${this.bulkSpecies}" to ${this.selectedIds.size} photos`, 'success');
        this.closeModal('bulk-modal');
        this.clearSelection();
        this.applyFilters();
    },

    // ============================================
    // Populate UI Elements
    // ============================================

    populateLocationFilter() {
        const select = document.getElementById('location-filter');
        Store.locations.forEach(loc => {
            const opt = document.createElement('option');
            opt.value = loc;
            opt.textContent = loc;
            select.appendChild(opt);
        });
    },

    populateLocationButtons() {
        const container = document.getElementById('location-quick-buttons');
        const topLocations = Store.locations.slice(0, 6);

        container.innerHTML = topLocations.map(loc => `
            <button class="location-quick-btn" data-location="${loc}">${loc}</button>
        `).join('');

        container.querySelectorAll('.location-quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.getElementById('camera-location-select').value = btn.dataset.location;
                this.updateLocationButtons(btn.dataset.location);
            });
        });

        // Also populate the select
        const select = document.getElementById('camera-location-select');
        Store.locations.forEach(loc => {
            const opt = document.createElement('option');
            opt.value = loc;
            opt.textContent = loc;
            select.appendChild(opt);
        });
    },

    updateLocationButtons(activeLocation) {
        document.querySelectorAll('.location-quick-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.location === activeLocation);
        });
    },

    populateDeerSelect() {
        const select = document.getElementById('deer-id-select');
        Store.deer.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d.id;
            opt.textContent = d.name;
            select.appendChild(opt);
        });
    },

    updateRecentBucks() {
        const container = document.getElementById('quick-buck-buttons');
        container.innerHTML = Store.recentBucks.slice(0, 9).map(id => {
            const deer = Store.deer.find(d => d.id === id);
            const name = deer ? deer.name : id;
            return `<button class="quick-buck-btn" data-id="${id}">${name}</button>`;
        }).join('');

        container.querySelectorAll('.quick-buck-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.getElementById('deer-id-select').value = btn.dataset.id;
            });
        });
    },

    // ============================================
    // Modals
    // ============================================

    closeModal(id) {
        document.getElementById(id).classList.remove('active');
    },

    // ============================================
    // CuddeLink Download
    // ============================================

    openCuddelinkModal() {
        this.resetCuddelinkModal();

        // Set default dates (last 7 days)
        const today = new Date().toISOString().split('T')[0];
        const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
        document.getElementById('cudde-from').value = weekAgo;
        document.getElementById('cudde-to').value = today;

        // Load saved credentials
        const savedEmail = localStorage.getItem('cudde_email');
        if (savedEmail) {
            document.getElementById('cudde-email').value = savedEmail;
        }

        document.getElementById('cuddelink-modal').classList.add('active');
    },

    closeCuddelinkModal() {
        document.getElementById('cuddelink-modal').classList.remove('active');
        if (this.cuddelinkPollInterval) {
            clearInterval(this.cuddelinkPollInterval);
            this.cuddelinkPollInterval = null;
        }
    },

    resetCuddelinkModal() {
        document.getElementById('cuddelink-form-container').style.display = 'block';
        document.getElementById('cuddelink-progress-container').style.display = 'none';
        document.getElementById('cuddelink-complete-container').style.display = 'none';
        document.getElementById('cuddelink-error-container').style.display = 'none';
    },

    async startCuddelinkDownload() {
        const email = document.getElementById('cudde-email').value.trim();
        const password = document.getElementById('cudde-password').value;
        const dateFrom = document.getElementById('cudde-from').value;
        const dateTo = document.getElementById('cudde-to').value;

        if (!email || !password) {
            this.showToast('Please enter email and password', 'error');
            return;
        }

        // Save email for next time
        localStorage.setItem('cudde_email', email);

        // Show progress
        document.getElementById('cuddelink-form-container').style.display = 'none';
        document.getElementById('cuddelink-progress-container').style.display = 'block';
        document.getElementById('progress-message').textContent = 'Starting...';
        document.getElementById('cudde-progress-fill').style.width = '0%';

        try {
            const response = await fetch('/api/cuddelink/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, dateFrom, dateTo })
            });

            const data = await response.json();

            if (data.error) {
                this.showCuddelinkError(data.error);
                return;
            }

            // Start polling for status
            this.pollCuddelinkStatus();

        } catch (error) {
            this.showCuddelinkError(error.message);
        }
    },

    pollCuddelinkStatus() {
        this.cuddelinkPollInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/cuddelink/status');
                const status = await response.json();

                document.getElementById('progress-message').textContent = status.message;
                document.getElementById('cudde-progress-fill').style.width = `${status.progress}%`;

                if (status.photos_found > 0) {
                    document.getElementById('progress-details').textContent =
                        `Found ${status.photos_found} photos`;
                }

                if (status.status === 'complete') {
                    clearInterval(this.cuddelinkPollInterval);
                    this.showCuddelinkComplete(status.message, status.photos_downloaded);
                } else if (status.status === 'error') {
                    clearInterval(this.cuddelinkPollInterval);
                    this.showCuddelinkError(status.error || status.message);
                }

            } catch (error) {
                console.error('Poll error:', error);
            }
        }, 1000);
    },

    showCuddelinkComplete(message, count) {
        document.getElementById('cuddelink-progress-container').style.display = 'none';
        document.getElementById('cuddelink-complete-container').style.display = 'block';
        document.getElementById('complete-message').textContent =
            count > 0 ? `Downloaded ${count} photos!` : message;

        this.showToast(`Downloaded ${count} photos`, 'success');

        // Reload photos to show the new ones
        if (count > 0) {
            this.reloadPhotos();
        }
    },

    async reloadPhotos() {
        try {
            // Re-fetch photos.json with cache busting
            const response = await fetch('photos.json?t=' + Date.now());
            const photosData = await response.json();

            // Update store
            Store.photos = photosData.map(p => ({
                id: p.id,
                filename: p.file_path.split('/').pop(),
                filePath: p.file_path,
                date: p.date_taken,
                location: p.camera_location || '',
                species: p.tags ? p.tags.split(',')[0] : null,
                allTags: p.tags ? p.tags.split(',') : [],
                thumbnail: p.thumbnail_path,
                notes: p.notes || '',
                reviewed: p.tags !== null,
                cameraModel: p.camera_model,
                sex: null,
                deerId: null,
                age: null,
                leftPoints: null,
                rightPoints: null,
                characteristics: []
            }));

            // Load any local modifications
            Store.loadModifications();

            // Re-apply filters to refresh the list
            this.applyFilters();

            this.showToast('Photo list updated', 'success');
        } catch (error) {
            console.error('Failed to reload photos:', error);
            this.showToast('Failed to reload photos', 'error');
        }
    },

    showCuddelinkError(message) {
        document.getElementById('cuddelink-progress-container').style.display = 'none';
        document.getElementById('cuddelink-error-container').style.display = 'block';
        document.getElementById('error-message').textContent = message;
    },

    // ============================================
    // Utilities
    // ============================================

    getImageUrl(filePath) {
        if (!filePath) return '';
        if (filePath.includes('.thumbnails')) {
            return '/thumbnails/' + filePath.split('/').pop();
        }
        const match = filePath.match(/TrailCamLibrary\/(.+)/);
        return match ? '/photos/' + match[1] : filePath;
    },

    handleKeyboard(e) {
        // Don't intercept if typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
            return;
        }

        switch (e.key) {
            case 'ArrowLeft':
            case 'ArrowUp':
                e.preventDefault();
                this.navigate(-1);
                break;
            case 'ArrowRight':
            case 'ArrowDown':
                e.preventDefault();
                this.navigate(1);
                break;
            case 's':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    this.saveCurrentPhoto();
                }
                break;
            case '1': this.selectSpecies('buck'); break;
            case '2': this.selectSpecies('doe'); break;
            case '3': this.selectSpecies('fawn'); break;
            case '4': this.selectSpecies('turkey'); break;
            case '5': this.selectSpecies('coyote'); break;
            case '0': this.selectSpecies('empty'); break;
        }
    },

    showToast(message, type = 'success') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 200);
        }, 2500);
    }
};


// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => UI.init());

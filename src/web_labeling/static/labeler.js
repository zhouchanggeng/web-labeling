// ============ State ============
const S = {
  images: [], currentIdx: -1, annotation: null, dirty: false,
  shapes: [], selectedIdx: -1,
  tool: 'select', // select | rect | polygon
  // Canvas transform
  scale: 1, offsetX: 0, offsetY: 0,
  // Drawing state
  drawing: false, drawPoints: [],
  // Dragging / resizing
  dragging: false, dragStart: null, dragType: null, dragHandleIdx: -1,
  // Pan
  panning: false, panStart: null,
  // Image
  img: null, imgW: 0, imgH: 0,
  // Mouse position for crosshair
  mouseX: -1, mouseY: -1,
  // Label history
  labelHistory: [],
  // Label filter
  labelMap: {},        // {label: [img_name, ...]}
  filterLabel: null,   // active label filter, null = show all
  // Lazy loading
  totalImages: 0,      // total count from server (with current filter/search)
  loadingMore: false,   // currently fetching a page
  // Display options
  showLabels: true,     // whether to show label text on shapes
};

const COLORS = ['#f38ba8','#a6e3a1','#89b4fa','#fab387','#cba6f7','#f9e2af','#94e2d5','#f2cdcd','#89dceb','#eba0ac'];

// ============ Undo ============
const _undoStack = [];
const UNDO_MAX = 50;
function pushUndo() {
  _undoStack.push(JSON.stringify(S.shapes.map(exportShape)));
  if (_undoStack.length > UNDO_MAX) _undoStack.shift();
}
function undo() {
  if (!_undoStack.length) return;
  S.shapes = JSON.parse(_undoStack.pop()).map(normalizeShape);
  S.selectedIdx = -1;
  S.dirty = true;
  renderShapeList();
  updateEditor();
  draw();
}
function colorFor(label) {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = ((h << 5) - h + label.charCodeAt(i)) | 0;
  return COLORS[Math.abs(h) % COLORS.length];
}

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvas-wrap');

// ============ API (lazy loading) ============
const PAGE_SIZE = 200;

async function fetchImages(reset = true) {
  if (reset) {
    S.images = [];
    S.totalImages = 0;
    S.currentIdx = -1;
  }
  await _loadImagePage(0);
  renderImageList();
}

async function _loadImagePage(offset) {
  if (S.loadingMore) return;
  S.loadingMore = true;
  try {
    const params = new URLSearchParams({ offset, limit: PAGE_SIZE });
    const search = document.getElementById('search').value.trim();
    if (search) params.set('search', search);
    if (S.filterLabel) params.set('label', S.filterLabel);
    const r = await fetch('/api/images?' + params);
    const data = await r.json();
    S.totalImages = data.total;
    if (data.offset === 0) {
      S.images = data.images;
    } else if (data.offset === S.images.length) {
      S.images.push(...data.images);
    }
  } finally {
    S.loadingMore = false;
  }
}

async function _loadMoreIfNeeded() {
  if (S.loadingMore || S.images.length >= S.totalImages) return;
  await _loadImagePage(S.images.length);
  renderImageList();
}

async function fetchLabels() {
  const r = await fetch('/api/labels');
  S.labelMap = await r.json();
  renderLabelFilter();
}

async function setLabelFilter(label) {
  S.filterLabel = S.filterLabel === label ? null : label;
  await fetchLabels();
  renderLabelFilter();
  fetchImages();
}

function renderLabelFilter() {
  const container = document.getElementById('label-filter');
  const labels = Object.keys(S.labelMap).sort();
  container.innerHTML = labels.map(l => {
    const count = S.labelMap[l].length;
    const active = S.filterLabel === l ? ' active' : '';
    const c = colorFor(l);
    return `<span class="label-tag${active}" data-label="${l}" style="--tag-color:${c}">${l} (${count})</span>`;
  }).join('');
  container.querySelectorAll('.label-tag').forEach(el => {
    el.addEventListener('click', () => setLabelFilter(el.dataset.label));
  });
}

// ============ Image preload cache ============
const _preloadCache = new Map();
const PRELOAD_MAX = 10;

function _preloadAdjacent(idx) {
  const toPreload = [1, 2, -1].map(d => idx + d)
    .filter(p => p >= 0 && p < S.images.length)
    .map(p => S.images[p]);
  for (const name of toPreload) {
    if (_preloadCache.has(name)) continue;
    const img = new window.Image();
    img.src = '/api/image/' + encodeURIComponent(name);
    _preloadCache.set(name, img);
  }
  if (_preloadCache.size > PRELOAD_MAX) {
    const keep = new Set(toPreload);
    if (idx < S.images.length) keep.add(S.images[idx]);
    for (const [k] of _preloadCache) {
      if (!keep.has(k)) { _preloadCache.delete(k); if (_preloadCache.size <= PRELOAD_MAX) break; }
    }
  }
}

async function loadImage(idx) {
  if (S.dirty) await saveAnnotation();
  _undoStack.length = 0;
  S.currentIdx = idx;
  S.selectedIdx = -1;
  S.drawing = false;
  S.drawPoints = [];
  const name = S.images[idx];
  document.getElementById('img-name').textContent = name;

  // Clear current display immediately
  S.shapes = [];
  S.img = null;
  S.imgW = 0;
  S.imgH = 0;
  invalidateImgCache();
  renderShapeList();
  document.getElementById('img-resolution').textContent = '';
  draw();

  // Use preloaded image if available
  const preloaded = _preloadCache.get(name);

  // Load image and annotation in parallel
  const [img, ann] = await Promise.all([
    preloaded && preloaded.complete && preloaded.naturalWidth > 0
      ? Promise.resolve(preloaded)
      : new Promise((resolve) => {
          const image = preloaded || new window.Image();
          if (image.complete && image.naturalWidth > 0) { resolve(image); return; }
          image.onload = () => resolve(image);
          image.onerror = () => resolve(null);
          if (!preloaded) image.src = '/api/image/' + encodeURIComponent(name);
        }),
    fetch('/api/annotation/' + encodeURIComponent(name)).then(r => r.json()),
  ]);

  // Check if user already switched to another image
  if (S.currentIdx !== idx) return;

  // Apply annotation
  S.annotation = ann;
  S.shapes = (ann.shapes || []).map(normalizeShape);
  S.dirty = false;

  // Apply image
  if (img) {
    S.img = img;
    S.imgW = img.naturalWidth;
    S.imgH = img.naturalHeight;
    invalidateImgCache();
    document.getElementById('img-resolution').textContent = S.imgW + ' × ' + S.imgH;
    fitView();
  }

  renderShapeList();
  renderImageList();

  // Preload adjacent images
  _preloadAdjacent(idx);
  setTool('select');
}

async function saveAnnotation() {
  if (S.currentIdx < 0) return;
  const name = S.images[S.currentIdx];
  const ann = {
    version: "2.4.0",
    flags: S.annotation.flags || {},
    shapes: S.shapes.map(exportShape),
    imagePath: name,
    imageData: null,
    imageHeight: S.imgH,
    imageWidth: S.imgW,
  };
  await fetch('/api/annotation/' + encodeURIComponent(name), {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(ann),
  });
  S.dirty = false;
  renderImageList();
}

// ============ Shape helpers ============
function normalizeShape(s) {
  // X-AnyLabeling format: shape_type, points, label, group_id, flags, description, difficult, attributes
  let points = s.points || [];
  // Convert 4-point rectangle to 2-point format (top-left, bottom-right)
  if (s.shape_type === 'rectangle' && points.length === 4) {
    let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
    for (const [x,y] of points) { minX=Math.min(minX,x); minY=Math.min(minY,y); maxX=Math.max(maxX,x); maxY=Math.max(maxY,y); }
    points = [[minX, minY], [maxX, maxY]];
  }
  return {
    label: s.label || '',
    points: points,
    group_id: s.group_id ?? null,
    description: s.description || '',
    difficult: s.difficult || false,
    shape_type: s.shape_type || 'rectangle',
    flags: s.flags || {},
    attributes: s.attributes || {},
    kie_linking: s.kie_linking || [],
  };
}

function exportShape(s) {
  return {
    label: s.label,
    points: s.points,
    group_id: s.group_id,
    description: s.description,
    difficult: s.difficult,
    shape_type: s.shape_type,
    flags: s.flags,
    attributes: s.attributes,
    kie_linking: s.kie_linking,
  };
}

function shapeRect(s) {
  if (s.shape_type === 'rectangle' && s.points.length >= 2) {
    if (s.points.length === 2) {
      const [x1,y1] = s.points[0], [x2,y2] = s.points[1];
      return { x: Math.min(x1,x2), y: Math.min(y1,y2), w: Math.abs(x2-x1), h: Math.abs(y2-y1) };
    }
    // 4-point rectangle format
    let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
    for (const [x,y] of s.points) { minX=Math.min(minX,x); minY=Math.min(minY,y); maxX=Math.max(maxX,x); maxY=Math.max(maxY,y); }
    return { x:minX, y:minY, w:maxX-minX, h:maxY-minY };
  }
  // Bounding box for polygon
  let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
  for (const [x,y] of s.points) { minX=Math.min(minX,x); minY=Math.min(minY,y); maxX=Math.max(maxX,x); maxY=Math.max(maxY,y); }
  return { x:minX, y:minY, w:maxX-minX, h:maxY-minY };
}

// ============ Canvas transform ============
function toCanvas(px, py) {
  return [px * S.scale + S.offsetX, py * S.scale + S.offsetY];
}
function toImage(cx, cy) {
  return [(cx - S.offsetX) / S.scale, (cy - S.offsetY) / S.scale];
}

function fitView() {
  const wrapW = wrap.clientWidth;
  const wrapH = wrap.clientHeight;
  if (wrapW <= 0 || wrapH <= 0) return;
  canvas.width = wrapW;
  canvas.height = wrapH;
  const pad = 40;
  const cw = wrapW - pad*2, ch = wrapH - pad*2;
  if (!S.imgW || !S.imgH || cw <= 0 || ch <= 0) return;
  const scaleW = cw / S.imgW;
  const scaleH = ch / S.imgH;
  S.scale = Math.min(scaleW, scaleH);
  S.offsetX = (wrapW - S.imgW * S.scale) / 2;
  S.offsetY = (wrapH - S.imgH * S.scale) / 2;
  invalidateImgCache();
  draw();
}

function resizeCanvas() {
  canvas.width = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  draw();
}

// ============ Image cache ============
let _imgCache = null, _imgCacheScale = 0;
function _cacheImage(dw, dh) {
  if (dw < 1 || dh < 1 || dw > 8000 || dh > 8000) return;
  const oc = new OffscreenCanvas(Math.round(dw), Math.round(dh));
  const octx = oc.getContext('2d');
  octx.drawImage(S.img, 0, 0, Math.round(dw), Math.round(dh));
  _imgCache = oc;
  _imgCacheScale = S.scale;
}
function invalidateImgCache() { _imgCache = null; _imgCacheScale = 0; }

// ============ Drawing ============
let _rafId = 0;
function draw() {
  if (!_rafId) _rafId = requestAnimationFrame(_draw);
}
function _draw() {
  _rafId = 0;
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw image
  if (S.img) {
    const [ix, iy] = toCanvas(0, 0);
    const dw = S.imgW * S.scale, dh = S.imgH * S.scale;
    ctx.drawImage(S.img, ix, iy, dw, dh);
  }

  // Draw shapes
  S.shapes.forEach((s, i) => {
    if (s._hidden) return;
    const selected = i === S.selectedIdx;
    const color = colorFor(s.label || 'default');
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = selected ? 4 : 3;
    ctx.fillStyle = color + '40';

    if (s.shape_type === 'rectangle' && s.points.length >= 2) {
      let rx, ry, rw, rh;
      if (s.points.length === 2) {
        const [x1,y1] = toCanvas(...s.points[0]);
        const [x2,y2] = toCanvas(...s.points[1]);
        rx = Math.min(x1,x2); ry = Math.min(y1,y2);
        rw = Math.abs(x2-x1); rh = Math.abs(y2-y1);
      } else {
        // 4-point rectangle format: compute bounding box
        let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
        for (const p of s.points) {
          const [cx,cy] = toCanvas(...p);
          if (cx < minX) minX = cx; if (cy < minY) minY = cy;
          if (cx > maxX) maxX = cx; if (cy > maxY) maxY = cy;
        }
        rx = minX; ry = minY; rw = maxX - minX; rh = maxY - minY;
      }
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeRect(rx, ry, rw, rh);
      // Draw outer glow for contrast on any background
      if (selected) {
        ctx.strokeStyle = 'rgba(0,0,0,0.5)';
        ctx.lineWidth = 6;
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(rx, ry, rw, rh);
      }
    } else if (s.points.length > 0) {
      ctx.beginPath();
      const [sx,sy] = toCanvas(...s.points[0]);
      ctx.moveTo(sx, sy);
      for (let j = 1; j < s.points.length; j++) {
        const [px,py] = toCanvas(...s.points[j]);
        ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
      if (selected) {
        ctx.strokeStyle = 'rgba(0,0,0,0.5)';
        ctx.lineWidth = 6;
        ctx.stroke();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
      }
      ctx.stroke();
    }

    // Label text with background
    if (S.showLabels && s.points.length > 0) {
      const r = shapeRect(s);
      const [lx, ly] = toCanvas(r.x, r.y);
      const fontSize = Math.max(13, 15 * Math.min(S.scale, 2));
      const text = s.label || '(no label)';
      ctx.font = `bold ${fontSize}px sans-serif`;
      const tm = ctx.measureText(text);
      const pad = 3;
      // Background pill
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.beginPath();
      ctx.roundRect(lx - 1, ly - fontSize - pad * 2, tm.width + pad * 2 + 2, fontSize + pad * 2, 3);
      ctx.fill();
      // Text
      ctx.fillStyle = '#fff';
      ctx.fillText(text, lx + pad, ly - pad - 2);
    }

    // Handles for selected
    if (selected) {
      s.points.forEach(p => {
        const [hx, hy] = toCanvas(...p);
        ctx.fillStyle = '#fff';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(hx, hy, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    }
    ctx.restore();
  });

  // Drawing in progress
  if (S.drawing && S.drawPoints.length > 0) {
    ctx.save();
    ctx.strokeStyle = '#f9e2af';
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 4]);
    if (S.tool === 'rect' && S.drawPoints.length === 2) {
      const [x1,y1] = toCanvas(...S.drawPoints[0]);
      const [x2,y2] = toCanvas(...S.drawPoints[1]);
      ctx.strokeRect(Math.min(x1,x2), Math.min(y1,y2), Math.abs(x2-x1), Math.abs(y2-y1));
    } else if (S.tool === 'polygon') {
      ctx.beginPath();
      const [sx,sy] = toCanvas(...S.drawPoints[0]);
      ctx.moveTo(sx, sy);
      for (let j = 1; j < S.drawPoints.length; j++) {
        const [px,py] = toCanvas(...S.drawPoints[j]);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
      // Draw vertices
      S.drawPoints.forEach(p => {
        const [hx,hy] = toCanvas(...p);
        ctx.fillStyle = '#f9e2af';
        ctx.beginPath();
        ctx.arc(hx, hy, 4, 0, Math.PI*2);
        ctx.fill();
      });
    }
    ctx.restore();
  }

  // Draw crosshair at mouse position when in drawing tool mode
  if ((S.tool === 'rect' || S.tool === 'polygon') && S.mouseX >= 0 && S.mouseY >= 0) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    // Vertical line
    ctx.beginPath();
    ctx.moveTo(S.mouseX, 0);
    ctx.lineTo(S.mouseX, canvas.height);
    ctx.stroke();
    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(0, S.mouseY);
    ctx.lineTo(canvas.width, S.mouseY);
    ctx.stroke();
    ctx.restore();
  }
}

// ============ Hit testing ============
function hitTest(mx, my) {
  const [ix, iy] = toImage(mx, my);
  // Check in reverse order (top shapes first)
  for (let i = S.shapes.length - 1; i >= 0; i--) {
    const s = S.shapes[i];
    if (s.shape_type === 'rectangle' && s.points.length >= 2) {
      const r = shapeRect(s);
      if (ix >= r.x && ix <= r.x+r.w && iy >= r.y && iy <= r.y+r.h) return i;
    } else if (s.points.length >= 3) {
      if (pointInPolygon(ix, iy, s.points)) return i;
    }
  }
  return -1;
}

function pointInPolygon(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi,yi] = pts[i], [xj,yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < (xj-xi)*(y-yi)/(yj-yi)+xi) inside = !inside;
  }
  return inside;
}

function hitHandle(mx, my, shapeIdx) {
  if (shapeIdx < 0) return -1;
  const s = S.shapes[shapeIdx];
  const threshold = 8 / S.scale;
  for (let i = 0; i < s.points.length; i++) {
    const [px,py] = s.points[i];
    const [ix,iy] = toImage(mx, my);
    if (Math.abs(ix-px) < threshold && Math.abs(iy-py) < threshold) return i;
  }
  return -1;
}

// ============ Mouse events ============
canvas.addEventListener('mousedown', e => {
  if (SAM3.active && SAM3.pointMode && e.button === 0 && !e.altKey) return; // let click handler handle it
  const mx = e.offsetX, my = e.offsetY;

  // Middle button or space+click = pan
  if (e.button === 1 || (e.button === 0 && e.altKey)) {
    S.panning = true;
    S.panStart = { x: mx, y: my, ox: S.offsetX, oy: S.offsetY };
    canvas.style.cursor = 'grabbing';
    return;
  }

  if (e.button !== 0) return;

  if (S.tool === 'select') {
    // Check handle drag first
    const hi = hitHandle(mx, my, S.selectedIdx);
    if (hi >= 0) {
      pushUndo();
      S.dragging = true;
      S.dragType = 'handle';
      S.dragHandleIdx = hi;
      S.dragStart = toImage(mx, my);
      return;
    }
    // Check shape hit
    const idx = hitTest(mx, my);
    if (idx >= 0) {
      S.selectedIdx = idx;
      pushUndo();
      S.dragging = true;
      S.dragType = 'move';
      S.dragStart = toImage(mx, my);
      renderShapeList();
      updateEditor();
      draw();
      return;
    }
    S.selectedIdx = -1;
    renderShapeList();
    updateEditor();
    draw();
  } else if (S.tool === 'rect') {
    const [ix, iy] = toImage(mx, my);
    S.drawing = true;
    S.drawPoints = [[ix, iy], [ix, iy]];
  } else if (S.tool === 'polygon') {
    const [ix, iy] = toImage(mx, my);
    if (!S.drawing) {
      S.drawing = true;
      S.drawPoints = [[ix, iy]];
    } else {
      // Check if close to first point to finish
      const [fx, fy] = S.drawPoints[0];
      if (S.drawPoints.length >= 3 && Math.abs(ix-fx) < 10/S.scale && Math.abs(iy-fy) < 10/S.scale) {
        finishPolygon();
      } else {
        S.drawPoints.push([ix, iy]);
      }
    }
    draw();
  }
});

canvas.addEventListener('mousemove', e => {
  const mx = e.offsetX, my = e.offsetY;

  // Track mouse for crosshair
  S.mouseX = mx;
  S.mouseY = my;

  // Update coordinate display
  const [imgX, imgY] = toImage(mx, my);
  if (S.img && imgX >= 0 && imgX <= S.imgW && imgY >= 0 && imgY <= S.imgH) {
    document.getElementById('mouse-coords').textContent = `坐标: ${Math.round(imgX)}, ${Math.round(imgY)}`;
  } else {
    document.getElementById('mouse-coords').textContent = '坐标: -';
  }

  if (S.panning) {
    S.offsetX = S.panStart.ox + (mx - S.panStart.x);
    S.offsetY = S.panStart.oy + (my - S.panStart.y);
    draw();
    return;
  }

  if (S.dragging && S.dragType === 'handle') {
    const [ix, iy] = toImage(mx, my);
    S.shapes[S.selectedIdx].points[S.dragHandleIdx] = [ix, iy];
    S.dirty = true;
    draw();
    return;
  }

  if (S.dragging && S.dragType === 'move') {
    const [ix, iy] = toImage(mx, my);
    const dx = ix - S.dragStart[0], dy = iy - S.dragStart[1];
    const s = S.shapes[S.selectedIdx];
    s.points = s.points.map(([px,py]) => [px+dx, py+dy]);
    S.dragStart = [ix, iy];
    S.dirty = true;
    draw();
    return;
  }

  if (S.drawing && S.tool === 'rect') {
    const [ix, iy] = toImage(mx, my);
    S.drawPoints[1] = [ix, iy];
    draw();
    return;
  }

  // Redraw for crosshair update in drawing tool modes
  if (S.tool === 'rect' || S.tool === 'polygon') {
    draw();
    return;
  }

  // Hover-select in select mode
  if (S.tool === 'select' && !S.dragging) {
    const hi = hitHandle(mx, my, S.selectedIdx);
    if (hi >= 0) { canvas.style.cursor = 'grab'; return; }
    const idx = hitTest(mx, my);
    canvas.style.cursor = idx >= 0 ? 'move' : 'default';
    if (idx !== S.selectedIdx) {
      S.selectedIdx = idx;
      renderShapeList();
      updateEditor();
      draw();
    }
  }
});

canvas.addEventListener('mouseleave', () => {
  S.mouseX = -1;
  S.mouseY = -1;
  document.getElementById('mouse-coords').textContent = '坐标: -';
  draw();
});

canvas.addEventListener('mouseup', e => {
  if (S.panning) {
    S.panning = false;
    canvas.style.cursor = S.tool === 'select' ? 'default' : 'crosshair';
    return;
  }
  if (S.dragging) {
    S.dragging = false;
    S.dragType = null;
    return;
  }
  if (S.drawing && S.tool === 'rect') {
    const [x1,y1] = S.drawPoints[0], [x2,y2] = S.drawPoints[1];
    if (Math.abs(x2-x1) > 3 && Math.abs(y2-y1) > 3) {
      promptLabel(label => {
        pushUndo();
        S.shapes.push(normalizeShape({
          label, shape_type: 'rectangle',
          points: [[Math.min(x1,x2),Math.min(y1,y2)],[Math.max(x1,x2),Math.max(y2,y1)]],
        }));
        S.selectedIdx = S.shapes.length - 1;
        S.dirty = true;
        addLabelHistory(label);
        renderShapeList();
        updateEditor();
        draw();
      });
    }
    S.drawing = false;
    S.drawPoints = [];
    draw();
  }
});

// Polygon: right-click or double-click to finish
canvas.addEventListener('dblclick', e => {
  if (S.drawing && S.tool === 'polygon' && S.drawPoints.length >= 3) {
    finishPolygon();
  }
});
canvas.addEventListener('contextmenu', e => {
  e.preventDefault();
  if (S.drawing && S.tool === 'polygon' && S.drawPoints.length >= 3) {
    finishPolygon();
    return;
  }
  // Right-click context menu for shapes
  const mx = e.offsetX, my = e.offsetY;
  const idx = hitTest(mx, my);
  if (idx >= 0) {
    S.selectedIdx = idx;
    renderShapeList();
    updateEditor();
    draw();
    showContextMenu(e.clientX, e.clientY);
  }
});

function finishPolygon() {
  const pts = [...S.drawPoints];
  S.drawing = false;
  S.drawPoints = [];
  promptLabel(label => {
    pushUndo();
    S.shapes.push(normalizeShape({ label, shape_type: 'polygon', points: pts }));
    S.selectedIdx = S.shapes.length - 1;
    S.dirty = true;
    addLabelHistory(label);
    renderShapeList();
    updateEditor();
    draw();
  });
  draw();
}

// Zoom
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const mx = e.offsetX, my = e.offsetY;
  const [ix, iy] = toImage(mx, my);
  const factor = e.deltaY < 0 ? 1.1 : 0.9;
  S.scale *= factor;
  S.scale = Math.max(0.05, Math.min(50, S.scale));
  S.offsetX = mx - ix * S.scale;
  S.offsetY = my - iy * S.scale;
  draw();
}, { passive: false });

// ============ Label dialog ============
function promptLabel(cb) {
  const dlg = document.getElementById('label-dialog');
  const inp = document.getElementById('dialog-label');
  dlg.style.display = 'block';
  inp.value = S.labelHistory[0] || '';
  inp.focus();
  inp.select();
  updateLabelSuggestions();

  // Render quick-pick label buttons
  const picks = document.getElementById('dialog-label-picks');
  const globalLabels = Object.keys(S.labelMap);
  const labels = [...new Set([...S.labelHistory, ...globalLabels, ...S.shapes.map(s => s.label).filter(Boolean)])];
  picks.innerHTML = labels.map(l => {
    const c = colorFor(l);
    return `<span class="label-pick" style="border-color:${c}" data-label="${l}">${l}</span>`;
  }).join('');
  picks.querySelectorAll('.label-pick').forEach(el => {
    el.addEventListener('click', () => {
      inp.value = el.dataset.label;
      done(true);
    });
  });

  function done(ok) {
    dlg.style.display = 'none';
    cleanup();
    canvas.focus();
    if (ok && inp.value.trim()) cb(inp.value.trim());
  }
  function onKey(e) { if (e.key === 'Enter') done(true); if (e.key === 'Escape') done(false); }
  function cleanup() {
    document.getElementById('dialog-ok').removeEventListener('click', okH);
    document.getElementById('dialog-cancel').removeEventListener('click', cancelH);
    inp.removeEventListener('keydown', onKey);
  }
  const okH = () => done(true);
  const cancelH = () => done(false);
  document.getElementById('dialog-ok').addEventListener('click', okH);
  document.getElementById('dialog-cancel').addEventListener('click', cancelH);
  inp.addEventListener('keydown', onKey);
}

function addLabelHistory(label) {
  if (!label) return;
  S.labelHistory = [label, ...S.labelHistory.filter(l => l !== label)].slice(0, 20);
  updateLabelSuggestions();
}

function updateLabelSuggestions() {
  const dl = document.getElementById('label-suggestions');
  dl.innerHTML = '';
  // Collect from global labels + history + current shapes
  const globalLabels = Object.keys(S.labelMap);
  const labels = [...new Set([...S.labelHistory, ...globalLabels, ...S.shapes.map(s => s.label).filter(Boolean)])];
  labels.forEach(l => { const o = document.createElement('option'); o.value = l; dl.appendChild(o); });
}

// ============ UI rendering ============
// ============ Virtual scroll for image list ============
const VITEM_H = 30;

function renderImageList() {
  const list = document.getElementById('img-list');
  const totalH = S.totalImages * VITEM_H;
  list.innerHTML = '';
  let spacer = list._vspacer;
  if (!spacer) {
    spacer = document.createElement('div');
    spacer.style.cssText = 'width:1px;pointer-events:none;';
    list._vspacer = spacer;
  }
  spacer.style.height = totalH + 'px';
  list.appendChild(spacer);
  _renderVisibleSlice(list);

  const pos = S.currentIdx >= 0 ? S.currentIdx + 1 : 0;
  const filterInfo = S.filterLabel ? ` [${S.filterLabel}]` : '';
  document.getElementById('img-counter').textContent = `${pos} / ${S.totalImages}${filterInfo}`;

  // Scroll active item into view
  if (S.currentIdx >= 0 && S.currentIdx < S.images.length) {
    const itemTop = S.currentIdx * VITEM_H;
    const listH = list.clientHeight;
    if (itemTop < list.scrollTop || itemTop + VITEM_H > list.scrollTop + listH) {
      list.scrollTop = itemTop - listH / 2 + VITEM_H / 2;
    }
  }
}

function _renderVisibleSlice(list) {
  if (!list) list = document.getElementById('img-list');
  const old = list.querySelectorAll('.vitem');
  old.forEach(el => el.remove());

  const scrollTop = list.scrollTop;
  const listH = list.clientHeight;
  const overscan = 10;
  const startIdx = Math.max(0, Math.floor(scrollTop / VITEM_H) - overscan);
  const endIdx = Math.min(S.images.length, Math.ceil((scrollTop + listH) / VITEM_H) + overscan);

  const frag = document.createDocumentFragment();
  for (let vi = startIdx; vi < endIdx; vi++) {
    const name = S.images[vi];
    const div = document.createElement('div');
    div.className = 'vitem item' + (vi === S.currentIdx ? ' active' : '') + (hasAnnotation(name) ? ' has-ann' : '');
    div.dataset.idx = vi;
    div.textContent = name;
    div.style.cssText = `position:absolute;top:${vi * VITEM_H}px;left:0;right:0;height:${VITEM_H}px;`;
    div.addEventListener('click', () => loadImage(vi));
    frag.appendChild(div);
  }

  if (S.images.length < S.totalImages && endIdx >= S.images.length) {
    const ld = document.createElement('div');
    ld.className = 'vitem';
    ld.textContent = '加载中...';
    ld.style.cssText = `position:absolute;top:${S.images.length * VITEM_H}px;left:0;right:0;height:${VITEM_H}px;color:#6c7086;padding:6px 12px;font-size:13px;`;
    frag.appendChild(ld);
  }
  list.appendChild(frag);

  // Trigger lazy load when near the loaded boundary
  const viewBottom = scrollTop + listH;
  const loadedBottom = S.images.length * VITEM_H;
  if (viewBottom > loadedBottom - VITEM_H * 20 && S.images.length < S.totalImages) {
    _loadMoreIfNeeded();
  }
}

document.getElementById('img-list').addEventListener('scroll', () => {
  _renderVisibleSlice();
});

function hasAnnotation(name) {
  // Simple check: if current image, check shapes length
  if (S.images[S.currentIdx] === name) return S.shapes.length > 0;
  return false; // Can't know without loading
}

function renderShapeList() {
  const list = document.getElementById('shape-list');
  list.innerHTML = S.shapes.map((s, i) => {
    const c = colorFor(s.label || 'default');
    const hidden = s._hidden;
    const eyeIcon = hidden ? '👁‍🗨' : '👁';
    const dimStyle = hidden ? ' opacity:0.4;' : '';
    const r = shapeRect(s);
    const sizeText = r.w > 0 ? `${Math.round(r.w)}×${Math.round(r.h)}` : '';
    return `<div class="shape-item${i === S.selectedIdx ? ' selected' : ''}" data-idx="${i}" style="${dimStyle}">
      <span><span class="color-dot" style="background:${c}"></span>${s.label || '(no label)'} <span class="shape-size">${sizeText}</span></span>
      <span style="display:flex;align-items:center;gap:4px">
        <span style="color:#6c7086;font-size:11px">${s.shape_type}</span>
        <button class="btn-vis-shape" data-idx="${i}" title="${hidden ? '显示' : '隐藏'}" style="background:none;border:none;color:#a6adc8;cursor:pointer;font-size:13px;padding:0 2px;">${eyeIcon}</button>
        <button class="btn-del-shape" data-idx="${i}" title="删除" style="background:none;border:none;color:#e06c75;cursor:pointer;font-size:14px;padding:0 2px;">✕</button>
      </span>
    </div>`;
  }).join('');
  list.querySelectorAll('.shape-item').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-del-shape') || e.target.classList.contains('btn-vis-shape')) return;
      S.selectedIdx = parseInt(el.dataset.idx);
      renderShapeList();
      updateEditor();
      draw();
    });
  });
  list.querySelectorAll('.btn-vis-shape').forEach(el => {
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(el.dataset.idx);
      S.shapes[idx]._hidden = !S.shapes[idx]._hidden;
      renderShapeList();
      draw();
    });
  });
  list.querySelectorAll('.btn-del-shape').forEach(el => {
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(el.dataset.idx);
      S.shapes.splice(idx, 1);
      if (S.selectedIdx === idx) S.selectedIdx = -1;
      else if (S.selectedIdx > idx) S.selectedIdx--;
      S.dirty = true;
      renderShapeList();
      updateEditor();
      draw();
    });
  });
}

document.getElementById('btn-toggle-all-vis').addEventListener('click', () => {
  const allHidden = S.shapes.length > 0 && S.shapes.every(s => s._hidden);
  S.shapes.forEach(s => s._hidden = !allHidden);
  document.getElementById('btn-toggle-all-vis').textContent = allHidden ? '👁' : '👁‍🗨';
  renderShapeList();
  draw();
});

document.getElementById('btn-toggle-labels').addEventListener('click', () => {
  S.showLabels = !S.showLabels;
  document.getElementById('btn-toggle-labels').style.color = S.showLabels ? '#a6adc8' : '#585b70';
  draw();
});

function updateEditor() {
  const editor = document.getElementById('shape-editor');
  if (S.selectedIdx < 0) { editor.style.display = 'none'; return; }
  editor.style.display = 'block';
  const s = S.shapes[S.selectedIdx];
  document.getElementById('edit-label').value = s.label;
  document.getElementById('edit-group').value = s.group_id ?? '';
}

document.getElementById('edit-label').addEventListener('change', e => {
  if (S.selectedIdx < 0) return;
  pushUndo();
  S.shapes[S.selectedIdx].label = e.target.value;
  S.dirty = true;
  addLabelHistory(e.target.value);
  renderShapeList();
  draw();
});
document.getElementById('edit-group').addEventListener('change', e => {
  if (S.selectedIdx < 0) return;
  const v = e.target.value;
  S.shapes[S.selectedIdx].group_id = v ? (isNaN(v) ? v : parseInt(v)) : null;
  S.dirty = true;
});

// ============ Toolbar ============
function setTool(t) {
  S.tool = t;
  S.drawing = false;
  S.drawPoints = [];
  document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + (t === 'select' ? 'select' : t === 'rect' ? 'rect' : 'poly')).classList.add('active');
  canvas.style.cursor = t === 'select' ? 'default' : 'crosshair';
  draw();
}

document.getElementById('btn-select').addEventListener('click', () => setTool('select'));
document.getElementById('btn-rect').addEventListener('click', () => setTool('rect'));
document.getElementById('btn-poly').addEventListener('click', () => setTool('polygon'));
document.getElementById('btn-delete').addEventListener('click', deleteSelected);
document.getElementById('btn-save').addEventListener('click', saveAnnotation);
document.getElementById('btn-prev').addEventListener('click', () => navFiltered(-1));
document.getElementById('btn-next').addEventListener('click', () => navFiltered(1));

function navFiltered(dir) {
  if (S.images.length === 0 && S.totalImages === 0) return;
  let next = S.currentIdx + dir;
  if (next < 0) next = 0;
  if (next >= S.images.length) {
    if (S.images.length < S.totalImages) {
      _loadMoreIfNeeded().then(() => {
        if (next < S.images.length) loadImage(next);
      });
      return;
    }
    return;
  }
  loadImage(next);
}
document.getElementById('btn-zoomin').addEventListener('click', () => { zoomCenter(1.2); });
document.getElementById('btn-zoomout').addEventListener('click', () => { zoomCenter(0.8); });
document.getElementById('btn-fit').addEventListener('click', () => { fitView(); draw(); });
let _searchTimer = 0;
document.getElementById('search').addEventListener('input', () => {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(() => fetchImages(), 250);
});

function zoomCenter(factor) {
  const cx = wrap.clientWidth / 2, cy = wrap.clientHeight / 2;
  const [ix, iy] = toImage(cx, cy);
  S.scale *= factor;
  S.scale = Math.max(0.05, Math.min(50, S.scale));
  S.offsetX = cx - ix * S.scale;
  S.offsetY = cy - iy * S.scale;
  draw();
}

function deleteSelected() {
  if (S.selectedIdx < 0) return;
  pushUndo();
  S.shapes.splice(S.selectedIdx, 1);
  S.selectedIdx = -1;
  S.dirty = true;
  renderShapeList();
  updateEditor();
  draw();
}

// ============ Context menu ============
const ctxMenu = document.getElementById('context-menu');

function showContextMenu(x, y) {
  ctxMenu.style.left = x + 'px';
  ctxMenu.style.top = y + 'px';
  ctxMenu.style.display = 'block';
  // Adjust if overflows viewport
  const rect = ctxMenu.getBoundingClientRect();
  if (rect.right > window.innerWidth) ctxMenu.style.left = (x - rect.width) + 'px';
  if (rect.bottom > window.innerHeight) ctxMenu.style.top = (y - rect.height) + 'px';
}

function hideContextMenu() { ctxMenu.style.display = 'none'; }

document.addEventListener('mousedown', e => {
  if (!ctxMenu.contains(e.target)) hideContextMenu();
});

ctxMenu.querySelectorAll('.ctx-item').forEach(el => {
  el.addEventListener('click', () => {
    const action = el.dataset.action;
    const idx = S.selectedIdx;
    hideContextMenu();
    if (idx < 0) return;
    switch (action) {
      case 'change-label':
        promptLabel(label => {
          pushUndo();
          S.shapes[idx].label = label;
          S.dirty = true;
          addLabelHistory(label);
          renderShapeList();
          updateEditor();
          draw();
        });
        break;
      case 'delete':
        deleteSelected();
        break;
      case 'duplicate': {
        pushUndo();
        const copy = JSON.parse(JSON.stringify(S.shapes[idx]));
        // Offset the copy slightly so it's visible
        copy.points = copy.points.map(([x, y]) => [x + 10, y + 10]);
        S.shapes.push(copy);
        S.selectedIdx = S.shapes.length - 1;
        S.dirty = true;
        renderShapeList();
        updateEditor();
        draw();
        break;
      }
      case 'to-front': {
        pushUndo();
        const shape = S.shapes.splice(idx, 1)[0];
        S.shapes.push(shape);
        S.selectedIdx = S.shapes.length - 1;
        S.dirty = true;
        renderShapeList();
        draw();
        break;
      }
      case 'to-back': {
        pushUndo();
        const shape = S.shapes.splice(idx, 1)[0];
        S.shapes.unshift(shape);
        S.selectedIdx = 0;
        S.dirty = true;
        renderShapeList();
        draw();
        break;
      }
    }
  });
});

// ============ Keyboard shortcuts ============
document.addEventListener('keydown', e => {
  // Don't handle when typing in inputs
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.ctrlKey && e.key === 's') { e.preventDefault(); saveAnnotation(); return; }
  if (e.ctrlKey && e.key === 'z') { e.preventDefault(); undo(); return; }
  switch(e.key) {
    case 'v': setTool('select'); break;
    case 'r': setTool('rect'); break;
    case 'p': setTool('polygon'); break;
    case 'a': navFiltered(-1); break;
    case 'd': navFiltered(1); break;
    case 'f': fitView(); draw(); break;
    case 'j': if (S.images.length > 0) loadImage(Math.floor(Math.random() * S.images.length)); break;
    case 'Delete': case 'Backspace': deleteSelected(); break;
    case 'Escape':
      if (S.drawing) { S.drawing = false; S.drawPoints = []; draw(); }
      else { S.selectedIdx = -1; renderShapeList(); updateEditor(); draw(); }
      break;
    case '=': case '+': zoomCenter(1.2); break;
    case '-': zoomCenter(0.8); break;
  }
});

// ============ Folder browser ============
document.getElementById('btn-folder').addEventListener('click', openFolderDialog);

let _browseTarget = 'data'; // 'data' or 'label'

async function openFolderDialog() {
  const dlg = document.getElementById('folder-dialog');
  dlg.style.display = 'block';
  const r = await fetch('/api/datadir');
  const info = await r.json();
  document.getElementById('folder-path-input').value = info.path;
  document.getElementById('folder-label-input').value = info.label_path || '';
  document.getElementById('folder-format').value = info.format || 'auto';
  _browseTarget = 'data';
  browseTo(info.path, 'folder-list');
}

// browseTo: listId can be a DOM element or a string id; inputEl can be passed directly
async function browseTo(dirPath, listId, inputEl) {
  const r = await fetch('/api/browse?path=' + encodeURIComponent(dirPath));
  const data = await r.json();
  const inputMap = {
    'folder-list': 'folder-path-input',
    'folder-label-list': 'folder-label-input',
    'cmp-gt-list': 'cmp-gt-path',
  };
  const list = (typeof listId === 'string') ? document.getElementById(listId) : listId;
  if (!inputEl) {
    const mapKey = (typeof listId === 'string') ? listId : '';
    inputEl = document.getElementById(inputMap[mapKey] || 'folder-path-input');
  }
  inputEl.value = data.path;
  if (data.error) {
    list.innerHTML = `<div class="folder-error">${data.error}</div>`;
    return;
  }
  const paths = [data.path + '/..'];
  let html = `<div class="folder-item" data-idx="0">📁 ..</div>`;
  data.dirs.forEach((d, i) => {
    paths.push(data.path + '/' + d.name);
    const imgInfo = d.img_count > 0 ? `<span class="folder-img-count">${d.img_count} 张图片</span>` : '';
    html += `<div class="folder-item" data-idx="${i + 1}">
      <span class="folder-item-name">📁 ${d.name}${imgInfo}</span>
      <button class="btn-pick-folder" data-idx="${i + 1}" title="选择此目录">✔</button>
    </div>`;
  });
  list.innerHTML = html;
  list.querySelectorAll('.folder-item-name').forEach(el => {
    const item = el.closest('.folder-item');
    el.addEventListener('click', () => browseTo(paths[parseInt(item.dataset.idx)], listId, inputEl));
  });
  // ".." item has no pick button, just navigate
  const dotdot = list.querySelector('.folder-item[data-idx="0"]');
  if (dotdot) dotdot.addEventListener('click', () => browseTo(paths[0], listId, inputEl));
  list.querySelectorAll('.btn-pick-folder').forEach(el => {
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      const p = paths[parseInt(el.dataset.idx)];
      inputEl.value = p;
      list.innerHTML = '';
    });
  });
}

document.getElementById('folder-go').addEventListener('click', () => {
  browseTo(document.getElementById('folder-path-input').value, 'folder-list');
});
document.getElementById('folder-path-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') browseTo(e.target.value, 'folder-list');
});
document.getElementById('folder-label-go').addEventListener('click', () => {
  browseTo(document.getElementById('folder-label-input').value || document.getElementById('folder-path-input').value, 'folder-label-list');
});
document.getElementById('folder-label-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') browseTo(e.target.value, 'folder-label-list');
});

document.getElementById('folder-select').addEventListener('click', async () => {
  const path = document.getElementById('folder-path-input').value;
  const labelPath = document.getElementById('folder-label-input').value;
  const format = document.getElementById('folder-format').value;
  const r = await fetch('/api/datadir', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, label_path: labelPath, format }),
  });
  const data = await r.json();
  if (data.ok) {
    document.getElementById('folder-dialog').style.display = 'none';
    S.currentIdx = -1;
    S.filterLabel = null;
    await fetchImages();
    await fetchLabels();
    if (S.images.length > 0) loadImage(0);
  } else {
    alert(data.error || '无法打开目录');
  }
});

document.getElementById('folder-cancel').addEventListener('click', () => {
  document.getElementById('folder-dialog').style.display = 'none';
});

// ============ Analysis ============
document.getElementById('btn-analyze').addEventListener('click', async () => {
  const dlg = document.getElementById('analysis-dialog');
  const content = document.getElementById('analysis-content');
  dlg.style.display = 'block';
  content.innerHTML = '扫描标注中...';
  try {
    const r = await fetch('/api/analysis');
    const d = await r.json();
    if (!d.ok) { content.innerHTML = '分析失败'; return; }
    content.innerHTML = renderAnalysis(d);
  } catch (e) {
    content.innerHTML = '请求失败: ' + e.message;
  }
});

document.getElementById('analysis-close').addEventListener('click', () => {
  document.getElementById('analysis-dialog').style.display = 'none';
});

function renderAnalysis(d) {
  const cc = d.class_counts;
  const cs = d.class_stats;
  const labels = Object.keys(cc).sort((a, b) => cc[b] - cc[a]);
  const maxCount = Math.max(...Object.values(cc), 1);
  const sd = d.size_distribution;

  // Overview stats
  let html = `<div class="analysis-grid">
    <div class="analysis-stat"><div class="val">${d.total_images}</div><div class="lbl">总图片数</div></div>
    <div class="analysis-stat"><div class="val">${d.images_with_ann}</div><div class="lbl">有标注图片</div></div>
    <div class="analysis-stat"><div class="val">${d.total_boxes}</div><div class="lbl">总标注框数</div></div>
    <div class="analysis-stat"><div class="val">${d.avg_boxes_per_image}</div><div class="lbl">平均每张框数</div></div>
  </div>`;

  // Class distribution
  html += `<div class="analysis-section"><h4>类别分布</h4><table class="analysis-table">
    <tr><th>类别</th><th>数量</th><th>占比</th><th>分布</th></tr>`;
  labels.forEach(l => {
    const cnt = cc[l];
    const pct = (cnt / d.total_boxes * 100).toFixed(1);
    const barW = (cnt / maxCount * 100).toFixed(1);
    const c = colorFor(l);
    html += `<tr><td>${l}</td><td>${cnt}</td><td>${pct}%</td>
      <td><div class="analysis-bar-wrap"><div class="analysis-bar" style="width:${barW}%;background:${c}"></div></div></td></tr>`;
  });
  html += `</table></div>`;

  // Size distribution
  const sdTotal = sd.small + sd.medium + sd.large || 1;
  html += `<div class="analysis-section"><h4>框尺寸分布 (按面积占比)</h4><table class="analysis-table">
    <tr><th>类型</th><th>数量</th><th>占比</th><th>说明</th></tr>
    <tr><td>小目标</td><td>${sd.small}</td><td>${(sd.small/sdTotal*100).toFixed(1)}%</td><td>面积 &lt; 1%</td></tr>
    <tr><td>中目标</td><td>${sd.medium}</td><td>${(sd.medium/sdTotal*100).toFixed(1)}%</td><td>1% ≤ 面积 &lt; 10%</td></tr>
    <tr><td>大目标</td><td>${sd.large}</td><td>${(sd.large/sdTotal*100).toFixed(1)}%</td><td>面积 ≥ 10%</td></tr>
  </table></div>`;

  // Per-class size stats
  html += `<div class="analysis-section"><h4>各类别框尺寸 (归一化)</h4><table class="analysis-table">
    <tr><th>类别</th><th>数量</th><th>平均宽</th><th>平均高</th><th>宽范围</th><th>高范围</th></tr>`;
  labels.forEach(l => {
    const s = cs[l];
    if (!s) return;
    html += `<tr><td>${l}</td><td>${s.count}</td>
      <td>${s.avg_w}</td><td>${s.avg_h}</td>
      <td>${s.min_w} ~ ${s.max_w}</td><td>${s.min_h} ~ ${s.max_h}</td></tr>`;
  });
  html += `</table></div>`;

  // Width-Height scatter (simple canvas-based)
  if (d.box_widths.length > 0) {
    html += `<div class="analysis-section"><h4>宽高分布散点图</h4>
      <canvas id="analysis-scatter" width="500" height="300" style="background:#11111b;border-radius:6px;width:100%;"></canvas></div>`;
  }

  // Use setTimeout to draw scatter after DOM update
  setTimeout(() => drawScatter(d.box_widths, d.box_heights, labels, cc), 50);
  return html;
}

function drawScatter(widths, heights) {
  const canvas = document.getElementById('analysis-scatter');
  if (!canvas) return;
  const ctx2 = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = 40;

  // Axes
  ctx2.strokeStyle = '#45475a';
  ctx2.lineWidth = 1;
  ctx2.beginPath();
  ctx2.moveTo(pad, pad);
  ctx2.lineTo(pad, H - pad);
  ctx2.lineTo(W - pad, H - pad);
  ctx2.stroke();

  // Labels
  ctx2.fillStyle = '#6c7086';
  ctx2.font = '11px sans-serif';
  ctx2.fillText('归一化宽度', W / 2 - 20, H - 8);
  ctx2.save();
  ctx2.translate(12, H / 2 + 20);
  ctx2.rotate(-Math.PI / 2);
  ctx2.fillText('归一化高度', 0, 0);
  ctx2.restore();

  // Ticks
  for (let i = 0; i <= 4; i++) {
    const v = (i * 0.25).toFixed(2);
    const x = pad + (W - 2 * pad) * (i / 4);
    const y = H - pad - (H - 2 * pad) * (i / 4);
    ctx2.fillStyle = '#45475a';
    ctx2.fillText(v, x - 10, H - pad + 14);
    ctx2.fillText(v, 4, y + 4);
  }

  // Points
  const maxW = Math.max(...widths, 0.01);
  const maxH = Math.max(...heights, 0.01);
  const scaleX = (W - 2 * pad) / Math.min(maxW * 1.1, 1);
  const scaleY = (H - 2 * pad) / Math.min(maxH * 1.1, 1);

  ctx2.fillStyle = 'rgba(137, 180, 250, 0.4)';
  for (let i = 0; i < widths.length; i++) {
    const x = pad + widths[i] * scaleX;
    const y = H - pad - heights[i] * scaleY;
    ctx2.beginPath();
    ctx2.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx2.fill();
  }
}

// ============ Conflict detection ============
document.getElementById('btn-conflicts').addEventListener('click', async () => {
  document.getElementById('btn-conflicts').textContent = '⚠ 检测中...';
  document.getElementById('btn-conflicts').disabled = true;
  try {
    const r = await fetch('/api/conflicts?iou=0.5');
    const data = await r.json();
    if (!data.ok || data.total_images === 0) {
      alert(`未发现冲突标注 (共扫描 ${S.totalImages} 张图片)`);
      return;
    }
    // Show only conflict images by filtering via a special search
    // Store conflict image names for reference
    const conflictNames = data.conflicts.map(c => c.image);
    S.images = conflictNames;
    S.totalImages = conflictNames.length;
    renderImageList();
    alert(`发现 ${data.total_images} 张图片存在冲突标注 (共 ${data.total_pairs} 对), 已过滤显示`);
    if (S.images.length > 0) loadImage(0);
  } finally {
    document.getElementById('btn-conflicts').textContent = '⚠ 冲突';
    document.getElementById('btn-conflicts').disabled = false;
  }
});

// ============ Evaluate mode ============
let _evalCharts = []; // track Chart.js instances for cleanup

document.getElementById('btn-compare').addEventListener('click', () => {
  document.getElementById('compare-dialog').style.display = 'block';
  // Pre-fill GT with current label dir
  const gtInput = document.getElementById('cmp-gt-path');
  if (!gtInput.value) {
    gtInput.value = document.getElementById('folder-label-input')?.value || '';
  }
});

// GT browse
document.getElementById('cmp-gt-browse').addEventListener('click', () => {
  browseTo(document.getElementById('cmp-gt-path').value || '~', 'cmp-gt-list');
});
document.getElementById('cmp-gt-path').addEventListener('keydown', e => {
  if (e.key === 'Enter') browseTo(e.target.value, 'cmp-gt-list');
});

// Wire up browse for model entries (works for dynamically added ones too)
function wireModelEntry(entry) {
  const browseBtn = entry.querySelector('.cmp-model-browse');
  const pathInput = entry.querySelector('.cmp-model-path');
  const listDiv = entry.querySelector('.cmp-model-list');
  browseBtn.addEventListener('click', () => {
    browseTo(pathInput.value || '~', listDiv, pathInput);
  });
  pathInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') browseTo(e.target.value, listDiv, pathInput);
  });
}
// Wire initial model entry
document.querySelectorAll('.cmp-model-entry').forEach(wireModelEntry);

// Add model button
let _modelIdx = 1;
document.getElementById('cmp-add-model').addEventListener('click', () => {
  _modelIdx++;
  const container = document.getElementById('cmp-models-container');
  const div = document.createElement('div');
  div.className = 'cmp-model-entry';
  div.dataset.idx = _modelIdx - 1;
  div.innerHTML = `
    <label class="cmp-label">模型 ${_modelIdx} 标注目录: <span class="cmp-model-remove" title="移除" style="float:right;cursor:pointer;color:#f38ba8;">✕</span></label>
    <div class="folder-path-row">
      <input class="cmp-model-path" type="text" placeholder="模型推理结果目录">
      <button class="cmp-model-browse">浏览</button>
    </div>
    <div class="cmp-model-list"></div>
    <label class="cmp-label">格式:</label>
    <select class="cmp-model-format">
      <option value="auto">自动检测</option>
      <option value="json">X-AnyLabeling (JSON)</option>
      <option value="voc">VOC (XML)</option>
      <option value="coco">COCO (JSON)</option>
      <option value="yolo">YOLO (TXT)</option>
    </select>`;
  container.appendChild(div);
  wireModelEntry(div);
  div.querySelector('.cmp-model-remove').addEventListener('click', () => div.remove());
});

document.getElementById('cmp-cancel').addEventListener('click', () => {
  document.getElementById('compare-dialog').style.display = 'none';
});

// Start evaluation
document.getElementById('cmp-start').addEventListener('click', async () => {
  const gtPath = document.getElementById('cmp-gt-path').value.trim();
  const gtFmt = document.getElementById('cmp-gt-format').value;
  if (!gtPath) { alert('请填写 GT 标注目录'); return; }

  const entries = document.querySelectorAll('.cmp-model-entry');
  const models = [];
  entries.forEach((el, i) => {
    const p = el.querySelector('.cmp-model-path').value.trim();
    const f = el.querySelector('.cmp-model-format').value;
    if (p) models.push({ path: p, format: f, name: p.split('/').filter(Boolean).pop() || ('模型' + (i+1)) });
  });
  if (models.length === 0) { alert('至少需要一个模型目录'); return; }

  const btn = document.getElementById('cmp-start');
  btn.textContent = '评估中...';
  btn.disabled = true;

  // Show progress bar
  const dlgBox = document.querySelector('.compare-dialog-box');
  let progWrap = document.getElementById('cmp-progress');
  if (!progWrap) {
    progWrap = document.createElement('div');
    progWrap.id = 'cmp-progress';
    progWrap.innerHTML = '<div class="cmp-prog-text"></div><div class="cmp-prog-bar-wrap"><div class="cmp-prog-bar"></div></div>';
    dlgBox.querySelector('.dialog-btns').before(progWrap);
  }
  progWrap.style.display = 'block';
  const progText = progWrap.querySelector('.cmp-prog-text');
  const progBar = progWrap.querySelector('.cmp-prog-bar');
  progText.textContent = '准备中...';
  progBar.style.width = '0%';

  try {
    const r = await fetch('/api/evaluate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ gt_path: gtPath, gt_format: gtFmt, models, iou_thresh: 0.5 }),
    });
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let finalData = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const msg = JSON.parse(line.slice(6));
        if (msg.type === 'progress') {
          const pct = ((msg.current + 0.5) / msg.total * 100).toFixed(0);
          progText.textContent = `评估中 (${msg.current + 1}/${msg.total}): ${msg.name}`;
          progBar.style.width = pct + '%';
        } else if (msg.type === 'done') {
          finalData = msg;
          progBar.style.width = '100%';
          progText.textContent = '完成';
        }
      }
    }
    if (!finalData || !finalData.ok) { alert(finalData?.error || '评估失败'); return; }
    document.getElementById('compare-dialog').style.display = 'none';
    showEvalResults(finalData.results);
  } finally {
    btn.textContent = '开始评估';
    btn.disabled = false;
    progWrap.style.display = 'none';
  }
});

// Render evaluation results
function showEvalResults(results) {
  // Destroy old charts
  _evalCharts.forEach(c => c.destroy());
  _evalCharts = [];

  const content = document.getElementById('eval-content');
  const modelNames = Object.keys(results);
  let html = '';

  modelNames.forEach((name, mi) => {
    const ev = results[name];
    if (ev.error) {
      html += `<div class="eval-model-section"><h4>${name}</h4><p style="color:#f38ba8;">${ev.error}</p></div>`;
      return;
    }
    const labels = ev.labels || [];
    html += `<div class="eval-model-section"><h4>📊 ${name}</h4>`;
    // Metrics summary
    html += `<div class="eval-metrics">
      <div class="eval-metric"><div class="val">${(ev.mAP50 * 100).toFixed(1)}%</div><div class="lbl">mAP@50</div></div>
      <div class="eval-metric"><div class="val">${ev.total_gt}</div><div class="lbl">GT 框数</div></div>
      <div class="eval-metric"><div class="val">${ev.total_pred}</div><div class="lbl">预测框数</div></div>
      <div class="eval-metric"><div class="val">${labels.length}</div><div class="lbl">类别数</div></div>
    </div>`;
    // Per-class AP table
    html += `<div class="eval-ap-table"><table class="analysis-table"><tr><th>类别</th><th>AP@50</th></tr>`;
    labels.forEach(l => {
      const ap = ev.ap_per_class[l] || 0;
      const barW = (ap * 100).toFixed(1);
      html += `<tr><td>${l}</td><td><div class="eval-ap-bar-wrap"><div class="eval-ap-bar" style="width:${barW}%"></div><span>${(ap*100).toFixed(1)}%</span></div></td></tr>`;
    });
    html += `</table></div>`;
    // Chart canvases
    html += `<div class="eval-charts">
      <div class="eval-chart-wrap"><h5>PR 曲线</h5><canvas id="eval-pr-${mi}"></canvas></div>
      <div class="eval-chart-wrap"><h5>F1 曲线</h5><canvas id="eval-f1-${mi}"></canvas></div>
      <div class="eval-chart-wrap eval-chart-full"><h5>混淆矩阵</h5><canvas id="eval-cm-${mi}"></canvas></div>
    </div>`;
    html += `</div>`;
  });

  content.innerHTML = html;
  document.getElementById('eval-dialog').style.display = 'block';

  // Draw charts after DOM update
  setTimeout(() => {
    modelNames.forEach((name, mi) => {
      const ev = results[name];
      if (ev.error) return;
      drawPRChart(mi, ev);
      drawF1Chart(mi, ev);
      drawConfusionMatrix(mi, ev);
    });
  }, 50);
}

function drawPRChart(mi, ev) {
  const el = document.getElementById('eval-pr-' + mi);
  if (!el) return;
  const datasets = ev.labels.map((lbl, i) => {
    const pr = ev.pr_curves[lbl];
    const points = pr.recall.map((r, j) => ({ x: r, y: pr.precision[j] }));
    return { label: lbl + ' (AP=' + ((ev.ap_per_class[lbl]||0)*100).toFixed(1) + '%)', data: points,
      borderColor: COLORS[i % COLORS.length], backgroundColor: 'transparent', pointRadius: 0, borderWidth: 1.5, tension: 0.1 };
  });
  const chart = new Chart(el, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, showLine: true,
      scales: { x: { title: { display: true, text: 'Recall', color: '#6c7086' }, min: 0, max: 1, ticks: { color: '#6c7086' }, grid: { color: '#31324422' } },
                y: { title: { display: true, text: 'Precision', color: '#6c7086' }, min: 0, max: 1, ticks: { color: '#6c7086' }, grid: { color: '#31324422' } } },
      plugins: { legend: { labels: { color: '#cdd6f4', font: { size: 10 } } } }
    }
  });
  _evalCharts.push(chart);
}

function drawF1Chart(mi, ev) {
  const el = document.getElementById('eval-f1-' + mi);
  if (!el) return;
  const datasets = ev.labels.map((lbl, i) => {
    const fc = ev.f1_curves[lbl];
    return { label: lbl, data: fc.confidence.map((c, j) => ({ x: c, y: fc.f1[j] })),
      borderColor: COLORS[i % COLORS.length], backgroundColor: 'transparent', pointRadius: 0, borderWidth: 1.5, tension: 0.1 };
  });
  const chart = new Chart(el, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, showLine: true,
      scales: { x: { title: { display: true, text: 'Confidence', color: '#6c7086' }, min: 0, max: 1, reverse: true, ticks: { color: '#6c7086' }, grid: { color: '#31324422' } },
                y: { title: { display: true, text: 'F1', color: '#6c7086' }, min: 0, max: 1, ticks: { color: '#6c7086' }, grid: { color: '#31324422' } } },
      plugins: { legend: { labels: { color: '#cdd6f4', font: { size: 10 } } } }
    }
  });
  _evalCharts.push(chart);
}

function drawConfusionMatrix(mi, ev) {
  const el = document.getElementById('eval-cm-' + mi);
  if (!el) return;
  const labels = [...ev.labels, 'BG'];
  const matrix = ev.confusion_matrix;
  const n = labels.length;
  // Find max value for color scaling
  let maxVal = 1;
  matrix.forEach(row => row.forEach(v => { if (v > maxVal) maxVal = v; }));

  const ctx2 = el.getContext('2d');
  const pad = { top: 30, left: 70, right: 20, bottom: 60 };
  const cellSize = Math.min(Math.floor((600 - pad.left - pad.right) / n), 50);
  const W = pad.left + cellSize * n + pad.right;
  const H = pad.top + cellSize * n + pad.bottom;
  el.width = W; el.height = H;
  el.style.width = W + 'px'; el.style.height = H + 'px';

  // Draw cells
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = matrix[r]?.[c] || 0;
      const intensity = v / maxVal;
      const x = pad.left + c * cellSize;
      const y = pad.top + r * cellSize;
      // Color: diagonal=green, off-diagonal=red
      if (r === c) {
        ctx2.fillStyle = `rgba(166,227,161,${0.1 + intensity * 0.8})`;
      } else {
        ctx2.fillStyle = v > 0 ? `rgba(243,139,168,${0.15 + intensity * 0.7})` : 'rgba(49,50,68,0.3)';
      }
      ctx2.fillRect(x, y, cellSize, cellSize);
      ctx2.strokeStyle = '#45475a';
      ctx2.strokeRect(x, y, cellSize, cellSize);
      // Value text
      if (v > 0) {
        ctx2.fillStyle = intensity > 0.5 ? '#1e1e2e' : '#cdd6f4';
        ctx2.font = `${Math.max(9, cellSize * 0.35)}px sans-serif`;
        ctx2.textAlign = 'center';
        ctx2.textBaseline = 'middle';
        ctx2.fillText(v, x + cellSize / 2, y + cellSize / 2);
      }
    }
  }
  // Row labels (GT)
  ctx2.fillStyle = '#a6adc8';
  ctx2.font = `${Math.max(9, cellSize * 0.3)}px sans-serif`;
  ctx2.textAlign = 'right';
  ctx2.textBaseline = 'middle';
  for (let r = 0; r < n; r++) {
    ctx2.fillText(labels[r], pad.left - 4, pad.top + r * cellSize + cellSize / 2);
  }
  // Column labels (Pred)
  ctx2.textAlign = 'center';
  ctx2.textBaseline = 'top';
  for (let c = 0; c < n; c++) {
    ctx2.save();
    ctx2.translate(pad.left + c * cellSize + cellSize / 2, pad.top + n * cellSize + 4);
    ctx2.rotate(Math.PI / 4);
    ctx2.fillText(labels[c], 0, 0);
    ctx2.restore();
  }
  // Axis titles
  ctx2.fillStyle = '#6c7086';
  ctx2.font = '11px sans-serif';
  ctx2.textAlign = 'center';
  ctx2.fillText('预测类别 (Predicted)', pad.left + cellSize * n / 2, H - 6);
  ctx2.save();
  ctx2.translate(12, pad.top + cellSize * n / 2);
  ctx2.rotate(-Math.PI / 2);
  ctx2.fillText('真实类别 (GT)', 0, 0);
  ctx2.restore();
}

document.getElementById('eval-close').addEventListener('click', () => {
  document.getElementById('eval-dialog').style.display = 'none';
  _evalCharts.forEach(c => c.destroy());
  _evalCharts = [];
});

// ============ SAM3 AI-assisted labeling ============
const SAM3 = { active: false, pointMode: false, points: [] };

document.getElementById('btn-sam3').addEventListener('click', async () => {
  const bar = document.getElementById('sam3-bar');
  if (SAM3.active) { closeSam3(); return; }
  // Check availability
  const status = document.getElementById('sam3-status');
  status.textContent = '检查模型...';
  bar.style.display = 'flex';
  try {
    const r = await fetch('/api/sam3/status');
    const d = await r.json();
    if (!d.available) { status.textContent = '模型文件不存在'; return; }
    status.textContent = d.loaded ? '已就绪' : '首次使用需加载模型';
    SAM3.active = true;
  } catch (e) { status.textContent = '连接失败'; }
});

function closeSam3() {
  SAM3.active = false;
  SAM3.pointMode = false;
  SAM3.points = [];
  document.getElementById('sam3-bar').style.display = 'none';
  document.getElementById('sam3-point-btn').classList.remove('active');
  canvas.style.cursor = S.tool === 'select' ? 'default' : 'crosshair';
  draw();
}
document.getElementById('sam3-close').addEventListener('click', closeSam3);

// Text prompt
document.getElementById('sam3-run').addEventListener('click', sam3RunText);
document.getElementById('sam3-text').addEventListener('keydown', e => {
  if (e.key === 'Enter') sam3RunText();
});

async function sam3RunText() {
  if (S.currentIdx < 0) return;
  const text = document.getElementById('sam3-text').value.trim();
  if (!text) { alert('请输入文本提示'); return; }
  const btn = document.getElementById('sam3-run');
  const status = document.getElementById('sam3-status');
  btn.disabled = true;
  status.textContent = '推理中...';
  try {
    const r = await fetch('/api/sam3/text', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        image: S.images[S.currentIdx], text,
        show_masks: document.getElementById('sam3-mask').checked,
        conf: 0.3,
      }),
    });
    const d = await r.json();
    if (!d.ok) { alert(d.error); status.textContent = '失败'; return; }
    sam3AddShapes(d.shapes);
    status.textContent = `检测到 ${d.shapes.length} 个目标`;
  } catch (e) { status.textContent = '请求失败'; }
  finally { btn.disabled = false; }
}

// Batch inference
document.getElementById('sam3-batch').addEventListener('click', sam3RunBatch);

async function sam3RunBatch() {
  const text = document.getElementById('sam3-text').value.trim();
  if (!text) { alert('请输入文本提示'); return; }
  // Use current loaded images list
  const imgs = [...S.images];
  if (imgs.length === 0) { alert('没有图片'); return; }
  if (!confirm(`将对 ${imgs.length} 张图片进行批量推理\n文本: ${text}\n已有标注的图片将跳过\n\n继续？`)) return;

  const btn = document.getElementById('sam3-batch');
  const runBtn = document.getElementById('sam3-run');
  const status = document.getElementById('sam3-status');
  btn.disabled = true;
  runBtn.disabled = true;
  status.textContent = '批量推理中 0/' + imgs.length;

  try {
    const r = await fetch('/api/sam3/batch', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        text, images: imgs,
        show_masks: document.getElementById('sam3-mask').checked,
        conf: 0.3, skip_existing: true,
      }),
    });
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const msg = JSON.parse(line.slice(6));
        if (msg.type === 'progress') {
          status.textContent = `批量推理 ${msg.current + 1}/${msg.total}: ${msg.image}`;
        } else if (msg.type === 'done') {
          status.textContent = `完成: ${msg.done}/${msg.total} 张, 共 ${msg.objects} 个目标`;
          // Reload current image annotation
          await fetchLabels();
          if (S.currentIdx >= 0) loadImage(S.currentIdx);
        }
      }
    }
  } catch (e) { status.textContent = '批量推理失败'; }
  finally { btn.disabled = false; runBtn.disabled = false; }
}

// Point mode
document.getElementById('sam3-point-btn').addEventListener('click', () => {
  SAM3.pointMode = !SAM3.pointMode;
  SAM3.points = [];
  document.getElementById('sam3-point-btn').classList.toggle('active', SAM3.pointMode);
  canvas.style.cursor = SAM3.pointMode ? 'crosshair' : (S.tool === 'select' ? 'default' : 'crosshair');
  draw();
});

// Intercept canvas click for SAM3 point mode
canvas.addEventListener('click', async (e) => {
  if (!SAM3.active || !SAM3.pointMode) return;
  if (S.panning || S.dragging || S.drawing) return;
  const [ix, iy] = toImage(e.offsetX, e.offsetY);
  if (ix < 0 || iy < 0 || ix > S.imgW || iy > S.imgH) return;

  const positive = !e.shiftKey; // shift+click = negative point
  SAM3.points.push({ x: ix, y: iy, positive });
  draw(); // draw point markers

  // Send to backend
  const status = document.getElementById('sam3-status');
  status.textContent = '推理中...';
  try {
    const r = await fetch('/api/sam3/point', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        image: S.images[S.currentIdx],
        points: SAM3.points,
        text: document.getElementById('sam3-text').value.trim(),
        show_masks: document.getElementById('sam3-mask').checked,
        conf: 0.3,
      }),
    });
    const d = await r.json();
    if (!d.ok) { status.textContent = d.error; return; }
    // Show preview - replace previous SAM3 point results
    SAM3._previewShapes = d.shapes;
    status.textContent = `${d.shapes.length} 个目标 (Enter确认, Esc取消)`;
    draw();
  } catch (e) { status.textContent = '请求失败'; }
}, true);

// Confirm/cancel point results
document.addEventListener('keydown', (e) => {
  if (!SAM3.active || !SAM3.pointMode) return;
  if (e.key === 'Enter' && SAM3._previewShapes?.length) {
    sam3AddShapes(SAM3._previewShapes);
    SAM3.points = [];
    SAM3._previewShapes = null;
    document.getElementById('sam3-status').textContent = '已添加';
    draw();
    e.preventDefault();
  } else if (e.key === 'Escape' && SAM3.pointMode) {
    SAM3.points = [];
    SAM3._previewShapes = null;
    document.getElementById('sam3-status').textContent = '已取消';
    draw();
    e.preventDefault();
  }
});

function sam3AddShapes(shapes) {
  pushUndo();
  for (const s of shapes) {
    S.shapes.push(normalizeShape(s));
  }
  S.dirty = true;
  renderShapeList();
  updateEditor();
  draw();
}

// Draw SAM3 point markers and preview shapes
const _origDrawSam3 = _draw;
function _drawWithSam3() {
  _origDrawSam3();
  if (!SAM3.active) return;
  // Draw point markers
  if (SAM3.pointMode && SAM3.points.length > 0) {
    for (const pt of SAM3.points) {
      const [cx, cy] = toCanvas(pt.x, pt.y);
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = pt.positive ? '#a6e3a1' : '#f38ba8';
      ctx.fill();
      ctx.strokeStyle = '#1e1e2e';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  // Draw preview shapes
  if (SAM3._previewShapes) {
    for (const s of SAM3._previewShapes) {
      const pts = s.points || [];
      if (pts.length < 2) continue;
      ctx.save();
      ctx.strokeStyle = '#a6e3a1';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      if (s.shape_type === 'rectangle') {
        const [x1, y1] = toCanvas(pts[0][0], pts[0][1]);
        const [x2, y2] = toCanvas(pts[1][0], pts[1][1]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = 'rgba(166,227,161,0.12)';
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      } else {
        ctx.beginPath();
        const [sx, sy] = toCanvas(pts[0][0], pts[0][1]);
        ctx.moveTo(sx, sy);
        for (let j = 1; j < pts.length; j++) {
          const [px, py] = toCanvas(pts[j][0], pts[j][1]);
          ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fillStyle = 'rgba(166,227,161,0.15)';
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }
  }
}
_draw = _drawWithSam3;

// ============ Resize ============
window.addEventListener('resize', () => {
  if (S.img) { fitView(); } else { resizeCanvas(); }
});

// ============ Init ============
(async () => {
  await fetchImages();
  await fetchLabels();
  resizeCanvas();
  if (S.images.length > 0) loadImage(0);
})();

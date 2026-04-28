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
  filteredImages: [],   // indices into S.images matching filter
};

const COLORS = ['#f38ba8','#a6e3a1','#89b4fa','#fab387','#cba6f7','#f9e2af','#94e2d5','#f2cdcd','#89dceb','#eba0ac'];
function colorFor(label) {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = ((h << 5) - h + label.charCodeAt(i)) | 0;
  return COLORS[Math.abs(h) % COLORS.length];
}

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvas-wrap');

// ============ API ============
async function fetchImages() {
  const r = await fetch('/api/images');
  S.images = await r.json();
  updateFilteredImages();
  renderImageList();
}

async function fetchLabels() {
  const r = await fetch('/api/labels');
  S.labelMap = await r.json();
  renderLabelFilter();
}

function updateFilteredImages() {
  if (!S.filterLabel) {
    S.filteredImages = S.images.map((_, i) => i);
  } else {
    const allowed = new Set(S.labelMap[S.filterLabel] || []);
    S.filteredImages = S.images.map((name, i) => ({ name, i }))
      .filter(x => allowed.has(x.name))
      .map(x => x.i);
  }
}

function setLabelFilter(label) {
  S.filterLabel = S.filterLabel === label ? null : label;
  updateFilteredImages();
  renderLabelFilter();
  renderImageList();
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

async function loadImage(idx) {
  if (S.dirty && !confirm('当前标注未保存，是否切换？')) return;
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

  // Load image and annotation in parallel
  const [img, ann] = await Promise.all([
    new Promise((resolve) => {
      const image = new window.Image();
      image.onload = () => resolve(image);
      image.onerror = () => resolve(null);
      image.src = '/api/image/' + encodeURIComponent(name);
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
  // Use the actual CSS layout size of the wrapper
  const wrapW = wrap.clientWidth;
  const wrapH = wrap.clientHeight;
  console.log('[fitView] wrap size:', wrapW, 'x', wrapH, 'img size:', S.imgW, 'x', S.imgH);
  if (wrapW <= 0 || wrapH <= 0) return;
  canvas.width = wrapW;
  canvas.height = wrapH;
  const pad = 40;
  const cw = wrapW - pad*2, ch = wrapH - pad*2;
  if (!S.imgW || !S.imgH || cw <= 0 || ch <= 0) return;
  // Ensure image fits completely
  const scaleW = cw / S.imgW;
  const scaleH = ch / S.imgH;
  S.scale = Math.min(scaleW, scaleH);
  S.offsetX = (wrapW - S.imgW * S.scale) / 2;
  S.offsetY = (wrapH - S.imgH * S.scale) / 2;
  console.log('[fitView] scale:', S.scale, 'offset:', S.offsetX, S.offsetY);
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

  // Draw image from offscreen cache
  if (S.img) {
    const [ix, iy] = toCanvas(0, 0);
    const dw = S.imgW * S.scale, dh = S.imgH * S.scale;
    // Use cached bitmap when available
    if (_imgCache && Math.abs(_imgCacheScale - S.scale) < 0.001) {
      ctx.drawImage(_imgCache, ix, iy);
    } else {
      ctx.drawImage(S.img, ix, iy, dw, dh);
      // Cache at current scale (async, won't block)
      _cacheImage(dw, dh);
    }
  }

  // Draw shapes
  S.shapes.forEach((s, i) => {
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
    if (s.points.length > 0) {
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

  // Cursor
  if (S.tool === 'select' && !S.dragging) {
    const hi = hitHandle(mx, my, S.selectedIdx);
    if (hi >= 0) { canvas.style.cursor = 'grab'; return; }
    const idx = hitTest(mx, my);
    canvas.style.cursor = idx >= 0 ? 'move' : 'default';
  }
});

canvas.addEventListener('mouseleave', () => {
  S.mouseX = -1;
  S.mouseY = -1;
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
  }
});

function finishPolygon() {
  const pts = [...S.drawPoints];
  S.drawing = false;
  S.drawPoints = [];
  promptLabel(label => {
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

  function done(ok) {
    dlg.style.display = 'none';
    cleanup();
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
  // Collect from current shapes + history
  const labels = [...new Set([...S.labelHistory, ...S.shapes.map(s => s.label).filter(Boolean)])];
  labels.forEach(l => { const o = document.createElement('option'); o.value = l; dl.appendChild(o); });
}

// ============ UI rendering ============
function renderImageList() {
  const list = document.getElementById('img-list');
  const search = document.getElementById('search').value.toLowerCase();
  const filtered = S.filteredImages
    .map(i => ({ name: S.images[i], i }))
    .filter(x => x.name.toLowerCase().includes(search));
  list.innerHTML = filtered.map(({ name, i }) =>
    `<div class="item${i === S.currentIdx ? ' active' : ''}${hasAnnotation(name) ? ' has-ann' : ''}" data-idx="${i}">${name}</div>`
  ).join('');
  list.querySelectorAll('.item').forEach(el => {
    el.addEventListener('click', () => loadImage(parseInt(el.dataset.idx)));
  });
  const total = S.filteredImages.length;
  const pos = S.currentIdx >= 0 ? S.filteredImages.indexOf(S.currentIdx) + 1 : 0;
  const filterInfo = S.filterLabel ? ` [${S.filterLabel}]` : '';
  document.getElementById('img-counter').textContent = `${pos} / ${total}${filterInfo}`;
}

function hasAnnotation(name) {
  // Simple check: if current image, check shapes length
  if (S.images[S.currentIdx] === name) return S.shapes.length > 0;
  return false; // Can't know without loading
}

function renderShapeList() {
  const list = document.getElementById('shape-list');
  list.innerHTML = S.shapes.map((s, i) => {
    const c = colorFor(s.label || 'default');
    return `<div class="shape-item${i === S.selectedIdx ? ' selected' : ''}" data-idx="${i}">
      <span><span class="color-dot" style="background:${c}"></span>${s.label || '(no label)'}</span>
      <span style="color:#6c7086;font-size:11px">${s.shape_type}</span>
    </div>`;
  }).join('');
  list.querySelectorAll('.shape-item').forEach(el => {
    el.addEventListener('click', () => {
      S.selectedIdx = parseInt(el.dataset.idx);
      renderShapeList();
      updateEditor();
      draw();
    });
  });
}

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
  if (S.filteredImages.length === 0) return;
  const curPos = S.filteredImages.indexOf(S.currentIdx);
  let next;
  if (curPos < 0) {
    next = dir > 0 ? 0 : S.filteredImages.length - 1;
  } else {
    next = curPos + dir;
  }
  if (next >= 0 && next < S.filteredImages.length) {
    loadImage(S.filteredImages[next]);
  }
}
document.getElementById('btn-zoomin').addEventListener('click', () => { zoomCenter(1.2); });
document.getElementById('btn-zoomout').addEventListener('click', () => { zoomCenter(0.8); });
document.getElementById('btn-fit').addEventListener('click', () => { fitView(); draw(); });
document.getElementById('search').addEventListener('input', renderImageList);

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
  S.shapes.splice(S.selectedIdx, 1);
  S.selectedIdx = -1;
  S.dirty = true;
  renderShapeList();
  updateEditor();
  draw();
}

// ============ Keyboard shortcuts ============
document.addEventListener('keydown', e => {
  // Don't handle when typing in inputs
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.ctrlKey && e.key === 's') { e.preventDefault(); saveAnnotation(); return; }
  switch(e.key) {
    case 'v': setTool('select'); break;
    case 'r': setTool('rect'); break;
    case 'p': setTool('polygon'); break;
    case 'a': navFiltered(-1); break;
    case 'd': navFiltered(1); break;
    case 'f': fitView(); draw(); break;
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

async function browseTo(dirPath, listId) {
  const r = await fetch('/api/browse?path=' + encodeURIComponent(dirPath));
  const data = await r.json();
  const inputId = listId === 'folder-list' ? 'folder-path-input' : 'folder-label-input';
  document.getElementById(inputId).value = data.path;
  const list = document.getElementById(listId);
  if (data.error) {
    list.innerHTML = `<div class="folder-error">${data.error}</div>`;
    return;
  }
  const paths = [data.path + '/..'];
  let html = `<div class="folder-item" data-idx="0">📁 ..</div>`;
  data.dirs.forEach((d, i) => {
    paths.push(data.path + '/' + d.name);
    const imgInfo = d.img_count > 0 ? `<span class="folder-img-count">${d.img_count} 张图片</span>` : '';
    html += `<div class="folder-item" data-idx="${i + 1}">📁 ${d.name}${imgInfo}</div>`;
  });
  list.innerHTML = html;
  list.querySelectorAll('.folder-item').forEach(el => {
    el.addEventListener('click', () => browseTo(paths[parseInt(el.dataset.idx)], listId));
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

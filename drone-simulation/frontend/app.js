/* =============================================================
   STENDR Sim — frontend
   - Leaflet map centered on Oslo, Norway
   - Smooth 60 fps interpolation between server frames (10 Hz)
   - Fused tracks (lime) shown next to ground truth (magenta) so the
     visual gap between estimate and reality is obvious
   - Click a row in the right-hand track list OR a marker on the map
     to inspect a track in the left drawer
   ============================================================= */
'use strict';

// ---------------------------------------------------------------
// Constants & globals
// ---------------------------------------------------------------
const DEFAULT_CENTER = [59.9139, 10.7522];   // Oslo, Norway
let baseLat = DEFAULT_CENTER[0];
let baseLon = DEFAULT_CENTER[1];

let map = null;
let baseMarker = null;
let rangeRings = [];
let cameraFovLayer = null;

// State arriving from the server
// We keep TWO snapshots — previous and current — and interpolate between them.
let prevSnap = null;   // { t, tracks_by_id, truth_by_id }
let currSnap = null;
let serverDt = 0.1;    // updated from each frame's `dt`
let lastFrameRecvMs = 0;

// Persistent on-map state
const trackMarkers = new Map();        // trackId -> { marker, headingLine, lastLevel, lastSelected }
const trackTrails  = new Map();        // trackId -> { polyline, latlngs[] }
const truthMarkers = new Map();        // truthId -> Leaflet marker
const measMarkers  = [];               // wiped & redrawn every frame

const TRAIL_LEN = 60;

let latest = null;     // most recent server payload (for the sidebar / drawer reads)
let selectedId = null;
let lastTrackListSig = '';   // memoised stringified IDs to avoid pointless DOM churn

// ---------------------------------------------------------------
// Coordinate conversion (local meters <-> lat/lon)
// ---------------------------------------------------------------
const METERS_PER_DEG_LAT = 111_320;
function metersPerDegLon(lat) {
  return METERS_PER_DEG_LAT * Math.cos(lat * Math.PI / 180);
}
function localToLatLng(x, y) {
  const lat = baseLat + (y / METERS_PER_DEG_LAT);
  const lon = baseLon + (x / metersPerDegLon(baseLat));
  return [lat, lon];
}

// ---------------------------------------------------------------
// Map setup
// ---------------------------------------------------------------
function initMap() {
  map = L.map('map', {
    center: DEFAULT_CENTER,
    zoom: 14,
    zoomControl: true,
    attributionControl: true,
    preferCanvas: true,
    zoomAnimation: true,
  });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(map);

  setTimeout(() => map.invalidateSize(), 100);
}

function drawStaticOverlays() {
  if (baseMarker) map.removeLayer(baseMarker);
  baseMarker = L.marker([baseLat, baseLon], {
    icon: L.divIcon({
      className: '',
      iconSize: [28, 28],
      iconAnchor: [14, 14],
      html: `
        <svg width="28" height="28" viewBox="0 0 28 28">
          <circle cx="14" cy="14" r="11" fill="none" stroke="#d4f15c" stroke-width="1.4"/>
          <circle cx="14" cy="14" r="3" fill="#d4f15c"/>
          <line x1="14" y1="2"  x2="14" y2="6"  stroke="#d4f15c" stroke-width="1.4"/>
          <line x1="14" y1="22" x2="14" y2="26" stroke="#d4f15c" stroke-width="1.4"/>
          <line x1="2"  y1="14" x2="6"  y2="14" stroke="#d4f15c" stroke-width="1.4"/>
          <line x1="22" y1="14" x2="26" y2="14" stroke="#d4f15c" stroke-width="1.4"/>
        </svg>`,
    }),
    interactive: false,
  }).addTo(map);

  rangeRings.forEach(r => map.removeLayer(r));
  rangeRings = [];
  for (const r of [500, 1000, 1500, 2000, 2500]) {
    const ring = L.circle([baseLat, baseLon], {
      radius: r, color: '#3b4651', weight: 1, opacity: 0.35,
      fillOpacity: 0, dashArray: '4 6', interactive: false,
    }).addTo(map);
    rangeRings.push(ring);

    const [labelLat,] = localToLatLng(0, r);
    const label = L.marker([labelLat, baseLon], {
      icon: L.divIcon({
        className: '',
        iconSize: [40, 14],
        iconAnchor: [-4, 7],
        html: `<div style="font-family:JetBrains Mono;font-size:9px;color:#6c7a86;letter-spacing:0.1em;">${r}m</div>`,
      }),
      interactive: false,
    }).addTo(map);
    rangeRings.push(label);
  }

  if (cameraFovLayer) map.removeLayer(cameraFovLayer);
  const fovPts = [[baseLat, baseLon]];
  const r = 600;
  for (let degFromNorth = -30; degFromNorth <= 30; degFromNorth += 2) {
    const rad = degFromNorth * Math.PI / 180;
    const x = r * Math.sin(rad);
    const y = r * Math.cos(rad);
    fovPts.push(localToLatLng(x, y));
  }
  cameraFovLayer = L.polygon(fovPts, {
    color: '#ffd166', weight: 1, opacity: 0.25,
    fillColor: '#ffd166', fillOpacity: 0.04, interactive: false,
  }).addTo(map);

  map.setView([baseLat, baseLon], 14);
}

// ---------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { try { ws.send('hi'); } catch (e) {} };
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (!latest || data.base_lat !== baseLat || data.base_lon !== baseLon) {
      baseLat = data.base_lat || DEFAULT_CENTER[0];
      baseLon = data.base_lon || DEFAULT_CENTER[1];
      drawStaticOverlays();
    }
    latest = data;
    serverDt = data.dt || 0.1;

    // Swap snapshots for interpolation
    prevSnap = currSnap;
    currSnap = snapFromPayload(data);
    lastFrameRecvMs = performance.now();

    updateStatus(data);
    renderMeasurements(data.measurements);
    updateTrackList(data.tracks);

    if (selectedId !== null) {
      const t = data.tracks.find(t => t.id === selectedId);
      if (t) updateDrawer(t);
    }
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

function snapFromPayload(data) {
  const tracks = {};
  for (const t of data.tracks) tracks[t.id] = t;
  const truth = {};
  for (const d of (data.ground_truth || [])) truth[d.id] = d;
  return { t: data.t, tracks, truth };
}

// ---------------------------------------------------------------
// 60 fps animation loop — interpolates between snapshots so motion looks smooth.
// Server runs at 10 Hz; without this the UI looks choppy.
// ---------------------------------------------------------------
function animLoop() {
  requestAnimationFrame(animLoop);
  if (!currSnap || !prevSnap) {
    // If we only have one snapshot, just render at its positions.
    if (currSnap) renderSnapshot(currSnap.tracks, currSnap.truth, 1);
    return;
  }
  // Time since last server frame, normalised to expected DT.
  const elapsedS = (performance.now() - lastFrameRecvMs) / 1000.0;
  // Cap alpha at 1.0 so we don't extrapolate past the last frame.
  const alpha = Math.min(1, elapsedS / serverDt);
  renderInterpolated(prevSnap, currSnap, alpha);
}

function renderInterpolated(prev, curr, alpha) {
  // Build interpolated track positions.
  const interpTracks = {};
  for (const id in curr.tracks) {
    const c = curr.tracks[id];
    const p = prev.tracks[id];
    if (p) {
      interpTracks[id] = {
        ...c,
        x: p.x + (c.x - p.x) * alpha,
        y: p.y + (c.y - p.y) * alpha,
      };
    } else {
      interpTracks[id] = c;
    }
  }
  const interpTruth = {};
  for (const id in curr.truth) {
    const c = curr.truth[id];
    const p = prev.truth[id];
    if (p) {
      interpTruth[id] = {
        ...c,
        x: p.x + (c.x - p.x) * alpha,
        y: p.y + (c.y - p.y) * alpha,
      };
    } else {
      interpTruth[id] = c;
    }
  }
  renderSnapshot(interpTracks, interpTruth, alpha);
}

function renderSnapshot(tracksById, truthById, alpha) {
  // Convert dict back to array for rendering
  const tracks = Object.values(tracksById);
  const truth  = Object.values(truthById);
  renderTracks(tracks);
  renderTruth(truth);
}

// ---------------------------------------------------------------
// Status / health / control state
// ---------------------------------------------------------------
function updateStatus(s) {
  document.getElementById('stat-time').textContent     = s.t.toFixed(1) + 's';
  document.getElementById('stat-tracks').textContent   = String(s.n_tracks).padStart(2, '0');
  document.getElementById('stat-truth').textContent    = String(s.n_truth).padStart(2, '0');
  document.getElementById('stat-latency').textContent  = s.latency_ms.toFixed(1) + ' ms';
  document.getElementById('stat-mode').textContent     = s.mode.toUpperCase();

  document.getElementById('mode-kalman').classList.toggle('ctl-active', s.mode === 'kalman');
  document.getElementById('mode-jpda').classList.toggle('ctl-active',   s.mode === 'jpda');

  // Speed buttons: highlight the closest one
  const cur = s.sim_speed || 3;
  document.querySelectorAll('.ctl-speed').forEach(btn => {
    btn.classList.toggle('ctl-active', Number(btn.dataset.speed) === Math.round(cur));
  });

  setJamBtn('jam-radar', s.jamming.radar);
  setJamBtn('jam-rf',    s.jamming.rf);
  setJamBtn('jam-cam',   s.jamming.camera_spoof);

  document.getElementById('btn-pause').textContent = s.paused ? 'RESUME' : 'PAUSE';

  setHealth('h-radar',    s.jamming.radar       ? 'down' : '');
  setHealth('h-rf',       s.jamming.rf          ? 'down' : '');
  setHealth('h-camera',   s.jamming.camera_spoof? 'degraded' : '');
  setHealth('h-acoustic', '');
}

function setJamBtn(id, on) {
  const el = document.getElementById(id);
  el.classList.toggle('ctl-active', on);
  el.querySelector('.ctl-jam-status').textContent = on ? 'ON' : 'OFF';
}
function setHealth(id, cls) {
  const el = document.getElementById(id);
  el.classList.remove('degraded', 'down');
  if (cls) el.classList.add(cls);
}

// ---------------------------------------------------------------
// Track list in the right panel
// ---------------------------------------------------------------
function updateTrackList(tracks) {
  const root = document.getElementById('track-list');
  // Sort by threat score (highest first) so operators see the worst threat first.
  const sorted = tracks.slice().sort((a, b) => (b.score || 0) - (a.score || 0));

  // Memoise — only re-render when the set of IDs (or selection) changes.
  const sig = sorted.map(t => `${t.id}:${t.level}:${t.id === selectedId ? 1 : 0}`).join('|');
  if (sig === lastTrackListSig) {
    // Still need to update the live error values on existing rows
    for (const t of sorted) {
      const row = root.querySelector(`[data-track-id="${t.id}"]`);
      if (row) {
        const meta = row.querySelector('.tl-meta');
        const acc = t.accuracy;
        const errStr = acc ? ` · err ${acc.pos_error_m.toFixed(0)}m` : '';
        meta.textContent = `${t.speed_mps.toFixed(1)} m/s · ${t.range_m.toFixed(0)}m${errStr}`;
      }
    }
    return;
  }
  lastTrackListSig = sig;

  if (sorted.length === 0) {
    root.innerHTML = '<div class="track-list-empty">no tracks</div>';
    return;
  }
  root.innerHTML = '';
  for (const t of sorted) {
    const row = document.createElement('div');
    row.className = 'track-list-item threat-' + (t.level || 'low');
    if (t.id === selectedId) row.classList.add('selected');
    row.dataset.trackId = t.id;
    const acc = t.accuracy;
    const errStr = acc ? ` · err ${acc.pos_error_m.toFixed(0)}m` : '';
    row.innerHTML = `
      <span class="tl-dot"></span>
      <span>
        <span class="tl-id">T${String(t.id).padStart(2, '0')}</span>
        <div class="tl-meta">${t.speed_mps.toFixed(1)} m/s · ${t.range_m.toFixed(0)}m${errStr}</div>
      </span>
      <span style="font-size:9px;letter-spacing:0.15em;">${(t.level || 'low').toUpperCase()}</span>
    `;
    row.addEventListener('click', () => selectTrack(t.id));
    root.appendChild(row);
  }
}

// ---------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------
function measColor(sensor, spoofed) {
  if (spoofed) return '#ff6a6a';
  return { radar: '#6ab7ff', camera: '#ffd166', rf: '#b388ff', acoustic: '#67e8c1' }[sensor] || '#888';
}
function renderMeasurements(meas) {
  for (const m of measMarkers) map.removeLayer(m);
  measMarkers.length = 0;
  if (!meas) return;
  for (const m of meas) {
    const [lat, lon] = localToLatLng(m.x, m.y);
    const color = measColor(m.sensor, m.spoofed);
    const marker = L.circleMarker([lat, lon], {
      radius: m.spoofed ? 3 : 3.5,
      color, weight: 1,
      opacity: m.spoofed ? 0.55 : 0.85,
      fillColor: color,
      fillOpacity: m.spoofed ? 0.25 : 0.65,
      interactive: false,
    }).addTo(map);
    measMarkers.push(marker);
  }
}

// ---------------------------------------------------------------
// Ground truth markers (magenta diamonds — the "real" drones)
// ---------------------------------------------------------------
function truthIcon() {
  return L.divIcon({
    className: '',
    iconSize: [16, 16],
    iconAnchor: [8, 8],
    html: `
      <svg width="16" height="16" viewBox="0 0 16 16">
        <polygon points="8,2 14,8 8,14 2,8" fill="none" stroke="#ff66c4" stroke-width="1.4"/>
        <circle cx="8" cy="8" r="1.5" fill="#ff66c4"/>
      </svg>`,
  });
}

function renderTruth(truthList) {
  const seen = new Set();
  for (const d of truthList) {
    seen.add(d.id);
    const [lat, lon] = localToLatLng(d.x, d.y);
    let marker = truthMarkers.get(d.id);
    if (!marker) {
      marker = L.marker([lat, lon], { icon: truthIcon(), interactive: false }).addTo(map);
      truthMarkers.set(d.id, marker);
    } else {
      marker.setLatLng([lat, lon]);
    }
  }
  for (const id of Array.from(truthMarkers.keys())) {
    if (!seen.has(id)) {
      map.removeLayer(truthMarkers.get(id));
      truthMarkers.delete(id);
    }
  }
}

// ---------------------------------------------------------------
// Tracks (the fused estimate — lime / threat-coloured)
// ---------------------------------------------------------------
function threatColor(level) {
  if (level === 'critical') return '#ff5555';
  if (level === 'high')     return '#ffb86c';
  if (level === 'medium')   return '#ffd166';
  return '#d4f15c';
}

function trackIcon(t, isSelected) {
  const color = threatColor(t.level);
  const r = isSelected ? 12 : 9;
  const cs = isSelected ? `<line x1="2"  y1="16" x2="9"  y2="16" stroke="${color}" stroke-width="1"/>
                           <line x1="23" y1="16" x2="30" y2="16" stroke="${color}" stroke-width="1"/>
                           <line x1="16" y1="2"  x2="16" y2="9"  stroke="${color}" stroke-width="1"/>
                           <line x1="16" y1="23" x2="16" y2="30" stroke="${color}" stroke-width="1"/>` : '';
  return L.divIcon({
    className: 'drone-marker',
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    html: `
      <svg width="32" height="32" viewBox="0 0 32 32">
        ${cs}
        <circle cx="16" cy="16" r="${r}" fill="none" stroke="${color}" stroke-width="${isSelected ? 2 : 1.4}"/>
        <circle cx="16" cy="16" r="3" fill="${color}"/>
      </svg>
      <div style="position:absolute;left:36px;top:8px;font-family:JetBrains Mono;font-size:10px;color:${isSelected ? '#fff' : 'rgba(216,225,232,0.75)'};letter-spacing:0.08em;white-space:nowrap;text-shadow:0 0 4px #000;">T${String(t.id).padStart(2,'0')}</div>
    `,
  });
}

function renderTracks(tracks) {
  const seen = new Set();
  for (const t of tracks) {
    seen.add(t.id);
    const [lat, lon] = localToLatLng(t.x, t.y);
    const isSelected = t.id === selectedId;

    let entry = trackMarkers.get(t.id);
    if (!entry) {
      const marker = L.marker([lat, lon], { icon: trackIcon(t, isSelected) }).addTo(map);
      marker.on('click', () => selectTrack(t.id));
      entry = { marker, headingLine: null, lastLevel: t.level, lastSelected: isSelected };
      trackMarkers.set(t.id, entry);
    } else {
      entry.marker.setLatLng([lat, lon]);
      if (entry.lastLevel !== t.level || entry.lastSelected !== isSelected) {
        entry.marker.setIcon(trackIcon(t, isSelected));
        entry.lastLevel = t.level;
        entry.lastSelected = isSelected;
      }
    }

    // Heading vector (2 seconds ahead)
    const tipX = t.x + (t.vx || 0) * 2.0;
    const tipY = t.y + (t.vy || 0) * 2.0;
    const tip = localToLatLng(tipX, tipY);
    const headPts = [[lat, lon], tip];
    if (!entry.headingLine) {
      entry.headingLine = L.polyline(headPts, {
        color: threatColor(t.level),
        weight: 1.2, opacity: 0.45, interactive: false,
      }).addTo(map);
    } else {
      entry.headingLine.setLatLngs(headPts);
      entry.headingLine.setStyle({ color: threatColor(t.level) });
    }

    // Trail
    let trail = trackTrails.get(t.id);
    if (!trail) {
      trail = { polyline: null, latlngs: [] };
      trackTrails.set(t.id, trail);
    }
    // Only append a trail point when the server gives us a new frame, not every
    // animation tick — otherwise the trail explodes with thousands of points.
    if (currSnap && trail.lastT !== currSnap.t) {
      const tr = currSnap.tracks[t.id];
      if (tr) {
        const [trLat, trLon] = localToLatLng(tr.x, tr.y);
        trail.latlngs.push([trLat, trLon]);
        if (trail.latlngs.length > TRAIL_LEN) trail.latlngs.shift();
        trail.lastT = currSnap.t;
      }
    }
    if (!trail.polyline) {
      trail.polyline = L.polyline(trail.latlngs, {
        color: '#d4f15c', weight: 1.4, opacity: 0.35, interactive: false,
      }).addTo(map);
    } else {
      trail.polyline.setLatLngs(trail.latlngs);
    }
  }

  // GC tracks that disappeared
  for (const id of Array.from(trackMarkers.keys())) {
    if (!seen.has(id)) {
      const e = trackMarkers.get(id);
      if (e.marker) map.removeLayer(e.marker);
      if (e.headingLine) map.removeLayer(e.headingLine);
      trackMarkers.delete(id);
      const tr = trackTrails.get(id);
      if (tr && tr.polyline) map.removeLayer(tr.polyline);
      trackTrails.delete(id);
      if (selectedId === id) closeDrawer();
    }
  }
}

// ---------------------------------------------------------------
// Selection + drawer
// ---------------------------------------------------------------
function selectTrack(id) {
  selectedId = id;
  document.querySelector('.layout').classList.add('drawer-open');
  if (latest) {
    const t = latest.tracks.find(t => t.id === id);
    if (t) {
      updateDrawer(t);
      lastTrackListSig = '';  // force re-render to update selection highlight
      updateTrackList(latest.tracks);
      setTimeout(() => map.invalidateSize(), 280);
    }
  }
}

function accuracyGrade(errM) {
  if (errM == null) return { label: 'UNMATCHED', cls: 'unmatched' };
  if (errM < 15)  return { label: 'EXCELLENT', cls: 'excellent' };
  if (errM < 40)  return { label: 'GOOD',      cls: 'good' };
  if (errM < 100) return { label: 'FAIR',      cls: 'fair' };
  return                 { label: 'POOR',      cls: 'poor' };
}

function updateDrawer(t) {
  document.getElementById('d-id').textContent      = 'T' + String(t.id).padStart(2, '0');
  const lev = document.getElementById('d-level');
  lev.className = 'threat-pill ' + (t.level || 'low');
  lev.textContent = (t.level || 'low').toUpperCase();

  document.getElementById('d-pos').textContent     = `${t.x.toFixed(0)}, ${t.y.toFixed(0)} m`;
  document.getElementById('d-range').textContent   = `${t.range_m.toFixed(0)} m`;
  document.getElementById('d-speed').textContent   = `${t.speed_mps.toFixed(1)} m/s`;
  document.getElementById('d-heading').textContent = `${t.heading.toFixed(0)}°`;
  document.getElementById('d-closing').textContent = `${t.closing_mps.toFixed(1)} m/s`;
  document.getElementById('d-ttb').textContent     = t.ttb_s != null ? `${t.ttb_s.toFixed(1)} s` : '— (not closing)';

  document.getElementById('d-sensors').textContent = (t.sensors && t.sensors.length) ? t.sensors.join(' · ') : 'none';
  document.getElementById('d-type').textContent    = t.drone_type || 'unknown';
  document.getElementById('d-hits').textContent    = `${t.hits} / ${t.misses}`;
  document.getElementById('d-age').textContent     = `${t.age.toFixed(1)} s`;
  document.getElementById('d-cov').textContent     = t.cov_trace.toFixed(1);

  // Accuracy section
  const acc = t.accuracy;
  const grade = accuracyGrade(acc ? acc.pos_error_m : null);
  const gradeEl = document.getElementById('d-grade');
  gradeEl.className = 'accuracy-grade ' + grade.cls;
  gradeEl.textContent = grade.label;
  if (acc) {
    document.getElementById('d-truth-id').textContent = 'D' + String(acc.truth_id).padStart(2, '0');
    document.getElementById('d-pos-err').textContent  = `${acc.pos_error_m.toFixed(1)} m`;
    document.getElementById('d-vel-err').textContent  = `${acc.vel_error_mps.toFixed(2)} m/s`;
    document.getElementById('d-hd-err').textContent   = `${acc.heading_error_deg.toFixed(0)}°`;
  } else {
    document.getElementById('d-truth-id').textContent = '—';
    document.getElementById('d-pos-err').textContent  = '—';
    document.getElementById('d-vel-err').textContent  = '—';
    document.getElementById('d-hd-err').textContent   = '—';
  }

  const pct = Math.min(100, Math.max(0, t.score * 100));
  const bar = document.getElementById('d-bar');
  bar.style.width = pct + '%';
  bar.style.background = threatColor(t.level);
  document.getElementById('d-score').textContent = t.score.toFixed(3);
}

function closeDrawer() {
  selectedId = null;
  document.querySelector('.layout').classList.remove('drawer-open');
  lastTrackListSig = '';
  if (latest) updateTrackList(latest.tracks);
  setTimeout(() => map.invalidateSize(), 280);
}
document.getElementById('drawer-close').addEventListener('click', closeDrawer);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });

// ---------------------------------------------------------------
// Controls
// ---------------------------------------------------------------
async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : '{}',
  });
  return r.json();
}

document.getElementById('btn-launch').addEventListener('click',
  () => post('/api/launch', { n: 12, pattern: 'pincer' }));
document.getElementById('btn-launch-line').addEventListener('click',
  () => post('/api/launch', { n: 10, pattern: 'line' }));
document.getElementById('btn-pause').addEventListener('click',
  () => post('/api/pause'));
document.getElementById('btn-clear').addEventListener('click',
  () => { post('/api/clear'); closeDrawer(); });

document.querySelectorAll('.ctl-toggle').forEach(btn => {
  btn.addEventListener('click', () => post('/api/mode', { mode: btn.dataset.mode }));
});
document.querySelectorAll('.ctl-jam').forEach(btn => {
  btn.addEventListener('click', () => {
    const wasActive = btn.classList.contains('ctl-active');
    post('/api/jam', { target: btn.dataset.jam, on: !wasActive });
  });
});
document.querySelectorAll('.ctl-speed').forEach(btn => {
  btn.addEventListener('click', () => post('/api/speed', { speed: Number(btn.dataset.speed) }));
});

// ---------------------------------------------------------------
// Boot
// ---------------------------------------------------------------
initMap();
drawStaticOverlays();
connect();
requestAnimationFrame(animLoop);

window.addEventListener('resize', () => map && map.invalidateSize());

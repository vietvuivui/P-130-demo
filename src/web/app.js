/* AutoLabel 2D — UI review by exception (không cần build, gọi thẳng FastAPI) */
'use strict';

// ?project=<id>: làm việc trong một dự án (data/projects/<id>), API cùng đường dẫn nhưng có tiền tố /p/<id>
const PROJECT = new URLSearchParams(location.search).get('project');
const API = PROJECT ? `/p/${encodeURIComponent(PROJECT)}/api/v1` : '/api/v1';
const LEVELS = ['high', 'medium', 'low'];
const LEVEL_NAME = { low: 'Low', medium: 'Medium', high: 'High' };
const RISK_COLOR = { low: '#0ca30c', medium: '#fab219', high: '#d03b3b' };
const HUMAN_COLOR = '#3987e5';
const STATUS_TEXT = { auto: 'Chưa mở', editing: 'Đang sửa', approved: 'Đã duyệt', rejected: 'Bị trả lại' };

const S = {
  cfg: null,
  queue: [],
  sort: 'risk',
  statusFilter: '',
  frame: null,
  img: null,
  sweepImgs: {},
  lidar: null,
  gt: null,
  selected: null,
  viewOffset: 0,
  showLidar: false,
  showGt: false,
  showLow: true,
  showMask: true,
  minScore: 0, // FR-06: ẩn box model có score thấp hơn (chỉ đổi cách xem)
  lowOpen: false,
  mode: 'view', // view | edit | add
  editBox: null,
  drag: null,
  timers: {},
  timerStart: null,
  autoProp: true,
  viewMode: 'image', // image | video
  videos: [],
  video: null, // VideoDetail đang mở
  playTimer: null,
  videoPoll: null,
  uploads: new Set(), // video vừa tải lên, đang chờ auto-label xong để báo
  zoom: 1, // 1 = vừa khung; phóng to tới 8x
  dragging: false, // đang kéo thẻ frame từ timeline
  timelineStale: false,
  scrolledTo: null,
};

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

// ---------- tiện ích ----------

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
const fx = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d));

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(d?.message || (typeof d === 'string' ? d : JSON.stringify(d)) || res.statusText);
  }
  return data;
}

let toastTimer;
function toast(msg, error = false) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast' + (error ? ' error' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), error ? 5000 : 2200);
}

// mỗi dự án nhớ chế độ / video đang mở riêng; tên người duyệt dùng chung
const storageKey = (k) => (PROJECT && k !== 'reviewer' ? `pj.${PROJECT}.${k}` : k);
function storageGet(k, fallback) {
  try { return localStorage.getItem(storageKey(k)) ?? fallback; } catch { return fallback; }
}
function storageSet(k, v) {
  try { localStorage.setItem(storageKey(k), v); } catch { /* bỏ qua: private mode */ }
}

const reviewer = () => $('reviewer').value.trim() || S.cfg?.reviewer || 'annotator';

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = src;
  });
}

function levelOf(o) { return o.qa?.level || 'low'; }
const isProp = (o) => o.source === 'propagated';
const isAutoDeleted = (o) => o.review.action === 'PROPAGATED_DELETE';
/* Nhãn lan truyền hiện c_prop thay cho score detector */
function scoreText(o) {
  if (o.source === 'human') return 'người vẽ';
  return isProp(o) ? `↦ c ${fx(o.propagation.prop_conf)}` : fx(o.score);
}
function finalBox(o) { return o.review.final_bbox || o.bbox; }
function finalLabel(o) { return o.review.final_label || o.label; }

/* Thứ tự duyệt: object chờ (high -> medium -> low, risk giảm dần) rồi tới object đã xử lý */
function orderedObjects() {
  if (!S.frame) return [];
  const pending = S.frame.objects.filter((o) => o.review.status === 'pending');
  const rank = { high: 0, medium: 1, low: 2 };
  pending.sort((a, b) => rank[levelOf(a)] - rank[levelOf(b)] || (b.qa?.risk || 0) - (a.qa?.risk || 0));
  return pending.concat(S.frame.objects.filter((o) => o.review.status !== 'pending'));
}
const getObj = (id) => S.frame?.objects.find((o) => o.object_id === id);

// ---------- hàng đợi ----------

async function loadQueue() {
  const q = new URLSearchParams({ sort: S.sort });
  if (S.statusFilter) q.set('status', S.statusFilter);
  S.queue = await api('/frames?' + q);
  renderQueue();
  $('empty-state').classList.toggle('hidden', S.queue.length > 0 || !!S.frame);
}

function riskLevel(r) {
  const lv = S.cfg.levels;
  return r >= lv.high ? 'high' : r >= lv.medium ? 'medium' : 'low';
}

function renderQueue() {
  const list = $('queue-list');
  list.innerHTML = S.queue.map((f) => {
    const lv = riskLevel(f.frame_risk);
    const statusText = STATUS_TEXT[f.status];
    return `<li class="queue-item ${S.frame?.frame_id === f.frame_id ? 'active' : ''}" data-id="${esc(f.frame_id)}">
      <div class="qi-top"><span class="qi-id">${esc(f.frame_id)}</span>${f.propagated_from && f.status === 'auto' ? `<span class="prop-tag" title="Nhãn lan truyền từ ${esc(f.propagated_from)}">↦</span>` : ''}<span class="status-pill ${f.status}">${statusText}</span></div>
      <div class="risk-meter" title="Frame risk ${fx(f.frame_risk)} (${LEVEL_NAME[lv]})"><span style="width:${Math.max(4, f.frame_risk * 100)}%;background:${RISK_COLOR[lv]}"></span></div>
      <div class="qi-meta">
        <span class="lv" title="High"><span class="dot high"></span>${f.counts.high}</span>
        <span class="lv" title="Medium"><span class="dot medium"></span>${f.counts.medium}</span>
        <span class="lv" title="Low"><span class="dot low"></span>${f.counts.low}</span>
        <span style="margin-left:auto">${f.pending ? f.pending + ' chờ' : '✓'}</span>
      </div></li>`;
  }).join('');
}

// ---------- frame ----------

function stopTimer() {
  if (S.frame && S.timerStart) {
    S.timers[S.frame.frame_id] = (S.timers[S.frame.frame_id] || 0) + (Date.now() - S.timerStart) / 1000;
  }
  S.timerStart = null;
}
function elapsed() {
  if (!S.frame) return 0;
  return (S.timers[S.frame.frame_id] || 0) + (S.timerStart ? (Date.now() - S.timerStart) / 1000 : 0);
}

async function openFrame(id, { preview = false } = {}) {
  if (!preview) stopPlay();
  stopTimer();
  const frame = await api(`/frames/${encodeURIComponent(id)}`);
  S.frame = frame;
  S.img = null;
  S.lidar = S.gt = null;
  S.sweepImgs = {};
  S.viewOffset = 0;
  S.mode = 'view';
  S.editBox = null;
  S.selected = orderedObjects()[0]?.object_id || null;
  if (frame.status !== 'approved' && !preview) S.timerStart = Date.now();
  $('empty-state').classList.add('hidden');
  renderAll();
  highlightTimeline();
  const img = await loadImage(`${API}/frames/${encodeURIComponent(id)}/image`);
  if (S.frame?.frame_id !== id) return; // người dùng đã chuyển sang frame khác
  S.img = img;
  fitCanvas();
  if (S.showLidar) await ensureLidar();
  if (S.showGt) await ensureGt();
  renderAll();
  loadSweeps(frame);
}

async function loadSweeps(frame) {
  await Promise.all(frame.sweeps.map(async (s) => {
    try {
      const im = await loadImage(`${API}/frames/${encodeURIComponent(frame.frame_id)}/image?offset=${s.offset}`);
      if (S.frame?.frame_id === frame.frame_id) S.sweepImgs[s.offset] = im;
    } catch { /* thiếu sweep: để trống */ }
  }));
  renderFilmstrip();
}

async function ensureLidar() {
  if (!S.lidar && S.frame) S.lidar = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/lidar`);
}
async function ensureGt() {
  if (!S.gt && S.frame) S.gt = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/gt`);
}

function renderAll() {
  renderHeader();
  draw();
  renderPanel();
  renderFilmstrip();
  renderQueue();
  renderRejectBanner();
  refreshHistory();
}

// ---------- trả lại frame (FR-15), hoàn tác / làm lại (FR-09) ----------

function renderRejectBanner() {
  const f = S.frame;
  const box = $('reject-banner');
  const show = f && f.reject_reason && f.status !== 'approved';
  box.classList.toggle('hidden', !show);
  if (show) {
    const when = (f.rejected_at || '').replace('T', ' ').replace('+00:00', ' UTC');
    box.innerHTML = `<b>Bị trả lại</b> bởi ${esc(f.rejected_by || '?')} · ${esc(when)}<br>${esc(f.reject_reason)}`;
  }
}

let historyToken = 0;
async function refreshHistory() {
  const f = S.frame;
  const token = ++historyToken;
  if (!f) { $('btn-undo').disabled = $('btn-redo').disabled = true; return; }
  try {
    const h = await api(`/frames/${encodeURIComponent(f.frame_id)}/history`);
    if (token !== historyToken) return; // đã mở frame khác
    const locked = S.frame?.status === 'approved';
    $('btn-undo').disabled = locked || !h.undo;
    $('btn-redo').disabled = locked || !h.redo;
    $('btn-undo').title = `Hoàn tác (Ctrl+Z) · còn ${h.undo} bước`;
    $('btn-redo').title = `Làm lại (Ctrl+Y) · còn ${h.redo} bước`;
  } catch { /* máy chủ cũ không có lịch sử */ }
}

async function undoRedo(dir) {
  if (!S.frame || S.frame.status === 'approved') return;
  if (S.mode !== 'view') cancelEdit();
  try {
    S.frame = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/${dir}`, { method: 'POST', body: { reviewer: reviewer() } });
    if (!getObj(S.selected)) S.selected = null;
    toast(dir === 'undo' ? 'Đã hoàn tác' : 'Đã làm lại');
    renderAll();
    refreshLists();
  } catch (err) {
    toast(err.message, true);
  }
}

function openReject() {
  if (!S.frame) return;
  $('reject-box').classList.remove('hidden');
  $('reject-reason').focus();
}

async function sendReject() {
  const reason = $('reject-reason').value.trim();
  if (reason.length < 3) { toast('Ghi lý do trả lại (ít nhất 3 ký tự)', true); $('reject-reason').focus(); return; }
  try {
    S.frame = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/reject`, { method: 'POST', body: { reason, reviewer: reviewer() } });
    $('reject-reason').value = '';
    $('reject-box').classList.add('hidden');
    toast(`Đã trả lại ${S.frame.frame_id}`);
    renderAll();
    refreshLists();
  } catch (err) {
    toast(err.message, true);
  }
}

function renderHeader() {
  const f = S.frame;
  $('frame-id').textContent = f ? f.frame_id : '—';
  const pill = $('frame-status');
  pill.className = 'status-pill ' + (f?.status || '');
  pill.textContent = f ? STATUS_TEXT[f.status] : '';
  let where = '';
  if (f && S.viewMode === 'video' && S.video) {
    const i = S.video.frames.findIndex((x) => x.frame_id === f.frame_id);
    if (i >= 0) where = ` · frame ${i + 1}/${S.video.frames.length} · t=${S.video.frames[i].t.toFixed(1)}s`;
  }
  $('frame-detectors').textContent = f
    ? `Detector: ${f.detectors.join(' + ')}${where}${f.propagated_from ? ` · ↦ lan truyền từ ${frameRef(f.propagated_from)}` : ''}`
    : '';
}

// ---------- canvas ----------

function fitCanvas() {
  const W = S.img?.naturalWidth || 1600;
  const H = S.img?.naturalHeight || 900;
  canvas.width = W;
  canvas.height = H;
  const wrap = $('canvas-wrap');
  const scale = Math.min(wrap.clientWidth / W, wrap.clientHeight / H) * S.zoom;
  canvas.style.width = `${Math.floor(W * scale)}px`;
  canvas.style.height = `${Math.floor(H * scale)}px`;
  $('canvas-scroll').classList.toggle('zoomed', S.zoom > 1);
  $('btn-zoom-reset').textContent = `${Math.round(S.zoom * 100)}%`;
}

// ---------- zoom ----------

const ZOOM_MIN = 1;
const ZOOM_MAX = 8;

// Đổi mức zoom, giữ nguyên điểm ảnh dưới (cx, cy) — toạ độ màn hình; mặc định là tâm khung
function setZoom(z, cx, cy) {
  const next = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.round(z * 100) / 100));
  if (next === S.zoom) return;
  const box = $('canvas-scroll');
  const view = box.getBoundingClientRect();
  if (cx === undefined) { cx = view.left + view.width / 2; cy = view.top + view.height / 2; }
  const r = canvas.getBoundingClientRect();
  const fx = (cx - r.left) / r.width;
  const fy = (cy - r.top) / r.height;
  S.zoom = next;
  fitCanvas();
  draw();
  const r2 = canvas.getBoundingClientRect();
  // vị trí canvas trong vùng cuộn (margin auto khi canvas nhỏ hơn khung ở một chiều)
  const offX = r2.left - view.left + box.scrollLeft;
  const offY = r2.top - view.top + box.scrollTop;
  box.scrollLeft = offX + fx * r2.width - (cx - view.left);
  box.scrollTop = offY + fy * r2.height - (cy - view.top);
}
const zoomBy = (factor, cx, cy) => setZoom(S.zoom * factor, cx, cy);
const px = () => canvas.width / (canvas.getBoundingClientRect().width || canvas.width);

function toImg(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height];
}

function depthColor(d) {
  // gần = đỏ, xa = xanh (0 -> 60 m)
  const t = Math.min(1, d / 60);
  return `hsl(${Math.round(t * 230)}, 90%, 55%)`;
}

const belowScore = (o) => o.source !== 'human' && o.score < S.minScore;

// Mask sơ bộ của model (FR-04): chỉ khi người chưa sửa / vẽ lại box (mask không còn khớp box mới)
function drawMask(o, color, sel) {
  const m = o.mask;
  if (!m || m.length < 6 || ['EDIT_BOX', 'ADD_BOX'].includes(o.review.action)) return;
  ctx.save();
  ctx.globalAlpha = sel ? 0.35 : 0.2;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(m[0], m[1]);
  for (let i = 2; i < m.length; i += 2) ctx.lineTo(m[i], m[i + 1]);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawBox(b, color, { lw = 2, dash = null, label = null, alpha = 1, textColor = '#fff' } = {}) {
  const k = px();
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lw * k;
  if (dash) ctx.setLineDash(dash.map((v) => v * k));
  ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
  ctx.setLineDash([]);
  if (label) {
    ctx.font = `${600} ${12 * k}px 'Plus Jakarta Sans', sans-serif`;
    const w = ctx.measureText(label).width + 8 * k;
    const h = 17 * k;
    const y = b[1] - h >= 0 ? b[1] - h : b[1];
    ctx.fillStyle = color;
    ctx.fillRect(b[0] - (lw * k) / 2, y, w, h);
    ctx.fillStyle = textColor;
    ctx.fillText(label, b[0] + 4 * k - (lw * k) / 2, y + 12.5 * k);
  }
  ctx.restore();
}

function draw() {
  drawCanvas();
  window.bev2d?.draw(); // khung BEV (bev2d.js) theo cùng frame / box đang chọn / box đang sửa
}

function drawCanvas() {
  const f = S.frame;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!f) return;
  const img = S.viewOffset === 0 ? S.img : S.sweepImgs[S.viewOffset];
  if (img) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const k = px();

  if (S.showLidar && S.lidar && S.viewOffset === 0) {
    const r = 2.2 * k;
    for (let i = 0; i < S.lidar.u.length; i++) {
      ctx.fillStyle = depthColor(S.lidar.d[i]);
      ctx.fillRect(S.lidar.u[i] - r / 2, S.lidar.v[i] - r / 2, r, r);
    }
  }
  if (S.showGt && S.gt && S.viewOffset === 0) {
    for (const g of S.gt) {
      drawBox(g.bbox, g.ignore ? 'rgba(255,255,255,.35)' : '#ffffff', { lw: 1.2, dash: [4, 3], label: g.ignore ? null : `GT ${g.label}`, textColor: '#0b1020' });
    }
  }

  if (S.viewOffset !== 0) {
    // Xem một sweep: vẽ detection của sweep đó mờ + box track của object đang chọn
    const sweep = f.sweeps.find((s) => s.offset === S.viewOffset);
    for (const d of sweep?.detections || []) drawBox(d.bbox, '#94a3b8', { lw: 1, alpha: 0.7 });
    const o = getObj(S.selected);
    const tb = o?.track?.[String(S.viewOffset)];
    if (tb) drawBox(tb, '#3987e5', { lw: 3, label: `#${o.object_id} ${o.label} @t${S.viewOffset > 0 ? '+' : ''}${S.viewOffset}` });
    return;
  }

  let shown = 0, total = 0;
  for (const o of f.objects) {
    if (o.review.status === 'deleted') continue;
    total++;
    const lv = levelOf(o);
    const pending = o.review.status === 'pending';
    if (pending && lv === 'low' && !S.showLow && o.object_id !== S.selected) continue;
    if (belowScore(o) && o.object_id !== S.selected) continue;
    shown++;
    const sel = o.object_id === S.selected;
    if (sel && S.mode === 'edit') continue;
    const color = o.source === 'human' ? HUMAN_COLOR : RISK_COLOR[lv];
    if (S.showMask) drawMask(o, color, sel);
    // Nhãn đầy đủ chỉ cho box đang chọn / high / người vẽ; medium chỉ hiện #id để ảnh không bị che kín
    const full = `${pending ? '' : '✓ '}#${o.object_id} ${finalLabel(o)}${o.source === 'human' ? '' : ' ' + scoreText(o)}`;
    let tag = null;
    if (sel || o.source === 'human' || (pending && lv === 'high')) tag = full;
    else if (pending && lv === 'medium') tag = `#${o.object_id}`;
    drawBox(finalBox(o), color, {
      lw: sel ? 3.5 : pending ? 2 : 1.4,
      dash: o.source === 'track' ? [8, 5] : isProp(o) && !o.propagation.matched ? [3, 3] : null,
      label: tag,
      alpha: sel || S.selected == null ? 1 : 0.85,
    });
  }

  $('min-score-count').textContent = S.minScore > 0 ? ` · ${shown}/${total} box` : '';

  if (S.editBox) {
    drawBox(S.editBox, '#7c5cd6', { lw: 3, dash: S.mode === 'add' ? [6, 4] : null });
    if (S.mode === 'edit') {
      ctx.fillStyle = '#fff';
      ctx.strokeStyle = '#7c5cd6';
      ctx.lineWidth = 2 * k;
      for (const [hx, hy] of handles(S.editBox)) {
        ctx.fillRect(hx - 5 * k, hy - 5 * k, 10 * k, 10 * k);
        ctx.strokeRect(hx - 5 * k, hy - 5 * k, 10 * k, 10 * k);
      }
    }
  }
}

function handles(b) {
  const [x1, y1, x2, y2] = b;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return [[x1, y1], [mx, y1], [x2, y1], [x2, my], [x2, y2], [mx, y2], [x1, y2], [x1, my]];
}
// Mỗi handle ứng với cạnh nào bị kéo: [x1, y1, x2, y2]
const HANDLE_EDGES = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]];

function hitObject(x, y) {
  // Box nhỏ nhất chứa điểm click được chọn (để chọn được object nằm trong object khác)
  let best = null;
  let bestArea = Infinity;
  for (const o of S.frame.objects) {
    if (o.review.status === 'deleted') continue;
    if (o.review.status === 'pending' && levelOf(o) === 'low' && !S.showLow) continue;
    const b = finalBox(o);
    if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) {
      const a = (b[2] - b[0]) * (b[3] - b[1]);
      if (a < bestArea) { best = o; bestArea = a; }
    }
  }
  return best;
}

canvas.addEventListener('mousedown', (e) => {
  if (!S.frame || S.viewOffset !== 0) return;
  const [x, y] = toImg(e);
  if (S.mode === 'add') {
    S.drag = { kind: 'draw', x0: x, y0: y };
    S.editBox = [x, y, x, y];
    return;
  }
  if (S.mode === 'edit' && S.editBox) {
    const k = px();
    const hi = handles(S.editBox).findIndex(([hx, hy]) => Math.abs(hx - x) <= 8 * k && Math.abs(hy - y) <= 8 * k);
    if (hi >= 0) { S.drag = { kind: 'handle', edges: HANDLE_EDGES[hi] }; return; }
    const b = S.editBox;
    if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) { S.drag = { kind: 'move', x0: x, y0: y, box: b.slice() }; return; }
    return;
  }
  const o = hitObject(x, y);
  if (o) select(o.object_id);
  else if (S.zoom > 1) {
    // Kéo vùng trống để di chuyển ảnh khi đang phóng to
    const box = $('canvas-scroll');
    S.pan = { x: e.clientX, y: e.clientY, left: box.scrollLeft, top: box.scrollTop };
    box.classList.add('panning');
  }
});

window.addEventListener('mousemove', (e) => {
  if (!S.pan) return;
  const box = $('canvas-scroll');
  box.scrollLeft = S.pan.left - (e.clientX - S.pan.x);
  box.scrollTop = S.pan.top - (e.clientY - S.pan.y);
});
window.addEventListener('mouseup', () => {
  if (!S.pan) return;
  S.pan = null;
  $('canvas-scroll').classList.remove('panning');
});

// Lăn chuột trên ảnh = phóng to/thu nhỏ quanh con trỏ
$('canvas-scroll').addEventListener('wheel', (e) => {
  if (!S.frame) return;
  e.preventDefault();
  zoomBy(e.deltaY < 0 ? 1.25 : 0.8, e.clientX, e.clientY);
}, { passive: false });
$('btn-zoom-in').addEventListener('click', () => zoomBy(1.5));
$('btn-zoom-out').addEventListener('click', () => zoomBy(1 / 1.5));
$('btn-zoom-reset').addEventListener('click', () => setZoom(1));

window.addEventListener('mousemove', (e) => {
  if (!S.drag) return;
  let [x, y] = toImg(e);
  x = Math.max(0, Math.min(canvas.width, x));
  y = Math.max(0, Math.min(canvas.height, y));
  const d = S.drag;
  if (d.kind === 'draw') {
    S.editBox = [Math.min(d.x0, x), Math.min(d.y0, y), Math.max(d.x0, x), Math.max(d.y0, y)];
  } else if (d.kind === 'handle') {
    const b = S.editBox;
    if (d.edges[0]) b[0] = Math.min(x, b[2] - 4);
    if (d.edges[1]) b[1] = Math.min(y, b[3] - 4);
    if (d.edges[2]) b[2] = Math.max(x, b[0] + 4);
    if (d.edges[3]) b[3] = Math.max(y, b[1] + 4);
  } else if (d.kind === 'move') {
    const dx = x - d.x0;
    const dy = y - d.y0;
    S.editBox = [d.box[0] + dx, d.box[1] + dy, d.box[2] + dx, d.box[3] + dy];
  }
  draw();
});

window.addEventListener('mouseup', () => {
  if (!S.drag) return;
  const d = S.drag;
  S.drag = null;
  if (d.kind === 'draw') {
    const b = S.editBox;
    if (b[2] - b[0] < 4 || b[3] - b[1] < 4) { S.editBox = null; draw(); return; }
    showEditBar('add');
  }
});

function startEdit() {
  const o = getObj(S.selected);
  if (!o || S.frame.status === 'approved') return;
  S.mode = 'edit';
  S.editBox = finalBox(o).slice();
  showEditBar('edit');
  draw();
}

function startAdd() {
  if (!S.frame || S.frame.status === 'approved') return;
  S.viewOffset = 0;
  S.mode = 'add';
  S.editBox = null;
  canvas.classList.add('drawing');
  $('btn-add').classList.add('active');
  showEditBar('draw');
  draw();
}

function cancelEdit() {
  S.mode = 'view';
  S.editBox = null;
  S.drag = null;
  canvas.classList.remove('drawing');
  $('btn-add').classList.remove('active');
  $('edit-bar').classList.add('hidden');
  draw();
}

function showEditBar(kind) {
  $('edit-bar').classList.remove('hidden');
  $('add-class').classList.toggle('hidden', kind !== 'add');
  $('edit-save').classList.toggle('hidden', kind === 'draw');
  $('edit-hint').textContent = {
    draw: 'Kéo chuột trên ảnh để vẽ box mới',
    add: 'Chọn lớp cho box mới',
    edit: 'Kéo góc/cạnh hoặc kéo cả box để sửa',
  }[kind];
  if (kind === 'add') $('add-class').focus();
}

async function saveEdit() {
  const box = S.editBox.map((v) => Math.round(v * 10) / 10);
  if (S.mode === 'add') {
    await act({ action: 'ADD_BOX', bbox: box, label: $('add-class').value });
  } else if (S.mode === 'edit') {
    await act({ action: 'EDIT_BOX', object_id: S.selected, bbox: box });
  }
  cancelEdit();
}

// ---------- hành động review ----------

async function act(body) {
  try {
    const prevOrder = orderedObjects().map((o) => o.object_id);
    const frame = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/actions`, {
      method: 'POST',
      body: { ...body, reviewer: reviewer() },
    });
    S.frame = frame;
    if (body.action === 'ADD_BOX') {
      S.selected = frame.objects[frame.objects.length - 1].object_id;
    } else if (['KEEP', 'DELETE', 'CHANGE_CLASS'].includes(body.action)) {
      advanceFrom(body.object_id, prevOrder);
    }
    renderAll();
    refreshLists();
  } catch (err) {
    toast(err.message, true);
  }
}

/* Sau khi xử lý xong một object, nhảy sang object chờ kế tiếp */
function advanceFrom(id, prevOrder) {
  const pending = new Set(S.frame.objects.filter((o) => o.review.status === 'pending').map((o) => o.object_id));
  const i = prevOrder.indexOf(id);
  const next = prevOrder.slice(i + 1).find((x) => pending.has(x)) || prevOrder.find((x) => pending.has(x));
  S.selected = next || id;
}

async function approveLow() {
  if (!S.frame) return;
  try {
    S.frame = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/approve-low-risk`, { method: 'POST', body: { reviewer: reviewer() } });
    S.selected = orderedObjects().find((o) => o.review.status === 'pending')?.object_id || S.selected;
    renderAll();
    refreshLists();
    toast('Đã duyệt nhóm rủi ro thấp');
  } catch (err) {
    toast(err.message, true);
  }
}

async function approveFrame() {
  if (!S.frame) return;
  try {
    const t = Math.round(elapsed());
    S.frame = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/approve`, {
      method: 'POST',
      body: { reviewer: reviewer(), review_time_s: t },
    });
    stopTimer();
    toast(`Đã approve ${S.frame.frame_id} (${fmtTime(t)})`);
    const cur = S.frame;
    if (S.viewMode === 'video') {
      // Video: lan truyền sang các frame sau rồi mở frame kế tiếp (nơi vừa nhận nhãn lan truyền)
      if (S.autoProp) await propagate(cur.frame_id);
      await refreshVideo();
      const frames = S.video?.frames || [];
      const i = frames.findIndex((f) => f.frame_id === cur.frame_id);
      const next = frames.slice(i + 1).find((f) => f.status !== 'approved') || frames.find((f) => f.status !== 'approved');
      if (next) openFrame(next.frame_id); else renderAll();
      return;
    }
    await loadQueue();
    const next = S.queue.find((f) => f.status !== 'approved' && f.frame_id !== cur.frame_id);
    if (next) openFrame(next.frame_id); else renderAll();
  } catch (err) {
    toast(err.message, true);
  }
}

async function propagate(frameId) {
  try {
    const r = await api(`/frames/${encodeURIComponent(frameId)}/propagate`, { method: 'POST', body: {} });
    const n = r.frames_updated.length;
    const extra = r.objects_suppressed ? `, tự xoá ${r.objects_suppressed} box người đã xoá` : '';
    toast(n ? `Lan truyền sang ${n} frame (${r.objects_propagated} nhãn${extra}). ${r.stop_reason || ''}` : `Không lan truyền: ${r.stop_reason || 'không có frame phù hợp'}`);
    return r;
  } catch (err) {
    toast('Lan truyền lỗi: ' + err.message, true);
    return null;
  }
}

async function propagateCurrent() {
  if (S.viewMode !== 'video' || !S.frame || S.frame.status !== 'approved') return;
  await propagate(S.frame.frame_id);
  await refreshVideo();
}

async function reopenFrame() {
  S.frame = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/reopen`, { method: 'POST' });
  S.timerStart = Date.now();
  renderAll();
  refreshLists();
}

function select(id) {
  if (S.mode !== 'view') cancelEdit();
  S.selected = id;
  const o = getObj(id);
  if (o && levelOf(o) === 'low' && o.review.status === 'pending') S.lowOpen = true;
  renderPanel();
  draw();
  renderFilmstrip();
  document.querySelector(`[data-oid="${CSS.escape(id)}"]`)?.scrollIntoView({ block: 'nearest' });
}

// ---------- panel phải ----------

function classOptions(selected) {
  return Object.keys(S.cfg.classes).map((c) => `<option value="${c}" ${c === selected ? 'selected' : ''}>${c}</option>`).join('');
}

function objectCard(o) {
  const lv = levelOf(o);
  const issues = o.qa?.issues || [];
  const lid = o.qa?.lidar || {};
  const tmp = o.qa?.temporal || {};
  const locked = S.frame.status === 'approved';
  const done = o.review.status !== 'pending';
  const pr = o.propagation;
  const srcNote = o.source === 'track' ? ' · box nội suy' : o.source === 'human' ? ' · người vẽ' : isProp(o) ? ` · ↦ từ ${esc(frameRef(pr.keyframe_id))}` : '';
  const facts = [];
  if (isProp(o)) {
    facts.push(`c_prop ${fx(pr.prop_conf)}`);
    facts.push(pr.matched ? `detector: ${esc(pr.detector_label)} ${fx(pr.detector_score)}` : 'detector không thấy, box dự đoán');
  } else if (o.source !== 'human') facts.push(`score ${fx(o.score)}`);
  if (lid.available) facts.push(`${lid.n_points ?? 0} điểm LiDAR${lid.depth_m != null ? ` · ${fx(lid.depth_m, 1)} m` : ''}${lid.est_height_m != null ? ` · cao ~${fx(lid.est_height_m, 1)} m` : ''}`);
  if (tmp.available) facts.push(`sweep ${tmp.support}/${tmp.available}`);
  const status = !done
    ? ''
    : isAutoDeleted(o)
      ? `<span class="done-tag deleted" title="Người đã xoá object này ở ${esc(pr?.keyframe_id)}. Bấm Keep nếu đây là object thật.">✗ Tự xoá theo ${esc(frameRef(pr?.keyframe_id))}</span>`
      : `<span class="done-tag ${o.review.status}">${o.review.status === 'deleted' ? '✗ Đã xoá' : `✓ ${o.review.action}${o.review.final_label !== o.label ? ' → ' + esc(o.review.final_label) : ''}`}</span>`;
  return `<div class="obj-card ${lv} ${o.object_id === S.selected ? 'selected' : ''}" data-oid="${esc(o.object_id)}">
    <div class="oc-body">
      <canvas class="oc-crop" width="192" height="144" data-crop="${esc(o.object_id)}"></canvas>
      <div class="oc-info">
        <div class="oc-title"><span>${esc(finalLabel(o))} <span class="oid">#${esc(o.object_id)}${srcNote}</span></span>
          ${o.qa ? `<span class="risk-badge ${lv}">${LEVEL_NAME[lv]} ${fx(o.qa.risk)}</span>` : ''}</div>
        <div class="oc-stats">${facts.join(' · ')}</div>
        ${status}
        ${issues.length ? `<ul class="issues">${issues.map((i) => `<li title="${esc(S.cfg.issue_help[i.code] || '')}"><span class="issue-code">${esc(i.code)}</span> <span class="issue-msg">${esc(i.message)}</span></li>`).join('')}</ul>` : ''}
      </div>
    </div>
    ${locked ? '' : `<div class="oc-actions">
      <button class="btn btn-sm btn-keep" data-act="KEEP">✓ Keep</button>
      <button class="btn btn-sm btn-del" data-act="DELETE">🗑 Delete</button>
      <select data-class>${classOptions(finalLabel(o))}</select>
      <button class="btn btn-sm btn-class" data-act="CHANGE_CLASS">Đổi lớp</button>
      <button class="btn btn-sm btn-edit" data-act="EDIT">✎ Sửa box</button>
    </div>`}
  </div>`;
}

function compactRow(o) {
  const done = o.review.status !== 'pending';
  const tag = done
    ? isAutoDeleted(o)
      ? '<span class="done-tag deleted">✗ tự xoá</span>'
      : `<span class="done-tag ${o.review.status}">${o.review.status === 'deleted' ? '✗ xoá' : '✓ ' + o.review.action.toLowerCase().replace('_', ' ')}</span>`
    : `<span class="muted">risk ${fx(o.qa?.risk)}</span>`;
  return `<div class="low-row ${o.object_id === S.selected ? 'selected' : ''}" data-oid="${esc(o.object_id)}">
    <span><span class="dot ${levelOf(o)}"></span> #${esc(o.object_id)} ${esc(finalLabel(o))} <span class="muted">${scoreText(o)}</span></span>${tag}</div>`;
}

function renderPanel() {
  const f = S.frame;
  const objs = f ? orderedObjects() : [];
  const pending = objs.filter((o) => o.review.status === 'pending');
  const done = objs.filter((o) => o.review.status !== 'pending');
  const byLevel = { high: [], medium: [], low: [] };
  pending.forEach((o) => byLevel[levelOf(o)].push(o));

  const all = f ? f.objects.filter((o) => o.qa) : [];
  const lv = S.cfg?.levels || { medium: 0.3, high: 0.6 };
  const ranges = { low: `0.00–${fx(lv.medium)}`, medium: `${fx(lv.medium)}–${fx(lv.high)}`, high: `${fx(lv.high)}–1.00` };
  $('risk-summary').innerHTML = ['low', 'medium', 'high'].map((l) => `<div class="rs ${l}">
      <div class="rs-label"><span class="dot ${l}"></span>${LEVEL_NAME[l]} risk</div>
      <div class="rs-value">${all.filter((o) => levelOf(o) === l).length}</div>
      <div class="rs-range">${ranges[l]}</div></div>`).join('');

  const sel = S.selected;
  $('list-high').innerHTML = byLevel.high.map(objectCard).join('') || '<p class="muted">Không có object rủi ro cao chờ duyệt</p>';
  $('list-medium').innerHTML = byLevel.medium.map(objectCard).join('') || '<p class="muted">—</p>';
  $('list-low').innerHTML = byLevel.low.map((o) => (o.object_id === sel ? objectCard(o) : compactRow(o))).join('');
  $('list-low').classList.toggle('collapsed', !S.lowOpen);
  $('toggle-low').setAttribute('aria-expanded', String(S.lowOpen));
  $('list-done').innerHTML = done.map((o) => (o.object_id === sel ? objectCard(o) : compactRow(o))).join('');
  $('count-high').textContent = byLevel.high.length;
  $('count-medium').textContent = byLevel.medium.length;
  $('count-low').textContent = byLevel.low.length;
  $('count-done').textContent = done.length;

  const locked = f?.status === 'approved';
  const btnLow = $('btn-approve-low');
  btnLow.disabled = !f || locked || byLevel.low.length === 0;
  btnLow.innerHTML = `✓ Approve all low-risk (${byLevel.low.length}) <kbd>A</kbd>`;

  const total = objs.length;
  $('progress-bar').style.width = total ? `${(done.length / total) * 100}%` : '0';
  $('progress-text').textContent = f ? `${done.length}/${total} object đã xử lý` : '';
  const btn = $('btn-approve-frame');
  if (locked) {
    btn.disabled = false;
    btn.innerHTML = 'Mở lại để sửa';
    btn.onclick = reopenFrame;
  } else {
    btn.disabled = !f || pending.length > 0;
    btn.innerHTML = 'Approve frame <kbd>Enter</kbd>';
    btn.onclick = approveFrame;
  }
  $('btn-add').disabled = !f || locked;
  $('btn-propagate').disabled = !locked || S.viewMode !== 'video';
  drawCrops();
}

function drawCrops() {
  if (!S.img) return;
  document.querySelectorAll('canvas[data-crop]').forEach((c) => {
    const o = getObj(c.dataset.crop);
    if (!o) return;
    const [x1, y1, x2, y2] = finalBox(o);
    const pad = 0.15 * Math.max(x2 - x1, y2 - y1) + 6;
    const sx = Math.max(0, x1 - pad);
    const sy = Math.max(0, y1 - pad);
    const sw = Math.min(S.img.naturalWidth, x2 + pad) - sx;
    const sh = Math.min(S.img.naturalHeight, y2 + pad) - sy;
    const cc = c.getContext('2d');
    cc.fillStyle = '#0b1020';
    cc.fillRect(0, 0, c.width, c.height);
    const s = Math.min(c.width / sw, c.height / sh);
    const ox = (c.width - sw * s) / 2;
    const oy = (c.height - sh * s) / 2;
    cc.drawImage(S.img, sx, sy, sw, sh, ox, oy, sw * s, sh * s);
    cc.strokeStyle = o.source === 'human' ? HUMAN_COLOR : RISK_COLOR[levelOf(o)];
    cc.lineWidth = 3;
    if (o.source === 'track' || (isProp(o) && !o.propagation.matched)) cc.setLineDash([6, 4]);
    cc.strokeRect(ox + (x1 - sx) * s, oy + (y1 - sy) * s, (x2 - x1) * s, (y2 - y1) * s);
  });
}

document.querySelector('.review-panel').addEventListener('click', (e) => {
  const card = e.target.closest('[data-oid]');
  if (!card) return;
  const id = card.dataset.oid;
  const btn = e.target.closest('[data-act]');
  if (e.target.closest('select')) return;
  if (!btn) { select(id); return; }
  const a = btn.dataset.act;
  if (a === 'EDIT') { S.selected = id; startEdit(); renderPanel(); return; }
  if (a === 'CHANGE_CLASS') {
    act({ action: 'CHANGE_CLASS', object_id: id, label: card.querySelector('[data-class]').value });
    return;
  }
  act({ action: a, object_id: id });
});

// ---------- filmstrip (temporal) ----------

function renderFilmstrip() {
  const f = S.frame;
  const strip = $('filmstrip');
  if (!f || !S.cfg) { strip.innerHTML = ''; return; }
  const offsets = [...new Set([...S.cfg.sweep_offsets, 0])].sort((a, b) => a - b);
  const o = getObj(S.selected);
  $('film-caption').textContent = o ? `#${o.object_id} ${finalLabel(o)} qua t-2 … t+2 (click để xem sweep)` : 'Chọn một object để xem track qua các sweep';
  strip.innerHTML = offsets.map((off) => {
    const name = off === 0 ? 't (keyframe)' : `t${off > 0 ? '+' : ''}${off}`;
    const has = off === 0 || f.sweeps.some((s) => s.offset === off);
    let st = '<span class="muted">—</span>';
    if (o && has) {
      if (off === 0) st = o.source === 'track' ? '<span class="miss">missing</span>' : '<span class="det">detected</span>';
      else st = o.track?.[String(off)] ? '<span class="det">detected</span>' : '<span class="miss">missing</span>';
    } else if (!has) st = '<span class="muted">không có</span>';
    return `<div class="film ${off === 0 ? 'key' : ''} ${S.viewOffset === off ? 'viewing' : ''}" data-off="${off}">
      <canvas width="320" height="180" data-film="${off}"></canvas><div class="ft"><span>${name}</span>${st}</div></div>`;
  }).join('');
  strip.querySelectorAll('canvas[data-film]').forEach((c) => {
    const off = Number(c.dataset.film);
    const img = off === 0 ? S.img : S.sweepImgs[off];
    const cc = c.getContext('2d');
    cc.fillStyle = '#0b1020';
    cc.fillRect(0, 0, c.width, c.height);
    if (!img) return;
    cc.drawImage(img, 0, 0, c.width, c.height);
    if (!o) return;
    const b = off === 0 ? finalBox(o) : o.track?.[String(off)];
    if (!b) return;
    const sx = c.width / img.naturalWidth;
    const sy = c.height / img.naturalHeight;
    cc.strokeStyle = off === 0 && o.source === 'track' ? '#d03b3b' : '#0ca30c';
    cc.lineWidth = 3;
    if (off === 0 && o.source === 'track') cc.setLineDash([6, 4]);
    cc.strokeRect(b[0] * sx, b[1] * sy, (b[2] - b[0]) * sx, (b[3] - b[1]) * sy);
  });
}

$('filmstrip').addEventListener('click', (e) => {
  const film = e.target.closest('[data-off]');
  if (!film) return;
  const off = Number(film.dataset.off);
  if (off !== 0 && !S.sweepImgs[off]) return;
  if (S.mode !== 'view') cancelEdit();
  S.viewOffset = S.viewOffset === off ? 0 : off;
  draw();
  renderFilmstrip();
});

// ---------- chế độ Ảnh / Video ----------

function refreshLists() {
  return S.viewMode === 'video' ? refreshVideo() : loadQueue();
}

async function setMode(mode) {
  stopPlay();
  S.viewMode = mode;
  storageSet('viewMode', mode);
  document.querySelectorAll('.mode').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  // Chế độ 3D có panel riêng (app3d.js); tab Review hiển thị panel của chế độ đang chọn
  const reviewTab = document.querySelector('.tab.active')?.dataset.tab === 'review';
  $('tab-review').classList.toggle('hidden', mode === '3d' || !reviewTab);
  $('tab-review3d').classList.toggle('hidden', mode !== '3d' || !reviewTab);
  $('metrics3d').classList.toggle('hidden', mode !== '3d');
  window.dispatchEvent(new CustomEvent('autolabel:mode', { detail: mode }));
  if (mode === '3d') return;
  const video = mode === 'video';
  $('queue-panel').classList.toggle('hidden', video);
  $('video-panel').classList.toggle('hidden', !video);
  $('video-controls').classList.toggle('hidden', !video);
  $('timeline-panel').classList.toggle('hidden', !video);
  $('filmstrip-head').classList.toggle('hidden', video);
  $('filmstrip').classList.toggle('hidden', video);
  fitCanvas();
  if (video) {
    await loadVideos();
    const want = S.video?.video_id || storageGet('videoId', '');
    const pick = S.videos.find((v) => v.video_id === want && v.n_frames) || S.videos.find((v) => v.n_frames);
    if (pick) await openVideo(pick.video_id);
    else { S.video = null; renderTimeline(); }
  } else {
    await loadQueue();
    if (!S.frame && S.queue.length) await openFrame(S.queue[0].frame_id);
  }
  renderAll();
}

async function loadVideos() {
  S.videos = await api('/videos');
  renderVideoList();
  for (const id of [...S.uploads]) {
    const v = S.videos.find((x) => x.video_id === id);
    if (v?.status === 'processing') continue;
    S.uploads.delete(id);
    if (v?.status === 'ready') {
      toast(`${v.name} đã auto-label xong (${v.n_frames} frame) — bấm vào video bên trái để duyệt`);
      if (S.viewMode === 'video' && !S.video) await openVideo(id);
    } else if (v) toast(`${v.name} lỗi: ${v.message || ''}`, true);
  }
  clearTimeout(S.videoPoll);
  // Video tải lên đang auto-label: hỏi lại tiến độ
  if (S.videos.some((v) => v.status === 'processing')) {
    S.videoPoll = setTimeout(async () => {
      const before = S.video?.video_id;
      await loadVideos();
      if (S.viewMode === 'video' && before && S.videos.find((v) => v.video_id === before)) await refreshVideo();
    }, 1500);
  }
}

function renderVideoList() {
  $('video-list').innerHTML = S.videos.map((v) => {
    const pct = v.n_frames ? Math.round((v.approved / v.n_frames) * 100) : 0;
    const state = v.status === 'processing'
      ? `Đang xử lý ${Math.round(v.progress * 100)}%`
      : v.status === 'error' ? 'Lỗi' : `${v.approved}/${v.n_frames} frame đã duyệt`;
    return `<li class="queue-item video-item ${S.video?.video_id === v.video_id ? 'active' : ''}" data-video="${esc(v.video_id)}">
      <div class="qi-top"><span class="vi-name" title="${esc(v.video_id)}">${esc(v.name)}</span><span class="vi-src ${v.source}">${v.source === 'upload' ? 'mp4' : 'nuScenes'}</span></div>
      <div class="risk-meter"><span style="width:${v.status === 'processing' ? Math.round(v.progress * 100) : Math.max(3, pct)}%;background:${v.status === 'processing' ? 'var(--accent)' : 'var(--low)'}"></span></div>
      <div class="qi-meta"><span>${state}</span><span style="margin-left:auto">${v.duration_s.toFixed(1)} s</span></div>
      ${v.propagated ? `<div class="qi-meta"><span class="prop-tag">↦ ${v.propagated} frame lan truyền</span></div>` : ''}
      ${v.status === 'error' ? `<div class="vi-err">${esc(v.message || '')}</div>` : ''}
    </li>`;
  }).join('') || '<li class="muted panel-note">Chưa có video. Chạy auto-label một scene nuScenes hoặc tải lên mp4.</li>';
}

async function openVideo(id) {
  stopPlay();
  // Mở lại đúng video đang xem thì giữ frame hiện tại; mở video khác (hoặc lần đầu vào chế độ Video) thì nhảy tới
  // frame đầu tiên chưa duyệt — chỗ bắt đầu gán nhãn / lan truyền
  const same = S.video?.video_id === id;
  S.video = await api(`/videos/${encodeURIComponent(id)}`);
  S.scrolledTo = null; // vẽ timeline mới: cuộn tới frame đang mở
  storageSet('videoId', id);
  renderVideoList();
  renderTimeline();
  const frames = S.video.frames;
  const inVideo = same && frames.some((f) => f.frame_id === S.frame?.frame_id);
  if (!inVideo && frames.length) {
    const start = frames.find((f) => f.status !== 'approved') || frames[0];
    await openFrame(start.frame_id);
  }
}

async function refreshVideo() {
  if (!S.video) return;
  S.video = await api(`/videos/${encodeURIComponent(S.video.video_id)}`);
  const i = S.videos.findIndex((v) => v.video_id === S.video.video_id);
  if (i >= 0) S.videos[i] = S.video;
  renderVideoList();
  renderTimeline();
}

function frameRef(id) {
  // Trong chế độ Video, gọi frame theo vị trí trên timeline cho dễ đọc; ngoài ra dùng frame_id
  const i = S.viewMode === 'video' ? (S.video?.frames || []).findIndex((f) => f.frame_id === id) : -1;
  return i >= 0 ? `frame #${i + 1} (${S.video.frames[i].t.toFixed(1)}s)` : id;
}

function renderTimeline() {
  // Đang kéo một thẻ thì không vẽ lại (thẻ bị thay giữa chừng sẽ làm hỏng thao tác kéo); vẽ sau khi thả
  if (S.dragging) { S.timelineStale = true; return; }
  const v = S.video;
  $('timeline-title').textContent = v ? v.name : 'Timeline';
  $('timeline-meta').textContent = v ? `${v.frames.length} frame · ${v.duration_s.toFixed(1)} s · ${v.approved} đã duyệt · ${v.propagated} lan truyền` : '';
  if (!v) { $('timeline').innerHTML = '<p class="muted">Chọn một video bên trái.</p>'; return; }
  $('timeline').innerHTML = v.frames.map((f, i) => {
    const lv = riskLevel(f.frame_risk);
    const marks = [];
    if (f.status === 'approved') marks.push('<span class="tl-mark approved" title="Người đã duyệt (keyframe cho lan truyền)">★</span>');
    else if (f.status === 'editing') marks.push('<span class="tl-mark editing" title="Đang sửa">✎</span>');
    if (f.propagated_from && f.status === 'auto') marks.push(`<span class="tl-mark propagated" title="Nhãn lan truyền từ ${esc(f.propagated_from)}">↦</span>`);
    return `<div class="tl-card ${f.status}" draggable="true" data-id="${esc(f.frame_id)}" data-from="${esc(f.propagated_from || '')}" title="${esc(f.frame_id)} · ${f.pending} object chờ duyệt">
      <div class="tl-badges">${marks.join('')}</div>
      <img loading="lazy" src="${API}/frames/${encodeURIComponent(f.frame_id)}/image" alt="">
      <div class="tl-risk" style="background:${f.status === 'approved' ? 'var(--low)' : RISK_COLOR[lv]}"></div>
      <div class="tl-cap"><span>#${i + 1}</span><span>${f.t.toFixed(1)}s</span><span>${f.pending ? f.pending + ' chờ' : '✓'}</span></div>
    </div>`;
  }).join('');
  highlightTimeline();
}

function highlightTimeline() {
  if (S.viewMode !== 'video') return;
  const id = S.frame?.frame_id;
  let current = null;
  document.querySelectorAll('.tl-card').forEach((c) => {
    const on = c.dataset.id === id;
    c.classList.toggle('current', on);
    // Frame nhận nhãn lan truyền từ frame đang mở được gạch chân
    c.classList.toggle('from-current', !!id && c.dataset.from === id);
    if (on) current = c;
  });
  // Chỉ cuộn khi đổi frame, để timeline không giật về mỗi lần cập nhật tiến độ
  if (current && id !== S.scrolledTo) current.scrollIntoView({ block: 'nearest', inline: 'center' });
  S.scrolledTo = id;
}

function stepVideo(delta) {
  const frames = S.video?.frames || [];
  if (!frames.length) return;
  const i = frames.findIndex((f) => f.frame_id === S.frame?.frame_id);
  const j = Math.min(frames.length - 1, Math.max(0, i + delta));
  if (j !== i) openFrame(frames[j].frame_id);
}

function stopPlay() {
  const wasPlaying = !!S.playTimer;
  if (S.playTimer) clearInterval(S.playTimer);
  S.playTimer = null;
  const b = $('btn-play');
  if (b) b.textContent = '▶ Phát';
  // Frame mở lúc phát là bản xem nhanh (không bấm giờ); dừng ở frame nào thì bắt đầu tính giờ duyệt frame đó (M1)
  if (wasPlaying && S.frame && S.frame.status !== 'approved' && !S.timerStart) S.timerStart = Date.now();
}

function togglePlay() {
  if (S.playTimer) { stopPlay(); return; }
  const frames = S.video?.frames || [];
  if (frames.length < 2) return;
  $('btn-play').textContent = '⏸ Dừng';
  let busy = false;
  S.playTimer = setInterval(async () => {
    if (busy) return;
    const i = frames.findIndex((f) => f.frame_id === S.frame?.frame_id);
    if (i >= frames.length - 1) { stopPlay(); return; }
    busy = true;
    try { await openFrame(frames[i + 1].frame_id, { preview: true }); } finally { busy = false; }
  }, 700);
}

async function uploadVideo(file) {
  const fd = new FormData();
  fd.append('file', file);
  toast(`Đang tải lên ${file.name}…`);
  try {
    const res = await fetch(`${API}/videos/upload`, { method: 'POST', body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail?.message || res.statusText);
    // Giữ nguyên video đang xem; danh sách bên trái hiện tiến độ, xong thì báo (loadVideos)
    toast(`Đã cắt ${file.name}: auto-label đang chạy nền, xong sẽ báo`);
    S.uploads.add(data.video_id);
    await loadVideos();
  } catch (err) {
    toast('Tải lên lỗi: ' + err.message, true);
  }
}

document.querySelectorAll('.mode').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
$('video-list').addEventListener('click', (e) => {
  const li = e.target.closest('[data-video]');
  if (!li) return;
  const v = S.videos.find((x) => x.video_id === li.dataset.video);
  if (!v?.n_frames) { toast(v?.status === 'processing' ? 'Video đang auto-label, đợi chút' : 'Video chưa có frame', !!v && v.status === 'error'); return; }
  openVideo(li.dataset.video).catch((err) => toast(err.message, true));
});
$('btn-upload').addEventListener('click', () => $('upload-input').click());
$('upload-input').addEventListener('change', (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (f) uploadVideo(f);
});
$('btn-prev').addEventListener('click', () => stepVideo(-1));
$('btn-next').addEventListener('click', () => stepVideo(1));
$('btn-play').addEventListener('click', togglePlay);
$('timeline').addEventListener('click', (e) => {
  const card = e.target.closest('.tl-card');
  if (card) openFrame(card.dataset.id).catch((err) => toast(err.message, true));
});
// Kéo một frame từ timeline thả lên khung ảnh để mở ra chỉnh sửa như frame 2D
$('timeline').addEventListener('dragstart', (e) => {
  const card = e.target.closest('.tl-card');
  if (!card) return;
  e.dataTransfer.setData('text/x-frame-id', card.dataset.id);
  e.dataTransfer.effectAllowed = 'copy';
  S.dragging = true;
  $('drop-hint').classList.remove('hidden');
});
function endDrag() {
  if (!S.dragging) return;
  S.dragging = false;
  $('drop-hint').classList.add('hidden');
  if (S.timelineStale) { S.timelineStale = false; renderTimeline(); }
}
// Bắt ở document: thẻ đang kéo có thể đã bị vẽ lại (timeline cập nhật tiến độ) nên dragend không tới #timeline
document.addEventListener('dragend', endDrag);
document.addEventListener('drop', endDrag);
$('canvas-wrap').addEventListener('dragover', (e) => {
  if (e.dataTransfer.types.includes('text/x-frame-id')) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }
});
$('canvas-wrap').addEventListener('drop', (e) => {
  const id = e.dataTransfer.getData('text/x-frame-id');
  $('drop-hint').classList.add('hidden');
  if (id) { e.preventDefault(); openFrame(id).catch((err) => toast(err.message, true)); }
});

// ---------- correction log ----------

async function loadLog() {
  const fid = $('log-frame').value.trim();
  const rows = await api('/corrections' + (fid ? `?frame_id=${encodeURIComponent(fid)}` : ''));
  const tbody = $('log-table').querySelector('tbody');
  tbody.innerHTML = rows.map((r, i) => `<tr data-i="${i}">
    <td>${esc((r.timestamp || '').replace('T', ' ').replace('+00:00', ''))}</td>
    <td>${esc(r.frame_id)}</td><td>#${esc(r.object_id)}</td><td><strong>${esc(r.human_action)}</strong></td>
    <td>${esc(r.prediction.class)}</td><td>${r.human_action === 'DELETE' ? '<span class="done-tag deleted">xoá</span>' : esc(r.final_class)}</td>
    <td class="num">${fx(r.prediction.score)}</td><td class="num">${fx(r.qa.risk)}</td>
    <td>${esc(r.qa.issues.join(', '))}</td></tr>`).join('') || '<tr><td colspan="9" class="muted">Chưa có chỉnh sửa nào</td></tr>';
  tbody.onclick = (e) => {
    const tr = e.target.closest('tr[data-i]');
    if (!tr) return;
    tbody.querySelectorAll('tr').forEach((x) => x.classList.remove('selected'));
    tr.classList.add('selected');
    $('log-json').textContent = JSON.stringify(rows[Number(tr.dataset.i)], null, 2);
  };
}

// ---------- metrics & export ----------

function meterCell(v) {
  return `<span class="meter"><span style="width:${v == null ? 0 : Math.round(v * 100)}%"></span></span>${pct(v)}`;
}

async function loadMetrics() {
  const m = await api('/metrics');
  const tiles = [
    ['Frame đã duyệt', `${m.frames.approved}/${m.frames.total}`, `${m.frames.editing} đang sửa · ${m.frames.rejected ?? 0} bị trả lại`],
    ['M4 · tỉ lệ nhãn phải sửa', pct(m.m4_correction_rate), `${m.model_objects_fixed}/${m.model_objects_reviewed} object máy sinh`],
    ['Flag recall', pct(m.flag_recall), 'object bị sửa đã được agent gắn cờ'],
    ['Flag precision', pct(m.flag_precision), 'object gắn cờ thật sự bị sửa'],
    ['M1 · thời gian TB/frame', m.m1_avg_review_time_s == null ? '—' : fmtTime(m.m1_avg_review_time_s), 'đo từ lúc mở tới lúc approve'],
    ['Box tracking được giữ', `${m.track_proposals.accepted}/${m.track_proposals.proposed}`, `${m.track_proposals.rejected} bị xoá`],
    ['Box người vẽ thêm', String(m.human_added_boxes), 'object model và tracking đều sót'],
    ['Nhãn lan truyền phải sửa', pct(m.propagation.m4_correction_rate), `${m.propagation.fixed}/${m.propagation.reviewed} đã duyệt · ${m.propagation.auto_suppressed} tự xoá`],
  ];
  $('tiles').innerHTML = tiles.map(([l, v, n]) => `<div class="tile"><div class="tile-label">${l}</div><div class="tile-value">${v}</div><div class="tile-note">${n}</div></div>`).join('');

  $('level-table').innerHTML = '<thead><tr><th>Nhóm</th><th class="num">Đã duyệt</th><th class="num">Bị sửa</th><th>Tỉ lệ sửa</th></tr></thead><tbody>' +
    LEVELS.map((l) => {
      const v = m.fix_rate_by_level[l];
      return `<tr><td><span class="dot ${l}"></span> ${LEVEL_NAME[l]}</td><td class="num">${v.reviewed}</td><td class="num">${v.fixed}</td><td>${meterCell(v.rate)}</td></tr>`;
    }).join('') + '</tbody>';

  const issues = Object.entries(m.fix_rate_by_issue);
  $('issue-table').innerHTML = '<thead><tr><th>Issue</th><th class="num">Gắn cờ</th><th class="num">Bị sửa</th><th>Precision</th></tr></thead><tbody>' +
    (issues.map(([c, v]) => `<tr><td><code>${esc(c)}</code></td><td class="num">${v.flagged}</td><td class="num">${v.fixed}</td><td>${meterCell(v.rate)}</td></tr>`).join('') ||
      '<tr><td colspan="4" class="muted">Chưa có object gắn issue nào được duyệt</td></tr>') + '</tbody>';
  renderProductivity(m.productivity, '', API + '/report.csv');
  loadExports();
}

// Năng suất người duyệt + throughput auto-label (FR-27); link CSV (FR-19). Dùng chung với app3d.js qua window.AL
function renderProductivity(prod, p, csvBase) {
  $(p + 'csv-frames').href = csvBase + (csvBase.includes('?') ? '&' : '?') + 'kind=frames';
  $(p + 'csv-summary').href = csvBase + (csvBase.includes('?') ? '&' : '?') + 'kind=summary';
  const n = (v, d = 1) => (v == null ? '—' : Number(v).toFixed(d));
  const rows = prod?.reviewers || [];
  $(p + 'prod-table').innerHTML = '<thead><tr><th>Người duyệt</th><th class="num">Frame duyệt</th><th class="num">Object đã xử lý</th><th class="num">Trả lại</th><th class="num">Hoàn tác</th><th class="num">Thời gian duyệt</th><th class="num">Frame/giờ</th><th>Phiên</th></tr></thead><tbody>' +
    (rows.map((r) => `<tr><td>${esc(r.reviewer)}</td><td class="num">${r.frames_approved}</td><td class="num">${r.objects_handled}</td>
      <td class="num">${r.frames_rejected}</td><td class="num">${r.undo_redo}</td><td class="num">${fmtTime(r.review_time_s)}</td>
      <td class="num"><b>${n(r.frames_per_hour)}</b></td>
      <td class="prod-sessions">${r.sessions.map((s) => `${esc(s.start.slice(0, 16).replace('T', ' '))}: ${s.frames} frame, ${n(s.frames_per_hour)}/giờ`).join('<br>') || '—'}</td></tr>`).join('') ||
      '<tr><td colspan="8" class="muted">Chưa có frame nào được approve</td></tr>') + '</tbody>';
  const inf = prod?.inference || [];
  $(p + 'infer-table').innerHTML = '<thead><tr><th>Phiên auto-label</th><th>Thiết bị</th><th class="num">Frame</th><th class="num">s / frame</th><th class="num">Frame/giờ</th></tr></thead><tbody>' +
    (inf.map((r) => `<tr><td><code>${esc(r.run)}</code></td><td>${esc(r.device || '—')}</td><td class="num">${r.frames}</td><td class="num">${n(r.s_per_frame, 2)}</td><td class="num"><b>${n(r.frames_per_hour, 0)}</b></td></tr>`).join('') ||
      '<tr><td colspan="5" class="muted">Chưa có số đo (frame auto-label trước bản này không ghi thời gian; chạy lại <code>run --overwrite</code> hoặc tạo dự án mới)</td></tr>') + '</tbody>';
}

async function loadExports() {
  const ids = await api('/exports');
  $('export-list').innerHTML = ids.map((id) => `<li><strong>${esc(id)}</strong> — ${['coco.json', 'labels.jsonl', 'corrections.jsonl', 'manifest.json']
    .map((f) => `<a href="${API}/exports/${encodeURIComponent(id)}/${f}">${f}</a>`).join('')}</li>`).join('') || '<li class="muted">Chưa xuất lần nào</li>';
}

async function doExport() {
  try {
    const r = await api('/export', { method: 'POST' });
    $('export-result').innerHTML = `✓ Đã xuất <strong>${r.n_frames}</strong> frame, <strong>${r.n_objects}</strong> object → <code>${esc(r.export_id)}</code>`;
    loadExports();
  } catch (err) {
    $('export-result').innerHTML = `<span class="done-tag deleted">${esc(err.message)}</span>`;
  }
}

// ---------- điều hướng & phím tắt ----------

function fmtTime(s) {
  s = Math.round(s);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
setInterval(() => { $('timer').textContent = fmtTime(elapsed()); }, 500);

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
  ['review', 'log', 'metrics'].forEach((t) => $('tab-' + t).classList.toggle('hidden', t !== tab || (t === 'review' && S.viewMode === '3d')));
  $('tab-review3d').classList.toggle('hidden', !(tab === 'review' && S.viewMode === '3d'));
  window.dispatchEvent(new CustomEvent('autolabel:tab', { detail: tab }));
  if (S.viewMode === '3d' && tab !== 'log') return;
  if (tab === 'log') loadLog().catch((e) => toast(e.message, true));
  if (tab === 'metrics') loadMetrics().catch((e) => toast(e.message, true));
  if (tab === 'review') { fitCanvas(); draw(); }
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));

function stepFrame(delta) {
  if (S.viewMode === 'video') { stepVideo(delta); return; }
  if (!S.queue.length) return;
  const i = S.queue.findIndex((f) => f.frame_id === S.frame?.frame_id);
  const next = S.queue[(i + delta + S.queue.length) % S.queue.length];
  if (next) openFrame(next.frame_id);
}

function stepObject(delta) {
  const objs = orderedObjects().filter((o) => o.review.status !== 'deleted' || o.object_id === S.selected);
  if (!objs.length) return;
  const i = objs.findIndex((o) => o.object_id === S.selected);
  select(objs[(i + delta + objs.length) % objs.length].object_id);
}

document.addEventListener('keydown', (e) => {
  if ($('tab-review').classList.contains('hidden')) return;
  const inField = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);
  if (e.key === 'Escape') { cancelEdit(); document.activeElement?.blur(); return; }
  if (inField) {
    if (e.key === 'Enter' && document.activeElement.id === 'add-class') saveEdit();
    return;
  }
  if (!S.frame) return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (S.playTimer && key !== ' ') stopPlay();
  if ((e.ctrlKey || e.metaKey) && (key === 'z' || key === 'y')) {
    e.preventDefault();
    undoRedo(key === 'y' || e.shiftKey ? 'redo' : 'undo');
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  // Điều hướng: dùng được cả khi frame đã approve
  const nav = {
    ArrowDown: () => stepObject(1),
    ArrowUp: () => stepObject(-1),
    n: () => stepFrame(1),
    p: () => stepFrame(-1),
    l: () => $('show-lidar').click(),
    v: () => $('show-bev2d').click(),
    g: () => $('show-gt').click(),
    t: propagateCurrent,
    r: openReject,
    ' ': () => S.viewMode === 'video' && togglePlay(),
    '+': () => zoomBy(1.5),
    '=': () => zoomBy(1.5),
    '-': () => zoomBy(1 / 1.5),
    0: () => setZoom(1),
  };
  if (nav[key]) { nav[key](); e.preventDefault(); return; }
  if (S.frame.status === 'approved') return;

  const sel = S.selected;
  const edit = {
    k: () => sel && act({ action: 'KEEP', object_id: sel }),
    d: () => sel && act({ action: 'DELETE', object_id: sel }),
    Delete: () => sel && act({ action: 'DELETE', object_id: sel }),
    c: () => sel && document.querySelector(`[data-oid="${CSS.escape(sel)}"] [data-class]`)?.focus(),
    e: startEdit,
    b: startAdd,
    a: approveLow,
    Enter: () => (S.mode !== 'view' && S.editBox ? saveEdit() : approveFrame()),
  };
  if (edit[key]) { edit[key](); e.preventDefault(); }
});

// Đổi lớp bằng bàn phím: C -> chọn trong dropdown -> Enter
document.querySelector('.review-panel').addEventListener('keydown', (e) => {
  if (e.key !== 'Enter' || !e.target.matches('[data-class]')) return;
  const card = e.target.closest('[data-oid]');
  act({ action: 'CHANGE_CLASS', object_id: card.dataset.oid, label: e.target.value });
});

$('queue-list').addEventListener('click', (e) => {
  const li = e.target.closest('[data-id]');
  if (li) openFrame(li.dataset.id).catch((err) => toast(err.message, true));
});
$('queue-sort').addEventListener('change', (e) => { S.sort = e.target.value; loadQueue(); });
$('queue-filter').addEventListener('click', (e) => {
  const b = e.target.closest('[data-status]');
  if (!b) return;
  S.statusFilter = b.dataset.status;
  document.querySelectorAll('#queue-filter .chip').forEach((c) => c.classList.toggle('active', c === b));
  loadQueue();
});
$('show-lidar').addEventListener('change', async (e) => { S.showLidar = e.target.checked; if (S.showLidar) await ensureLidar(); draw(); });
$('show-gt').addEventListener('change', async (e) => { S.showGt = e.target.checked; if (S.showGt) await ensureGt(); draw(); });
$('show-low').addEventListener('change', (e) => { S.showLow = e.target.checked; draw(); });
$('show-mask').addEventListener('change', (e) => { S.showMask = e.target.checked; draw(); });
$('min-score').addEventListener('input', (e) => {
  S.minScore = Number(e.target.value);
  $('min-score-val').textContent = S.minScore.toFixed(2);
  draw();
});
$('toggle-low').addEventListener('click', () => { S.lowOpen = !S.lowOpen; renderPanel(); });
$('btn-approve-low').addEventListener('click', approveLow);
$('btn-undo').addEventListener('click', () => undoRedo('undo'));
$('btn-redo').addEventListener('click', () => undoRedo('redo'));
$('btn-reject').addEventListener('click', openReject);
$('reject-send').addEventListener('click', sendReject);
$('reject-cancel').addEventListener('click', () => $('reject-box').classList.add('hidden'));
$('reject-reason').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendReject(); });
$('btn-propagate').addEventListener('click', propagateCurrent);
$('auto-prop').addEventListener('change', (e) => { S.autoProp = e.target.checked; storageSet('autoProp', S.autoProp ? '1' : '0'); });
$('btn-add').addEventListener('click', () => (S.mode === 'add' ? cancelEdit() : startAdd()));
$('edit-save').addEventListener('click', saveEdit);
$('edit-cancel').addEventListener('click', cancelEdit);
$('log-refresh').addEventListener('click', loadLog);
$('log-frame').addEventListener('change', loadLog);
$('metrics-refresh').addEventListener('click', loadMetrics);
$('btn-export').addEventListener('click', doExport);
$('reviewer').addEventListener('change', (e) => storageSet('reviewer', e.target.value.trim()));
new ResizeObserver(() => { fitCanvas(); draw(); }).observe($('canvas-wrap'));
// cho bev2d.js (module) dùng chung trạng thái
window.AL = { get S() { return S; }, select, ensureLidar, renderProductivity };

// Dự án: hiện tên + link quay lại, ẩn chế độ không có dữ liệu (không LiDAR -> không 3D)
async function initProject() {
  const r = await fetch(`/api/v1/projects/${encodeURIComponent(PROJECT)}`);
  if (!r.ok) throw new Error(`Không tìm thấy dự án ${PROJECT}`);
  const p = await r.json();
  document.title = `${p.name} — AutoLabel 3D`;
  document.body.classList.add('in-project');
  $('project-link').classList.remove('hidden');
  $('project-name').textContent = p.name;
  const modes = ['image', 'video'];
  const done3d = p.steps.some((s) => s.name === 'predict3d' && s.status === 'done');
  if (p.stats?.has_lidar && done3d) modes.push('3d');
  document.querySelectorAll('.mode').forEach((b) => b.classList.toggle('hidden', !modes.includes(b.dataset.mode)));
  return modes;
}

(async function init() {
  try {
    S.cfg = await api('/config');
    $('reviewer').value = storageGet('reviewer', S.cfg.reviewer);
    S.autoProp = storageGet('autoProp', '1') === '1';
    $('auto-prop').checked = S.autoProp;
    $('add-class').innerHTML = classOptions('car');
    let modes = ['image', 'video', '3d'];
    if (PROJECT) modes = await initProject();
    const urlMode = new URLSearchParams(location.search).get('mode');
    const saved = modes.includes(urlMode) ? urlMode : storageGet('viewMode', modes[0]);
    await setMode(modes.includes(saved) ? saved : modes[0]);
  } catch (err) {
    toast('Không tải được dữ liệu: ' + err.message, true);
  }
})();

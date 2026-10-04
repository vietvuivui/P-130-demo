/* AutoLabel 2D — UI review by exception (không cần build, gọi thẳng FastAPI) */
'use strict';

// ?project=<id>: làm việc trong một dự án (data/projects/<id>), API cùng đường dẫn nhưng có tiền tố /p/<id>
const PROJECT = new URLSearchParams(location.search).get('project');
const API = PROJECT ? `/p/${encodeURIComponent(PROJECT)}/api/v1` : '/api/v1';
const LEVELS = ['high', 'medium', 'low'];
const LEVEL_NAME = { low: 'Low', medium: 'Medium', high: 'High' };
const RISK_COLOR = { low: '#10b981', medium: '#f59e0b', high: '#ef4444' };
const HUMAN_COLOR = '#2563eb';
const QC_COLOR = '#c026d3';
const STATUS_TEXT = { auto: 'Chưa mở', editing: 'Đang sửa', approved: 'Đã duyệt', rejected: 'Bị trả lại' };

const S = {
  cfg: null,
  queue: [],
  sort: 'order', // mặc định theo thứ tự frame; 'risk' = khó nhất trước
  objSort: 'id-asc',
  statusFilter: '',
  frame: null,
  img: null,
  sweepImgs: {},
  lidar: null,
  gt: null,
  selected: null,
  viewOffset: 0,
  sweepSel: null, // box đang chọn khi xem một sweep (sửa tự do ở t-2 … t+2)
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
  playback: null, // đang phát video: {items, i, imgs, ...}
  me: null, // người đang đăng nhập ({id, name, email, admin}) hoặc null
  presence: { locks: {}, assignments: {}, users: {} }, // ai đang mở frame nào, frame giao cho ai (GET /presence)
  lockedBy: null, // frame đang mở bị người khác giữ -> chỉ xem
  mineOnly: false, // hàng đợi: chỉ frame giao cho tôi
  videoPoll: null,
  uploads: new Set(), // video vừa tải lên, đang chờ auto-label xong để báo
  zoom: 1, // 1 = vừa khung; phóng to tới 8x
  dragging: false, // đang kéo thẻ frame từ timeline
  timelineStale: false,
  scrolledTo: null,
  qc: null, // QC nhãn cuối của frame đang mở: { findings, open }
  qcReport: null,
  audit: null, // { items, summary }
  auditCur: null,
  auditImgs: {},
  quick: null, // kết quả Quick Check gần nhất
  hideTags: false,
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
    const err = new Error(d?.message || (typeof d === 'string' ? d : JSON.stringify(d)) || res.statusText);
    err.code = d?.code;
    throw err;
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

/* Thứ tự duyệt: theo sort hoặc object chờ (high -> medium -> low, risk giảm dần) rồi tới object đã xử lý */
function orderedObjects() {
  if (!S.frame) return [];
  let objs = [...S.frame.objects];
  if (S.objSort === 'id-asc') {
    objs.sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0) || String(a.object_id).localeCompare(String(b.object_id), undefined, { numeric: true }));
    return objs;
  } else if (S.objSort === 'id-desc') {
    objs.sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0) || String(b.object_id).localeCompare(String(a.object_id), undefined, { numeric: true }));
    return objs;
  }
  const pending = objs.filter((o) => o.review.status === 'pending');
  const rank = { high: 0, medium: 1, low: 2 };
  pending.sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0) || rank[levelOf(a)] - rank[levelOf(b)] || (b.qa?.risk || 0) - (a.qa?.risk || 0));
  return pending.concat(objs.filter((o) => o.review.status !== 'pending'));
}
const getObj = (id) => S.frame?.objects.find((o) => o.object_id === id);

// ---------- sweep t-2 … t+2: sửa tự do để QA / score của keyframe đúng hơn (không xuất ra) ----------
const inSweep = () => !!S.frame && S.viewOffset !== 0;
const curSweep = () => S.frame?.sweeps.find((s) => s.offset === S.viewOffset);
function sweepBoxes(s) {
  if (!s) return [];
  return s.boxes || s.detections.map((d, i) => ({ box_id: String(i + 1), bbox: d.bbox, label: d.label, score: d.score, source: 'model', review: { status: 'pending' } }));
}
const getSweepBox = (id) => sweepBoxes(curSweep()).find((b) => b.box_id === id);
const offName = (off) => `t${off > 0 ? '+' : '−'}${Math.abs(off)}`;
/* Object keyframe nào đang được ghép với box sweep này (qua check temporal) */
function linkedObject(b) {
  const box = finalBox(b);
  return S.frame.objects.find((o) => {
    const t = o.track?.[String(S.viewOffset)];
    return t && Math.abs(t[0] - box[0]) + Math.abs(t[1] - box[1]) + Math.abs(t[2] - box[2]) + Math.abs(t[3] - box[3]) < 2;
  });
}
function hitSweep(x, y) {
  let best = null;
  let bestArea = Infinity;
  for (const b of sweepBoxes(curSweep())) {
    if (b.review.status === 'deleted' && b.box_id !== S.sweepSel) continue;
    const [x1, y1, x2, y2] = finalBox(b);
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && (x2 - x1) * (y2 - y1) < bestArea) { best = b; bestArea = (x2 - x1) * (y2 - y1); }
  }
  return best;
}
function setView(off) {
  if (S.mode !== 'view') cancelEdit();
  S.viewOffset = off;
  S.sweepSel = null;
  draw();
  renderPanel();
  renderFilmstrip();
}
async function sweepAct(body) {
  try {
    const frame = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/sweeps/${S.viewOffset}/actions`, {
      method: 'POST',
      body: { ...body, reviewer: reviewer() },
    });
    S.frame = frame;
    const boxes = sweepBoxes(curSweep());
    if (body.action === 'ADD_BOX') S.sweepSel = boxes[boxes.length - 1]?.box_id || null;
    else if (['KEEP', 'DELETE', 'CHANGE_CLASS'].includes(body.action)) {
      const i = boxes.findIndex((b) => b.box_id === body.box_id);
      const next = boxes.slice(i + 1).concat(boxes.slice(0, i)).find((b) => b.review.status === 'pending');
      S.sweepSel = next ? next.box_id : body.box_id;
    }
    renderAll();
    refreshLists();
  } catch (err) {
    toast(err.message, true);
  }
}
function sweepPanel() {
  const sw = curSweep();
  const boxes = sweepBoxes(sw);
  const locked = S.frame.status === 'approved';
  const edited = boxes.filter((b) => b.review.status !== 'pending').length;
  const rows = boxes.map((b) => {
    const sel = b.box_id === S.sweepSel;
    const o = linkedObject(b);
    const st = b.review.status === 'deleted' ? '<span class="done-tag deleted">✗ xoá</span>'
      : b.review.status === 'approved' ? `<span class="done-tag approved">✓ ${esc(b.review.action.toLowerCase().replace('_', ' '))}</span>`
        : `<span class="muted">score ${fx(b.score)}</span>`;
    const actions = !sel || locked ? '' : b.review.status === 'deleted'
      ? '<div class="oc-actions"><button class="btn btn-sm" data-sact="RESTORE">↺ Khôi phục</button></div>'
      : `<div class="oc-actions">
          <button class="btn btn-sm btn-keep" data-sact="KEEP">✓ Keep</button>
          <button class="btn btn-sm btn-del" data-sact="DELETE">🗑 Delete</button>
          <select data-sclass>${classOptions(finalLabel(b))}</select>
          <button class="btn btn-sm btn-class" data-sact="CHANGE_CLASS">Đổi lớp</button>
          <button class="btn btn-sm btn-edit" data-sact="EDIT">✎ Sửa box</button>
        </div>`;
    return `<div class="sw-row ${sel ? 'selected' : ''} ${b.review.status}" data-sbid="${esc(b.box_id)}">
      <div class="sw-line"><span>#${esc(b.box_id)} ${esc(finalLabel(b))}${b.source === 'human' ? ' <span class="muted">· người vẽ</span>' : ''}${o ? ` <span class="muted">↔ #${esc(o.object_id)} ở keyframe</span>` : ''}</span>${st}</div>${actions}</div>`;
  }).join('');
  return `<h3>Sweep ${offName(S.viewOffset)} <small>${boxes.length} box · đã sửa ${edited}</small>
      <button class="btn btn-ghost btn-sm" data-sweep-back title="Quay về keyframe (Esc)">← Keyframe</button></h3>
    <p class="muted sw-note">Box ở sweep không được xuất; sửa ở đây để cờ FLICKER / RECOVERED, score và lan truyền của keyframe đúng hơn. Keyframe được tính lại ngay sau mỗi thao tác.</p>
    ${locked ? '' : '<button class="btn btn-sm wide" data-sweep-add>+ Vẽ box ở sweep này <kbd>B</kbd></button>'}
    <div class="sw-list">${rows || '<p class="muted">Detector không thấy gì ở sweep này — vẽ thêm nếu có vật bị sót.</p>'}</div>`;
}

// ---------- hàng đợi ----------

async function loadQueue() {
  const q = new URLSearchParams({ sort: S.sort });
  if (S.statusFilter) q.set('status', S.statusFilter);
  S.queue = await api('/frames?' + q);
  if (S.mineOnly && S.me) S.queue = S.queue.filter((f) => S.presence.assignments[f.frame_id] === S.me.id);
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
    const lock = S.presence.locks[f.frame_id];
    const who = S.presence.assignments[f.frame_id];
    const whoName = who ? (S.presence.users[who] || who) : '';
    const tag = lock && lock.user_id !== S.me?.id
      ? `<span class="qi-who" title="${esc(lock.name)} đang mở"><span class="av lock">🔒</span>${esc(lock.name)}</span>`
      : who ? `<span class="qi-who" title="Giao cho ${esc(whoName)}"><span class="av ${who === S.me?.id ? 'me' : ''}">${esc(initials(whoName))}</span>${who === S.me?.id ? 'tôi' : esc(whoName)}</span>` : '';
    return `<li class="queue-item ${S.frame?.frame_id === f.frame_id ? 'active' : ''}" data-id="${esc(f.frame_id)}">
      <div class="qi-top"><span class="qi-id">${esc(f.frame_id)}</span>${tag}${f.propagated_from && f.status === 'auto' ? `<span class="prop-tag" title="Nhãn lan truyền từ ${esc(f.propagated_from)}">↦</span>` : ''}<span class="status-pill ${f.status}">${statusText}</span></div>
      <div class="risk-meter" title="Frame risk ${fx(f.frame_risk)} (${LEVEL_NAME[lv]})"><span style="width:${Math.max(4, f.frame_risk * 100)}%;background:${RISK_COLOR[lv]}"></span></div>
      <div class="qi-meta">
        <span class="lv" title="High"><span class="dot high"></span>${f.counts.high}</span>
        <span class="lv" title="Medium"><span class="dot medium"></span>${f.counts.medium}</span>
        <span class="lv" title="Low"><span class="dot low"></span>${f.counts.low}</span>
        <span style="margin-left:auto">${f.pending ? f.pending + ' chờ' : '✓'}</span>
      </div></li>`;
  }).join('');
  // đổi frame (N/P, bấm filmstrip…) thì cuộn danh sách tới frame đang mở; không giật khi chỉ làm mới
  const cur = S.frame?.frame_id || null;
  if (cur !== S.queueShown) {
    S.queueShown = cur;
    list.querySelector('.queue-item.active')?.scrollIntoView({ block: 'nearest' });
  }
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
  if (!preview) stopPlay({ reopen: false });
  stopTimer();
  const frame = await api(`/frames/${encodeURIComponent(id)}`);
  S.frame = frame;
  if (!preview && S.lockedBy) { S.lockedBy = null; renderLockBanner(); }
  if (!preview) acquireLock(id);
  S.img = null;
  S.lidar = S.gt = null;
  S.sweepImgs = {};
  S.viewOffset = 0;
  S.sweepSel = null;
  S.mode = 'view';
  S.editBox = null;
  S.selected = orderedObjects()[0]?.object_id || null;
  S.qc = null;
  if (frame.status !== 'approved' && !preview) S.timerStart = Date.now();
  $('empty-state').classList.add('hidden');
  renderAll();
  highlightTimeline();
  if (!preview) loadFrameQC();
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
  // Box đã sửa tay thì mask cũ không còn khớp; box thêm bằng bấm-để-chọn-vật (ADD_BOX có mask) vẫn hiện mask
  if (!m || m.length < 6 || o.review.action === 'EDIT_BOX') return;
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

function hexToRgba(hex, alpha = 1) {
  if (!hex || !hex.startsWith('#')) return hex;
  const h = hex.replace('#', '');
  if (h.length === 3) {
    const r = parseInt(h[0] + h[0], 16), g = parseInt(h[1] + h[1], 16), b = parseInt(h[2] + h[2], 16);
    return `rgba(${r},${g},${b},${alpha})`;
  } else if (h.length === 6) {
    const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  return hex;
}

function drawBox(b, color, { lw = 2, dash = null, label = null, alpha = 1, textColor = '#fff', fill = null } = {}) {
  const k = px();
  ctx.save();
  ctx.globalAlpha = alpha;
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fillRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
  }
  if (S.showBorder !== false) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw * k;
    if (dash) ctx.setLineDash(dash.map((v) => v * k));
    ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    ctx.setLineDash([]);
  }
  if (label) {
    ctx.font = `${600} ${12 * k}px 'Inter', sans-serif`;
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
}

function drawCanvas() {
  const f = S.frame;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (S.playback?.drawn) { drawPlayback(); return; }
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
    // Xem một sweep: box của sweep (sửa được); xám = máy sinh chưa xem, xanh dương = người đã xác nhận / vẽ
    for (const b of sweepBoxes(f.sweeps.find((s) => s.offset === S.viewOffset))) {
      const sel = b.box_id === S.sweepSel;
      if (sel && S.mode === 'edit') continue;
      if (b.review.status === 'deleted') {
        if (sel) drawBox(finalBox(b), '#d03b3b', { lw: 1.5, dash: [4, 4], label: `#${b.box_id} đã xoá` });
        continue;
      }
      const done = b.review.status === 'approved' || b.source === 'human';
      drawBox(finalBox(b), done ? HUMAN_COLOR : '#94a3b8', {
        lw: sel ? 3.5 : 1.6,
        label: S.hideTags ? null : `#${b.box_id} ${finalLabel(b)}${done ? '' : ' ' + fx(b.score)}`,
        alpha: sel || !S.sweepSel ? 1 : 0.8,
      });
    }
    drawClickPreview();
    if (S.editBox) drawBox(S.editBox, '#7c5cd6', { lw: 3, dash: S.mode === 'add' ? [6, 4] : null });
    if (S.editBox && S.mode === 'edit') {
      ctx.fillStyle = '#fff';
      ctx.strokeStyle = '#7c5cd6';
      ctx.lineWidth = 2 * k;
      for (const [hx, hy] of handles(S.editBox)) {
        ctx.fillRect(hx - 5 * k, hy - 5 * k, 10 * k, 10 * k);
        ctx.strokeRect(hx - 5 * k, hy - 5 * k, 10 * k, 10 * k);
      }
    }
    return;
  }

  const qcFlagged = new Set(qcOpen().flatMap((x) => [x.object_id, x.other_object_id]).filter(Boolean));
  let shown = 0, total = 0;
  for (const o of f.objects) {
    if (o.review.status === 'deleted') continue;
    if (S.hideAllBoxes) continue;
    if (o.hidden) continue;
    total++;
    const lv = levelOf(o);
    const pending = o.review.status === 'pending';
    if (pending && lv === 'low' && !S.showLow && o.object_id !== S.selected && !qcFlagged.has(o.object_id)) continue;
    // Box đang có lỗi QC luôn hiện (phải sửa hoặc xác nhận trước khi approve), kể cả khi dưới ngưỡng score
    if (belowScore(o) && o.object_id !== S.selected && !qcFlagged.has(o.object_id)) continue;
    shown++;
    const sel = o.object_id === S.selected;
    if (sel && S.mode === 'edit') continue;
    if (qcFlagged.has(o.object_id)) {
      // Viền ngoài nét đứt: nhãn cuối còn lỗi QC chưa xử lý
      const [x1, y1, x2, y2] = finalBox(o);
      const m = 5 * k;
      drawBox([x1 - m, y1 - m, x2 + m, y2 + m], QC_COLOR, { lw: 1.6, dash: [5, 3] });
    }
    let color = o.source === 'human' ? HUMAN_COLOR : RISK_COLOR[lv];
    if (S.colorMode === 'label' && S.cfg?.classes?.[finalLabel(o)]) {
      color = S.cfg.classes[finalLabel(o)];
    } else if (S.colorMode === 'object') {
      const hash = String(o.object_id).split('').reduce((acc, c) => acc * 31 + c.charCodeAt(0), 0);
      color = `hsl(${Math.abs(hash) % 360}, 80%, 55%)`;
    } else if (S.colorMode === 'group') {
      color = RISK_COLOR[lv];
    }
    if (S.showMask) drawMask(o, color, sel);
    // Nhãn đầy đủ chỉ cho box đang chọn / high / người vẽ; medium chỉ hiện #id để ảnh không bị che kín
    const full = `${pending ? '' : '✓ '}#${o.object_id} ${finalLabel(o)}${o.source === 'human' ? '' : ' ' + scoreText(o)}`;
    let tag = null;
    if (!S.hideTags) {
      if (sel || o.source === 'human' || (pending && lv === 'high')) tag = full;
      else if (pending && lv === 'medium') tag = `#${o.object_id}`;
    }
    const fillAlpha = sel ? (S.selectedOpacity ?? 0.25) : (S.boxFillOpacity ?? 0.05);
    let fill = null;
    if (fillAlpha > 0) {
      if (color.startsWith('#')) fill = hexToRgba(color, fillAlpha);
      else if (color.startsWith('hsl(')) fill = color.replace('hsl(', 'hsla(').replace(')', `, ${fillAlpha})`);
      else fill = color;
    }
    drawBox(finalBox(o), color, {
      lw: sel ? 3.5 : pending ? 2 : 1.4,
      dash: o.source === 'track' ? [8, 5] : isProp(o) && !o.propagation.matched ? [3, 3] : null,
      label: tag,
      alpha: sel || S.selected == null ? 1 : 0.85,
      fill: fill,
    });
  }

  $('min-score-count').textContent = S.minScore > 0 ? ` · ${shown}/${total} box` : '';

  drawClickPreview();
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
  if (!S.frame) return;
  const [x, y] = toImg(e);
  if (S.mode === 'click') { // bấm = điểm; kéo = box thô quanh vật (xử lý ở mouseup)
    S.drag = { kind: 'prompt', x0: x, y0: y, x1: x, y1: y, negative: e.shiftKey || e.button === 2 };
    e.preventDefault();
    return;
  }
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
  const o = inSweep() ? null : hitObject(x, y);
  const sb = inSweep() ? hitSweep(x, y) : null;
  if (sb) { S.sweepSel = sb.box_id; renderPanel(); draw(); }
  else if (o) select(o.object_id);
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
  } else if (d.kind === 'prompt') {
    d.x1 = x;
    d.y1 = y;
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
  if (d.kind === 'prompt') {
    const k = px();
    const dragged = Math.abs(d.x1 - d.x0) > 8 * k && Math.abs(d.y1 - d.y0) > 8 * k;
    if (dragged && !d.negative) clickSegment(null, null, false, [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)]);
    else clickSegment(d.x0, d.y0, d.negative);
    return;
  }
  if (d.kind === 'draw') {
    const b = S.editBox;
    if (b[2] - b[0] < 4 || b[3] - b[1] < 4) { S.editBox = null; draw(); return; }
    showEditBar('add');
  }
});

function startEdit() {
  if (inSweep()) {
    const b = getSweepBox(S.sweepSel);
    if (!b || b.review.status === 'deleted' || S.frame.status === 'approved') return;
    S.mode = 'edit';
    S.editBox = finalBox(b).slice();
    showEditBar('edit');
    draw();
    return;
  }
  const o = getObj(S.selected);
  if (!o || S.frame.status === 'approved') return;
  S.mode = 'edit';
  S.editBox = finalBox(o).slice();
  showEditBar('edit');
  draw();
}

function startAdd() {
  if (!S.frame || S.frame.status === 'approved') return;
  S.mode = 'add';
  S.editBox = null;
  canvas.classList.add('drawing');
  $('btn-add').classList.add('active');
  showEditBar('draw');
  draw();
}

// ---------- bấm để chọn vật (SAM 2.1; máy chủ không có GPU thì GrabCut) ----------
// Bấm lên vật -> POST /frames/{id}/segment -> box + đa giác mask xem trước; bấm thêm để tinh chỉnh (Shift+bấm hoặc
// chuột phải = vùng không thuộc vật); chọn lớp rồi Enter / Lưu để thêm như một box người vẽ (kèm mask).
function startClick() {
  if (!S.frame || S.frame.status === 'approved') return;
  if (S.mode !== 'view') cancelEdit();
  S.mode = 'click';
  S.click = { points: [], labels: [], box: null, polygon: null, busy: false, engine: null };
  // Mã hoá ảnh trước ở nền (SAM ONNX) để lần bấm đầu không phải chờ
  api(`/frames/${encodeURIComponent(S.frame.frame_id)}/segment/preload?offset=${S.viewOffset || 0}`, { method: 'POST' }).catch(() => {});
  S.editBox = null;
  canvas.classList.add('drawing');
  $('btn-click').classList.add('active');
  $('tool-poly')?.classList.add('active');
  showEditBar('click');
  draw();
}

async function clickSegment(x, y, negative, box = null) {
  const c = S.click;
  if (!c || c.busy) return;
  if (negative && !c.points.length && !c.box) { toast('Bấm vào vật (hoặc kéo box quanh vật) trước; Shift+bấm để loại vùng thừa sau đó'); return; }
  const undo = { points: c.points.slice(), labels: c.labels.slice(), box: c.box };
  if (box) { // box thô mới thay cho prompt cũ
    c.box = box.map((v) => Math.round(v * 10) / 10);
    c.points = [];
    c.labels = [];
  } else {
    c.points.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
    c.labels.push(negative ? 0 : 1);
  }
  c.busy = true;
  canvas.classList.add('busy');
  draw();
  try {
    const r = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/segment`, {
      method: 'POST', body: { points: c.points, labels: c.labels, box: c.box || undefined, offset: S.viewOffset || 0 },
    });
    if (S.click !== c) return; // đã huỷ trong lúc chờ
    c.polygon = r.polygon;
    c.engine = r.engine;
    S.editBox = r.bbox;
    showEditBar('click');
  } catch (err) {
    Object.assign(c, undo);
    toast(err.message, true);
  } finally {
    c.busy = false;
    canvas.classList.remove('busy');
    draw();
  }
}

function drawClickPreview() {
  const c = S.mode === 'click' ? S.click : null;
  if (!c) return;
  const k = px();
  if (c.polygon && c.polygon.length >= 6) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(c.polygon[0], c.polygon[1]);
    for (let i = 2; i < c.polygon.length; i += 2) ctx.lineTo(c.polygon[i], c.polygon[i + 1]);
    ctx.closePath();
    ctx.fillStyle = 'rgba(124, 92, 214, .35)';
    ctx.fill();
    ctx.strokeStyle = '#7c5cd6';
    ctx.lineWidth = 1.5 * k;
    ctx.stroke();
    ctx.restore();
  }
  const pr = S.drag?.kind === 'prompt' ? [S.drag.x0, S.drag.y0, S.drag.x1, S.drag.y1] : c.box;
  if (pr) { // box thô đang kéo / đã dùng làm prompt
    ctx.save();
    ctx.setLineDash([6 * k, 4 * k]);
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 1.5 * k;
    ctx.strokeRect(Math.min(pr[0], pr[2]), Math.min(pr[1], pr[3]), Math.abs(pr[2] - pr[0]), Math.abs(pr[3] - pr[1]));
    ctx.restore();
  }
  c.points.forEach(([x, y], i) => {
    ctx.beginPath();
    ctx.arc(x, y, 5 * k, 0, Math.PI * 2);
    ctx.fillStyle = c.labels[i] ? '#22c55e' : '#ef4444';
    ctx.fill();
    ctx.lineWidth = 2 * k;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  });
}

function cancelEdit() {
  S.mode = 'view';
  S.editBox = null;
  S.drag = null;
  S.click = null;
  $('btn-click')?.classList.remove('active');
  $('tool-poly')?.classList.remove('active');
  canvas.classList.remove('drawing');
  $('btn-add').classList.remove('active');
  $('edit-bar').classList.add('hidden');
  draw();
}

function showEditBar(kind) {
  $('edit-bar').classList.remove('hidden');
  const picked = kind === 'click' && !!S.editBox;
  $('add-class').classList.toggle('hidden', kind !== 'add' && !picked);
  $('edit-save').classList.toggle('hidden', kind === 'draw' || (kind === 'click' && !picked));
  $('edit-hint').textContent = {
    click: picked
      ? `${S.click?.engine === 'grabcut' ? 'GrabCut (chưa có SAM) · ' : ''}Bấm thêm: tinh chỉnh · Shift+bấm: loại vùng · Enter: lưu`
      : 'Bấm vào vật, hoặc kéo một box quanh vật',
    draw: 'Kéo chuột trên ảnh để vẽ box mới',
    add: 'Chọn lớp cho box mới',
    edit: 'Kéo góc/cạnh hoặc kéo cả box để sửa',
  }[kind];
  if (kind === 'add') $('add-class').focus();
}

async function saveEdit() {
  const box = S.editBox.map((v) => Math.round(v * 10) / 10);
  if (inSweep()) {
    if (S.mode === 'add' || S.mode === 'click') await sweepAct({ action: 'ADD_BOX', bbox: box, label: $('add-class').value });
    else if (S.mode === 'edit') await sweepAct({ action: 'EDIT_BOX', box_id: S.sweepSel, bbox: box });
    cancelEdit();
    return;
  }
  if (S.mode === 'click') {
    const poly = S.click?.polygon;
    await act({ action: 'ADD_BOX', bbox: box, label: $('add-class').value, mask: poly && poly.length >= 6 ? poly.slice(0, 400) : undefined });
    const again = S.frame && S.frame.status !== 'approved';
    cancelEdit();
    if (again) startClick(); // chọn tiếp vật khác, Esc để thoát
    return;
  }
  if (S.mode === 'add') {
    await act({ action: 'ADD_BOX', bbox: box, label: $('add-class').value });
  } else if (S.mode === 'edit') {
    await act({ action: 'EDIT_BOX', object_id: S.selected, bbox: box });
  }
  cancelEdit();
}

// ---------- hành động review ----------

async function act(body) {
  if (body.object_id && S.frame) {
    const target = S.frame.objects.find(o => o.object_id === body.object_id);
    if (target?.locked && body.action !== 'UNLOCK') {
      toast(`Đối tượng #${body.object_id} đang bị khóa, hãy mở khóa để sửa`, true);
      return;
    }
  }
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
    loadFrameQC(); // nhãn cuối vừa đổi: kiểm lại ngay
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
    loadFrameQC();
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
      if (S.autoProp && (await slowProp())) {
        // Luồng chậm (DAM4SAM) chạy nền: không chờ và không tự mở frame kế (frame người đã mở sẽ không nhận nhãn lan
        // truyền); xong thì tải lại timeline
        propagate(cur.frame_id).then(() => refreshVideo());
        await refreshVideo();
        renderAll();
        return;
      }
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
    if (err.code === 'QC_FINDINGS') {
      // Server chặn vì nhãn cuối còn lỗi QC: hiện danh sách lên đầu panel
      await loadFrameQC();
      $('review-scroll').scrollTop = 0;
    }
    toast(err.message, true);
  }
}

// ---------- luồng lan truyền (người dùng chọn như chọn model): Nhanh = theo Cài đặt; Chính xác = DAM4SAM + BoT-SORT ----------
function updateTopModelSwitch(activeId) {
  const topSwitch = $('topbar-model-switch');
  if (!topSwitch) return;
  topSwitch.querySelectorAll('.model-btn').forEach((btn) => {
    const isAct = btn.dataset.engine === activeId;
    btn.classList.toggle('active', isAct);
    const eng = (S.engines || []).find((e) => e.id === btn.dataset.engine);
    if (eng) {
      btn.disabled = !eng.available;
      if (!eng.available && eng.reason) {
        btn.title = `${eng.label} chưa dùng được: ${eng.reason}`;
      }
    }
  });
}

function setEngine(id) {
  const eng = (S.engines || []).find((e) => e.id === id);
  if (eng && !eng.available) {
    toast(`Mô hình ${eng.label} chưa dùng được: ${eng.reason || 'thiếu cấu hình / GPU'}`, true);
    return;
  }
  const sel = $('prop-engine');
  if (sel) {
    sel.value = id;
    sel.title = engineInfo().detail || '';
  }
  storageSet('propEngine', id);
  updateTopModelSwitch(id);
  const label = id === 'dam4sam' ? '🎯 Chính xác (DAM4SAM)' : '⚡ Nhanh';
  toast(`Đã chuyển sang mô hình: ${label}`);
}

async function loadEngines() {
  const sel = $('prop-engine');
  try {
    S.engines = await api('/propagation/engines');
  } catch { S.engines = [{ id: 'default', label: 'Nhanh', available: true, detail: '' }]; }
  const saved = storageGet('propEngine', 'default');
  const validSaved = S.engines.some((e) => e.id === saved && e.available) ? saved : 'default';
  if (sel) {
    sel.innerHTML = S.engines.map((e) => `<option value="${esc(e.id)}" ${e.available ? '' : 'disabled'} title="${esc(e.detail)}${e.reason ? '\nChưa dùng được: ' + esc(e.reason) : ''}">${e.id === 'default' ? '⚡' : '🎯'} ${esc(e.label)}</option>`).join('');
    sel.value = validSaved;
    sel.title = engineInfo().detail || '';
  }
  updateTopModelSwitch(validSaved);
  loadMainModels();
  resumePropJob();
}
// Luồng lan truyền chỉ còn chọn ở tab Cài đặt (Dự đoán chuyển động + Ghép với detection): mọi lần bấm dùng cấu hình đó
const engineInfo = () => (S.engines || []).find((e) => e.id === 'default') || { id: 'default' };
// Cài đặt đang chọn DAM4SAM (chậm, cần GPU)? Hỏi lại máy chủ mỗi lần vì Cài đặt có thể vừa đổi
async function slowProp() {
  try { S.engines = await api('/propagation/engines'); } catch { /* giữ danh sách cũ */ }
  return !!engineInfo().slow;
}

function showPropJob(job) {
  const el = $('prop-job');
  const running = job?.state === 'running';
  el.classList.toggle('hidden', !running);
  $('btn-propagate').classList.toggle('busy', running);
  if (running) el.textContent = `🎯 ${job.message || 'Đang lan truyền…'} ${fmtTime((Date.now() / 1000) - (job.started || Date.now() / 1000))}`;
}

// Hỏi tiến độ việc lan truyền chạy nền tới khi xong; trả về kết quả (hoặc ném lỗi)
async function waitPropJob() {
  for (;;) {
    const job = await api('/propagation/job');
    showPropJob(job);
    if (job.state === 'done') return job.result;
    if (job.state === 'error') throw new Error(job.message);
    if (job.state !== 'running') return null;
    await new Promise((r) => setTimeout(r, 2000));
  }
}

async function resumePropJob() { // mở lại trang khi việc nền còn chạy: hiện tiến độ, xong thì tải lại timeline
  try {
    const job = await api('/propagation/job');
    if (job.state !== 'running') return;
    const r = await waitPropJob();
    if (r) { toast(`Lan truyền (${r.engine}) xong: ${r.frames_updated.length} frame, ${r.objects_propagated} nhãn`); refreshLists(); }
  } catch (err) { toast('Lan truyền lỗi: ' + err.message, true); }
}

async function propagate(frameId) {
  try {
    const engine = 'default';
    let r;
    if (!(await slowProp())) {
      r = await api(`/frames/${encodeURIComponent(frameId)}/propagate`, { method: 'POST', body: {} });
    } else {
      const job = await api(`/frames/${encodeURIComponent(frameId)}/propagate-async`, { method: 'POST', body: { engine } });
      if (job.state === 'running' && job.message && !job.message.includes(frameId)) toast('Đang có một lần lan truyền khác chạy nền, đợi nó xong');
      else toast('Lan truyền bằng DAM4SAM: chạy nền, vài phút. Bạn vẫn duyệt tiếp được.');
      r = await waitPropJob();
      if (!r) return null;
    }
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
  loadFrameQC();
}

// ---------- QC nhãn cuối (kiểm lại sau khi người sửa) ----------

function qcOpen() { return (S.qc?.findings || []).filter((x) => !x.acked); }
function qcFor(id) { return qcOpen().filter((x) => x.object_id === id || x.other_object_id === id); }

async function loadFrameQC() {
  const f = S.frame;
  if (!f) return;
  try {
    const r = await api(`/frames/${encodeURIComponent(f.frame_id)}/qc`);
    if (S.frame?.frame_id !== f.frame_id) return;
    S.qc = r;
  } catch (err) {
    toast('QC: ' + err.message, true);
    return;
  }
  renderPanel();
  draw();
}

function suggestionText(s) {
  if (!s) return '';
  return {
    DELETE: `Xoá #${s.object_id}`,
    EDIT_BOX: 'Cắt box về biên ảnh',
    ADD_BOX: `Thêm box ${s.label}`,
    CHANGE_CLASS: `Đổi lớp → ${s.label}`,
  }[s.action] || s.action;
}

function qcRow(x) {
  const locked = S.frame?.status === 'approved';
  const ref = (id) => `<button class="linklike qc-obj" data-qc-select="${esc(id)}">#${esc(id)}</button>`;
  const who = x.object_id ? ref(x.object_id) + (x.other_object_id ? ` ↔ ${ref(x.other_object_id)}` : '') : '';
  const apply = x.suggestion && !locked
    ? `<button class="btn btn-sm btn-primary" data-qc-apply>${esc(suggestionText(x.suggestion))}</button>` : '';
  const ack = x.severity === 'warning'
    ? `<input type="text" class="qc-note" maxlength="300" placeholder="lý do giữ nguyên (tuỳ chọn)"><button class="btn btn-sm btn-ghost" data-qc-ack>Đã kiểm, giữ nguyên</button>`
    : '<span class="muted">lỗi: phải sửa</span>';
  return `<div class="qc-row ${x.severity}" data-qc-key="${esc(x.key)}">
    <div class="qc-top"><span class="sev ${x.severity}">${x.severity === 'error' ? 'Lỗi' : 'Cảnh báo'}</span><span class="issue-code">${esc(x.code)}</span>${who}</div>
    <div class="issue-msg" title="${esc(S.cfg.issue_help[x.code] || '')}">${esc(x.message)}</div>
    <div class="qc-row-actions">${apply}${ack}</div>
  </div>`;
}

async function ackQC(x, note) {
  try {
    S.qc = await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/qc/ack`, {
      method: 'POST',
      body: { key: x.key, fingerprint: x.fingerprint, note, reviewer: reviewer() },
    });
    renderPanel();
    draw();
    toast(`Đã xác nhận ${x.code}${S.qc.open ? ` · còn ${S.qc.open} lỗi QC` : ' · frame sạch QC'}`);
  } catch (err) {
    toast(err.message, true);
    loadFrameQC();
  }
}

function applySuggestion(x) {
  const s = x.suggestion;
  const body = { action: s.action };
  for (const k of ['object_id', 'bbox', 'label']) if (s[k] != null) body[k] = s[k];
  return act(body);
}

$('list-qc').addEventListener('click', (e) => {
  const sel = e.target.closest('[data-qc-select]');
  if (sel) { select(sel.dataset.qcSelect); return; }
  const row = e.target.closest('[data-qc-key]');
  const x = row && qcOpen().find((f) => f.key === row.dataset.qcKey);
  if (!x) return;
  if (e.target.closest('[data-qc-ack]')) ackQC(x, row.querySelector('.qc-note')?.value.trim() || '');
  else if (e.target.closest('[data-qc-apply]')) applySuggestion(x);
});

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
  // Thẻ gọn kiểu CVAT: một dòng tiêu đề (#id · lớp · mức rủi ro · nút), một dòng số liệu, lỗi QA dạng chip (rê chuột xem
  // giải thích); ảnh cắt + chi tiết chỉ mở cho thẻ đang chọn
  const lv = levelOf(o);
  const issues = o.qa?.issues || [];
  const lid = o.qa?.lidar || {};
  const tmp = o.qa?.temporal || {};
  const locked = S.frame.status === 'approved';
  const done = o.review.status !== 'pending';
  const pr = o.propagation;
  const sel = o.object_id === S.selected;
  const facts = [];
  if (isProp(o)) facts.push(`<span title="Độ tin cậy lan truyền từ ${esc(frameRef(pr.keyframe_id))}">↦ ${fx(pr.prop_conf)}</span>`, pr.matched ? `<span title="Detector thấy ở frame này">det ${esc(pr.detector_label)} ${fx(pr.detector_score)}</span>` : '<span title="Detector không thấy, box dự đoán từ chuyển động">dự đoán</span>');
  else if (o.source === 'track') facts.push('<span title="Box nội suy từ các sweep lân cận (RECOVERED_BY_TRACK)">nội suy</span>');
  else if (o.source === 'human') facts.push('người vẽ');
  else facts.push(`<span title="${o.det_score != null && Math.abs(o.det_score - o.score) >= 0.005 ? `Detector ${fx(o.det_score)}, tính lại theo sweep lân cận` : 'Score detector'}">score ${fx(o.score)}</span>`);
  if (lid.available) facts.push(`<span title="${lid.n_points ?? 0} điểm LiDAR trong box${lid.est_height_m != null ? `, cao ~${fx(lid.est_height_m, 1)} m` : ''}">${lid.depth_m != null ? `${fx(lid.depth_m, 0)} m` : `${lid.n_points ?? 0} pt`}</span>`);
  if (tmp.available) facts.push(`<span title="Thấy ở ${tmp.support}/${tmp.available} sweep lân cận">${tmp.support}/${tmp.available} sw</span>`);
  const status = !done
    ? ''
    : isAutoDeleted(o)
      ? `<span class="done-tag deleted" title="Người đã xoá object này ở ${esc(pr?.keyframe_id)}. Bấm Giữ nếu đây là object thật.">✗ tự xoá</span>`
      : `<span class="done-tag ${o.review.status}">${o.review.status === 'deleted' ? '✗ đã xoá' : `✓ ${o.review.final_label && o.review.final_label !== o.label ? '→ ' + esc(o.review.final_label) : 'giữ'}`}</span>`;
  const chips = issues.map((i) => `<span class="chip-issue" title="${esc(i.message)}${S.cfg.issue_help[i.code] ? '\n\n' + esc(S.cfg.issue_help[i.code]) : ''}">${esc(i.code)}</span>`).join('');
  const qc = qcFor(o.object_id).map((x) => `<span class="chip-issue qc" title="QC: ${esc(x.message)}">${esc(x.code)}</span>`).join('');
  return `<div class="obj-card ${lv} ${sel ? 'selected' : ''}" data-oid="${esc(o.object_id)}">
    <div class="oc-row">
      <span class="oid">#${esc(o.object_id)}</span>
      <select data-class class="card-select" ${locked ? 'disabled' : ''} title="Đổi lớp (phím C)">${classOptions(finalLabel(o))}</select>
      ${o.qa ? `<span class="risk-badge ${lv}" title="Mức rủi ro ${fx(o.qa.risk)}">${fx(o.qa.risk)}</span>` : ''}
      <span class="oc-icons">
        <button class="card-icon-action-btn ${o.locked ? 'active' : ''}" data-act="LOCK" title="${o.locked ? 'Mở khóa' : 'Khóa'}"><i class="${o.locked ? 'ri-lock-fill' : 'ri-lock-line'}"></i></button>
        <button class="card-icon-action-btn ${o.assigned ? 'active' : ''}" data-act="ASSIGN" title="${o.assigned ? 'Người phụ trách: ' + esc(o.assigned) : 'Người phụ trách'}"><i class="ri-user-line"></i></button>
        <button class="card-icon-action-btn ${o.hidden ? 'active' : ''}" data-act="VISIBILITY" title="${o.hidden ? 'Hiện box' : 'Ẩn box'}"><i class="${o.hidden ? 'ri-eye-off-line' : 'ri-eye-line'}"></i></button>
        <button class="card-icon-action-btn ${o.pinned ? 'active' : ''}" data-act="PIN" title="${o.pinned ? 'Bỏ ghim' : 'Ghim'}"><i class="${o.pinned ? 'ri-pushpin-fill' : 'ri-pushpin-line'}"></i></button>
      </span>
      ${locked ? '' : `<span class="oc-btns">
        <button class="ib keep" data-act="KEEP" title="Giữ (K)">✓</button>
        <button class="ib del" data-act="DELETE" title="Xoá (D)">✕</button>
        <button class="ib" data-act="EDIT" title="Sửa box (E)">✎</button>
      </span>`}
    </div>
    <div class="oc-row oc-meta">${facts.join('<i>·</i>')}${status}${chips}${qc}</div>
    ${sel ? `<div class="oc-body"><canvas class="oc-crop" width="192" height="144" data-crop="${esc(o.object_id)}"></canvas>
      <div class="oc-info">${issues.length ? `<ul class="issues">${issues.map((i) => `<li><span class="issue-code">${esc(i.code)}</span> <span class="issue-msg">${esc(i.message)}</span></li>`).join('')}</ul>` : '<span class="muted">Không có lỗi QA</span>'}
      ${isProp(o) ? `<div class="muted">↦ lan truyền từ ${esc(frameRef(pr.keyframe_id))}</div>` : ''}</div></div>` : ''}
  </div>`;
}

function compactRow(o) {
  const done = o.review.status !== 'pending';
  const tag = done
    ? isAutoDeleted(o)
      ? '<span class="done-tag deleted">✗ tự xoá</span>'
      : `<span class="done-tag ${o.review.status}">${o.review.status === 'deleted' ? '✗ xoá' : '✓ ' + o.review.action.toLowerCase().replace('_', ' ')}</span>`
    : `<span class="muted">risk ${fx(o.qa?.risk)}</span>`;
  const qcTag = qcFor(o.object_id).length ? '<span class="qc-mark sm" title="Nhãn cuối còn lỗi QC">QC</span>' : '';
  return `<div class="low-row ${o.object_id === S.selected ? 'selected' : ''}" data-oid="${esc(o.object_id)}">
    <span><span class="dot ${levelOf(o)}"></span> #${esc(o.object_id)} HỘP <strong style="margin-left:4px;">${esc(finalLabel(o))}</strong> <span class="muted">${scoreText(o)}</span>${qcTag}</span>${tag}</div>`;
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

  const qcList = f ? qcOpen() : [];
  $('qc-section').classList.toggle('hidden', !qcList.length);
  $('count-qc').textContent = qcList.length;
  $('list-qc').innerHTML = qcList.map(qcRow).join('');

  renderLabelsTab();
  renderIssuesTab();

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
    btn.title = '';
    btn.onclick = reopenFrame;
  } else {
    const qcBlock = S.cfg?.qc?.gate_on_approve && qcList.length > 0;
    btn.disabled = !f || pending.length > 0 || qcBlock;
    btn.innerHTML = qcBlock && !pending.length ? `Còn ${qcList.length} lỗi QC` : 'Approve frame <kbd>Enter</kbd>';
    btn.title = qcBlock ? 'Sửa hoặc xác nhận các lỗi QC ở đầu panel trước khi approve' : '';
    btn.onclick = approveFrame;
  }
  $('btn-add').disabled = !f || locked;
  $('btn-propagate').disabled = !locked || S.viewMode !== 'video';
  const sweepMode = inSweep();
  $('review-scroll').classList.toggle('sweep-mode', sweepMode);
  $('sweep-panel').classList.toggle('hidden', !sweepMode);
  $('sweep-panel').innerHTML = sweepMode ? sweepPanel() : '';
  drawCrops();
}

function renderLabelsTab() {
  const wrap = $('labels-list-wrap');
  if (!wrap) return;
  const f = S.frame;
  if (!f) {
    wrap.innerHTML = '<p class="muted" style="padding:12px;">Chưa mở frame nào</p>';
    return;
  }
  const counts = {};
  f.objects.forEach((o) => {
    if (o.review.status === 'deleted') return;
    const lbl = finalLabel(o);
    counts[lbl] = (counts[lbl] || 0) + 1;
  });
  const classes = { ...(S.cfg?.classes || {}), ...counts };
  wrap.innerHTML = Object.keys(classes).map((lbl) => {
    const color = S.cfg?.classes?.[lbl] || '#3b82f6';
    const count = counts[lbl] || 0;
    return `<div class="label-stat-item" data-label="${esc(lbl)}" title="Bấm để lọc/chọn đối tượng lớp ${esc(lbl)}">
      <div class="label-stat-left">
        <span class="label-color-dot" style="background: ${color};"></span>
        <strong>${esc(lbl)}</strong>
      </div>
      <span class="label-stat-count">${count} đối tượng</span>
    </div>`;
  }).join('') || '<p class="muted" style="padding:12px;">Không có nhãn nào</p>';
}

function renderIssuesTab() {
  const wrap = $('issues-list-wrap');
  if (!wrap) return;
  const f = S.frame;
  if (!f) {
    wrap.innerHTML = '<p class="muted" style="padding:12px;">Chưa mở frame nào</p>';
    return;
  }
  const issues = [];
  f.objects.forEach((o) => {
    if (o.review.status === 'deleted') return;
    (o.qa?.issues || []).forEach((iss) => {
      issues.push({ obj: o, ...iss });
    });
  });
  wrap.innerHTML = issues.map((iss) => `
    <div class="issue-stat-item" data-oid="${esc(iss.obj.object_id)}" title="Bấm để xem object #${esc(iss.obj.object_id)}">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="issue-code" style="font-weight:600;">#${esc(iss.obj.object_id)} ${esc(iss.code)}</span>
        <span class="risk-badge ${levelOf(iss.obj)}">${LEVEL_NAME[levelOf(iss.obj)]}</span>
      </div>
      <div class="muted text-xs" style="margin-top:4px;">${esc(iss.message)}</div>
    </div>
  `).join('') || '<p class="muted" style="padding:12px;">✓ Không phát hiện sự cố (issue) nào trên frame này</p>';
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
  if (e.target.closest('[data-sweep-back]')) { setView(0); return; }
  if (e.target.closest('[data-sweep-add]')) { startAdd(); return; }
  const row = e.target.closest('[data-sbid]');
  if (row) {
    if (e.target.closest('select')) return;
    const id = row.dataset.sbid;
    const sbtn = e.target.closest('[data-sact]');
    if (!sbtn) { S.sweepSel = id; renderPanel(); draw(); return; }
    const a = sbtn.dataset.sact;
    if (a === 'EDIT') { S.sweepSel = id; startEdit(); return; }
    if (a === 'CHANGE_CLASS') { sweepAct({ action: 'CHANGE_CLASS', box_id: id, label: row.querySelector('[data-sclass]').value }); return; }
    sweepAct({ action: a, box_id: id });
    return;
  }
  const card = e.target.closest('[data-oid]');
  if (!card) return;
  const id = card.dataset.oid;
  const btn = e.target.closest('[data-act]');
  if (e.target.closest('select')) return;
  if (!btn) { select(id); return; }
  // Nút biểu tượng (khoá / người phụ trách / ẩn / ghim) chỉ đổi trạng thái hiển thị, xử lý ở index.html: không gọi API
  if (btn.classList.contains('card-icon-action-btn')) return;
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
  setView(S.viewOffset === off ? 0 : off);
});

// ---------- chế độ Ảnh / Video ----------

function refreshLists() {
  return S.viewMode === 'video' ? refreshVideo() : loadQueue();
}

async function setMode(mode) {
  stopPlay({ reopen: false });
  S.viewMode = mode;
  storageSet('viewMode', mode);
  markTabs();
  // Chỉ hiển thị player-controls trên topbar khi ở chế độ video
  const playerControls = document.getElementById('topbar-player-controls') || document.querySelector('.player-controls');
  if (playerControls) {
    const isVideo = mode === 'video';
    playerControls.classList.toggle('hidden', !isVideo);
    playerControls.style.display = isVideo ? 'flex' : 'none';
  }
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
  stopPlay({ reopen: false });
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

// ---------- phát video ----------
// Phát mọi ảnh của video (keyframe 2 fps, sweep t-2 … t+2 và ảnh nằm giữa hai cửa sổ sweep) theo đúng thời gian thực, box lấy sẵn từ
// GET /videos/{id}/playback trong một lần gọi. Lúc phát chỉ vẽ lại khung ảnh (không mở frame, không vẽ lại panel);
// ảnh phía trước được tải trước, ảnh chưa kịp tải thì đồng hồ đứng chờ. Dừng ở đâu thì mở keyframe gần đó để duyệt.
const PLAY_AHEAD = 15; // số ảnh tải trước
const PLAY_KEEP = 4; // số ảnh đã qua còn giữ (ảnh 2560×1440 giải nén ~15 MB, không giữ cả video)

function setPlayButtons(playing, label) {
  const b = $('btn-play');
  if (b) b.textContent = playing ? `⏸ ${label || 'Dừng'}` : '▶ Phát';
  const nb = $('btn-nav-play');
  if (nb) nb.innerHTML = `<i class="${playing ? 'ri-pause-fill' : 'ri-play-fill'}"></i>`;
}

function stopPlay({ reopen = true } = {}) {
  const pb = S.playback;
  if (!pb) return;
  S.playback = null;
  cancelAnimationFrame(pb.raf);
  setPlayButtons(false);
  const cur = pb.items[pb.i];
  // Mở keyframe của ảnh đang dừng (bắt đầu tính giờ duyệt frame đó, M1)
  if (reopen && cur && S.viewMode === 'video') openFrame(cur.frame_id).catch((err) => toast(err.message, true));
  else drawCanvas();
}

function playImage(pb, j) {
  const it = pb.items[j];
  if (!it) return null;
  let im = pb.imgs.get(j);
  if (!im) {
    im = new Image();
    im.decoding = 'async';
    // Ảnh nằm ngoài cửa sổ sweep của mọi keyframe (có "sd") lấy theo sd_token của video
    im.src = it.sd
      ? `${API}/videos/${encodeURIComponent(pb.videoId)}/image?sd=${encodeURIComponent(it.sd)}`
      : `${API}/frames/${encodeURIComponent(it.frame_id)}/image${it.offset ? `?offset=${it.offset}` : ''}`;
    im.decode?.().catch(() => {});
    pb.imgs.set(j, im);
  }
  return im;
}

function playTick(now) {
  const pb = S.playback;
  if (!pb) return;
  pb.raf = requestAnimationFrame(playTick);
  const items = pb.items;
  if (pb.last != null) {
    // Ảnh kế chưa tải xong thì đồng hồ đứng chờ (không nhảy cóc qua ảnh thiếu)
    const nxt = items[pb.i + 1];
    const im = nxt && playImage(pb, pb.i + 1);
    if (!nxt || (im.complete && im.naturalWidth)) pb.clock += Math.min(0.25, (now - pb.last) / 1000);
  }
  pb.last = now;
  let j = pb.i;
  while (j + 1 < items.length && items[j + 1].t <= pb.clock) {
    const im = playImage(pb, j + 1);
    if (!(im.complete && im.naturalWidth)) break;
    j++;
  }
  for (let k = j + 1; k <= Math.min(items.length - 1, j + PLAY_AHEAD); k++) playImage(pb, k);
  for (const k of pb.imgs.keys()) if (k < j - PLAY_KEEP) pb.imgs.delete(k);
  if (j !== pb.i || !pb.drawn) {
    pb.i = j;
    pb.drawn = true;
    drawCanvas();
    const it = items[j];
    setPlayButtons(true, `${it.t.toFixed(1)}s`);
    if (it.frame_id !== pb.frameId) { pb.frameId = it.frame_id; markTimeline(it.frame_id); }
  }
  if (j >= items.length - 1 && pb.clock >= items[j].t + 0.15) stopPlay();
}

// Đánh dấu keyframe đang phát trên timeline mà không mở frame
function markTimeline(id) {
  let cur = null;
  document.querySelectorAll('.tl-card').forEach((c) => { const on = c.dataset.id === id; c.classList.toggle('current', on); if (on) cur = c; });
  cur?.scrollIntoView({ block: 'nearest', inline: 'center' });
}

function drawPlayback() {
  const pb = S.playback;
  const it = pb.items[pb.i];
  const im = pb.imgs.get(pb.i);
  if (im?.complete && im.naturalWidth) ctx.drawImage(im, 0, 0, canvas.width, canvas.height);
  for (const b of it.boxes) {
    if (b.pending && b.level === 'low' && !S.showLow) continue;
    const color = b.source === 'human' ? HUMAN_COLOR : RISK_COLOR[b.level] || RISK_COLOR.low;
    const tag = b.source === 'human' || (b.pending && b.level === 'high') ? `#${b.id} ${b.label}` : null;
    drawBox(b.bbox, color, { lw: b.pending ? 2 : 1.4, dash: b.source === 'track' ? [8, 5] : null, label: tag });
  }
}

async function togglePlay() {
  if (S.playback) { stopPlay(); return; }
  const v = S.video;
  if (!v || (v.frames || []).length < 2) return;
  const pb = { items: [], i: 0, imgs: new Map(), raf: 0, clock: 0, last: null, drawn: false, frameId: null, videoId: v.video_id };
  S.playback = pb;
  setPlayButtons(true, '…');
  try {
    const res = await api(`/videos/${encodeURIComponent(v.video_id)}/playback`);
    if (S.playback !== pb) return;
    pb.items = res.items;
  } catch (err) {
    if (S.playback === pb) { S.playback = null; setPlayButtons(false); }
    toast('Không phát được: ' + err.message, true);
    return;
  }
  if (pb.items.length < 2) { S.playback = null; setPlayButtons(false); return; }
  // Phát tiếp từ keyframe đang mở; đang ở cuối thì phát lại từ đầu
  let start = pb.items.findIndex((it) => it.frame_id === S.frame?.frame_id && it.offset === 0);
  if (start < 0 || start >= pb.items.length - 2) start = 0;
  pb.i = start;
  pb.clock = pb.items[start].t;
  stopTimer();
  for (let k = start; k <= Math.min(pb.items.length - 1, start + PLAY_AHEAD); k++) playImage(pb, k);
  pb.raf = requestAnimationFrame(playTick);
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

// Bấm Ảnh / Video / 3D từ bất kỳ tab nào (Log, Cài đặt…) đều về màn hình duyệt của chế độ đó
document.querySelectorAll('.mode').forEach((b) => b.addEventListener('click', () => {
  const onReview = document.querySelector('.tab.active')?.dataset.tab === 'review';
  if (!onReview) switchTab('review');
  if (!onReview || S.viewMode !== b.dataset.mode) setMode(b.dataset.mode);
}));
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
$('btn-nav-play')?.addEventListener('click', () => { if (S.viewMode === 'video') togglePlay(); });
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
  api('/metrics/tracking').then((t) => {
    $('track-table').innerHTML = !t.frames
      ? '<tr><td class="muted">Workspace chưa có frame nào có nhãn gốc</td></tr>'
      : `<thead><tr><th>Frame</th><th class="num">HOTA</th><th class="num">DetA</th><th class="num">AssA</th><th class="num">MOTA</th><th class="num">IDF1</th><th class="num">Đổi ID</th><th class="num">Thừa</th><th class="num">Sót</th></tr></thead>
        <tbody><tr><td>${t.frames}</td><td class="num">${fx(t.HOTA)}</td><td class="num">${fx(t.DetA)}</td><td class="num">${fx(t.AssA)}</td><td class="num">${fx(t.MOTA)}</td><td class="num">${fx(t.IDF1)}</td><td class="num">${t.IDSW}</td><td class="num">${t.FP}</td><td class="num">${t.FN}</td></tr></tbody>`;
  }).catch(() => { $('track-table').innerHTML = ''; });
}

const EXPORT_FILES = ['coco.json', 'labels.jsonl', 'corrections.jsonl', 'qc_log.jsonl', 'qa_report.md', 'manifest.json'];
// Năng suất người duyệt + throughput auto-label (FR-27); link CSV (FR-19). Dùng chung với app3d.js qua window.AL
function renderProductivity(prod, p, csvBase) {
  $(p + 'csv-frames').href = csvBase + (csvBase.includes('?') ? '&' : '?') + 'kind=frames';
  $(p + 'csv-summary').href = csvBase + (csvBase.includes('?') ? '&' : '?') + 'kind=summary';
  const n = (v, d = 1) => (v == null ? '—' : Number(v).toFixed(d));
  const rows = prod?.reviewers || [];
  $(p + 'prod-table').innerHTML = '<thead><tr><th>Người duyệt</th><th class="num">Frame duyệt</th><th class="num">Object đã xử lý</th><th class="num">Box sweep đã sửa</th><th class="num">Trả lại</th><th class="num">Hoàn tác</th><th class="num">Thời gian duyệt</th><th class="num">Frame/giờ</th><th>Phiên</th></tr></thead><tbody>' +
    (rows.map((r) => `<tr><td>${esc(r.reviewer)}</td><td class="num">${r.frames_approved}</td><td class="num">${r.objects_handled}</td>
      <td class="num">${r.sweep_boxes_handled ?? 0}</td><td class="num">${r.frames_rejected}</td><td class="num">${r.undo_redo}</td><td class="num">${fmtTime(r.review_time_s)}</td>
      <td class="num"><b>${n(r.frames_per_hour)}</b></td>
      <td class="prod-sessions">${r.sessions.map((s) => `${esc(s.start.slice(0, 16).replace('T', ' '))}: ${s.frames} frame, ${n(s.frames_per_hour)}/giờ`).join('<br>') || '—'}</td></tr>`).join('') ||
      '<tr><td colspan="9" class="muted">Chưa có frame nào được approve</td></tr>') + '</tbody>';
  const inf = prod?.inference || [];
  $(p + 'infer-table').innerHTML = '<thead><tr><th>Phiên auto-label</th><th>Thiết bị</th><th class="num">Frame</th><th class="num">s / frame</th><th class="num">Frame/giờ</th></tr></thead><tbody>' +
    (inf.map((r) => `<tr><td><code>${esc(r.run)}</code></td><td>${esc(r.device || '—')}</td><td class="num">${r.frames}</td><td class="num">${n(r.s_per_frame, 2)}</td><td class="num"><b>${n(r.frames_per_hour, 0)}</b></td></tr>`).join('') ||
      '<tr><td colspan="5" class="muted">Chưa có số đo (frame auto-label trước bản này không ghi thời gian; chạy lại <code>run --overwrite</code> hoặc tạo dự án mới)</td></tr>') + '</tbody>';
}

async function loadExports() {
  const ids = await api('/exports');
  $('export-list').innerHTML = ids.map((id) => `<li><strong>${esc(id)}</strong> — ${EXPORT_FILES
    .map((f) => `<a href="${API}/exports/${encodeURIComponent(id)}/${f}">${f}</a>`).join('')}</li>`).join('') || '<li class="muted">Chưa xuất lần nào</li>';
}

async function doExport() {
  try {
    const r = await api('/export' + ($('export-ready').checked ? '?require_ready=true' : ''), { method: 'POST' });
    const skipped = r.frames_skipped_qc.length ? ` · bỏ ${r.frames_skipped_qc.length} frame còn lỗi QC` : '';
    $('export-result').innerHTML = `✓ Đã xuất <strong>${r.n_frames}</strong> frame, <strong>${r.n_objects}</strong> object → <code>${esc(r.export_id)}</code> <span class="release-pill ${r.release_status === 'READY' ? 'ready' : 'not-ready'}">${esc(r.release_status)}</span>${skipped}`;
    loadExports();
  } catch (err) {
    $('export-result').innerHTML = `<span class="done-tag deleted">${esc(err.message)}</span>`;
  }
}

// ---------- tab QC: sẵn sàng phát hành, audit ngẫu nhiên, Quick Check ----------

async function loadQC() {
  const [rep, aud] = await Promise.all([api('/qc/report'), api('/qc/audit')]);
  S.qcReport = rep;
  S.audit = aud;
  renderReadiness();
  renderOpenFindings();
  renderAudit();
}

function renderReadiness() {
  const rep = S.qcReport;
  const ready = rep.status === 'READY';
  const pill = $('qc-status');
  pill.className = 'release-pill ' + (ready ? 'ready' : 'not-ready');
  pill.textContent = ready ? '✓ READY' : '✗ NOT READY';
  const fr = rep.frames;
  $('qc-checks').innerHTML = rep.checks.map((c) => `<li class="${c.ok ? 'ok' : 'fail'}">
      <span class="ck">${c.ok ? '✓' : '✗'}</span><span class="ck-label">${esc(c.label)}</span><span class="ck-detail">${esc(c.detail)}</span></li>`).join('') +
    `<li class="info"><span class="ck">i</span><span class="ck-label">Độ phủ</span><span class="ck-detail">${fr.approved}/${fr.total} frame đã approve (${pct(rep.coverage)}) · ${fr.editing} đang sửa · ${fr.auto} chưa mở</span></li>`;
}

function renderOpenFindings() {
  const rep = S.qcReport;
  const rows = [
    ...rep.open_findings.map((x) => ({ frame: x.frame_id, obj: x.object_id, code: x.code, sev: x.severity, msg: x.message })),
    ...rep.track_findings.map((t) => {
      const [ref] = Object.values(t.classes)[0];
      const [frame, obj] = ref.split('#');
      return { frame, obj, code: t.code, sev: 'warning', msg: `track ${t.track_id}: ${t.message}` };
    }),
    ...rep.log_findings.map((e) => ({ frame: e.frame_id, obj: e.object_id, code: e.code, sev: 'error', msg: e.message })),
  ];
  $('qc-open-table').innerHTML = '<thead><tr><th>Frame</th><th>Object</th><th>Code</th><th>Mức</th><th>Mô tả</th></tr></thead><tbody>' +
    (rows.map((r) => `<tr data-frame="${esc(r.frame)}" data-obj="${esc(r.obj || '')}">
      <td>${esc(r.frame)}</td><td>${r.obj ? '#' + esc(r.obj) : '—'}</td><td><code>${esc(r.code)}</code></td>
      <td><span class="sev ${r.sev}">${r.sev === 'error' ? 'Lỗi' : 'Cảnh báo'}</span></td><td class="wrap">${esc(r.msg)}</td></tr>`).join('') ||
      '<tr><td colspan="5" class="muted">Không còn lỗi nào trên frame đã approve</td></tr>') + '</tbody>';
}

$('qc-open-table').addEventListener('click', async (e) => {
  const tr = e.target.closest('tr[data-frame]');
  if (!tr) return;
  switchTab('review');
  try {
    await openFrame(tr.dataset.frame);
    if (tr.dataset.obj) select(tr.dataset.obj);
  } catch (err) {
    toast(err.message, true);
  }
});

// ---- audit ----

function auditRow(name, a) {
  const ci = a.n ? `${pct(a.ci95[0])}–${pct(a.ci95[1])}` : '—';
  return `<tr><td>${name}</td><td class="num">${a.population}</td><td class="num">${a.n}${a.pending ? ` <span class="muted">(+${a.pending} chờ)</span>` : ''}</td>
    <td class="num">${a.errors}</td><td class="num">${pct(a.error_rate)}</td><td>${ci}</td>
    <td title="${esc(a.note)}">${a.passed ? '<span class="done-tag approved">✓ đạt</span>' : '<span class="done-tag deleted">✗ chưa</span>'}</td></tr>`;
}

function renderAudit() {
  const { items, summary } = S.audit;
  $('audit-table').innerHTML = '<thead><tr><th>Loại</th><th class="num">Tổng thể</th><th class="num">Đã audit</th><th class="num">Sai</th><th class="num">Tỉ lệ</th><th>CI95</th><th>Đạt</th></tr></thead><tbody>' +
    auditRow('Object duyệt theo lô', summary.object) + auditRow('Frame (vật bị sót)', summary.frame) + '</tbody>';
  const pending = items.filter((i) => !i.result);
  S.auditCur = pending[0] || null;
  $('audit-viewer').classList.toggle('hidden', !S.auditCur);
  if (S.auditCur) drawAuditItem(S.auditCur, pending.length).catch((e) => toast(e.message, true));
}

async function auditImage(frameId) {
  if (!S.auditImgs[frameId]) S.auditImgs[frameId] = await loadImage(`${API}/frames/${encodeURIComponent(frameId)}/image`);
  return S.auditImgs[frameId];
}

/* Vẽ box lên canvas phụ (audit / Quick Check): toạ độ ảnh -> canvas qua scale s và gốc (ox, oy) của vùng cắt */
function strokeOn(c2, b, color, s, ox, oy, { label = null, dash = null, lw = 2 } = {}) {
  c2.save();
  c2.strokeStyle = color;
  c2.lineWidth = lw;
  if (dash) c2.setLineDash(dash);
  const x = (b[0] - ox) * s;
  const y = (b[1] - oy) * s;
  c2.strokeRect(x, y, (b[2] - b[0]) * s, (b[3] - b[1]) * s);
  if (label) {
    c2.setLineDash([]);
    c2.font = "600 11px 'Plus Jakarta Sans', sans-serif";
    const w = c2.measureText(label).width + 8;
    const ty = y - 15 >= 0 ? y - 15 : y;
    c2.fillStyle = color;
    c2.fillRect(x, ty, w, 15);
    c2.fillStyle = '#fff';
    c2.fillText(label, x + 4, ty + 11);
  }
  c2.restore();
}

async function drawAuditItem(item, nPending) {
  const isObj = item.kind === 'object';
  $('audit-title').textContent = `Mẫu ${item.audit_id} · ${item.frame_id}${isObj ? ` #${item.object_id}` : ''}`;
  $('audit-progress').textContent = `${nPending} mẫu còn chờ`;
  $('audit-question').textContent = isObj
    ? `Box "${item.label}" này có đúng không? Đúng = đúng vật, đúng lớp, box ôm sát vật.`
    : 'Frame này có vật nào thuộc taxonomy mà chưa được gán nhãn không? Đúng = không sót vật nào.';
  const img = await auditImage(item.frame_id);
  const cv = $('audit-canvas');
  const c2 = cv.getContext('2d');
  c2.fillStyle = '#0b1020';
  c2.fillRect(0, 0, cv.width, cv.height);
  if (isObj) {
    // Cắt rộng quanh box để thấy cả ngữ cảnh
    const [x1, y1, x2, y2] = item.bbox;
    const pad = 0.8 * Math.max(x2 - x1, y2 - y1) + 30;
    const sx = Math.max(0, x1 - pad);
    const sy = Math.max(0, y1 - pad);
    const sw = Math.min(img.naturalWidth, x2 + pad) - sx;
    const sh = Math.min(img.naturalHeight, y2 + pad) - sy;
    const s = Math.min(cv.width / sw, cv.height / sh);
    const ox = sx - (cv.width / s - sw) / 2;
    const oy = sy - (cv.height / s - sh) / 2;
    c2.drawImage(img, sx, sy, sw, sh, (sx - ox) * s, (sy - oy) * s, sw * s, sh * s);
    strokeOn(c2, item.bbox, HUMAN_COLOR, s, ox, oy, { label: item.label, lw: 2.5 });
  } else {
    const frame = await api(`/frames/${encodeURIComponent(item.frame_id)}`);
    const s = Math.min(cv.width / img.naturalWidth, cv.height / img.naturalHeight);
    const ox = -(cv.width / s - img.naturalWidth) / 2;
    const oy = -(cv.height / s - img.naturalHeight) / 2;
    c2.drawImage(img, -ox * s, -oy * s, img.naturalWidth * s, img.naturalHeight * s);
    for (const o of frame.objects) {
      if (o.review.status !== 'approved') continue;
      strokeOn(c2, finalBox(o), HUMAN_COLOR, s, ox, oy, { label: finalLabel(o), lw: 1.5 });
    }
  }
}

async function auditSample(kind) {
  try {
    const before = S.audit?.items.length || 0;
    S.audit = await api('/qc/audit/sample', { method: 'POST', body: { kind } });
    toast(`Đã lấy ${S.audit.items.length - before} mẫu ${kind === 'object' ? 'object' : 'frame'}`);
    renderAudit();
  } catch (err) {
    toast(err.message, true);
  }
}

async function auditResult(result) {
  const item = S.auditCur;
  if (!item) return;
  try {
    S.audit = await api(`/qc/audit/${encodeURIComponent(item.audit_id)}`, {
      method: 'POST',
      body: { result, note: $('audit-note').value.trim(), reviewer: reviewer() },
    });
    $('audit-note').value = '';
    if (result === 'error') toast(`Đã mở lại ${item.frame_id} để sửa (${item.kind === 'object' ? 'object quay về chờ duyệt' : 'cần vẽ thêm box'})`);
    renderAudit();
    S.qcReport = await api('/qc/report');
    renderReadiness();
    renderOpenFindings();
  } catch (err) {
    toast(err.message, true);
  }
}

$('audit-sample-object').addEventListener('click', () => auditSample('object'));
$('audit-sample-frame').addEventListener('click', () => auditSample('frame'));
$('audit-ok').addEventListener('click', () => auditResult('ok'));
$('audit-error').addEventListener('click', () => auditResult('error'));
document.addEventListener('keydown', (e) => {
  if ($('tab-qc').classList.contains('hidden') || !S.auditCur) return;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === 'y') { auditResult('ok'); e.preventDefault(); }
  if (k === 'x') { auditResult('error'); e.preventDefault(); }
});

// ---- Quick Check ----

async function runQuickCheck() {
  const file = $('qc-file').files[0];
  if (!file) { toast('Chọn file nhãn (.json / .jsonl) trước', true); return; }
  const body = new FormData();
  body.append('file', file);
  $('qc-run').disabled = true;
  try {
    const res = await fetch(`${API}/qc/quick-check`, { method: 'POST', body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail?.message || res.statusText);
    S.quick = data;
    renderQuick();
  } catch (err) {
    toast('Quick Check: ' + err.message, true);
  } finally {
    $('qc-run').disabled = false;
  }
}

function renderQuick() {
  const q = S.quick;
  const tiles = [
    ['Frame', q.n_frames], ['Nhãn', q.n_labels], ['Lỗi', q.n_errors], ['Cảnh báo', q.n_warnings], ['Thời gian', `${q.elapsed_ms} ms`],
  ];
  $('qc-summary').innerHTML = `<div class="mini-tiles">${tiles.map(([l, v]) => `<div class="mini-tile"><span>${l}</span><strong>${v}</strong></div>`).join('')}</div>
    <p class="muted">${esc(q.file_name)} · định dạng ${esc(q.format)}${Object.keys(q.by_code).length ? ' · ' + Object.entries(q.by_code).map(([c, n]) => `<code>${esc(c)}</code> ${n}`).join(' · ') : ''}${q.n_acked ? ` · ${q.n_acked} cảnh báo file ghi là người đã xác nhận` : ''}</p>
    ${q.parse_errors.length ? `<ul class="qc-parse-errors">${q.parse_errors.slice(0, 20).map((m) => `<li>${esc(m)}</li>`).join('')}</ul>` : ''}`;
  const rows = [];
  q.frames.forEach((fr, fi) => fr.findings.forEach((x, xi) => rows.push({ fr, fi, x, xi })));
  rows.sort((a, b) => a.x.acked - b.x.acked); // lỗi còn mở lên trước
  const level = (x) => (x.acked ? '<span class="done-tag approved">đã xác nhận</span>'
    : `<span class="sev ${x.severity}">${x.severity === 'error' ? 'Lỗi' : 'Cảnh báo'}</span>`);
  const clean = !q.n_errors && !q.n_warnings
    ? `<tr><td colspan="6"><span class="done-tag approved">✓ ${q.n_labels} nhãn không còn lỗi QC mở</span></td></tr>` : '';
  $('qc-table').innerHTML = '<thead><tr><th>Frame</th><th>Nhãn</th><th>Code</th><th>Mức</th><th>Mô tả</th><th>Đề xuất</th></tr></thead><tbody>' + clean +
    rows.slice(0, 1000).map(({ fr, fi, x, xi }) => `<tr data-fi="${fi}" data-xi="${xi}" class="${x.acked ? 'acked' : ''}">
      <td>${esc(fr.frame_id || fr.file_name || '—')}</td><td>${x.object_id ? '#' + esc(x.object_id) : '—'}</td><td><code>${esc(x.code)}</code></td>
      <td>${level(x)}</td><td class="wrap">${esc(x.message)}</td>
      <td>${esc(suggestionText(x.suggestion))}</td></tr>`).join('') + '</tbody>';
  $('qc-canvas').classList.add('hidden');
  $('qc-legend').classList.add('hidden');
}

async function previewQuick(fr, x) {
  const cv = $('qc-canvas');
  const c2 = cv.getContext('2d');
  cv.classList.remove('hidden');
  $('qc-legend').classList.remove('hidden');
  let img = null;
  if (fr.in_workspace) {
    try { img = await auditImage(fr.frame_id); } catch { img = null; }
  }
  const allBoxes = fr.labels.map((l) => l.bbox).concat(fr.findings.filter((f) => f.suggestion?.bbox).map((f) => f.suggestion.bbox));
  const W = img?.naturalWidth || fr.width || Math.max(1, ...allBoxes.map((b) => b[2]));
  const H = img?.naturalHeight || fr.height || Math.max(1, ...allBoxes.map((b) => b[3]));
  const s = Math.min(cv.width / W, cv.height / H);
  const ox = -(cv.width / s - W) / 2;
  const oy = -(cv.height / s - H) / 2;
  c2.fillStyle = '#0b1020';
  c2.fillRect(0, 0, cv.width, cv.height);
  if (img) c2.drawImage(img, -ox * s, -oy * s, W * s, H * s);
  const bad = new Set(fr.findings.filter((f) => !f.acked).flatMap((f) => [f.object_id, f.other_object_id]).filter(Boolean));
  for (const l of fr.labels) {
    const hit = l.object_id === x.object_id || l.object_id === x.other_object_id;
    strokeOn(c2, l.bbox, bad.has(l.object_id) ? '#d03b3b' : HUMAN_COLOR, s, ox, oy, { label: hit ? `#${l.object_id} ${l.label}` : null, lw: hit ? 3 : 1.5 });
  }
  for (const f of fr.findings) {
    if (f.suggestion?.bbox) strokeOn(c2, f.suggestion.bbox, '#0ca30c', s, ox, oy, { dash: [6, 4], lw: f === x ? 3 : 1.5, label: f === x ? suggestionText(f.suggestion) : null });
  }
}

$('qc-run').addEventListener('click', runQuickCheck);
$('qc-table').addEventListener('click', (e) => {
  const tr = e.target.closest('tr[data-fi]');
  if (!tr || !S.quick) return;
  $('qc-table').querySelectorAll('tr').forEach((r) => r.classList.toggle('selected', r === tr));
  const fr = S.quick.frames[Number(tr.dataset.fi)];
  previewQuick(fr, fr.findings[Number(tr.dataset.xi)]).catch((err) => toast(err.message, true));
});
$('qc-refresh').addEventListener('click', () => loadQC().catch((e) => toast(e.message, true)));

// ---------- điều hướng & phím tắt ----------

function fmtTime(s) {
  s = Math.round(s);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
setInterval(() => { $('timer').textContent = fmtTime(elapsed()); }, 500);

// Thanh tab duy nhất: Ảnh / Video / 3D sáng khi đang ở màn hình duyệt của chế độ đó; Log / QC / Metrics / Cài đặt sáng
// khi đang ở tab đó. Lúc nào cũng chỉ một nút sáng.
function markTabs() {
  const tab = document.querySelector('.tab.active')?.dataset.tab || 'review';
  document.querySelectorAll('.mode').forEach((b) => b.classList.toggle('active', tab === 'review' && b.dataset.mode === S.viewMode));
}

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
  markTabs();
  ['review', 'log', 'qc', 'metrics', 'settings'].forEach((t) => $('tab-' + t).classList.toggle('hidden', t !== tab || (t === 'review' && S.viewMode === '3d')));
  $('tab-review3d').classList.toggle('hidden', !(tab === 'review' && S.viewMode === '3d'));
  window.dispatchEvent(new CustomEvent('autolabel:tab', { detail: tab }));
  if (tab === 'settings') { loadSettings().catch((e) => toast(e.message, true)); return; }
  if (tab === 'metrics') loadTiming().catch((e) => toast(e.message, true));
  if (S.viewMode === '3d' && tab !== 'log') return;
  if (tab === 'log') loadLog().catch((e) => toast(e.message, true));
  if (tab === 'qc') loadQC().catch((e) => toast(e.message, true));
  if (tab === 'metrics') loadMetrics().catch((e) => toast(e.message, true));
  if (tab === 'review') { fitCanvas(); draw(); }
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));

// ---------- ⚙ Cài đặt ----------

// Chọn mô hình ngay trên màn hình duyệt: hai ô chọn này ghi thẳng vào Cài đặt của dự án (một nguồn duy nhất), nên
// tab Cài đặt và màn hình duyệt luôn khớp nhau.
const PROP_PRESETS = [
  { flow: 'always', association: 'byte', label: 'Optical flow + ByteTrack' },
  { flow: 'always', association: 'botsort', label: 'Optical flow + BoT-SORT' },
  { flow: 'flow+dam4sam', association: 'byte', label: 'Optical flow + DAM4SAM', gpu: true },
  { flow: 'dam4sam', association: 'botsort', label: 'DAM4SAM + BoT-SORT', gpu: true },
  { flow: 'dam4sam', association: 'byte', label: 'DAM4SAM + ByteTrack', gpu: true },
  { flow: 'off', association: 'single', label: 'Vận tốc không đổi + IoU' },
];
function renderMainModels(fields) {
  const get = (path) => fields.find((f) => f.path === path);
  const det = get('detection.yoloe.weights');
  const m2 = $('model2d');
  if (m2 && det) {
    m2.innerHTML = Object.entries(det.choices).map(([k, lab]) => `<option value="${esc(k)}">${esc(lab)}</option>`).join('');
    m2.value = det.value;
    m2.dataset.value = det.value;
  }
  const pp = $('prop-preset');
  const flow = get('propagation.flow');
  const assoc = get('propagation.association');
  if (pp && flow && assoc) {
    const samOk = (S.engines || []).find((e) => e.id === 'dam4sam')?.available !== false;
    const cur = `${flow.value}|${assoc.value}`;
    const opts = PROP_PRESETS.map((p) => ({ ...p, id: `${p.flow}|${p.association}` }));
    if (!opts.some((p) => p.id === cur)) opts.push({ id: cur, label: `${flow.choices[flow.value] || flow.value} + ${assoc.choices[assoc.value] || assoc.value}` });
    pp.innerHTML = opts.map((p) => `<option value="${esc(p.id)}" ${p.gpu && !samOk ? 'disabled' : ''}>${esc(p.label)}${p.gpu ? (samOk ? ' · GPU' : ' · chưa cài') : ''}</option>`).join('');
    pp.value = cur;
    pp.dataset.value = cur;
  }
}
async function loadMainModels() {
  try { renderMainModels((await api('/settings')).fields); } catch { /* chưa có dự án: để trống */ }
}
async function waitRelabel() {
  for (;;) {
    const job = await api('/relabel');
    if (job.state !== 'running') return job;
    await new Promise((r) => setTimeout(r, 1500));
  }
}
$('model2d')?.addEventListener('change', async (e) => {
  const sel = e.target;
  const name = sel.options[sel.selectedIndex].textContent;
  sel.disabled = true;
  try {
    const r = await api('/settings', { method: 'PUT', body: { values: { 'detection.yoloe.weights': sel.value } } });
    renderMainModels(r.fields);
    toast(`Mô hình ${name}: đang áp dụng lại cho các frame chưa ai mở…`);
    await api('/relabel', { method: 'POST', body: {} });
    const job = await waitRelabel();
    if (job.state === 'error') throw new Error(job.message);
    toast(`Mô hình ${name}: đã áp dụng lại ${job.result?.relabeled ?? 0} frame`);
    await refreshLists();
    if (S.frame) openFrame(S.frame.frame_id, { preview: true }).catch(() => {});
  } catch (err) {
    sel.value = sel.dataset.value || sel.value;
    toast(err.message, true);
    loadMainModels();
  } finally { sel.disabled = false; }
});
$('prop-preset')?.addEventListener('change', async (e) => {
  const sel = e.target;
  const [flow, association] = sel.value.split('|');
  try {
    const r = await api('/settings', { method: 'PUT', body: { values: { 'propagation.flow': flow, 'propagation.association': association } } });
    renderMainModels(r.fields);
    toast(`Lan truyền: ${sel.options[sel.selectedIndex].textContent.replace(/ · .*/, '')}`);
  } catch (err) {
    sel.value = sel.dataset.value || sel.value;
    toast(err.message, true);
  }
});

const JOB_POLL = {};
async function loadSettings() {
  const r = await api('/settings');
  S.settings = r.fields;
  S.gtFrames = r.gt_frames;
  $('settings-scope').textContent = PROJECT ? `Dự án: ${$('project-name').textContent || PROJECT}` : 'Workspace mặc định';
  $('settings-form').innerHTML = r.fields.map(settingRow).join('');
  $('settings-save').disabled = true;
  $('eval-run').disabled = !r.gt_frames;
  $('eval-run').title = r.gt_frames ? `${r.gt_frames} frame có GT` : 'Cần dữ liệu có nhãn gốc (nuScenes có sample_annotation)';
  pollJob('relabel', true);
  pollJob('eval', true);
}
function fmtSetting(f, v) {
  if (f.kind === 'choice') return f.choices[v] || v;
  if (f.kind === 'bool') return v ? 'bật' : 'tắt';
  return String(v);
}
function settingRow(f, i, all) {
  // Dòng gọn: tên + ⓘ (rê chuột xem số đo làm căn cứ) + ghi chú một dòng; ô chọn bên phải
  const id = 'set-' + f.path.replace(/\./g, '-');
  let control;
  if (f.kind === 'choice') {
    control = `<select id="${id}" data-path="${esc(f.path)}">${Object.entries(f.choices).map(([k, lab]) => `<option value="${esc(k)}" ${k === f.value ? 'selected' : ''}>${esc(lab)}</option>`).join('')}</select>`;
  } else if (f.kind === 'bool') {
    control = `<label class="set-control"><input type="checkbox" id="${id}" data-path="${esc(f.path)}" ${f.value ? 'checked' : ''}> Bật</label>`;
  } else {
    control = `<input type="number" id="${id}" data-path="${esc(f.path)}" value="${f.value}" min="${f.min}" max="${f.max}" step="${f.step || 1}">`;
  }
  const group = f.group && (!i || all[i - 1].group !== f.group) ? `<h4 class="set-group">${esc(f.group)}</h4>` : '';
  const info = f.detail ? `<span class="set-info tip" tabindex="0" data-tip="${esc(f.detail)}"><i class="ri-information-line"></i></span>` : '';
  const tags = (f.changed ? `<span class="set-meta changed tip" data-tip="Mặc định: ${esc(fmtSetting(f, f.default))}">đã đổi</span>` : '')
    + (f.applies === 'relabel' ? '<span class="set-meta tip" data-tip="Có hiệu lực với frame gán nhãn mới. Frame cũ chưa ai mở: bấm Áp dụng lại.">cần Áp dụng lại</span>' : '');
  return `${group}<div class="set-row">
    <div class="set-name"><label for="${id}">${esc(f.label)}</label>${info}${tags}${f.help ? `<span class="set-help">${esc(f.help)}</span>` : ''}</div>
    <div>${control}</div>
  </div>`;
}
function settingValues() {
  const out = {};
  document.querySelectorAll('#settings-form [data-path]').forEach((el) => {
    const f = S.settings.find((x) => x.path === el.dataset.path);
    out[f.path] = f.kind === 'bool' ? el.checked : f.kind === 'choice' ? el.value : Number(el.value);
  });
  return out;
}
$('settings-form').addEventListener('input', () => { $('settings-save').disabled = false; });
$('settings-form').addEventListener('submit', (e) => e.preventDefault());
$('settings-save').addEventListener('click', async () => {
  try {
    const r = await api('/settings', { method: 'PUT', body: { values: settingValues() } });
    S.settings = r.fields;
    $('settings-form').innerHTML = r.fields.map(settingRow).join('');
    $('settings-save').disabled = true;
    renderMainModels(r.fields);
    const needs = r.fields.some((f) => f.applies === 'relabel' && f.changed);
    toast(needs ? 'Đã lưu. Bấm "Áp dụng lại" để frame chưa mở dùng cài đặt mới.' : 'Đã lưu.');
  } catch (err) { toast(err.message, true); }
});
$('settings-reset').addEventListener('click', async () => {
  try {
    const r = await api('/settings', { method: 'DELETE' });
    S.settings = r.fields;
    $('settings-form').innerHTML = r.fields.map(settingRow).join('');
    $('settings-save').disabled = true;
    renderMainModels(r.fields);
    toast('Đã về cài đặt mặc định.');
  } catch (err) { toast(err.message, true); }
});

const JOB_URL = { relabel: '/relabel', eval: '/eval-temporal' };
function renderJob(kind, job) {
  const box = $(kind + '-status');
  box.classList.toggle('error', job.state === 'error');
  $(kind + '-run').disabled = job.state === 'running' || (kind === 'eval' && !S.gtFrames);
  if (job.state === 'running') {
    box.innerHTML = `<div class="progress"><div class="job-bar" style="width:${(job.progress || 0) * 100}%"></div></div><span>${esc(job.message || '')}</span>`;
  } else if (job.state === 'error') {
    box.textContent = 'Lỗi: ' + job.message;
  } else if (job.state === 'done' && kind === 'relabel' && job.result) {
    const k = job.result.skipped;
    box.innerHTML = `✓ Đã áp dụng lại ${job.result.relabeled} frame. Bỏ qua: ${k.opened} đã mở / duyệt, ${k.propagated} có nhãn lan truyền, ${k.sweep_edited} có sweep đã sửa.`;
  } else if (job.state === 'done' && kind === 'eval') {
    box.textContent = '✓ Xong — xem bảng bên dưới.';
  } else box.textContent = '';
}
async function pollJob(kind, once = false) {
  clearTimeout(JOB_POLL[kind]);
  try {
    const r = await api(JOB_URL[kind]);
    const job = kind === 'eval' ? r.job : r;
    renderJob(kind, job);
    if (kind === 'eval') renderEvalResult(r.last);
    if (job.state === 'running' && !$('tab-settings').classList.contains('hidden')) JOB_POLL[kind] = setTimeout(() => pollJob(kind), 1500);
    else if (job.state === 'done' && !once && kind === 'relabel') { refreshLists(); if (S.frame) openFrame(S.frame.frame_id, { preview: true }).catch(() => {}); }
  } catch (err) { if (!once) toast(err.message, true); }
}
async function startJob(kind) {
  try {
    const job = await api(JOB_URL[kind], { method: 'POST', body: {} });
    renderJob(kind, job);
    JOB_POLL[kind] = setTimeout(() => pollJob(kind), 1000);
  } catch (err) { toast(err.message, true); }
}
$('relabel-run').addEventListener('click', () => startJob('relabel'));
$('eval-run').addEventListener('click', () => startJob('eval'));

const VARIANT_NAME = {
  base: 'Trước (không flow, score detector)', flow: 'So khớp sweep bằng flow', 'flow+mean': 'Flow + score trung bình 5 ảnh',
  'flow+linked': 'Flow + score trung bình các lần thấy', mean: 'Score trung bình 5 ảnh (không flow)',
};
const PROP_NAME = { off: 'Đoán theo vận tốc (trước)', missing: 'Flow ở ảnh chưa detect', always: 'Flow ở mọi ảnh', 'always+byte': 'Flow ở mọi ảnh + ghép 2 lượt (ByteTrack)' };
function renderEvalResult(last) {
  if (!last) {
    $('eval-result-note').textContent = 'Chưa chạy.';
    $('eval2d-table').innerHTML = $('evalprop-table').innerHTML = '';
    return;
  }
  const n = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d));
  const frames = last.rows2d[0]?.frames ?? '—';
  $('eval-result-note').innerHTML = `Chạy xong lúc ${esc(last.finished_at || '—')} trên ${frames} keyframe có GT. <b>Xem tay</b> = box rủi ro medium/high; <b>lọt qua</b> = box sai nằm trong nhóm low (duyệt theo lô không xem); <b>vẽ thêm</b> = vật có trong GT nhưng máy sót; <b>box sai</b> = phải xoá / sửa.`;
  const base = last.rows2d.find((r) => r.name === 'base');
  const delta = (v, b, lowerBetter) => {
    if (!base || v == null || b == null || v === b) return '';
    const good = lowerBetter ? v < b : v > b;
    return ` <span class="${good ? 'delta-good' : 'delta-bad'}">(${v > b ? '+' : ''}${typeof v === 'number' && !Number.isInteger(v) ? (v - b).toFixed(3) : v - b})</span>`;
  };
  $('eval2d-table').innerHTML = '<thead><tr><th>Cách xử lý 2D</th><th class="num">mAP50</th><th class="num">F1</th><th class="num">Xem tay</th><th class="num">Lọt qua</th><th class="num">Vẽ thêm</th><th class="num">Box sai</th><th class="num">s / frame</th></tr></thead><tbody>' +
    last.rows2d.map((r) => `<tr><td>${esc(VARIANT_NAME[r.name] || r.name)}</td><td class="num">${n(r.mAP50)}${delta(r.mAP50, base?.mAP50)}</td><td class="num">${n(r.f1)}</td>
      <td class="num">${r.review}${delta(r.review, base?.review, true)}</td><td class="num">${r.slip}${delta(r.slip, base?.slip, true)}</td>
      <td class="num">${r.fn}${delta(r.fn, base?.fn, true)}</td><td class="num">${r.fp}${delta(r.fp, base?.fp, true)}</td><td class="num">${n(r.s_per_frame, 2)}</td></tr>`).join('') + '</tbody>';
  const off = last.propagation.find((r) => r.mode === 'off');
  $('evalprop-table').innerHTML = '<thead><tr><th>Lan truyền nhãn</th><th class="num">Nhãn đúng</th><th class="num">Box ra</th><th class="num">Tỉ lệ đúng</th><th class="num">Đổi ID</th><th class="num">Mất dấu</th><th class="num">IoU TB</th><th class="num">Giây</th></tr></thead><tbody>' +
    (last.propagation.map((r) => `<tr><td>${esc(PROP_NAME[r.mode] || r.mode)}</td><td class="num">${r.correct}${off && r !== off ? ` <span class="${r.correct >= off.correct ? 'delta-good' : 'delta-bad'}">(${r.correct - off.correct >= 0 ? '+' : ''}${r.correct - off.correct})</span>` : ''}</td>
      <td class="num">${r.outputs}</td><td class="num">${n(r.correct_rate)}</td><td class="num">${r.id_switch}</td><td class="num">${r.lost}</td><td class="num">${n(r.mean_iou)}</td><td class="num">${n(r.seconds, 1)}</td></tr>`).join('') ||
      '<tr><td colspan="8" class="muted">Không có scene nào đủ keyframe để thử lan truyền</td></tr>') + '</tbody>';
}

// ---------- thời gian từng bước ----------
async function loadTiming() {
  const t = await api('/timing');
  const n = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d));
  const rows = [];
  const section = (title) => rows.push(`<tr class="timing-sec"><td colspan="5">${esc(title)}</td></tr>`);
  if (t.steps.length) {
    section('Các bước của dự án (tổng thời gian ÷ số frame)');
    t.steps.forEach((s) => rows.push(`<tr><td>${esc(s.label)}</td><td class="num">${s.frames ?? '—'}</td><td class="num"><b>${n(s.s_per_frame)}</b></td><td class="num">${n(s.seconds, 0)}</td><td></td></tr>`));
  }
  if (t.rows2d.length) {
    section(`Gán nhãn 2D — ${t.frames2d} frame, TB ${n(t.mean_total2d)} s/frame, ${n(t.model_images_per_frame, 1)} ảnh chạy model/frame (0 = dùng cache)`);
    t.rows2d.forEach((r) => rows.push(`<tr><td>${esc(r.name)}</td><td class="num">${r.frames}</td><td class="num"><b>${n(r.mean_s)}</b></td><td class="num">${n(r.total_s, 1)}</td><td>${meterCell(r.share)}</td></tr>`));
  }
  Object.entries(t.rows3d).forEach(([model, v]) => {
    section(`Kiểm chứng 3D (${model}) — ${v.frames} frame`);
    v.rows.forEach((r) => rows.push(`<tr><td>${esc(r.name)}</td><td class="num">${r.frames}</td><td class="num"><b>${n(r.mean_s)}</b></td><td class="num">${n(r.total_s, 1)}</td><td>${meterCell(r.share)}</td></tr>`));
  });
  const srv = Object.entries(t.server);
  if (srv.length) {
    section('Việc của server từ lúc khởi động (không gắn với frame)');
    srv.forEach(([, v]) => rows.push(`<tr><td>${esc(v.name)}</td><td class="num">${v.n} lần</td><td class="num"><b>${n(v.mean_s)}</b></td><td class="num">${n(v.total_s, 1)}</td><td class="muted">lâu nhất ${n(v.max_s)} s</td></tr>`));
  }
  $('timing-table').innerHTML = rows.length
    ? '<thead><tr><th>Bước</th><th class="num">Số frame / lần</th><th class="num">Giây TB</th><th class="num">Tổng (s)</th><th>Phần trăm</th></tr></thead><tbody>' + rows.join('') + '</tbody>'
    : '<tbody><tr><td class="muted">Chưa có số đo: frame auto-label trước bản này chưa ghi thời gian từng bước. Chạy lại (tab ⚙ Cài đặt → Áp dụng lại, hoặc tạo dự án mới).</td></tr></tbody>';
}
$('timing-refresh').addEventListener('click', () => loadTiming().catch((e) => toast(e.message, true)));

function stepFrame(delta) {
  if (S.viewMode === 'video') { stepVideo(delta); return; }
  if (!S.queue.length) return;
  const i = S.queue.findIndex((f) => f.frame_id === S.frame?.frame_id);
  const next = S.queue[(i + delta + S.queue.length) % S.queue.length];
  if (next) openFrame(next.frame_id);
}

function stepSweepBox(delta) {
  const boxes = sweepBoxes(curSweep());
  if (!boxes.length) return;
  const i = boxes.findIndex((b) => b.box_id === S.sweepSel);
  S.sweepSel = boxes[(i + delta + boxes.length) % boxes.length].box_id;
  renderPanel();
  draw();
  document.querySelector(`[data-sbid="${CSS.escape(S.sweepSel)}"]`)?.scrollIntoView({ block: 'nearest' });
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
  if (e.key === 'Escape') {
    if (S.mode === 'view' && inSweep() && !inField) setView(0);
    cancelEdit();
    document.activeElement?.blur();
    return;
  }
  if (inField) {
    if (e.key === 'Enter' && document.activeElement.id === 'add-class') saveEdit();
    return;
  }
  if (!S.frame) return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (S.playback && key !== ' ') { stopPlay(); e.preventDefault(); return; } // phím bất kỳ: chỉ dừng phát
  if ((e.ctrlKey || e.metaKey) && (key === 'z' || key === 'y')) {
    e.preventDefault();
    undoRedo(key === 'y' || e.shiftKey ? 'redo' : 'undo');
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  // Điều hướng: dùng được cả khi frame đã approve
  const nav = {
    ArrowDown: () => (inSweep() ? stepSweepBox(1) : stepObject(1)),
    ArrowUp: () => (inSweep() ? stepSweepBox(-1) : stepObject(-1)),
    n: () => stepFrame(1),
    p: () => stepFrame(-1),
    l: () => $('show-lidar').click(),
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

  if (inSweep()) {
    const sb = S.sweepSel;
    const sweepKeys = {
      k: () => sb && sweepAct({ action: 'KEEP', box_id: sb }),
      d: () => sb && sweepAct({ action: 'DELETE', box_id: sb }),
      Delete: () => sb && sweepAct({ action: 'DELETE', box_id: sb }),
      c: () => sb && document.querySelector(`[data-sbid="${CSS.escape(sb)}"] [data-sclass]`)?.focus(),
      e: startEdit,
      b: startAdd,
      m: startClick,
      Enter: () => S.mode !== 'view' && S.editBox && saveEdit(),
    };
    if (sweepKeys[key]) { sweepKeys[key](); e.preventDefault(); }
    return;
  }
  const sel = S.selected;
  const edit = {
    k: () => sel && act({ action: 'KEEP', object_id: sel }),
    d: () => sel && act({ action: 'DELETE', object_id: sel }),
    Delete: () => sel && act({ action: 'DELETE', object_id: sel }),
    c: () => sel && document.querySelector(`[data-oid="${CSS.escape(sel)}"] [data-class]`)?.focus(),
    e: startEdit,
    b: startAdd,
    m: startClick,
    a: approveLow,
    Enter: () => (S.mode !== 'view' && S.editBox ? saveEdit() : approveFrame()),
  };
  if (edit[key]) { edit[key](); e.preventDefault(); }
});

// Đổi lớp bằng bàn phím: C -> chọn trong dropdown -> Enter
document.querySelector('.review-panel').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.target.matches('[data-sclass]')) {
    sweepAct({ action: 'CHANGE_CLASS', box_id: e.target.closest('[data-sbid]').dataset.sbid, label: e.target.value });
    return;
  }
  if (e.key !== 'Enter' || !e.target.matches('[data-class]')) return;
  const card = e.target.closest('[data-oid]');
  act({ action: 'CHANGE_CLASS', object_id: card.dataset.oid, label: e.target.value });
});

// Đổi lớp bằng chuột: thẻ object kiểu mới không còn nút "Đổi lớp", chọn trong dropdown là áp dụng ngay.
// Bàn phím vẫn theo luồng cũ (C -> mũi tên -> Enter) nên chỉ nhận change đến từ chuột.
(() => {
  const panel = document.querySelector('.review-panel');
  let byPointer = false;
  panel.addEventListener('pointerdown', (e) => { if (e.target.closest('.obj-card [data-class]')) byPointer = true; });
  panel.addEventListener('keydown', (e) => { if (e.target.matches('.obj-card [data-class]')) byPointer = false; });
  panel.addEventListener('change', (e) => {
    if (!byPointer || !e.target.matches('.obj-card [data-class]')) return;
    byPointer = false;
    act({ action: 'CHANGE_CLASS', object_id: e.target.closest('[data-oid]').dataset.oid, label: e.target.value });
  });
})();

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
$('prop-engine')?.addEventListener('change', (e) => {
  storageSet('propEngine', e.target.value);
  e.target.title = engineInfo().detail || '';
  updateTopModelSwitch(e.target.value);
});
$('topbar-model-switch')?.addEventListener('click', (e) => {
  const btn = e.target.closest('.model-btn');
  if (btn && btn.dataset.engine) setEngine(btn.dataset.engine);
});
$('auto-prop').addEventListener('change', (e) => { S.autoProp = e.target.checked; storageSet('autoProp', S.autoProp ? '1' : '0'); });
$('btn-add').addEventListener('click', () => (S.mode === 'add' ? cancelEdit() : startAdd()));
$('btn-click').addEventListener('click', () => (S.mode === 'click' ? cancelEdit() : startClick()));
$('tool-poly')?.addEventListener('click', () => (S.mode === 'click' ? cancelEdit() : startClick()));
canvas.addEventListener('contextmenu', (e) => { if (S.mode === 'click') e.preventDefault(); });
$('edit-save').addEventListener('click', saveEdit);
$('edit-cancel').addEventListener('click', cancelEdit);
$('log-refresh').addEventListener('click', loadLog);
$('log-frame').addEventListener('change', loadLog);
$('metrics-refresh').addEventListener('click', loadMetrics);
$('btn-export').addEventListener('click', doExport);
$('reviewer').addEventListener('change', (e) => storageSet('reviewer', e.target.value.trim()));
new ResizeObserver(() => { fitCanvas(); draw(); }).observe($('canvas-wrap'));
// trạng thái dùng chung với app3d.js
window.AL = { get S() { return S; }, select, ensureLidar, renderProductivity };

// Dự án: hiện tên + link quay lại, ẩn chế độ không có dữ liệu (không LiDAR -> không 3D)
// ---------- nhiều người dùng: người đăng nhập, ai đang mở frame nào, khoá frame ----------
const initials = (name) => String(name || '?').trim().split(/\s+/).map((w) => w[0]).slice(-2).join('').toUpperCase();

async function initUser() {
  try {
    const me = await fetch('/api/v1/auth/me').then((r) => r.json());
    S.me = me.user;
    if (me.auth_required && !S.me) { location.href = '/ui/login.html'; return; }
  } catch { S.me = null; }
  const box = $('reviewer');
  if (S.me) {
    // Tên người duyệt lấy theo tài khoản (ghi vào lịch sử sửa / corrections.jsonl), không gõ tay
    box.value = S.me.name;
    box.readOnly = true;
    box.title = `${S.me.name} (${S.me.email}) — đăng xuất ở trang Dự án`;
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.id = 'chip-mine';
    chip.textContent = 'Của tôi';
    chip.title = 'Chỉ frame được giao cho tôi (chia việc ở trang Dự án → Thành viên)';
    chip.addEventListener('click', (e) => { e.stopPropagation(); S.mineOnly = !S.mineOnly; chip.classList.toggle('active', S.mineOnly); loadQueue(); });
    $('queue-filter').appendChild(chip);
  }
  await refreshPresence();
  setInterval(refreshPresence, 15000);
  setInterval(() => { if (S.frame && S.me && !S.lockedBy) acquireLock(S.frame.frame_id, true); }, 30000); // gia hạn khoá
  window.addEventListener('beforeunload', () => {
    if (S.frame && S.me && !S.lockedBy) fetch(`${API}/frames/${encodeURIComponent(S.frame.frame_id)}/lock`, { method: 'DELETE', keepalive: true }).catch(() => {});
  });
}

async function refreshPresence() {
  try {
    const p = await api('/presence');
    S.presence = { locks: p.locks || {}, assignments: p.assignments || {}, users: p.users || {} };
    if (p.me && !S.me) S.me = p.me;
    renderQueue();
    if (S.frame) renderLockBanner();
  } catch { /* server cũ */ }
}

async function acquireLock(id, quiet = false) {
  if (!S.me) return;
  try {
    await api(`/frames/${encodeURIComponent(id)}/lock`, { method: 'POST' });
    if (S.lockedBy) { S.lockedBy = null; renderLockBanner(); }
    S.presence.locks[id] = { user_id: S.me.id, name: S.me.name, at: Date.now() / 1000 };
  } catch (err) {
    if (err.code === 'FRAME_LOCKED') {
      const holder = S.presence.locks[id];
      S.lockedBy = holder?.name || 'người khác';
      renderLockBanner();
      if (!quiet) toast(`${S.lockedBy} đang mở frame này: bạn chỉ xem được`, true);
    }
  }
}

function renderLockBanner() {
  let b = $('lock-banner');
  if (!b) {
    b = document.createElement('div');
    b.id = 'lock-banner';
    b.className = 'lock-banner hidden';
    $('reject-banner').insertAdjacentElement('beforebegin', b);
    b.addEventListener('click', async (e) => {
      if (!e.target.closest('[data-take]') || !S.frame) return;
      try {
        await api(`/frames/${encodeURIComponent(S.frame.frame_id)}/lock?force=true`, { method: 'POST' });
        S.lockedBy = null; renderLockBanner(); refreshPresence(); toast('Bạn đã lấy quyền sửa frame này');
      } catch (err) { toast(err.message, true); }
    });
  }
  const locked = !!S.lockedBy && S.viewMode !== '3d';
  b.classList.toggle('hidden', !locked);
  document.body.classList.toggle('readonly', locked);
  if (locked) b.innerHTML = `🔒 <b>${esc(S.lockedBy)}</b> đang mở frame này — bạn chỉ xem. <button class="btn btn-ghost btn-sm" data-take title="Lấy quyền sửa (người kia sẽ bị chuyển sang chỉ xem)">Lấy quyền sửa</button>`;
}

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
    await initUser();
    loadEngines();
    S.autoProp = storageGet('autoProp', '1') === '1';
    $('auto-prop').checked = S.autoProp;
    $('add-class').innerHTML = classOptions('car');
    let modes = ['image', 'video', '3d'];
    if (PROJECT) modes = await initProject();
    const urlMode = new URLSearchParams(location.search).get('mode');
    const saved = modes.includes(urlMode) ? urlMode : storageGet('viewMode', modes[0]);
    await setMode(modes.includes(saved) ? saved : modes[0]);
    if (new URLSearchParams(location.search).get('tab') === 'settings') switchTab('settings');
  } catch (err) {
    toast('Không tải được dữ liệu: ' + err.message, true);
  }
})();

// Expose globally for topbar & left toolbar controls
window.S = S;
window.draw = draw;
window.setZoom = setZoom;
window.cancelEdit = cancelEdit;
window.startAdd = startAdd;
window.openFrame = openFrame;
window.togglePlay = togglePlay;
window.renderPanel = renderPanel;
window.renderLabelsTab = renderLabelsTab;
window.renderIssuesTab = renderIssuesTab;
window.select = select;
window.toast = toast;
window.reviewer = reviewer;
window.ensureLidar = ensureLidar;

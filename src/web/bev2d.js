// BEV cho chế độ Ảnh / Video: ảnh camera chiếu xuống mặt đường (homography) + box 2D đặt lên mặt đường.
// Vị trí box: theo điểm LiDAR rơi trong box (nét liền), không có LiDAR thì theo chân vật chạm đường (nét đứt).
// Dùng chung trạng thái với app.js qua window.AL (S, select, ensureLidar).

const API = '/api/v1';
const $ = (id) => document.getElementById(id);
const RISK = { low: '#0ca30c', medium: '#fab219', high: '#d03b3b' };
const HUMAN = '#3987e5';
// chiều dài trung bình (m) của từng lớp trên nuScenes: box 2D chỉ cho biết mặt trước của vật
const LENGTH = { car: 4.6, truck: 6.9, bus: 11, trailer: 12, construction_vehicle: 6.4, pedestrian: 0.73, motorcycle: 2.1, bicycle: 1.7, traffic_cone: 0.41, barrier: 0.5 };

const B = {
  on: false, fid: null, meta: null, img: null, loading: false, showImg: true, showPts: true,
  view: null, // {x, y, scale}: tâm khung nhìn (m, hệ ego) và px/m
  shapes: [], drag: null,
};
const storage = {
  get: (k, d) => { try { return localStorage.getItem('bev2d.' + k) ?? d; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem('bev2d.' + k, v); } catch { /* chế độ riêng tư */ } },
};

// ---------------------------------------------------------------- hình học
function inv3(m) {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
  const A = e * i - f * h, Bc = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * Bc + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [Bc / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}
function camToEgo(M, p) { // M = cam_from_ego: ego = R^T (p - t)
  const q = [p[0] - M[0][3], p[1] - M[1][3], p[2] - M[2][3]];
  return [0, 1, 2].map((j) => M[0][j] * q[0] + M[1][j] * q[1] + M[2][j] * q[2]);
}
function egoToCam(M, p) { return [0, 1, 2].map((i) => M[i][0] * p[0] + M[i][1] * p[1] + M[i][2] * p[2] + M[i][3]); }
function imgToGround(Hinv, u, v) {
  const x = Hinv[0][0] * u + Hinv[0][1] * v + Hinv[0][2];
  const y = Hinv[1][0] * u + Hinv[1][1] * v + Hinv[1][2];
  const w = Hinv[2][0] * u + Hinv[2][1] * v + Hinv[2][2];
  if (w <= 1e-9) return null; // trên đường chân trời
  return [x / w, y / w];
}
const quantile = (a, q) => { const s = [...a].sort((m, n) => m - n); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

function footprint(o, S) {
  const M = B.meta.cam_from_ego, K = S.frame.intrinsic;
  const [fx, fy, cx, cy] = [K[0][0], K[1][1], K[0][2], K[1][2]];
  const box = S.editBox && S.mode === 'edit' && o.object_id === S.selected ? S.editBox : (o.review.final_bbox || o.bbox);
  const [x1, y1, x2, y2] = box;
  let d = null, src = 'ground';
  if (S.lidar?.u?.length) {
    const mx = (x2 - x1) * 0.15, my = (y2 - y1) * 0.15, ds = [];
    for (let i = 0; i < S.lidar.u.length; i++) {
      const u = S.lidar.u[i], v = S.lidar.v[i];
      if (u >= x1 + mx && u <= x2 - mx && v >= y1 + my && v <= y2 - my) ds.push(S.lidar.d[i]);
    }
    if (ds.length >= 3) { d = quantile(ds, 0.3); src = 'lidar'; } // mặt gần nhất của vật, bỏ điểm nền phía sau
  }
  let P;
  if (d != null) {
    const uc = (x1 + x2) / 2, vc = (y1 + y2) / 2;
    P = camToEgo(M, [((uc - cx) * d) / fx, ((vc - cy) * d) / fy, d]);
  } else {
    const g = imgToGround(B.Hinv, (x1 + x2) / 2, y2);
    if (!g) return null;
    P = [g[0], g[1], 0];
    d = egoToCam(M, P)[2];
    if (!(d > 0.5) || d > 90) return null;
  }
  const C = B.camPos;
  let dx = P[0] - C[0], dy = P[1] - C[1];
  const n = Math.hypot(dx, dy) || 1;
  dx /= n; dy /= n;
  const W = Math.max(((x2 - x1) * d) / fx, 0.3), L = LENGTH[o.review.final_label || o.label] || 1;
  const px = -dy * (W / 2), py = dx * (W / 2);
  const far = [P[0] + dx * L, P[1] + dy * L];
  return {
    id: o.object_id, src, dist: Math.hypot(P[0], P[1]),
    pts: [[P[0] + px, P[1] + py], [far[0] + px, far[1] + py], [far[0] - px, far[1] - py], [P[0] - px, P[1] - py]],
  };
}

// ---------------------------------------------------------------- vẽ
function toScreen(x, y, c) { // hệ ego (x trước, y trái) -> pixel canvas: phía trước ở trên, bên trái ở bên trái
  return [c.width / 2 + (B.view.y - y) * B.view.scale, c.height / 2 + (B.view.x - x) * B.view.scale];
}
function toWorld(sx, sy, c) { return [B.view.x - (sy - c.height / 2) / B.view.scale, B.view.y - (sx - c.width / 2) / B.view.scale]; }
function fitView(c) {
  const [x0, x1] = B.meta?.x_range || [0, 50];
  B.view = { x: (x0 + x1) / 2 - 1, y: 0, scale: c.height / (x1 - x0 + 6) };
}
function heightColor(z) {
  const t = Math.min(1, Math.max(0, (z + 0.2) / 2.8));
  return `hsl(${Math.round(220 - 220 * t)}, 90%, ${Math.round(55 + 10 * t)}%)`;
}

async function ensure(fid) {
  if (B.fid === fid || B.loading === fid) return;
  B.loading = fid;
  try {
    const meta = await fetch(`${API}/frames/${encodeURIComponent(fid)}/bev/meta`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))));
    const img = new Image();
    img.src = `${API}/frames/${encodeURIComponent(fid)}/bev`;
    await img.decode().catch(() => null);
    await window.AL.ensureLidar().catch(() => null);
    if (window.AL.S.frame?.frame_id !== fid) return;
    B.meta = meta; B.img = img.naturalWidth ? img : null; B.fid = fid;
    B.Hinv = inv3(meta.homography);
    B.camPos = camToEgo(meta.cam_from_ego, [0, 0, 0]);
    const c = $('bev2d-canvas');
    if (!B.view) fitView(c);
    $('bev2d-note').textContent = meta.estimated
      ? `Không có calibration: giả định camera cao ${meta.camera_height} m, nhìn thẳng`
      : `camera cao ${meta.camera_height} m · kéo: di chuyển · lăn: zoom · bấm đúp: về mặc định`;
  } catch (e) {
    $('bev2d-note').textContent = 'Không tạo được BEV: ' + e.message;
  } finally {
    if (B.loading === fid) B.loading = false;
  }
  draw();
}

export function draw() {
  if (!B.on) return;
  const S = window.AL?.S;
  const c = $('bev2d-canvas');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
  if (!w || !h) return;
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; if (B.meta) fitView(c); }
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#0b1020'; ctx.fillRect(0, 0, w, h);
  B.shapes = [];
  const f = S?.frame;
  if (!f) return;
  if (B.fid !== f.frame_id) { ensure(f.frame_id); if (!B.meta || B.fid !== f.frame_id) return; }
  if (!B.view) fitView(c);
  const sc = B.view.scale;
  // ảnh BEV
  if (B.showImg && B.img) {
    const [x0, x1] = B.meta.x_range, [y0, y1] = B.meta.y_range;
    const [sx, sy] = toScreen(x1, y1, c);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(B.img, sx, sy, (y1 - y0) * sc, (x1 - x0) * sc);
  }
  // vòng khoảng cách + góc nhìn camera
  ctx.lineWidth = 1 * dpr;
  ctx.font = `${10 * dpr}px system-ui`;
  for (const r of [10, 20, 30, 40, 50]) {
    const [ox, oy] = toScreen(0, 0, c);
    ctx.strokeStyle = 'rgba(148,163,184,.35)';
    ctx.beginPath(); ctx.arc(ox, oy, r * sc, Math.PI, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = 'rgba(203,213,225,.8)';
    ctx.fillText(`${r} m`, ox + 3 * dpr, oy - r * sc + 11 * dpr);
  }
  const K = f.intrinsic, M = B.meta.cam_from_ego;
  const [cxs, cys] = toScreen(B.camPos[0], B.camPos[1], c);
  ctx.strokeStyle = 'rgba(56,189,248,.55)'; ctx.setLineDash([4 * dpr, 4 * dpr]);
  for (const u of [0, f.image.width]) {
    const dir = camToEgo(M, [(u - K[0][2]) / K[0][0], 0, 1]).map((v, i) => v - B.camPos[i]);
    const n = Math.hypot(dir[0], dir[1]) || 1;
    const [ex, ey] = toScreen(B.camPos[0] + (dir[0] / n) * 60, B.camPos[1] + (dir[1] / n) * 60, c);
    ctx.beginPath(); ctx.moveTo(cxs, cys); ctx.lineTo(ex, ey); ctx.stroke();
  }
  ctx.setLineDash([]);
  // điểm LiDAR (đổi ngược từ u, v, độ sâu về hệ ego)
  if (B.showPts && S.lidar?.u?.length) {
    const r = 2 * dpr;
    for (let i = 0; i < S.lidar.u.length; i++) {
      const d = S.lidar.d[i];
      const p = camToEgo(M, [((S.lidar.u[i] - K[0][2]) * d) / K[0][0], ((S.lidar.v[i] - K[1][2]) * d) / K[1][1], d]);
      const [sx, sy] = toScreen(p[0], p[1], c);
      if (sx < 0 || sy < 0 || sx > w || sy > h) continue;
      ctx.fillStyle = heightColor(p[2]);
      ctx.fillRect(sx - r / 2, sy - r / 2, r, r);
    }
  }
  // xe (ego): gốc hệ ego ở trục sau, thân xe từ -1 m tới 3.6 m
  {
    const pts = [[3.6, 0.95], [3.6, -0.95], [-1, -0.95], [-1, 0.95]].map(([x, y]) => toScreen(x, y, c));
    ctx.fillStyle = 'rgba(148,163,184,.9)';
    ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.fill();
  }
  // box
  const sel = S.selected;
  const objs = f.objects.filter((o) => o.review.status !== 'deleted');
  for (const o of objs) {
    const lv = o.qa?.level || 'low';
    if (o.review.status === 'pending' && lv === 'low' && !S.showLow && o.object_id !== sel) continue;
    const fp = footprint(o, S);
    if (!fp) continue;
    B.shapes.push({ ...fp, screen: fp.pts.map(([x, y]) => toScreen(x, y, c)), o });
  }
  B.shapes.sort((a, b) => (a.id === sel) - (b.id === sel)); // box đang chọn vẽ sau cùng
  for (const s of B.shapes) {
    const o = s.o, isSel = o.object_id === sel;
    const color = isSel ? '#ffffff' : o.source === 'human' ? HUMAN : RISK[o.qa?.level || 'low'];
    ctx.strokeStyle = color; ctx.lineWidth = (isSel ? 3 : 1.8) * dpr;
    ctx.setLineDash(s.src === 'lidar' ? [] : [5 * dpr, 3 * dpr]);
    ctx.fillStyle = isSel ? 'rgba(255,255,255,.18)' : color + '33';
    ctx.beginPath(); s.screen.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
    ctx.fill(); ctx.stroke();
    if (isSel || (o.review.status === 'pending' && (o.qa?.level === 'high'))) {
      const label = `#${o.object_id} ${o.review.final_label || o.label} · ${s.dist.toFixed(1)} m`;
      const [lx, ly] = s.screen[0];
      ctx.setLineDash([]);
      ctx.font = `${11 * dpr}px system-ui`;
      const tw = ctx.measureText(label).width + 8 * dpr;
      ctx.fillStyle = 'rgba(15,23,42,.85)'; ctx.fillRect(lx, ly + 3 * dpr, tw, 16 * dpr);
      ctx.fillStyle = '#fff'; ctx.fillText(label, lx + 4 * dpr, ly + 15 * dpr);
    }
  }
  ctx.setLineDash([]);
}

// ---------------------------------------------------------------- tương tác
function inside(pt, poly) {
  let ok = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) ok = !ok;
  }
  return ok;
}
function canvasPt(ev) {
  const c = $('bev2d-canvas'), r = c.getBoundingClientRect();
  return [((ev.clientX - r.left) / r.width) * c.width, ((ev.clientY - r.top) / r.height) * c.height];
}
function hit(ev) {
  const p = canvasPt(ev);
  for (let i = B.shapes.length - 1; i >= 0; i--) if (inside(p, B.shapes[i].screen)) return B.shapes[i];
  return null;
}

export function setOn(on) {
  B.on = on;
  storage.set('on', on ? '1' : '0');
  $('show-bev2d').checked = on;
  $('bev2d-wrap').classList.toggle('hidden', !on);
  $('stage-row').classList.toggle('with-bev', on);
  requestAnimationFrame(draw);
}

function bind() {
  const c = $('bev2d-canvas');
  $('show-bev2d').addEventListener('change', (e) => setOn(e.target.checked));
  $('bev2d-img').addEventListener('change', (e) => { B.showImg = e.target.checked; draw(); });
  $('bev2d-pts').addEventListener('change', (e) => { B.showPts = e.target.checked; draw(); });
  c.addEventListener('wheel', (e) => {
    if (!B.view) return;
    e.preventDefault();
    const p = canvasPt(e), before = toWorld(p[0], p[1], c);
    B.view.scale = Math.min(80, Math.max(2, B.view.scale * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));
    const after = toWorld(p[0], p[1], c);
    B.view.x += before[0] - after[0]; B.view.y += before[1] - after[1];
    draw();
  }, { passive: false });
  c.addEventListener('pointerdown', (e) => {
    if (!B.view) return;
    B.drag = { p: canvasPt(e), view: { ...B.view }, moved: false };
    c.setPointerCapture(e.pointerId);
  });
  c.addEventListener('pointermove', (e) => {
    if (B.drag) {
      const p = canvasPt(e), dx = p[0] - B.drag.p[0], dy = p[1] - B.drag.p[1];
      if (Math.hypot(dx, dy) > 3) B.drag.moved = true;
      if (B.drag.moved) { B.view.x = B.drag.view.x + dy / B.view.scale; B.view.y = B.drag.view.y + dx / B.view.scale; draw(); }
      return;
    }
    const s = hit(e);
    c.style.cursor = s ? 'pointer' : 'grab';
    c.title = s ? `#${s.o.object_id} ${s.o.review.final_label || s.o.label} · ${s.dist.toFixed(1)} m · ${s.src === 'lidar' ? 'vị trí theo LiDAR' : 'ước lượng theo chân vật chạm đường'}` : '';
  });
  c.addEventListener('pointerup', (e) => {
    const d = B.drag;
    B.drag = null;
    if (d && !d.moved) { const s = hit(e); if (s) window.AL.select(s.o.object_id); }
  });
  c.addEventListener('dblclick', () => { fitView(c); draw(); });
  new ResizeObserver(() => draw()).observe($('bev2d-wrap'));
}

bind();
window.bev2d = { draw, setOn, toggle: () => setOn(!B.on) };
setOn(storage.get('on', '0') === '1');

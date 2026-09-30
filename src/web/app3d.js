// Chế độ 3D: point cloud LiDAR + box 3D (three.js), chiếu box lên 6 camera, duyệt theo kết luận kiểm chứng.
import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const PROJECT = new URLSearchParams(location.search).get('project');
const BASE = PROJECT ? `/p/${encodeURIComponent(PROJECT)}/api/v1` : '/api/v1';
const API = `${BASE}/3d`;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const LEVELS = ['high', 'medium', 'low'];
const STATUS_TEXT = { auto: 'Chưa mở', editing: 'Đang sửa', approved: 'Đã duyệt', rejected: 'Bị trả lại' };
const LEVEL_NAME = { low: 'Low', medium: 'Medium', high: 'High' };
const COLOR = { low: 0x0ca30c, medium: 0xfab219, high: 0xd03b3b, approved: 0x3987e5, gt: 0xcbd5e1 };
const CSS_COLOR = { low: '#0ca30c', medium: '#fab219', high: '#d03b3b', approved: '#3987e5' };
const VERDICT_VI = {
  'DUNG': 'Đúng', 'DUNG VAT, BOX LECH': 'Đúng vật, box lệch', 'SAI LOP': 'Sai lớp',
  'NGHI BAO NHAM': 'Nghi báo nhầm', 'CAMERA KHONG XAC NHAN': 'Camera không xác nhận', 'CHUA DU THONG TIN': 'Chưa đủ thông tin',
};
const CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'];
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
// kích thước trung bình [w, l, h] (m) của từng lớp trên nuScenes: box vẽ thêm bắt đầu từ đây
const DEFAULT_SIZE = {
  car: [1.95, 4.6, 1.73], truck: [2.5, 6.9, 2.8], bus: [2.9, 11.0, 3.5], trailer: [2.9, 12.0, 3.9],
  construction_vehicle: [2.8, 6.4, 3.2], pedestrian: [0.67, 0.73, 1.77], motorcycle: [0.77, 2.1, 1.47],
  bicycle: [0.6, 1.7, 1.3], traffic_cone: [0.41, 0.41, 1.07], barrier: [2.5, 0.5, 0.98],
};
const EDIT_HEX = 0x22d3ee, EDIT_CSS = '#22d3ee';
const HINT = 'Kéo chuột trái: xoay · lăn chuột: zoom · chuột phải: di chuyển · bấm vào box để chọn';

const T = {
  active: false, models: [], model: null, queue: [], sort: 'risk', frame: null, points: null, gt: [],
  selected: null, cam: 'CAM_FRONT', camImg: {}, showGt: false, showLow: true, minScore: 0, color: 'height', bev: false,
  timerStart: null, classes: [], showBevImg: true, showPoints: true, groundZ: -1.84,
  tool: 'select', edit: null, // edit: {mode: 'add' | 'edit', oid, label, box: {center, size, yaw}}
};

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' }, ...opts, body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (opts.raw) {
    if (!res.ok) throw new Error(res.statusText);
    return res.arrayBuffer();
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail?.message || res.statusText);
  return data;
}
const toast = (m, err) => (window.toast ? window.toast(m, err) : console.log(m));
const reviewer = () => ($('reviewer')?.value || '').trim() || undefined;
const PREFIX = PROJECT ? `m3.${PROJECT}.` : 'm3.';
const storage = {
  get: (k, d) => { try { return localStorage.getItem(PREFIX + k) ?? d; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(PREFIX + k, v); } catch { /* chế độ riêng tư */ } },
};

// ---------------------------------------------------------------- hình học
function corners(box) {
  const [w, l, h] = box.size;
  const [cx, cy, cz] = box.center;
  const c = Math.cos(box.yaw), s = Math.sin(box.yaw);
  const xs = [1, 1, 1, 1, -1, -1, -1, -1].map((v) => (v * l) / 2);
  const ys = [1, -1, -1, 1, 1, -1, -1, 1].map((v) => (v * w) / 2);
  const zs = [1, 1, -1, -1, 1, 1, -1, -1].map((v) => (v * h) / 2);
  return xs.map((x, i) => [cx + c * x - s * ys[i], cy + s * x + c * ys[i], cz + zs[i]]);
}
const finalLabel = (o) => o.review.final_label || o.label;
const levelOf = (o) => o.verify?.level || 'medium';
function stateColor(o) {
  if (o.review.status === 'approved') return 'approved';
  return levelOf(o);
}

// ---------------------------------------------------------------- three.js
const V = { renderer: null, scene: null, cam3d: null, camBev: null, controls: null, pointsObj: null, boxGroup: null, gtGroup: null, pickables: [] };

function initViewer() {
  const canvas = $('m3-canvas');
  V.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  V.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  V.scene = new THREE.Scene();
  V.scene.background = new THREE.Color(0x0b1020);
  V.cam3d = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
  // hệ LiDAR nuScenes: x sang phải, y về phía trước, z lên
  V.cam3d.up.set(0, 0, 1);
  V.cam3d.position.set(-12, -22, 16);
  V.camBev = new THREE.OrthographicCamera(-40, 40, 25, -25, 0.1, 500);
  V.camBev.up.set(0, 1, 0); // đầu xe hướng lên trên màn hình
  V.camBev.position.set(0, 0, 120);
  // bắt pointerdown ở pha capture (trước OrbitControls) để tắt xoay khi đang vẽ / kéo box
  $('m3-stage').addEventListener('pointerdown', onPointerDown, { capture: true });
  V.controls = new OrbitControls(V.cam3d, canvas);
  V.controls.target.set(0, 8, 0);
  V.controls.update();
  V.boxGroup = new THREE.Group();
  V.gtGroup = new THREE.Group();
  V.editGroup = new THREE.Group();
  V.scene.add(V.boxGroup, V.gtGroup, V.editGroup);
  // xe (ego) + vòng 10 m
  const ego = new THREE.Mesh(new THREE.BoxGeometry(1.9, 4.5, 1.6), new THREE.MeshBasicMaterial({ color: 0x94a3b8, wireframe: true }));
  ego.position.set(0, 0, -0.9);
  V.scene.add(ego);
  for (const r of [10, 20, 30, 40, 50]) {
    const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
      Array.from({ length: 96 }, (_, i) => new THREE.Vector3(r * Math.cos((i / 96) * 2 * Math.PI), r * Math.sin((i / 96) * 2 * Math.PI), -1.8)),
    ), new THREE.LineBasicMaterial({ color: 0x1e293b }));
    V.scene.add(ring);
  }
  new ResizeObserver(resize).observe($('m3-stage'));
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointerleave', () => $('m3-tip').classList.add('hidden'));
  (function loop() {
    requestAnimationFrame(loop);
    if (!T.active) return;
    V.controls.update();
    V.renderer.render(V.scene, T.bev ? V.camBev : V.cam3d);
  })();
}

function resize() {
  const stage = $('m3-stage');
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h || !V.renderer) return;
  V.renderer.setSize(w, h, false);
  V.cam3d.aspect = w / h;
  V.cam3d.updateProjectionMatrix();
  const half = 32;
  V.camBev.left = -half * (w / h); V.camBev.right = half * (w / h); V.camBev.top = half; V.camBev.bottom = -half;
  V.camBev.updateProjectionMatrix();
  drawCamera();
}

function setView(bev) {
  T.bev = bev;
  storage.set('bev', bev ? '1' : '0');
  $('m3-view-3d').classList.toggle('active', !bev);
  $('m3-view-bev').classList.toggle('active', bev);
  V.controls.object = bev ? V.camBev : V.cam3d;
  V.controls.enableRotate = !bev;
  if (bev) { V.controls.target.set(V.camBev.position.x, V.camBev.position.y, 0); }
  V.controls.update();
}

function colormap(t) {
  // gần với "turbo": xanh dương -> xanh lá -> vàng -> đỏ
  t = Math.min(1, Math.max(0, t));
  const r = Math.min(1, Math.max(0, 1.6 * t - 0.3));
  const g = Math.min(1, Math.max(0, 1.4 - Math.abs(2.2 * t - 1.2)));
  const b = Math.min(1, Math.max(0, 1.1 - 1.8 * t));
  return [0.25 + 0.75 * r, 0.25 + 0.75 * g, 0.35 + 0.65 * b];
}

function buildPoints() {
  if (V.pointsObj) { V.scene.remove(V.pointsObj); V.pointsObj.geometry.dispose(); V.pointsObj = null; }
  if (!T.points) return;
  const n = T.points.length / 4;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = T.points[i * 4]; pos[i * 3 + 1] = T.points[i * 4 + 1]; pos[i * 3 + 2] = T.points[i * 4 + 2];
    const t = T.color === 'intensity' ? Math.sqrt(T.points[i * 4 + 3] / 100) : (T.points[i * 4 + 2] + 2.2) / 4.5;
    const [r, g, b] = colormap(t);
    col[i * 3] = r; col[i * 3 + 1] = g; col[i * 3 + 2] = b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  V.pointsObj = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.09, vertexColors: true }));
  V.pointsObj.visible = T.showPoints;
  V.scene.add(V.pointsObj);
}

function wireframe(box, color, opts = {}) {
  const k = corners(box);
  const pts = [];
  for (const [a, b] of EDGES) pts.push(new THREE.Vector3(...k[a]), new THREE.Vector3(...k[b]));
  // vạch hướng đầu xe: tâm -> giữa mặt trước
  const f = [0, 1, 2, 3].reduce((acc, i) => acc.map((v, j) => v + k[i][j] / 4), [0, 0, 0]);
  pts.push(new THREE.Vector3(...box.center), new THREE.Vector3(...f));
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = opts.dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.3, gapSize: 0.25, transparent: true, opacity: opts.opacity ?? 1 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity: opts.opacity ?? 1 });
  const line = new THREE.LineSegments(geo, mat);
  if (opts.dashed) line.computeLineDistances();
  return line;
}

function buildBoxes() {
  V.boxGroup.clear();
  V.pickables = [];
  if (!T.frame) return;
  for (const o of T.frame.objects) {
    if (o.review.status === 'deleted' || o.object_id === T.edit?.oid) continue;
    const lv = levelOf(o);
    if (!T.showLow && lv === 'low' && o.review.status === 'pending' && o.object_id !== T.selected) continue;
    if (o.source !== 'human' && o.score < T.minScore && o.object_id !== T.selected) continue;
    const sel = o.object_id === T.selected;
    const line = wireframe(o.box, sel ? 0xffffff : COLOR[stateColor(o)], { opacity: sel ? 1 : 0.95 });
    V.boxGroup.add(line);
    if (sel) {
      // khối mờ quanh box đang chọn
      const [w, l, h] = o.box.size;
      const m = new THREE.Mesh(new THREE.BoxGeometry(l, w, h), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.12, depthWrite: false }));
      m.position.set(...o.box.center); m.rotation.z = o.box.yaw;
      V.boxGroup.add(m);
    }
    const [w, l, h] = o.box.size;
    const pick = new THREE.Mesh(new THREE.BoxGeometry(l + 0.3, w + 0.3, h + 0.3), new THREE.MeshBasicMaterial({ visible: false }));
    pick.position.set(...o.box.center); pick.rotation.z = o.box.yaw;
    pick.userData.oid = o.object_id;
    V.boxGroup.add(pick);
    V.pickables.push(pick);
  }
  V.gtGroup.clear();
  if (T.showGt) for (const g of T.gt) V.gtGroup.add(wireframe(g, COLOR.gt, { dashed: true, opacity: g.num_pts > 0 ? 0.9 : 0.35 }));
  buildEdit();
}

const ray = new THREE.Raycaster();
function pickAt(ev) {
  const r = $('m3-canvas').getBoundingClientRect();
  const p = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(p, T.bev ? V.camBev : V.cam3d);
  const hit = ray.intersectObjects(V.pickables, false)[0];
  return hit ? T.frame.objects.find((o) => o.object_id === hit.object.userData.oid) : null;
}
let downAt = null;
let drag = null; // {kind: 'orbit' | 'draw' | 'move' | 'rotate' | 'corner', ...}

function onPointerDown(ev) {
  if (ev.target !== $('m3-canvas') || ev.button !== 0 || !T.frame) return;
  downAt = [ev.clientX, ev.clientY];
  const g = groundAt(ev);
  const grab = (d) => { drag = d; V.controls.enabled = false; $('m3-canvas').setPointerCapture(ev.pointerId); };
  if (T.tool === 'draw' && g) {
    const def = DEFAULT_SIZE[newClass()] || [1, 1, 1];
    startEditState('add', null, newClass(), { center: [g[0], g[1], T.groundZ + def[2] / 2], size: [...def], yaw: defaultYaw(g) });
    return grab({ kind: 'draw', p0: g });
  }
  if (T.edit) {
    const h = pickHandle(ev);
    if (h) return grab({ kind: h.kind, corner: h.corner });
    if (g && insideFootprint(T.edit.box, g)) return grab({ kind: 'move', last: g });
  }
  drag = { kind: 'orbit' };
}

function onPointerUp(ev) {
  const click = downAt && Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) < 4;
  const d = drag;
  drag = null; downAt = null;
  V.controls.enabled = true;
  if (!d) return;
  if (d.kind === 'draw') {
    setTool('select');
    fitZ(T.edit.box);
    T.cam = bestCamFor(T.edit.box.center) || T.cam;
    renderAll();
  } else if (d.kind !== 'orbit') {
    renderEdit(); drawCamera();
  } else if (click && !T.edit) {
    const o = pickAt(ev);
    if (o) select(o.object_id, true);
  }
}

function onDrag(ev) {
  const g = groundAt(ev);
  if (!g || !T.edit) return;
  const b = T.edit.box;
  if (drag.kind === 'draw') {
    const dx = g[0] - drag.p0[0], dy = g[1] - drag.p0[1], dist = Math.hypot(dx, dy);
    if (dist > 0.6) { // kéo từ đuôi tới đầu vật: hướng + chiều dài
      b.yaw = Math.atan2(dy, dx);
      b.size[1] = dist;
      b.center[0] = drag.p0[0] + dx / 2; b.center[1] = drag.p0[1] + dy / 2;
    }
  } else if (drag.kind === 'move') {
    b.center[0] += g[0] - drag.last[0]; b.center[1] += g[1] - drag.last[1];
    drag.last = g;
  } else if (drag.kind === 'rotate') {
    b.yaw = Math.atan2(g[1] - b.center[1], g[0] - b.center[0]);
  } else if (drag.kind === 'corner') {
    // góc đối diện đứng yên
    const [sx, sy] = drag.corner, c = Math.cos(b.yaw), s = Math.sin(b.yaw);
    const [w, l] = b.size;
    const fx = b.center[0] - c * (sx * l / 2) + s * (sy * w / 2);
    const fy = b.center[1] - s * (sx * l / 2) - c * (sy * w / 2);
    const lx = c * (g[0] - fx) + s * (g[1] - fy), ly = -s * (g[0] - fx) + c * (g[1] - fy);
    const nl = Math.max(Math.abs(lx), 0.2), nw = Math.max(Math.abs(ly), 0.2);
    const hx = Math.sign(lx || 1) * nl / 2, hy = Math.sign(ly || 1) * nw / 2;
    b.size[0] = nw; b.size[1] = nl;
    b.center[0] = fx + c * hx - s * hy; b.center[1] = fy + s * hx + c * hy;
  }
  buildEdit(); drawCamera(); renderEditFields();
}

const groundPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
function groundAt(ev) {
  const r = $('m3-canvas').getBoundingClientRect();
  const p = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(p, T.bev ? V.camBev : V.cam3d);
  groundPlane.constant = -(T.edit ? T.edit.box.center[2] - T.edit.box.size[2] / 2 : T.groundZ);
  const hit = new THREE.Vector3();
  return ray.ray.intersectPlane(groundPlane, hit) ? [hit.x, hit.y] : null;
}

function pickHandle(ev) {
  const r = $('m3-canvas').getBoundingClientRect();
  const p = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(p, T.bev ? V.camBev : V.cam3d);
  const hit = ray.intersectObjects(V.handles || [], false)[0];
  return hit ? hit.object.userData : null;
}

function onPointerMove(ev) {
  if (drag && drag.kind !== 'orbit') { onDrag(ev); return; }
  if (T.tool === 'draw' || T.edit) return;
  const tip = $('m3-tip');
  const o = T.frame && pickAt(ev);
  if (!o) { tip.classList.add('hidden'); return; }
  const r = $('m3-stage').getBoundingClientRect();
  tip.innerHTML = `<b>#${esc(o.object_id)} ${esc(finalLabel(o))}</b> ${o.score.toFixed(2)} · ${o.verify?.distance_m ?? '?'} m<br>${esc(VERDICT_VI[o.verify?.verdict] || '')}`;
  tip.style.left = `${Math.min(ev.clientX - r.left + 12, r.width - 330)}px`;
  tip.style.top = `${ev.clientY - r.top + 12}px`;
  tip.classList.remove('hidden');
}

// ---------------------------------------------------------------- camera
function loadCamImages() {
  T.camImg = {};
  if (!T.frame) return;
  const f = T.frame;
  for (const c of Object.keys(f.cameras)) {
    const img = new Image();
    img.onload = () => { if (T.frame === f) { if (c === T.cam) drawCamera(); } };
    img.src = `${API}/frames/${encodeURIComponent(f.model)}/${encodeURIComponent(f.frame_id)}/image/${c}`;
    T.camImg[c] = img;
  }
  renderStrip();
}

function renderStrip() {
  const f = T.frame;
  const sel = f?.objects.find((o) => o.object_id === T.selected);
  $('m3-cam-strip').innerHTML = f ? CAMS.filter((c) => f.cameras[c]).map((c, i) => `
    <div class="m3-thumb ${c === T.cam ? 'active' : ''}" data-cam="${c}" title="${c} (phím ${i + 1})">
      <img src="${esc(T.camImg[c]?.src || '')}" alt="">
      <span>${c.replace('CAM_', '')}</span>${sel?.verify?.camera === c ? '<b>chọn</b>' : ''}
    </div>`).join('') : '';
}

function project(K, M, p) {
  const x = M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + M[0][3];
  const y = M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + M[1][3];
  const z = M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + M[2][3];
  return [x, y, z];
}
function toPixel(K, q) { return [(K[0][0] * q[0]) / q[2] + K[0][2], (K[1][1] * q[1]) / q[2] + K[1][2]]; }

function drawCamera() {
  const canvas = $('m3-cam-canvas');
  const wrap = canvas.parentElement;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = wrap.clientWidth * dpr; canvas.height = wrap.clientHeight * dpr;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b1020'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const f = T.frame;
  $('m3-cam-name').textContent = f ? T.cam : '';
  if (!f || !f.cameras[T.cam]) return;
  const cam = f.cameras[T.cam];
  const img = T.camImg[T.cam];
  const s = Math.min(canvas.width / cam.width, canvas.height / cam.height);
  const ox = (canvas.width - cam.width * s) / 2, oy = (canvas.height - cam.height * s) / 2;
  if (img?.complete && img.naturalWidth) ctx.drawImage(img, ox, oy, cam.width * s, cam.height * s);
  const K = cam.intrinsic, M = cam.cam_from_lidar;
  const drawBox = (box, color, width, dash) => {
    const q = corners(box).map((p) => project(K, M, p));
    ctx.strokeStyle = color; ctx.lineWidth = width * dpr; ctx.setLineDash(dash ? [5 * dpr, 4 * dpr] : []);
    ctx.beginPath();
    for (const [a, b] of EDGES) {
      let A = q[a], B = q[b];
      if (A[2] < 0.1 && B[2] < 0.1) continue;
      if (A[2] < 0.1 || B[2] < 0.1) { // cắt theo mặt phẳng gần
        const t = (0.1 - A[2]) / (B[2] - A[2]);
        const C = A.map((v, i) => v + t * (B[i] - v));
        if (A[2] < 0.1) A = C; else B = C;
      }
      const pa = toPixel(K, A), pb = toPixel(K, B);
      ctx.moveTo(ox + pa[0] * s, oy + pa[1] * s); ctx.lineTo(ox + pb[0] * s, oy + pb[1] * s);
    }
    ctx.stroke();
  };
  if (T.showGt) for (const g of T.gt) drawBox(g, 'rgba(226,232,240,.8)', 1, true);
  let selObj = null;
  for (const o of f.objects) {
    if (o.review.status === 'deleted' || o.object_id === T.edit?.oid) continue;
    if (o.source !== 'human' && !o.verify?.bbox2d?.[T.cam]) continue;
    if (o.object_id === T.selected) { selObj = o; continue; }
    if (!T.showLow && levelOf(o) === 'low' && o.review.status === 'pending') continue;
    if (o.source !== 'human' && o.score < T.minScore && o.object_id !== T.selected) continue;
    drawBox(o.box, CSS_COLOR[stateColor(o)], 1.5, false);
  }
  if (T.edit) {
    drawBox(T.edit.box, EDIT_CSS, 2.5 * 1.2, false);
    return;
  }
  if (selObj && selObj.source === 'human') { drawBox(selObj.box, '#ffffff', 3, false); return; }
  if (selObj) {
    drawBox(selObj.box, '#ffffff', 3, false);
    const v = selObj.verify;
    if (v.det_bbox && v.det_camera === T.cam) { // box 2D của detector đã khớp
      const [x1, y1, x2, y2] = v.det_bbox;
      ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2 * dpr; ctx.setLineDash([6 * dpr, 4 * dpr]);
      ctx.strokeRect(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s);
    }
    const b = v.bbox2d[T.cam];
    ctx.setLineDash([]);
    ctx.font = `${12 * dpr}px system-ui`;
    const label = `#${selObj.object_id} ${finalLabel(selObj)}${v.det_label && v.det_camera === T.cam ? ` · 2D: ${v.det_label} ${v.det_score?.toFixed(2)}` : ''}`;
    const tw = ctx.measureText(label).width + 8 * dpr;
    ctx.fillStyle = 'rgba(15,23,42,.85)';
    ctx.fillRect(ox + b[0] * s, Math.max(0, oy + b[1] * s - 18 * dpr), tw, 17 * dpr);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, ox + b[0] * s + 4 * dpr, Math.max(13 * dpr, oy + b[1] * s - 5 * dpr));
  }
}

function setCam(c) {
  if (!T.frame?.cameras[c]) return;
  T.cam = c;
  renderStrip();
  drawCamera();
}

// ---------------------------------------------------------------- trả lại frame (FR-15), hoàn tác (FR-09)
const framePath = (f) => `/frames/${encodeURIComponent(f.model)}/${encodeURIComponent(f.frame_id)}`;

function renderRejectBanner() {
  const f = T.frame;
  const show = f && f.reject_reason && f.status !== 'approved';
  $('m3-reject-banner').classList.toggle('hidden', !show);
  if (show) {
    const when = (f.rejected_at || '').replace('T', ' ').replace('+00:00', ' UTC');
    $('m3-reject-banner').innerHTML = `<b>Bị trả lại</b> bởi ${esc(f.rejected_by || '?')} · ${esc(when)}<br>${esc(f.reject_reason)}`;
  }
}

let historyToken = 0;
async function refreshHistory() {
  const f = T.frame;
  const token = ++historyToken;
  if (!f) { $('m3-btn-undo').disabled = $('m3-btn-redo').disabled = true; return; }
  try {
    const h = await api(`${framePath(f)}/history`);
    if (token !== historyToken) return;
    const locked = T.frame?.status === 'approved';
    $('m3-btn-undo').disabled = locked || !h.undo;
    $('m3-btn-redo').disabled = locked || !h.redo;
  } catch { /* máy chủ cũ */ }
}

async function undoRedo(dir) {
  const f = T.frame;
  if (!f || f.status === 'approved' || T.edit) return;
  try {
    T.frame = await api(`${framePath(f)}/${dir}`, { method: 'POST', body: { reviewer: reviewer() } });
    if (!T.frame.objects.some((o) => o.object_id === T.selected)) T.selected = null;
    toast(dir === 'undo' ? 'Đã hoàn tác' : 'Đã làm lại');
    await refreshSummary();
    renderAll();
  } catch (e) { toast(e.message, true); }
}

function openReject() {
  if (!T.frame) return;
  $('m3-reject-box').classList.remove('hidden');
  $('m3-reject-reason').focus();
}

async function sendReject() {
  const reason = $('m3-reject-reason').value.trim();
  if (reason.length < 3) { toast('Ghi lý do trả lại (ít nhất 3 ký tự)', true); return; }
  try {
    T.frame = await api(`${framePath(T.frame)}/reject`, { method: 'POST', body: { reason, reviewer: reviewer() } });
    $('m3-reject-reason').value = '';
    $('m3-reject-box').classList.add('hidden');
    toast(`Đã trả lại ${T.frame.frame_id}`);
    await refreshSummary();
    renderAll();
  } catch (e) { toast(e.message, true); }
}


// ---------------------------------------------------------------- vẽ / sửa box
const newClass = () => $('m3-new-class').value || 'car';
const clone = (b) => ({ center: [...b.center], size: [...b.size], yaw: b.yaw });

function setTool(tool) {
  T.tool = tool;
  $('m3-draw').classList.toggle('active', tool === 'draw');
  $('m3-stage').classList.toggle('drawing', tool === 'draw');
  $('m3-hint').textContent = tool === 'draw'
    ? 'Kéo trên mặt đường từ đuôi tới đầu vật (hoặc bấm một điểm để đặt box cỡ mặc định) · Esc: thôi'
    : T.edit ? 'Kéo trong box: di chuyển · kéo chấm góc: đổi cỡ · kéo chấm đầu: xoay · Enter lưu · Esc huỷ' : HINT;
}

function toggleDraw() {
  if (!T.frame) return;
  if (T.frame.status === 'approved') { toast('Frame đã approve, mở lại trước khi sửa', true); return; }
  if (T.tool === 'draw') { setTool('select'); return; }
  cancelEdit(false);
  if (!T.bev) setView(true); // vẽ trên mặt đường dễ nhất khi nhìn từ trên
  setTool('draw');
}

function startEditState(mode, oid, label, box) {
  T.edit = { mode, oid, label, box };
  setTool(T.tool);
  buildBoxes(); renderEdit(); drawCamera();
}

function startEdit(oid) {
  const o = T.frame?.objects.find((x) => x.object_id === oid);
  if (!o || o.review.status === 'deleted') return;
  if (T.frame.status === 'approved') { toast('Frame đã approve, mở lại trước khi sửa', true); return; }
  setTool('select');
  startEditState('edit', oid, finalLabel(o), clone(o.box));
  T.selected = oid;
  renderAll();
}

function cancelEdit(render = true) {
  if (!T.edit) return;
  T.edit = null;
  setTool(T.tool);
  if (render) renderAll();
}

async function saveEdit() {
  const f = T.frame, e = T.edit;
  if (!f || !e) return;
  const box = { center: e.box.center.map((v) => +v.toFixed(3)), size: e.box.size.map((v) => +v.toFixed(3)), yaw: +e.box.yaw.toFixed(5) };
  const body = e.mode === 'add'
    ? { action: 'ADD_BOX', label: e.label, box, reviewer: reviewer() }
    : { action: 'EDIT_BOX', object_id: e.oid, label: e.label, box, reviewer: reviewer() };
  try {
    T.frame = await api(`/frames/${encodeURIComponent(f.model)}/${encodeURIComponent(f.frame_id)}/actions`, { method: 'POST', body });
    T.selected = e.mode === 'add' ? T.frame.objects[T.frame.objects.length - 1].object_id : e.oid;
    T.edit = null;
    setTool('select');
    toast(e.mode === 'add' ? `Đã thêm box ${e.label}` : `Đã sửa box #${e.oid}`);
    await refreshSummary();
    renderAll();
  } catch (err) { toast(err.message, true); }
}

function defaultYaw(g) {
  // cùng hướng với box gần nhất trong 10 m (xe đỗ, xe cùng làn thường song song), nếu không thì hướng thẳng về trước
  let best = null, bd = 10;
  for (const o of T.frame?.objects || []) {
    if (o.review.status === 'deleted') continue;
    const d = Math.hypot(o.box.center[0] - g[0], o.box.center[1] - g[1]);
    if (d < bd && o.box.size[1] > 1.2) { bd = d; best = o; }
  }
  return best ? best.box.yaw : Math.PI / 2;
}

function toLocal(b, x, y) {
  const c = Math.cos(b.yaw), s = Math.sin(b.yaw), dx = x - b.center[0], dy = y - b.center[1];
  return [c * dx + s * dy, -s * dx + c * dy];
}
function insideFootprint(b, g, margin = 0) {
  const [lx, ly] = toLocal(b, g[0], g[1]);
  return Math.abs(lx) <= b.size[1] / 2 + margin && Math.abs(ly) <= b.size[0] / 2 + margin;
}
function pointsNear(b, margin = 0, zmin = -Infinity, zmax = Infinity) {
  const out = [];
  const P = T.points;
  if (!P) return out;
  const R = Math.hypot(b.size[0], b.size[1]) / 2 + margin;
  for (let i = 0; i < P.length; i += 4) {
    const x = P[i], y = P[i + 1], z = P[i + 2];
    if (Math.abs(x - b.center[0]) > R || Math.abs(y - b.center[1]) > R || z < zmin || z > zmax) continue;
    const [lx, ly] = toLocal(b, x, y);
    if (Math.abs(lx) <= b.size[1] / 2 + margin && Math.abs(ly) <= b.size[0] / 2 + margin) out.push([lx, ly, z]);
  }
  return out;
}
const quantile = (arr, q) => { const a = [...arr].sort((m, n) => m - n); return a.length ? a[Math.min(a.length - 1, Math.floor(q * a.length))] : null; };
function countInside(b) {
  const zb = b.center[2] - b.size[2] / 2;
  return pointsNear(b, 0, zb, zb + b.size[2]).length;
}

function localGround(b) {
  // mặt đường ngay dưới box (vỉa hè, dốc): trung vị điểm thấp quanh box, nếu có
  const low = pointsNear(b, 1.0, T.groundZ - 0.6, T.groundZ + 0.4).map((p) => p[2]);
  return low.length >= 5 ? quantile(low, 0.5) : T.groundZ;
}

function fitZ(b) {
  // đáy box đặt lên mặt đường, chiều cao theo điểm LiDAR trong box (ít điểm thì dùng cỡ trung bình của lớp)
  const g = localGround(b);
  const zs = pointsNear(b, 0, g + 0.15, g + 5).map((p) => p[2]);
  const def = DEFAULT_SIZE[T.edit?.label]?.[2] ?? b.size[2];
  const h = zs.length >= 5 ? Math.max(quantile(zs, 0.99) - g + 0.05, 0.3) : def;
  b.size[2] = h;
  b.center[2] = g + h / 2;
}

function fitTight(b) {
  // co box khít đám điểm LiDAR (bỏ điểm mặt đường), giữ cạnh gần xe khi vật chỉ lộ một mặt
  const g = localGround(b);
  const pts = pointsNear(b, 0.4, g + 0.2, g + 4.5);
  if (pts.length < 8) { toast(`Chỉ có ${pts.length} điểm LiDAR quanh box: không đủ để co khít`, true); return; }
  const def = DEFAULT_SIZE[T.edit?.label] || b.size;
  const sensor = toLocal(b, 0, 0);
  const axis = (k, minLen) => {
    const vals = pts.map((p) => p[k]);
    let lo = quantile(vals, 0.02), hi = quantile(vals, 0.98);
    if (hi - lo < minLen) { if (sensor[k] < lo) hi = lo + minLen; else if (sensor[k] > hi) lo = hi - minLen; else { const m = (lo + hi) / 2; lo = m - minLen / 2; hi = m + minLen / 2; } }
    return [lo, hi];
  };
  const [x0, x1] = axis(0, 0.6 * def[1]);
  const [y0, y1] = axis(1, 0.8 * def[0]);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, c = Math.cos(b.yaw), s = Math.sin(b.yaw);
  b.center[0] += c * cx - s * cy; b.center[1] += s * cx + c * cy;
  // nhãn nuScenes bao ngoài bề mặt vật một chút: nới 5 cm mỗi phía
  b.size[1] = Math.max(x1 - x0 + 0.1, 0.2); b.size[0] = Math.max(y1 - y0 + 0.1, 0.2);
  fitZ(b);
}

function bestCamFor(p) {
  let best = null, bs = Infinity;
  for (const [name, cam] of Object.entries(T.frame?.cameras || {})) {
    const q = project(cam.intrinsic, cam.cam_from_lidar, p);
    if (q[2] < 0.5) continue;
    const [u, v] = toPixel(cam.intrinsic, q);
    if (u < 0 || v < 0 || u > cam.width || v > cam.height) continue;
    const off = Math.abs(u - cam.intrinsic[0][2]) / cam.intrinsic[0][0];
    if (off < bs) { bs = off; best = name; }
  }
  return best;
}

function buildEdit() {
  V.editGroup.clear();
  V.handles = [];
  const e = T.edit;
  if (!e) return;
  const b = e.box;
  V.editGroup.add(wireframe(b, EDIT_HEX));
  const [w, l, h] = b.size;
  const fill = new THREE.Mesh(new THREE.BoxGeometry(l, w, h), new THREE.MeshBasicMaterial({ color: EDIT_HEX, transparent: true, opacity: 0.15, depthWrite: false }));
  fill.position.set(...b.center); fill.rotation.z = b.yaw;
  V.editGroup.add(fill);
  const c = Math.cos(b.yaw), s = Math.sin(b.yaw), top = b.center[2] + h / 2;
  const at = (lx, ly) => [b.center[0] + c * lx - s * ly, b.center[1] + s * lx + c * ly, top];
  const r = Math.max(0.18, Math.min(0.45, Math.min(w, l) * 0.18));
  const handle = (pos, color, data) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 8), new THREE.MeshBasicMaterial({ color, depthTest: false }));
    m.position.set(...pos); m.renderOrder = 10; m.userData = data;
    V.editGroup.add(m); V.handles.push(m);
  };
  for (const [sx, sy] of [[1, 1], [1, -1], [-1, -1], [-1, 1]]) handle(at((sx * l) / 2, (sy * w) / 2), 0xffffff, { kind: 'corner', corner: [sx, sy] });
  handle(at(l / 2 + Math.max(0.8, l * 0.15), 0), 0xfacc15, { kind: 'rotate' });
}

const num = (v, d = 2) => Number(v).toFixed(d);
function renderEdit() {
  const el = $('m3-edit');
  const e = T.edit;
  el.classList.toggle('hidden', !e);
  if (!e) { el.innerHTML = ''; return; }
  const opts = T.classes.map((c) => `<option value="${c}" ${c === e.label ? 'selected' : ''}>${c}</option>`).join('');
  const f = (k, label, v, step) => `<label>${label}<input type="number" step="${step}" data-edit="${k}" value="${v}"></label>`;
  const b = e.box;
  el.innerHTML = `<h4><span>${e.mode === 'add' ? '＋ Box mới' : `✎ Sửa box #${esc(e.oid)}`}</span><select data-edit="label">${opts}</select></h4>
    <div class="m3-edit-grid">
      ${f('x', 'x (m)', num(b.center[0]), 0.1)}${f('y', 'y (m)', num(b.center[1]), 0.1)}${f('z', 'z (m)', num(b.center[2]), 0.05)}${f('yaw', 'hướng (°)', num((b.yaw * 180) / Math.PI, 1), 1)}
      ${f('l', 'dài (m)', num(b.size[1]), 0.1)}${f('w', 'rộng (m)', num(b.size[0]), 0.1)}${f('h', 'cao (m)', num(b.size[2]), 0.1)}
      <label>điểm LiDAR<input type="text" data-edit-count readonly value="${countInside(b)}"></label>
    </div>
    <div class="m3-edit-actions">
      <button class="btn btn-primary btn-sm" data-edit-act="save">Lưu <kbd>Enter</kbd></button>
      <button class="btn btn-ghost btn-sm" data-edit-act="fit" title="Co box khít đám điểm LiDAR">Khít điểm <kbd>F</kbd></button>
      <button class="btn btn-ghost btn-sm" data-edit-act="ground" title="Đặt đáy box lên mặt đường, chiều cao theo điểm LiDAR">Đặt lên đường <kbd>R</kbd></button>
      <button class="btn btn-ghost btn-sm" data-edit-act="cancel">Huỷ <kbd>Esc</kbd></button>
    </div>
    <div class="m3-edit-keys"><kbd>←</kbd><kbd>↑</kbd><kbd>→</kbd><kbd>↓</kbd> di chuyển theo màn hình · <kbd>Q</kbd>/<kbd>E</kbd> xoay · <kbd>[</kbd>/<kbd>]</kbd> dài · <kbd>;</kbd>/<kbd>'</kbd> rộng · <kbd>-</kbd>/<kbd>=</kbd> cao · giữ <kbd>Shift</kbd>: bước lớn</div>`;
}
function renderEditFields() {
  const e = T.edit;
  if (!e) return;
  const b = e.box;
  const vals = { x: num(b.center[0]), y: num(b.center[1]), z: num(b.center[2]), yaw: num((b.yaw * 180) / Math.PI, 1), l: num(b.size[1]), w: num(b.size[0]), h: num(b.size[2]) };
  for (const [k, v] of Object.entries(vals)) { const inp = document.querySelector(`#m3-edit [data-edit="${k}"]`); if (inp && document.activeElement !== inp) inp.value = v; }
  const cnt = document.querySelector('#m3-edit [data-edit-count]');
  if (cnt) cnt.value = countInside(b);
}
function onEditInput(ev) {
  const e = T.edit, k = ev.target.dataset.edit;
  if (!e || !k) return;
  if (k === 'label') {
    const old = DEFAULT_SIZE[e.label], nw = DEFAULT_SIZE[ev.target.value];
    e.label = ev.target.value;
    if (e.mode === 'add' && old && nw && e.box.size.every((v, i) => Math.abs(v - old[i]) < 1e-6)) { e.box.size = [...nw]; fitZ(e.box); }
  } else {
    const v = parseFloat(ev.target.value);
    if (!Number.isFinite(v)) return;
    const b = e.box;
    if (k === 'x') b.center[0] = v; else if (k === 'y') b.center[1] = v; else if (k === 'z') b.center[2] = v;
    else if (k === 'yaw') b.yaw = (v * Math.PI) / 180;
    else if (k === 'l') b.size[1] = Math.max(v, 0.1); else if (k === 'w') b.size[0] = Math.max(v, 0.1); else if (k === 'h') b.size[2] = Math.max(v, 0.1);
  }
  buildEdit(); drawCamera(); renderEditFields();
}

function editKey(e) {
  // trả về true nếu đã xử lý phím
  const b = T.edit.box, big = e.shiftKey ? 5 : 1;
  // hướng "lên" của màn hình trên mặt đất
  let fx = 0, fy = 1;
  if (!T.bev) { const d = new THREE.Vector3(); V.cam3d.getWorldDirection(d); const n = Math.hypot(d.x, d.y) || 1; fx = d.x / n; fy = d.y / n; }
  const move = (a, bb) => { b.center[0] += 0.1 * big * (a * fx + bb * fy); b.center[1] += 0.1 * big * (a * fy - bb * fx); };
  const acts = {
    ArrowUp: () => move(1, 0), ArrowDown: () => move(-1, 0), ArrowLeft: () => move(0, -1), ArrowRight: () => move(0, 1),
    KeyQ: () => { b.yaw += (Math.PI / 180) * (e.shiftKey ? 10 : 1); }, KeyE: () => { b.yaw -= (Math.PI / 180) * (e.shiftKey ? 10 : 1); },
    BracketLeft: () => { b.size[1] = Math.max(0.2, b.size[1] - 0.1 * big); }, BracketRight: () => { b.size[1] += 0.1 * big; },
    Semicolon: () => { b.size[0] = Math.max(0.2, b.size[0] - 0.1 * big); }, Quote: () => { b.size[0] += 0.1 * big; },
    Minus: () => { b.size[2] = Math.max(0.2, b.size[2] - 0.1 * big); }, Equal: () => { b.size[2] += 0.1 * big; },
    KeyF: () => fitTight(b), KeyR: () => fitZ(b),
  };
  if (e.key === 'Enter') { saveEdit(); return true; }
  if (e.key === 'Escape') { cancelEdit(); return true; }
  if (!acts[e.code]) return false;
  acts[e.code]();
  buildEdit(); drawCamera(); renderEditFields();
  return true;
}

// ---------------------------------------------------------------- ảnh BEV (homography mặt đường, ghép 6 camera)
function estimateGround(P) {
  if (!P) return -1.84;
  const bins = new Array(60).fill(0), vals = [];
  for (let i = 0; i < P.length; i += 4) {
    const r = Math.hypot(P[i], P[i + 1]), z = P[i + 2];
    if (r > 3 && r < 25 && z > -3.5 && z < -0.5) { bins[Math.floor((z + 3.5) / 0.05)]++; vals.push(z); }
  }
  if (vals.length < 50) return -1.84;
  const k = bins.indexOf(Math.max(...bins));
  const lo = -3.5 + (k - 1) * 0.05, hi = -3.5 + (k + 2) * 0.05;
  return quantile(vals.filter((z) => z >= lo && z < hi), 0.5);
}

async function loadBevImage() {
  if (V.bevPlane) { V.scene.remove(V.bevPlane); V.bevPlane.material.map?.dispose(); V.bevPlane.material.dispose(); V.bevPlane.geometry.dispose(); V.bevPlane = null; }
  const f = T.frame;
  if (!T.showBevImg || !f) return;
  try {
    const res = await fetch(`${API}/frames/${encodeURIComponent(f.model)}/${encodeURIComponent(f.frame_id)}/bev`);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail?.message || res.statusText);
    const z0 = parseFloat(res.headers.get('X-Ground-Z')), R = parseFloat(res.headers.get('X-BEV-Range') || '40');
    const url = URL.createObjectURL(await res.blob());
    const img = new Image();
    img.src = url;
    await img.decode();
    URL.revokeObjectURL(url);
    if (T.frame !== f || !T.showBevImg) return;
    const tex = new THREE.Texture(img);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    // ảnh: hàng trên cùng = y lớn nhất, cột trái = x nhỏ nhất -> khớp UV của PlaneGeometry
    V.bevPlane = new THREE.Mesh(new THREE.PlaneGeometry(2 * R, 2 * R), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
    V.bevPlane.position.set(0, 0, (Number.isFinite(z0) ? z0 : T.groundZ) - 0.03);
    V.bevPlane.renderOrder = -1;
    V.scene.add(V.bevPlane);
  } catch (e) { toast('Không tạo được ảnh BEV: ' + e.message, true); }
}

// ---------------------------------------------------------------- dữ liệu
async function loadModels() {
  T.models = await api('/models');
  const sel = $('m3-model');
  sel.innerHTML = T.models.map((m) => `<option value="${esc(m.model)}">${esc(m.label)}${m.NDS != null ? ` · NDS ${m.NDS.toFixed(3)} · mAP ${m.mAP.toFixed(3)}` : ''}</option>`).join('');
  const want = storage.get('model', '');
  T.model = T.models.find((m) => m.model === want)?.model || T.models[0]?.model || null;
  if (T.model) sel.value = T.model;
  renderModelInfo();
  $('m3-empty').classList.toggle('hidden', !!T.models.length);
}

function renderModelInfo() {
  const m = T.models.find((x) => x.model === T.model);
  $('m3-model-info').innerHTML = m
    ? `${esc(m.sensor || '')}${m.mAP != null ? ` · mAP <b>${m.mAP.toFixed(3)}</b> · NDS <b>${m.NDS.toFixed(3)}</b> (scene val)` : ''}<br>${m.approved}/${m.frames} keyframe đã duyệt`
    : 'Chưa có mô hình nào';
}

async function loadQueue() {
  if (!T.model) { T.queue = []; renderQueue(); return; }
  T.queue = await api(`/frames?model=${encodeURIComponent(T.model)}`);
  renderQueue();
}

function sortedQueue() {
  const q = [...T.queue];
  if (T.sort === 'risk') q.sort((a, b) => (a.status === 'approved') - (b.status === 'approved') || (b.status === 'rejected') - (a.status === 'rejected') || b.counts.high - a.counts.high || b.counts.medium - a.counts.medium);
  else q.sort((a, b) => a.frame_id.localeCompare(b.frame_id));
  return q;
}

function renderQueue() {
  $('m3-queue').innerHTML = sortedQueue().map((f) => `
    <li class="queue-item ${f.frame_id === T.frame?.frame_id ? 'active' : ''}" data-id="${esc(f.frame_id)}">
      <div class="qi-top"><span class="qi-id">${esc(f.frame_id)}</span>
        <span class="status-pill ${f.status}">${STATUS_TEXT[f.status] || f.status}</span></div>
      <div class="qi-meta">
        ${LEVELS.map((lv) => `<span class="lv"><span class="dot ${lv}"></span>${f.counts[lv]}</span>`).join('')}
        <span>· ${f.n_objects} box</span>${f.pending ? `<span>· ${f.pending} chờ</span>` : ''}
      </div>
    </li>`).join('') || '<li class="muted" style="padding:12px">Chưa có frame 3D</li>';
}

async function openFrame(fid) {
  const model = T.model;
  const f = await api(`/frames/${encodeURIComponent(model)}/${encodeURIComponent(fid)}`);
  if (model !== T.model) return;
  T.frame = f;
  T.edit = null;
  setTool('select');
  T.timerStart = f.status === 'approved' ? null : Date.now();
  storage.set('frame', fid);
  const pending = f.objects.filter((o) => o.review.status === 'pending');
  T.selected = (orderObjects(pending)[0] || f.objects[0])?.object_id || null;
  const sel = f.objects.find((o) => o.object_id === T.selected);
  if (sel?.verify?.camera) T.cam = sel.verify.camera;
  if (!f.cameras[T.cam]) T.cam = Object.keys(f.cameras)[0];
  const [buf, gt] = await Promise.all([
    api(`/frames/${encodeURIComponent(model)}/${encodeURIComponent(fid)}/points`, { raw: true }).catch(() => null),
    api(`/frames/${encodeURIComponent(model)}/${encodeURIComponent(fid)}/gt`).catch(() => []),
  ]);
  if (T.frame !== f) return;
  T.points = buf ? new Float32Array(buf) : null;
  T.groundZ = estimateGround(T.points);
  T.gt = gt;
  buildPoints();
  loadCamImages();
  loadBevImage();
  renderAll();
}

function orderObjects(objs) {
  const rank = { high: 0, medium: 1, low: 2 };
  return [...objs].sort((a, b) => rank[levelOf(a)] - rank[levelOf(b)] || b.score - a.score);
}

// ---------------------------------------------------------------- panel duyệt
function renderAll() {
  const f = T.frame;
  $('m3-frame-id').textContent = f ? `${f.frame_id}` : '—';
  $('m3-status').textContent = f ? STATUS_TEXT[f.status] : '';
  $('m3-status').className = `status-pill ${f?.status || ''}`;
  buildBoxes();
  renderReview();
  renderEdit();
  renderQueue();
  renderStrip();
  drawCamera();
  renderRejectBanner();
  refreshHistory();
}

function card(o) {
  const v = o.verify || {};
  const lv = levelOf(o);
  const opts = T.classes.map((c) => `<option value="${c}" ${c === finalLabel(o) ? 'selected' : ''}>${c}</option>`).join('');
  return `<div class="obj-card ${lv} ${o.object_id === T.selected ? 'selected' : ''}" data-oid="${esc(o.object_id)}">
    <div class="oc-title"><span>${esc(finalLabel(o))} <span class="oid">#${esc(o.object_id)}</span></span>
      <span class="verdict ${lv}">${esc(VERDICT_VI[v.verdict] || v.verdict || '')}</span></div>
    <div class="oc-stats">score ${o.score.toFixed(2)} · ${v.distance_m ?? '?'} m · ${v.lidar_points ?? 0} điểm LiDAR${v.camera ? ` · ${esc(v.camera.replace('CAM_', ''))}` : ''}${v.occlusion != null ? ` · che ${Math.round(v.occlusion * 100)}%` : ''}</div>
    <div class="m3-comment">${esc(v.comment || '')}</div>
    <div class="oc-actions">
      <button class="btn btn-keep btn-sm" data-act="KEEP">✓ Keep</button>
      <button class="btn btn-del btn-sm" data-act="DELETE">🗑 Delete</button>
      <select data-class>${opts}</select>
      <button class="btn btn-class btn-sm" data-act="CHANGE_CLASS">Đổi lớp</button>
      <button class="btn btn-ghost btn-sm" data-act="EDIT" title="Sửa vị trí / kích thước / hướng box (E)">✎ Sửa box</button>
    </div>
  </div>`;
}

function renderReview() {
  const f = T.frame;
  const objs = f?.objects || [];
  const pending = objs.filter((o) => o.review.status === 'pending');
  const by = { high: [], medium: [], low: [] };
  for (const o of orderObjects(pending)) by[levelOf(o)].push(o);
  const all = { high: 0, medium: 0, low: 0 };
  for (const o of objs) if (o.source !== 'human') all[levelOf(o)]++;
  $('m3-risk-summary').innerHTML = LEVELS.slice().reverse().map((lv) => `
    <div class="rs ${lv}"><div class="rs-label"><span class="dot ${lv}"></span>${LEVEL_NAME[lv]} risk</div>
    <div class="rs-value">${all[lv]}</div><div class="rs-range">${lv === 'low' ? 'camera xác nhận' : lv === 'medium' ? 'chưa chắc' : 'sai lớp / nghi báo nhầm'}</div></div>`).join('');
  for (const lv of ['high', 'medium']) {
    $(`m3-list-${lv}`).innerHTML = by[lv].map(card).join('') || '<p class="muted">Không có box chờ duyệt</p>';
    $(`m3-count-${lv}`).textContent = by[lv].length;
  }
  $('m3-count-low').textContent = by.low.length;
  $('m3-list-low').innerHTML = by.low.map((o) => o.object_id === T.selected ? card(o) : `
    <div class="low-row" data-oid="${esc(o.object_id)}"><span>#${esc(o.object_id)} ${esc(finalLabel(o))} · ${o.verify?.distance_m ?? '?'} m</span><span>${o.score.toFixed(2)}</span></div>`).join('');
  $('m3-approve-low').disabled = !by.low.length || f?.status === 'approved';
  $('m3-approve-low').innerHTML = `✓ Approve all low-risk (${by.low.length}) <kbd>A</kbd>`;
  const done = objs.filter((o) => o.review.status !== 'pending');
  $('m3-count-done').textContent = done.length;
  $('m3-list-done').innerHTML = done.map((o) => {
    const a = o.review.action;
    const tag = o.review.status === 'deleted' ? '<span class="done-tag deleted">✗ xoá</span>'
      : a === 'ADD_BOX' ? '<span class="done-tag added">＋ người thêm</span>'
        : `<span class="done-tag approved">${a === 'CHANGE_CLASS' ? `→ ${esc(o.review.final_label)}` : a === 'EDIT_BOX' ? `✎ sửa box${o.review.final_label !== o.label ? ` → ${esc(o.review.final_label)}` : ''}` : a === 'BATCH_APPROVE' ? 'batch approve' : 'keep'}</span>`;
    const dot = o.source === 'human' ? '<span class="dot" style="background:#22d3ee"></span>' : `<span class="dot ${levelOf(o)}"></span>`;
    return `<div class="low-row" data-oid="${esc(o.object_id)}"><span>${dot} #${esc(o.object_id)} ${esc(finalLabel(o))}${o.source === 'human' ? '' : ` ${o.score.toFixed(2)}`}</span>${tag}</div>`;
  }).join('');
  const n = objs.length, d = done.length;
  $('m3-progress-bar').style.width = n ? `${(100 * d) / n}%` : '0';
  $('m3-progress-text').textContent = f ? `${d}/${n} box đã xử lý` : '';
  const btn = $('m3-approve-frame');
  btn.disabled = !f || (f.status !== 'approved' && pending.length > 0);
  btn.innerHTML = f?.status === 'approved' ? 'Mở lại để sửa' : 'Approve frame <kbd>Enter</kbd>';
}

function select(oid, fromViewer = false) {
  T.selected = oid;
  const o = T.frame?.objects.find((x) => x.object_id === oid);
  if (o?.verify?.camera && (fromViewer || o.verify.bbox2d?.[T.cam] == null || true)) T.cam = o.verify.camera;
  renderAll();
  document.querySelector(`#tab-review3d .obj-card[data-oid="${CSS.escape(oid)}"]`)?.scrollIntoView({ block: 'nearest' });
}

async function act(action, oid, label) {
  const f = T.frame;
  if (!f) return;
  try {
    T.frame = await api(`/frames/${encodeURIComponent(f.model)}/${encodeURIComponent(f.frame_id)}/actions`, {
      method: 'POST', body: { action, object_id: oid, label, reviewer: reviewer() },
    });
    const next = orderObjects(T.frame.objects.filter((o) => o.review.status === 'pending'))[0];
    T.selected = next?.object_id || oid;
    if (next?.verify?.camera) T.cam = next.verify.camera;
    await refreshSummary();
    renderAll();
  } catch (e) { toast(e.message, true); }
}

async function approveLow() {
  const f = T.frame;
  if (!f) return;
  try {
    T.frame = await api(`/frames/${encodeURIComponent(f.model)}/${encodeURIComponent(f.frame_id)}/approve-low-risk`, { method: 'POST', body: { reviewer: reviewer() } });
    await refreshSummary();
    renderAll();
  } catch (e) { toast(e.message, true); }
}

async function approveFrame() {
  const f = T.frame;
  if (!f) return;
  try {
    if (f.status === 'approved') {
      T.frame = await api(`/frames/${encodeURIComponent(f.model)}/${encodeURIComponent(f.frame_id)}/reopen`, { method: 'POST', body: {} });
      T.timerStart = Date.now();
      await refreshSummary();
      renderAll();
      return;
    }
    const secs = T.timerStart ? Math.round((Date.now() - T.timerStart) / 1000) : undefined;
    T.frame = await api(`/frames/${encodeURIComponent(f.model)}/${encodeURIComponent(f.frame_id)}/approve`, { method: 'POST', body: { reviewer: reviewer(), review_time_s: secs } });
    toast(`Đã approve ${f.frame_id}`);
    T.timerStart = null;
    await refreshSummary();
    const next = sortedQueue().find((x) => x.status !== 'approved' && x.frame_id !== f.frame_id);
    if (next) await openFrame(next.frame_id); else renderAll();
  } catch (e) { toast(e.message, true); }
}

async function refreshSummary() {
  await loadQueue();
  const m = await api('/models').catch(() => null);
  if (m) { T.models = m; renderModelInfo(); }
}

function stepFrame(delta) {
  const q = sortedQueue();
  if (!q.length) return;
  const i = q.findIndex((f) => f.frame_id === T.frame?.frame_id);
  openFrame(q[(i + delta + q.length) % q.length].frame_id).catch((e) => toast(e.message, true));
}

function stepObject(delta) {
  const objs = orderObjects((T.frame?.objects || []).filter((o) => o.review.status !== 'deleted'));
  if (!objs.length) return;
  const i = objs.findIndex((o) => o.object_id === T.selected);
  select(objs[(i + delta + objs.length) % objs.length].object_id);
}

// ---------------------------------------------------------------- metrics 3D
const fmt = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d));
const pct = (v) => (v == null ? '—' : `${Math.round(v * 1000) / 10}%`);

async function loadMetrics3d() {
  const [cmp, m] = await Promise.all([
    fetch(API + '/compare').then((r) => r.json()).catch(() => ({})),
    T.model ? api(`/metrics?model=${encodeURIComponent(T.model)}`).catch(() => null) : null,
  ]);
  const models = Object.entries(cmp.models || {});
  const ok = models.filter(([, x]) => x.mAP != null);
  const best = (k, hi = true) => ok.length ? ok.reduce((a, b) => ((hi ? b[1][k] > a[1][k] : b[1][k] < a[1][k]) ? b : a))[0] : null;
  const bm = best('mAP'), bn = best('NDS'), bs = best('s_per_sample', false);
  $('m3-compare-note').textContent = cmp.scenes
    ? `Chấm chuẩn nuScenes (bộ chấm của devkit) trên ${cmp.scenes.length} scene val có trên máy, ${cmp.n_samples} keyframe. Tự duyệt / bắt lỗi: kết quả kiểm chứng bằng camera trên 3 scene demo, đối chiếu nhãn gốc.`
    : 'Chưa có kết quả: chạy tools3d/run3d.py trên máy có GPU rồi chép eval/results/det3d/ về.';
  $('m3-compare-table').innerHTML = models.length ? `<thead><tr><th>Mô hình</th><th>Cảm biến</th><th class="num">mAP</th><th class="num">NDS</th>
      <th class="num">mAP công bố</th><th class="num">s / keyframe</th><th class="num">GPU (GB)</th><th class="num">Tự duyệt</th><th class="num">Đúng trong nhóm tự duyệt</th><th class="num">Bắt được box sai</th></tr></thead><tbody>${models.map(([k, x]) => x.error
      ? `<tr><td>${esc(x.label || k)}</td><td colspan="9" class="muted">Lỗi: ${esc(x.error)}</td></tr>`
      : `<tr><td>${esc(x.label || k)}</td><td>${esc(x.sensor || '')}</td><td class="num ${k === bm ? 'best' : ''}">${fmt(x.mAP)}</td>
        <td class="num ${k === bn ? 'best' : ''}">${fmt(x.NDS)}</td><td class="num">${fmt((x.paper?.mAP ?? null) / 100)}</td>
        <td class="num ${k === bs ? 'best' : ''}">${fmt(x.s_per_sample, 2)}</td><td class="num">${fmt(x.peak_gpu_gb, 1)}</td>
        <td class="num">${pct(x.verify?.auto_share)}</td><td class="num">${pct(x.verify?.auto_precision)}</td><td class="num">${pct(x.verify?.error_recall)}</td></tr>`).join('')}</tbody>` : '';
  if (!m) return;
  const tiles = [
    ['Keyframe 3D đã duyệt', `${m.approved}/${m.frames}`, `${m.objects} box ${esc(T.model)}`],
    ['Tỉ lệ box phải sửa', pct(m.correction_rate), `${m.fixed}/${m.reviewed} box đã duyệt`],
    ['Flag precision', pct(m.flag_precision), 'box bị gắn cờ thật sự phải sửa'],
    ['Flag recall', pct(m.flag_recall), 'box phải sửa đã được gắn cờ'],
    ['Thời gian / keyframe', m.mean_review_s != null ? `${m.mean_review_s} s` : '—', 'trung bình các frame đã approve'],
  ];
  $('m3-tiles').innerHTML = tiles.map(([a, b, c]) => `<div class="tile"><div class="tile-label">${a}</div><div class="tile-value">${b}</div><div class="tile-note">${c}</div></div>`).join('');
  $('m3-verdict-table').innerHTML = `<thead><tr><th>Kết luận</th><th class="num">Tổng</th><th class="num">Đã duyệt</th><th class="num">Bị sửa</th><th class="num">Tỉ lệ sửa</th></tr></thead><tbody>${
    Object.keys(VERDICT_VI).map((v) => { const d = m.by_verdict[v] || { reviewed: 0, fixed: 0 };
      return `<tr><td>${VERDICT_VI[v]}</td><td class="num">${m.verdicts[v] || 0}</td><td class="num">${d.reviewed}</td><td class="num">${d.fixed}</td><td class="num">${d.reviewed ? pct(d.fixed / d.reviewed) : '—'}</td></tr>`; }).join('')}</tbody>`;
  window.AL?.renderProductivity(m.productivity, 'm3-', `${API}/report.csv?model=${encodeURIComponent(T.model)}`);
}

async function exportKitti() {
  try {
    const r = await api(`/export-kitti?model=${encodeURIComponent(T.model)}`, { method: 'POST' });
    $('m3-export-result').innerHTML = `<p>Đã xuất KITTI: ${r.frames} keyframe, ${r.labels} box nằm trong ảnh ${esc(r.camera)} · <a href="${esc(r.url)}" download>tải ${esc(r.file)}</a></p>`;
  } catch (e) { $('m3-export-result').innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

async function export3d() {
  try {
    const r = await api(`/export?model=${encodeURIComponent(T.model)}`, { method: 'POST' });
    $('m3-export-result').innerHTML = `<p>Đã xuất ${r.frames} keyframe, ${r.objects} box vào <code>${esc(r.dir)}</code></p>`;
  } catch (e) { $('m3-export-result').innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

// ---------------------------------------------------------------- sự kiện
function bind() {
  $('m3-model').addEventListener('change', async (e) => {
    T.model = e.target.value; storage.set('model', T.model); renderModelInfo();
    const keep = T.frame?.frame_id;
    await loadQueue();
    const f = T.queue.find((x) => x.frame_id === keep) || sortedQueue()[0];
    if (f) await openFrame(f.frame_id);
  });
  $('m3-sort').addEventListener('change', (e) => { T.sort = e.target.value; renderQueue(); });
  $('m3-queue').addEventListener('click', (e) => {
    const li = e.target.closest('[data-id]');
    if (li) openFrame(li.dataset.id).catch((err) => toast(err.message, true));
  });
  $('m3-show-gt').addEventListener('change', (e) => { T.showGt = e.target.checked; buildBoxes(); drawCamera(); });
  $('m3-show-low').addEventListener('change', (e) => { T.showLow = e.target.checked; buildBoxes(); drawCamera(); });
  $('m3-min-score').addEventListener('input', (e) => {
    T.minScore = Number(e.target.value);
    $('m3-min-score-val').textContent = T.minScore.toFixed(2);
    const objs = (T.frame?.objects || []).filter((o) => o.review.status !== 'deleted');
    const shown = objs.filter((o) => o.source === 'human' || o.score >= T.minScore).length;
    $('m3-min-score-count').textContent = T.minScore > 0 ? ` · ${shown}/${objs.length} box` : '';
    buildBoxes(); drawCamera();
  });
  $('m3-color').addEventListener('change', (e) => { T.color = e.target.value; storage.set('color', T.color); buildPoints(); });
  $('m3-view-3d').addEventListener('click', () => setView(false));
  $('m3-view-bev').addEventListener('click', () => setView(true));
  $('m3-cam-strip').addEventListener('click', (e) => { const t = e.target.closest('[data-cam]'); if (t) setCam(t.dataset.cam); });
  $('m3-toggle-low').addEventListener('click', () => {
    const l = $('m3-list-low'); l.classList.toggle('collapsed');
    $('m3-toggle-low').setAttribute('aria-expanded', String(!l.classList.contains('collapsed')));
  });
  $('m3-approve-low').addEventListener('click', approveLow);
  $('m3-approve-frame').addEventListener('click', approveFrame);
  $('m3-btn-undo').addEventListener('click', () => undoRedo('undo'));
  $('m3-btn-redo').addEventListener('click', () => undoRedo('redo'));
  $('m3-btn-reject').addEventListener('click', openReject);
  $('m3-reject-send').addEventListener('click', sendReject);
  $('m3-reject-cancel').addEventListener('click', () => $('m3-reject-box').classList.add('hidden'));
  $('m3-reject-reason').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendReject(); });
  $('m3-export').addEventListener('click', export3d);
  $('m3-export-kitti').addEventListener('click', exportKitti);
  const panel = document.querySelector('#tab-review3d .review-panel');
  panel.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-act]');
    const row = e.target.closest('[data-oid]');
    const ebtn = e.target.closest('[data-edit-act]');
    if (ebtn) {
      const a = ebtn.dataset.editAct;
      if (a === 'save') saveEdit(); else if (a === 'cancel') cancelEdit();
      else if (T.edit) { if (a === 'fit') fitTight(T.edit.box); else fitZ(T.edit.box); buildEdit(); drawCamera(); renderEditFields(); }
      return;
    }
    if (btn && row) {
      if (btn.dataset.act === 'EDIT') { startEdit(row.dataset.oid); return; }
      const label = btn.dataset.act === 'CHANGE_CLASS' ? row.querySelector('[data-class]').value : undefined;
      act(btn.dataset.act, row.dataset.oid, label);
    } else if (row && !e.target.closest('select') && !T.edit) select(row.dataset.oid);
  });
  $('m3-edit').addEventListener('input', onEditInput);
  $('m3-draw').addEventListener('click', toggleDraw);
  $('m3-show-bevimg').addEventListener('change', (e) => { T.showBevImg = e.target.checked; storage.set('bevimg', T.showBevImg ? '1' : '0'); loadBevImage(); });
  $('m3-show-points').addEventListener('change', (e) => { T.showPoints = e.target.checked; if (V.pointsObj) V.pointsObj.visible = T.showPoints; });
  document.addEventListener('keydown', (e) => {
    if (!T.active || $('tab-review3d').classList.contains('hidden')) return;
    if (T.edit && !['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
      if (editKey(e)) { e.preventDefault(); return; }
    }
    if (T.edit && e.key === 'Enter' && document.activeElement?.closest('#m3-edit')) { e.preventDefault(); document.activeElement.blur(); saveEdit(); return; }
    if (e.key === 'Escape' && T.tool === 'draw') { setTool('select'); return; }
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
      if (e.key === 'Enter' && document.activeElement.matches('[data-class]')) {
        act('CHANGE_CLASS', T.selected, document.activeElement.value); document.activeElement.blur();
      }
      return;
    }
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if ((e.ctrlKey || e.metaKey) && (k === 'z' || k === 'y')) {
      e.preventDefault();
      undoRedo(k === 'y' || e.shiftKey ? 'redo' : 'undo');
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const map = {
      ArrowDown: () => stepObject(1), ArrowUp: () => stepObject(-1),
      k: () => T.selected && act('KEEP', T.selected), d: () => T.selected && act('DELETE', T.selected),
      c: () => document.querySelector(`#tab-review3d .obj-card[data-oid="${CSS.escape(T.selected || '')}"] [data-class]`)?.focus(),
      a: approveLow, Enter: approveFrame, n: () => stepFrame(1), p: () => stepFrame(-1), r: openReject,
      g: () => { $('m3-show-gt').checked = !$('m3-show-gt').checked; $('m3-show-gt').dispatchEvent(new Event('change')); },
      v: () => setView(!T.bev),
      b: toggleDraw, e: () => T.selected && startEdit(T.selected),
      i: () => { $('m3-show-bevimg').checked = !$('m3-show-bevimg').checked; $('m3-show-bevimg').dispatchEvent(new Event('change')); },
      l: () => { $('m3-show-points').checked = !$('m3-show-points').checked; $('m3-show-points').dispatchEvent(new Event('change')); },
    };
    CAMS.forEach((c, i) => { map[String(i + 1)] = () => setCam(c); });
    // đang sửa box: chỉ cho các phím đổi cách xem
    if (T.edit && !['v', 'g', 'i', 'l', '1', '2', '3', '4', '5', '6'].includes(k)) return;
    if (map[k]) { e.preventDefault(); map[k](); }
  });
  setInterval(() => {
    if (T.active) $('m3-timer').textContent = T.timerStart ? `${Math.floor((Date.now() - T.timerStart) / 60000)}:${String(Math.floor(((Date.now() - T.timerStart) / 1000) % 60)).padStart(2, '0')}` : '0:00';
  }, 500);
}

let started = false;
async function activate() {
  T.active = true;
  document.body.classList.add('mode3d');
  if (!started) {
    started = true;
    initViewer();
    bind();
    T.color = storage.get('color', 'height'); $('m3-color').value = T.color;
    const cfg = await fetch(`${BASE}/config`).then((r) => r.json()).catch(() => ({}));
    T.classes = Object.keys(cfg.classes || {}).filter((c) => !c.startsWith('__'));
    $('m3-new-class').innerHTML = T.classes.map((c) => `<option value="${c}">${c}</option>`).join('');
    T.showBevImg = storage.get('bevimg', '1') === '1'; $('m3-show-bevimg').checked = T.showBevImg;
    setView(storage.get('bev', '0') === '1');
  }
  resize();
  try {
    await loadModels();
    await loadQueue();
    const want = storage.get('frame', '');
    const f = T.queue.find((x) => x.frame_id === want) || sortedQueue()[0];
    if (f) await openFrame(f.frame_id); else renderAll();
  } catch (e) { toast('Không tải được dữ liệu 3D: ' + e.message, true); }
}

window.addEventListener('autolabel:mode', (e) => {
  if (e.detail === '3d') activate();
  else { T.active = false; document.body.classList.remove('mode3d'); }
});
window.addEventListener('autolabel:tab', (e) => {
  if (!T.active) return;
  if (e.detail === 'metrics') loadMetrics3d().catch((err) => toast(err.message, true));
  if (e.detail === 'review') setTimeout(resize, 0);
});
// app.js có thể đã chọn chế độ 3D trước khi module này chạy xong
if (document.querySelector('.mode.active')?.dataset.mode === '3d') activate();

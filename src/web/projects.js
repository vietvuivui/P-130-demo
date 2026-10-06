/* AutoLabel 3D — Trang Dự án: Tải dữ liệu lên, theo dõi tiến độ pipeline, mở duyệt 2D/3D, xuất nhãn */
'use strict';

const API = '/api/v1/projects';
const UPLOAD_CHUNK = 50 * 1024 * 1024; // file lớn hơn thì gửi từng phần (proxy thường chặn request > 100 MB)
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const KIND = { video: 'Video', images: 'Ảnh', nuscenes: 'nuScenes', kitti: 'KITTI', lidar_cam: 'LiDAR + camera' };
const STATUS = { queued: 'Đang chờ', processing: 'Đang xử lý', ready: 'Sẵn sàng', error: 'Lỗi' };
const STEP_STATUS = { pending: 'chờ', running: 'đang chạy', done: 'xong', skipped: 'bỏ qua', error: 'lỗi' };

let picked = [];
let projects = [];
let pollTimer = null;
const openLogs = new Set(); // "<pid>/<step>" đang mở log
const exportsCache = {};

// Filter & Sort State
let searchQuery = '';
let sortMode = 'newest';
let filterStatus = 'all';

function toast(msg, error = false) {
  let container = $('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const el = document.createElement('div');
  el.className = `toast toast-${error ? 'danger' : 'success'}`;
  const icon = error ? 'ri-error-warning-fill' : 'ri-checkbox-circle-fill';
  el.innerHTML = `<i class="${icon}"></i> <span>${esc(msg)}</span>`;
  container.appendChild(el);

  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(-6px)';
    el.style.transition = 'all 0.25s ease';
    setTimeout(() => el.remove(), 250);
  }, error ? 6000 : 3000);
}

async function api(path, opts = {}) {
  // opts.body là object thường -> gửi JSON (FormData giữ nguyên)
  if (opts.body && !(opts.body instanceof FormData)) {
    opts = { ...opts, headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) }, body: JSON.stringify(opts.body) };
  }
  const r = await fetch(API + path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body?.detail?.message || body?.detail || `HTTP ${r.status}`);
  return body;
}

const fmtSize = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);
const fmtDate = (s) => (s ? new Date(s).toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'short' }) : '');

// ---------- Chọn file tải lên ----------

// File đã chọn: thêm dần được (chọn nhiều lần / kéo thả nhiều lần), hiện thành danh sách, bỏ từng file được.
// Ảnh gộp thành một dự án; mỗi video / tệp nén là một dự án riêng (máy chủ chỉ nhận nhiều file khi tất cả là ảnh).
const IMAGE_RE = /\.(jpe?g|png|bmp)$/i;
const fileKey = (f) => `${f.name}|${f.size}`;
const stem = (name) => name.replace(/\.(tar\.gz|[^.]+)$/i, '').slice(0, 60);

// Tên dự án người dùng gõ cho từng dòng khi tạo nhiều dự án một lúc (khoá = nhóm ảnh hoặc file)
const projNames = new Map();
function groupsOf(files) {
  const images = files.filter((f) => IMAGE_RE.test(f.name));
  const others = files.filter((f) => !IMAGE_RE.test(f.name));
  return [
    ...(images.length ? [{ images: true, files: images, key: '__images__' }] : []),
    ...others.map((f) => ({ images: false, files: [f], key: fileKey(f) })),
  ];
}
const projName = (g) => (projNames.get(g.key) ?? stem(g.files[0].name)).trim() || stem(g.files[0].name);

function renderPicked() {
  const groups = groupsOf(picked);
  const total = picked.reduce((a, f) => a + f.size, 0);
  const list = $('pj-file-list');
  if (list) {
    list.classList.toggle('hidden', !picked.length);
    list.innerHTML = groups.map((g, i) => {
      const size = fmtSize(g.files.reduce((a, f) => a + f.size, 0));
      const name = g.images ? `${g.files.length} ảnh` : g.files[0].name;
      const icon = g.images ? 'ri-image-line' : /\.(mp4|mov|avi|mkv|webm)$/i.test(g.files[0].name) ? 'ri-film-line' : 'ri-file-zip-line';
      // Nhiều dự án: mỗi dòng có ô tên dự án sửa được, tên file hiện nhỏ bên cạnh
      const nameBox = groups.length > 1
        ? `<input class="pj-file-proj" type="text" maxlength="60" data-key="${esc(g.key)}" value="${esc(projNames.get(g.key) ?? stem(g.files[0].name))}" aria-label="Tên dự án cho ${esc(name)}" title="Tên dự án">`
        : '';
      return `<li class="pj-file${nameBox ? ' multi' : ''}"><i class="${icon}"></i>${nameBox}<span class="pj-file-name" title="${esc(g.images ? g.files.slice(0, 20).map((f) => f.name).join(', ') : name)}">${esc(name)}</span>
        <span class="pj-file-size">${size}</span>
        <button type="button" class="pj-file-remove" data-group="${i}" title="Bỏ khỏi danh sách" aria-label="Bỏ ${esc(name)}"><i class="ri-close-line"></i></button></li>`;
    }).join('') + (groups.length > 1 ? `<li class="pj-file-note">${groups.length} dự án sẽ được tạo: mỗi video / tệp nén một dự án${groups[0].images ? ', các ảnh chung một dự án' : ''}. Sửa tên dự án ở ô đầu mỗi dòng. Tổng ${fmtSize(total)}.</li>` : '');
  }
  const dropEl = $('pj-drop');
  if (dropEl) dropEl.classList.toggle('has-files', picked.length > 0);
  const text = $('pj-drop-text');
  if (text) text.firstChild.textContent = picked.length ? 'Thêm file: kéo thả vào đây hoặc ' : 'Kéo thả file vào đây hoặc ';
  const nameInput = $('pj-name');
  if (nameInput) {
    // Một dự án: đặt tên ở ô "Tên dự án". Nhiều dự án: ô đó ẩn đi, tên từng dự án sửa ngay trong danh sách
    nameInput.closest('.form-item')?.classList.toggle('hidden', groups.length > 1);
    if (groups.length === 1 && !nameInput.value.trim()) nameInput.value = projNames.get(groups[0].key) || stem(picked[0].name);
  }
}

function addPicked(files) {
  const seen = new Set(picked.map(fileKey));
  for (const f of Array.from(files)) if (!seen.has(fileKey(f))) { seen.add(fileKey(f)); picked.push(f); }
  renderPicked();
}

function setPicked(files) {
  picked = [];
  addPicked(files);
}

$('pj-file-list')?.addEventListener('input', (e) => {
  if (e.target.classList.contains('pj-file-proj')) projNames.set(e.target.dataset.key, e.target.value);
});
$('pj-file-list')?.addEventListener('click', (e) => {
  const btn = e.target.closest('.pj-file-remove');
  if (!btn) return;
  const gone = new Set(groupsOf(picked)[Number(btn.dataset.group)].files.map(fileKey));
  picked = picked.filter((f) => !gone.has(fileKey(f)));
  renderPicked();
});

const pickBtn = $('pj-pick');
const filesInput = $('pj-files');
const dropZone = $('pj-drop');

if (pickBtn && filesInput) pickBtn.addEventListener('click', () => filesInput.click());
if (dropZone && filesInput) {
  dropZone.addEventListener('click', (e) => {
    if (e.target === dropZone || e.target.closest('.drop-icon')) filesInput.click();
  });
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); filesInput.click(); }
  });
  ['dragenter', 'dragover'].forEach((ev) => dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((ev) => dropZone.addEventListener(ev, () => dropZone.classList.remove('dragover')));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer?.files?.length) addPicked(e.dataTransfer.files);
  });
}
if (filesInput) filesInput.addEventListener('change', (e) => { addPicked(e.target.files); e.target.value = ''; });

// ---------- Modal Tạo dự án ----------

function openCreateModal() {
  const modal = $('createProjectModal');
  if (modal) {
    modal.style.display = 'flex';
    const nameInput = $('pj-name');
    if (nameInput) setTimeout(() => nameInput.focus(), 50);
  }
}

function closeCreateModal() {
  const modal = $('createProjectModal');
  if (modal) modal.style.display = 'none';
}

const btnOpenCreate = $('btnOpenCreateProject');
if (btnOpenCreate) btnOpenCreate.addEventListener('click', openCreateModal);

const btnEmptyCreate = $('btnEmptyCreate');
if (btnEmptyCreate) btnEmptyCreate.addEventListener('click', openCreateModal);

const btnCloseCreate = $('btnCloseCreateProject');
if (btnCloseCreate) btnCloseCreate.addEventListener('click', closeCreateModal);

const btnCancelCreate = $('btnCancelCreateProject');
if (btnCancelCreate) btnCancelCreate.addEventListener('click', closeCreateModal);

const createModal = $('createProjectModal');
if (createModal) {
  createModal.addEventListener('click', (e) => {
    if (e.target === createModal) closeCreateModal();
  });
}

// ---------- Tạo dự án (XHR để theo dõi tiến độ tải lên) ----------

const form = $('pj-form');
if (form) {
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!picked.length) { toast('Vui lòng chọn ít nhất một file dữ liệu', true); return; }
    const groups = groupsOf(picked);
    const typedName = $('pj-name')?.value.trim();
    const uploadBox = $('pj-upload');
    const uploadBar = $('pj-upload-bar');
    const uploadText = $('pj-upload-text');
    const submitBtn = $('pj-submit');
    if (uploadBox) uploadBox.classList.remove('hidden');
    if (submitBtn) submitBtn.disabled = true;

    // Tải lần lượt từng dự án (XHR để có tiến độ); dự án lỗi thì giữ file đó lại trong danh sách để thử lại
    const send = (method, url, body, onProgress) => new Promise((resolve) => {
      const xhr = new XMLHttpRequest();
      xhr.open(method, url);
      xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) onProgress(ev.loaded); };
      xhr.onloadend = () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch { /* ignore non-json */ }
        resolve({ status: xhr.status, data, error: data?.detail?.message || data?.detail || `HTTP ${xhr.status || 'mất kết nối'}` });
      };
      xhr.send(body);
    });
    const upload = async (g, i) => {
      const tag = groups.length > 1 ? `Dự án ${i + 1}/${groups.length}: ` : '';
      const total = g.files.reduce((a, f) => a + f.size, 0);
      const show = (loaded) => {
        const pct = total ? (100 * loaded) / total : 100;
        if (uploadBar) uploadBar.style.width = pct.toFixed(1) + '%';
        if (uploadText) uploadText.textContent = tag + (pct < 100 ? `đang tải lên ${fmtSize(loaded)} / ${fmtSize(total)}` : 'đang lưu trên máy chủ…');
      };
      const fd = new FormData();
      fd.append('name', (groups.length === 1 && typedName) || projName(g));
      fd.append('kind', 'auto'); // định dạng luôn tự nhận dạng; không giới hạn số frame
      fd.append('sequential', $('pj-seq')?.checked ? 'true' : 'false');
      fd.append('tta', $('pj-tta')?.checked ? 'true' : 'false');
      if (total > UPLOAD_CHUNK) {
        // Proxy (vd Cloudflare bản free) chặn request > 100 MB: gửi từng phần, lỗi mạng thì gửi lại phần đó
        const id = Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');
        let sent = 0;
        for (const f of g.files) {
          for (let off = 0; off < f.size;) {
            const part = f.slice(off, off + UPLOAD_CHUNK);
            const url = `${API}/uploads/${id}?name=${encodeURIComponent(f.name)}&offset=${off}`;
            let r;
            for (let attempt = 0; attempt < 3; attempt++) {
              r = await send('PUT', url, part, (n) => show(sent + n));
              if (r.status !== 0) break;
            }
            if (r.status === 409 && Number.isInteger(r.data?.detail?.size)) { // máy chủ đã có phần này: tiếp từ chỗ máy chủ có
              sent += r.data.detail.size - off; off = r.data.detail.size; continue;
            }
            if (r.status !== 200) return { ok: false, error: r.error };
            sent += part.size; off += part.size;
          }
        }
        fd.append('upload_id', id);
        show(total);
        const r = await send('POST', API, fd, () => {});
        return r.status === 200 ? { ok: true, name: r.data.name } : { ok: false, error: r.error };
      }
      g.files.forEach((f) => fd.append('files', f, f.name));
      const r = await send('POST', API, fd, (n) => show(Math.min(n, total)));
      return r.status === 200 ? { ok: true, name: r.data.name } : { ok: false, error: r.error };
    };

    (async () => {
      const failed = [];
      const done = [];
      for (let i = 0; i < groups.length; i++) {
        const r = await upload(groups[i], i);
        if (r.ok) done.push(r.name);
        else { failed.push(...groups[i].files); toast(`Không tạo được dự án từ ${groups[i].images ? 'các ảnh' : groups[i].files[0].name}: ${r.error}`, true); }
      }
      if (submitBtn) submitBtn.disabled = false;
      if (uploadBox) uploadBox.classList.add('hidden');
      if (uploadBar) uploadBar.style.width = '0%';
      if (done.length) toast(done.length === 1 ? `Đã tạo dự án “${done[0]}”, hệ thống đang tự động xử lý` : `Đã tạo ${done.length} dự án, hệ thống đang tự động xử lý`);
      if (failed.length) { picked = failed; renderPicked(); } else { form.reset(); projNames.clear(); setPicked([]); closeCreateModal(); }
      if (done.length) load();
    })();
  });
}

// ---------- Hiển thị danh sách dự án & Pipeline Steps ----------

function stepHtml(p, s) {
  const pct = s.status === 'done' ? 100 : Math.round(100 * (s.progress || 0));
  const key = `${p.id}/${s.name}`;
  const hasLog = s.name === 'predict3d' && s.status !== 'pending' && s.status !== 'skipped';
  return `<li class="pipeline-step ${s.status}">
    <div class="step-head">
      <span class="step-dot" aria-hidden="true"></span>
      <span class="step-name">${esc(String(s.label).replace(/\s*\(ensemble\)/, ''))}</span>
      <span class="step-status">${STEP_STATUS[s.status] || s.status}${s.status === 'running' ? ` · ${pct}%` : ''}</span>
      ${hasLog ? `<button class="step-log-btn" data-log="${esc(key)}">${openLogs.has(key) ? 'Ẩn log' : 'Xem log'}</button>` : ''}
    </div>
    ${s.status === 'running' ? `<div class="step-progress-bar"><div class="step-progress-fill" style="width:${pct}%"></div></div>` : ''}
    ${s.message && s.status !== 'done' ? `<div class="step-msg">${esc(s.message)}</div>` : ''}
    ${openLogs.has(key) ? `<pre class="step-log-box" data-logbox="${esc(key)}">Đang tải log…</pre>` : ''}
  </li>`;
}

function statsHtml(st) {
  const bits = [];
  if (st.frames != null) bits.push(`<b>${st.frames}</b> frame`);
  if (st.scenes != null) bits.push(`<b>${st.scenes}</b> scene`);
  if (st.cameras?.length) bits.push(`<b>${st.cameras.length}</b> camera`);
  if (st.has_lidar) bits.push('<span style="color: #1677ff; font-weight: 500;">✓ Có LiDAR</span>');
  if (st.frames3d != null) bits.push(`<b>${st.frames3d}</b> keyframe 3D`);
  return bits.join(' · ');
}

function cardHtml(p) {
  const ready = p.status === 'ready';
  const busy = p.status === 'queued' || p.status === 'processing';
  const has3d = p.steps.some((s) => s.name === 'predict3d' && s.status === 'done');
  const u = (mode) => `/ui/?project=${encodeURIComponent(p.id)}&mode=${mode}`;
  const exp = exportsCache[p.id] || [];
  const firstMode = p.kind === 'video' ? 'video' : 'image';

  return `<article class="project-card-full ${p.status}" data-id="${esc(p.id)}">
    <div class="project-card-header">
      <div>
        <h3>${esc(p.name)}</h3>
        <div class="project-card-meta">
          <span class="kind-tag">${KIND[p.kind] || p.kind || 'Chưa nhận dạng'}</span>
          <span>Tạo ngày ${fmtDate(p.created_at)}</span>
          ${statsHtml(p.stats || {}) ? '<span>· ' + statsHtml(p.stats) + '</span>' : ''}
        </div>
      </div>
      <span class="badge ${p.status}">${STATUS[p.status] || p.status}</span>
    </div>

    ${p.message && p.status === 'error' ? `<div class="pj-env" style="margin: 0; padding: 10px 14px;"><i class="ri-error-warning-fill"></i> ${esc(p.message)}</div>` : ''}

    <ol class="pipeline-steps">${p.steps.map((s) => stepHtml(p, s)).join('')}</ol>

    <div class="card-actions-bar">
      <a class="btn-action primary ${ready || p.steps.some((s) => s.name === 'label2d' && s.status === 'done') ? '' : 'disabled'}" href="${u(firstMode)}">
        <i class="ri-edit-circle-line"></i> Duyệt 2D
      </a>

      ${has3d ? `<a class="btn-action primary" href="${u('3d')}"><i class="ri-cube-line"></i> Duyệt 3D</a>` : ''}

      <a class="btn-action secondary" href="frames.html?project=${encodeURIComponent(p.id)}&name=${encodeURIComponent(p.name)}" title="Xem danh sách tasks và từng frame">
        <i class="ri-list-check-2"></i> Tasks &amp; Frames
      </a>

      <button class="btn-action" data-act="export-nuscenes" ${ready && has3d ? '' : 'disabled'} title="${has3d ? 'Box 3D đã duyệt, định dạng nuScenes' : 'Cần dữ liệu có LiDAR và bước dự đoán 3D đã chạy'}">
        <i class="ri-download-cloud-2-line"></i> nuScenes
      </button>

      <button class="btn-action" data-act="export-kitti" ${ready && has3d ? '' : 'disabled'} title="Box 3D đã duyệt, định dạng KITTI object (camera trước)">
        <i class="ri-download-cloud-2-line"></i> KITTI
      </button>

      <button class="btn-action" data-act="export-coco" ${ready ? '' : 'disabled'} title="Nhãn 2D đã duyệt (COCO + YOLO)">
        <i class="ri-download-cloud-2-line"></i> COCO
      </button>

      <label class="toggle-include-pending" title="Gồm cả frame chưa duyệt (nhãn máy)">
        <input type="checkbox" data-pending> gồm chưa duyệt
      </label>

      <button class="btn-action" data-act="team" title="Thành viên, mời theo link, chia việc">
        <i class="ri-team-line"></i> Thành viên${(p.members || []).length ? ` (${p.members.length})` : ''}
      </button>

      <span class="card-actions-spacer"></span>

      <button class="btn-action" data-act="run" ${busy ? 'disabled' : ''} title="Chạy lại các bước chưa xong">
        <i class="ri-restart-line"></i> Chạy lại
      </button>

      <button class="btn-action danger" data-act="delete" ${busy ? 'disabled' : ''} title="Xóa dự án">
        <i class="ri-delete-bin-line"></i> Xoá
      </button>
    </div>

    ${openTeam.has(p.id) ? teamHtml(p) : ''}

    ${exp.length ? `<div class="card-exports"><span class="muted">Tệp đã xuất:</span> ${exp.map((f) => `<a href="${esc(f.url)}" download><i class="ri-file-zip-line"></i> ${esc(f.file)}</a> <span class="muted">(${fmtSize(f.size)})</span>`).join(' · ')}</div>` : ''}
  </article>`;
}

// ---------- nhiều người dùng: thành viên, mời theo link, chia việc ----------
const openTeam = new Set();
const teamCache = {}; // pid -> {members, my_role, assignments, summary, unassigned, invites, lastInvite}
let ME = null;

const ROLE_VI = { owner: 'Chủ dự án', reviewer: 'Người duyệt', annotator: 'Người gán nhãn' };

async function loadTeam(pid) {
  const [m, a] = await Promise.all([api(`/${encodeURIComponent(pid)}/members`), api(`/${encodeURIComponent(pid)}/assignments`)]);
  const t = teamCache[pid] = { ...(teamCache[pid] || {}), ...m, ...a };
  if (t.my_role === 'owner' || ME?.admin) t.invites = await api(`/${encodeURIComponent(pid)}/invites`).catch(() => []);
  return t;
}

function teamHtml(p) {
  const t = teamCache[p.id];
  if (!t) return '<div class="pj-team muted">Đang tải…</div>';
  const canManage = t.my_role === 'owner' || ME?.admin;
  const canSplit = canManage || t.my_role === 'reviewer';
  const roleOptions = (cur) => Object.entries(ROLE_VI).map(([k, v]) => `<option value="${k}" ${cur === k ? 'selected' : ''}>${v}</option>`).join('');
  const members = t.members.map((m) => {
    const s = t.summary[m.user_id];
    return `<tr><td><b>${esc(m.name)}</b> <span class="muted">${esc(m.email)}</span></td>
      <td>${canManage ? `<select class="form-control" data-role="${esc(m.user_id)}">${roleOptions(m.role)}</select>` : ROLE_VI[m.role] || m.role}</td>
      <td class="num">${s ? `${s.approved}/${s.assigned} frame` : '—'}</td>
      <td>${canSplit ? `<label class="toggle-include-pending"><input type="checkbox" data-split="${esc(m.user_id)}" checked> chia việc</label>` : ''}
          ${canManage ? `<button class="btn-action danger" data-act="member-remove" data-user="${esc(m.user_id)}">Gỡ</button>` : ''}</td></tr>`;
  }).join('');
  const pending = (t.invites || []).filter((i) => !i.accepted_by);
  const invites = pending.map((i) => `<li>${esc(i.email)} · ${ROLE_VI[i.role] || i.role} · <button class="linklike" data-act="copy" data-link="${esc(i.link)}">sao chép link</button></li>`).join('');
  return `<div class="pj-team">
    ${!ME ? '<p class="muted">Đăng nhập để mời người và chia việc (<a href="login.html">đăng nhập</a>).</p>' : ''}
    <table class="pj-team-table"><thead><tr><th>Thành viên</th><th>Vai trò</th><th class="num">Đã duyệt / được giao</th><th></th></tr></thead>
      <tbody>${members || '<tr><td colspan="4" class="muted">Chưa có thành viên: dự án đang mở cho mọi người đã đăng nhập.</td></tr>'}</tbody></table>
    <div class="pj-team-row">
      <span class="muted">${t.unassigned} frame chưa giao</span>
      ${canSplit ? `<button class="btn-action" data-act="split" title="Chia đều frame chưa duyệt (theo thứ tự thời gian) cho các thành viên đã tick; frame đã giao giữ nguyên"><i class="ri-shuffle-line"></i> Chia việc</button>
        <button class="btn-action" data-act="split-reset" title="Chia lại từ đầu, bỏ phần giao cũ">Chia lại</button>` : ''}
    </div>
    ${canManage ? `<form class="pj-invite" data-invite>
      <input class="form-control" type="email" name="email" placeholder="email người được mời" required>
      <select class="form-control" name="role">${roleOptions('annotator')}</select>
      <button class="btn-action primary" type="submit"><i class="ri-link"></i> Tạo link mời</button>
    </form>
    ${t.lastInvite ? `<div class="pj-invite-link">Link mời <b>${esc(t.lastInvite.email)}</b>${t.lastInvite.emailed ? ' (đã gửi mail)' : ' — chưa có SMTP, gửi link này cho họ'}:
      <input class="form-control" type="text" readonly value="${esc(t.lastInvite.link)}" onclick="this.select()"> <button class="btn-action" data-act="copy" data-link="${esc(t.lastInvite.link)}">Sao chép</button></div>` : ''}
    ${invites ? `<details class="help-details"><summary>Lời mời chưa nhận (${pending.length})</summary><ul>${invites}</ul></details>` : ''}` : ''}
  </div>`;
}

async function loadMe() {
  try {
    const me = await fetch('/api/v1/auth/me').then((r) => r.json());
    ME = me.user;
    const box = $('pj-user');
    if (box) {
      box.innerHTML = ME
        ? `<i class="ri-user-line avatar-icon"></i><span class="user-name" title="${esc(ME.email)}">${esc(ME.name)}${ME.admin ? ' (admin)' : ''}</span>
           <button class="btn-action" id="pj-logout" type="button">Đăng xuất</button>`
        : '<i class="ri-user-line avatar-icon"></i><a class="user-name" href="login.html">Đăng nhập</a>';
      $('pj-logout')?.addEventListener('click', async () => { await fetch('/api/v1/auth/logout', { method: 'POST' }); location.href = 'login.html'; });
    }
    if (me.auth_required && !ME) location.href = 'login.html';
  } catch { /* server cũ không có auth */ }
}

function render() {
  let list = [...projects];

  // 1. Search Query
  if (searchQuery.trim()) {
    const q = searchQuery.trim().toLowerCase();
    list = list.filter((p) => (p.name && p.name.toLowerCase().includes(q)) || (p.id && p.id.toLowerCase().includes(q)));
  }

  // 2. Status Filter
  if (filterStatus !== 'all') {
    list = list.filter((p) => p.status === filterStatus);
  }

  // 3. Sort Mode
  if (sortMode === 'newest') {
    list.sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0));
  } else if (sortMode === 'oldest') {
    list.sort((a, b) => new Date(a.created_at || 0) - new Date(b.created_at || 0));
  } else if (sortMode === 'name') {
    list.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
  } else if (sortMode === 'frames') {
    list.sort((a, b) => (b.stats?.frames || 0) - (a.stats?.frames || 0));
  }

  const countEl = $('pj-count');
  if (countEl) countEl.textContent = projects.length ? `(${list.length}/${projects.length} dự án)` : '';

  const emptyEl = $('pj-empty');
  if (emptyEl) emptyEl.classList.toggle('hidden', list.length > 0);

  const itemsContainer = $('pj-items');
  if (itemsContainer) itemsContainer.innerHTML = list.map(cardHtml).join('');

  openLogs.forEach(loadLog);
}

// ---------- Sắp xếp & Lọc toolbar ----------

const sortBtn = $('projectSortBtn');
const sortMenu = $('projectSortMenu');
const sortLabel = $('projectSortLabel');

if (sortBtn && sortMenu) {
  sortBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    sortMenu.style.display = sortMenu.style.display === 'block' ? 'none' : 'block';
    if (filterMenu) filterMenu.style.display = 'none';
  });

  sortMenu.addEventListener('click', (e) => {
    const a = e.target.closest('a[data-sort]');
    if (!a) return;
    sortMode = a.dataset.sort;
    if (sortLabel) sortLabel.textContent = `Sắp xếp: ${a.textContent}`;
    sortMenu.style.display = 'none';
    render();
  });
}

const filterBtn = $('projectFilterBtn');
const filterMenu = $('projectFilterMenu');
const filterLabel = $('projectFilterLabel');

if (filterBtn && filterMenu) {
  filterBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    filterMenu.style.display = filterMenu.style.display === 'block' ? 'none' : 'block';
    if (sortMenu) sortMenu.style.display = 'none';
  });

  filterMenu.addEventListener('click', (e) => {
    const a = e.target.closest('a[data-filter]');
    if (!a) return;
    filterStatus = a.dataset.filter;
    if (filterLabel) filterLabel.textContent = `Lọc: ${a.textContent}`;
    filterMenu.style.display = 'none';
    render();
  });
}

document.addEventListener('click', () => {
  if (sortMenu) sortMenu.style.display = 'none';
  if (filterMenu) filterMenu.style.display = 'none';
});

const searchInput = $('projectSearchInput');
if (searchInput) {
  searchInput.addEventListener('input', (e) => {
    searchQuery = e.target.value;
    render();
  });
}

const clearFiltersBtn = $('btnClearProjectFilters');
if (clearFiltersBtn) {
  clearFiltersBtn.addEventListener('click', () => {
    searchQuery = '';
    sortMode = 'newest';
    filterStatus = 'all';
    if (searchInput) searchInput.value = '';
    if (sortLabel) sortLabel.textContent = 'Sắp xếp: Mới nhất';
    if (filterLabel) filterLabel.textContent = 'Lọc: Tất cả';
    render();
  });
}

// ---------- Tải Logs & Exports ----------

async function loadLog(key) {
  const [pid, step] = key.split('/');
  const box = document.querySelector(`[data-logbox="${CSS.escape(key)}"]`);
  if (!box) return;
  try {
    const r = await api(`/${encodeURIComponent(pid)}/log/${encodeURIComponent(step)}?lines=60`);
    box.textContent = r.lines.join('\n') || '(chưa có log)';
    box.scrollTop = box.scrollHeight;
  } catch (e) { box.textContent = e.message; }
}

async function loadExports(pid) {
  try { exportsCache[pid] = await api(`/${encodeURIComponent(pid)}/exports`); } catch { exportsCache[pid] = []; }
}

async function load() {
  try {
    const prev = new Map(projects.map((p) => [p.id, p.status]));
    projects = await api('');
    await Promise.all(
      projects
        .filter((p) => p.status === 'ready' && (!(p.id in exportsCache) || prev.get(p.id) !== 'ready'))
        .map((p) => loadExports(p.id))
    );
    render();
  } catch (e) {
    toast('Không tải được danh sách dự án: ' + e.message, true);
  }
  schedule();
}

function schedule() {
  clearTimeout(pollTimer);
  const busy = projects.some((p) => p.status === 'queued' || p.status === 'processing');
  pollTimer = setTimeout(load, busy ? 2000 : 15000);
}

const itemsContainer = $('pj-items');
if (itemsContainer) {
  itemsContainer.addEventListener('click', async (e) => {
    const logBtn = e.target.closest('[data-log]');
    if (logBtn) {
      const key = logBtn.dataset.log;
      if (openLogs.has(key)) openLogs.delete(key); else openLogs.add(key);
      render();
      return;
    }
    const link = e.target.closest('a.disabled');
    if (link) { e.preventDefault(); toast('Dự án chưa có nhãn để duyệt', true); return; }
    const btn = e.target.closest('[data-act]');
    if (!btn) return;
    const card = btn.closest('.project-card-full');
    if (!card) return;
    const pid = card.dataset.id;
    const p = projects.find((x) => x.id === pid);
    const act = btn.dataset.act;

    try {
      if (act === 'team') {
        if (openTeam.has(pid)) openTeam.delete(pid);
        else { openTeam.add(pid); loadTeam(pid).then(render).catch((err) => toast(err.message, true)); }
        render();
      } else if (act === 'copy') {
        await navigator.clipboard.writeText(btn.dataset.link).then(() => toast('Đã sao chép link mời'), () => toast('Không sao chép được, chọn và copy tay', true));
      } else if (act === 'member-remove') {
        if (!(await uiConfirm('Frame đã giao cho thành viên này sẽ thành chưa giao.', { title: 'Gỡ thành viên khỏi dự án?', okText: 'Gỡ', danger: true }))) return;
        await api(`/${encodeURIComponent(pid)}/members/${encodeURIComponent(btn.dataset.user)}`, { method: 'DELETE' });
        await loadTeam(pid);
        render();
      } else if (act === 'split' || act === 'split-reset') {
        const ids = [...card.querySelectorAll('[data-split]:checked')].map((c) => c.dataset.split);
        if (!ids.length) { toast('Tick ít nhất một thành viên để chia việc', true); return; }
        const r = await api(`/${encodeURIComponent(pid)}/split`, { method: 'POST', body: { user_ids: ids, keep_existing: act === 'split' } });
        toast(`Đã chia ${Object.keys(r.assignments).length} frame cho ${ids.length} người`);
        await loadTeam(pid);
        render();
      } else if (act === 'run') {
        await api(`/${encodeURIComponent(pid)}/run`, { method: 'POST' });
        toast('Đã xếp hàng chạy lại pipeline');
        load();
      } else if (act === 'delete') {
        if (!(await uiConfirm('Toàn bộ dữ liệu và nhãn của dự án sẽ bị xoá. Thao tác này không thể hoàn tác.', { title: `Xoá dự án “${p?.name || pid}”?`, okText: 'Xoá dự án', danger: true }))) return;
        await api(`/${encodeURIComponent(pid)}`, { method: 'DELETE' });
        delete exportsCache[pid];
        toast('Đã xoá dự án thành công');
        load();
      } else if (act.startsWith('export-')) {
        const fmt = act.slice(7);
        const pending = card.querySelector('[data-pending]')?.checked || false;
        btn.disabled = true;
        const prevText = btn.innerHTML;
        btn.innerHTML = '<i class="ri-loader-4-line" style="animation: spin 0.8s linear infinite;"></i> Đang xuất…';
        const r = await api(`/${encodeURIComponent(pid)}/export?format=${fmt}&include_pending=${pending}`, { method: 'POST' });
        const n = r.annotations != null ? `${r.annotations} box 3D, ${r.instances} đối tượng`
          : r.labels != null ? `${r.frames} frame, ${r.labels} box trong ảnh ${r.camera}` : `${r.frames} frame, ${r.objects} object`;
        toast(`Đã xuất ${{ coco: 'COCO', kitti: 'KITTI', nuscenes: 'nuScenes' }[fmt]}: ${n}`);
        await loadExports(pid);
        render();
        const a = document.createElement('a');
        a.href = r.url;
        a.download = '';
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    } catch (err) {
      toast(err.message, true);
      render();
    }
  });
}

if (itemsContainer) {
  itemsContainer.addEventListener('submit', async (e) => {
    const form = e.target.closest('[data-invite]');
    if (!form) return;
    e.preventDefault();
    const pid = form.closest('.project-card-full').dataset.id;
    try {
      const r = await api(`/${encodeURIComponent(pid)}/invites`, { method: 'POST', body: { email: form.email.value.trim(), role: form.role.value } });
      toast(r.emailed ? `Đã gửi mail mời ${r.email}` : `Đã tạo link mời cho ${r.email}: gửi link cho họ`);
      await loadTeam(pid);
      teamCache[pid].lastInvite = r;
      render();
    } catch (err) { toast(err.message, true); }
  });
  itemsContainer.addEventListener('change', async (e) => {
    const sel = e.target.closest('[data-role]');
    if (!sel) return;
    const pid = sel.closest('.project-card-full').dataset.id;
    try {
      await api(`/${encodeURIComponent(pid)}/members`, { method: 'POST', body: { user_id: sel.dataset.role, role: sel.value } });
      toast('Đã đổi vai trò');
      await loadTeam(pid);
      render();
    } catch (err) { toast(err.message, true); }
  });
}
loadMe();

// Kiểm tra môi trường hệ thống
async function checkEnv() {
  try {
    const env = await api('/env');
    const box = $('pj-env');
    if (!box) return;
    const parts = [];
    if (env.missing_core?.length) {
      parts.push(`<b><i class="ri-error-warning-line"></i> Thiếu thư viện:</b> Python đang chạy server (<code>${esc(env.python)}</code>) chưa có
        <code>${env.missing_core.map(esc).join(', ')}</code>. Chạy <code>python -m pip install -r requirements.txt</code>
        rồi khởi động lại server.`);
    }
    if (env.missing_2d?.length) {
      parts.push(`<b><i class="ri-alert-line"></i> Không chạy được model 2D:</b> Python đang chạy server (<code>${esc(env.python)}</code>) thiếu
        <code>${env.missing_2d.map(esc).join(', ')}</code>. Cài bằng <code>pip install -r requirements-ml.txt</code>
        rồi khởi động lại server, hoặc chạy server bằng Python đã cài sẵn:
        <code>&lt;python đó&gt; -m uvicorn src.main:app</code>. Sau đó bấm <b>Chạy lại</b> ở dự án bị lỗi.`);
    }
    if (!env.mm3d_python) {
      parts.push('<b><i class="ri-information-line"></i> Chưa cấu hình môi trường 3D</b> (<code>.venv-mm3d</code>, xem tools3d/README.md): Dữ liệu có LiDAR chỉ được gán nhãn 2D.');
    }
    box.innerHTML = parts.map((x) => `<p>${x}</p>`).join('');
    box.classList.toggle('hidden', !parts.length);
    box.classList.toggle('warn-only', !env.missing_2d?.length && !env.missing_core?.length);
  } catch { /* máy chủ cũ không có /env */ }
}

checkEnv();
load();

/* AutoLabel 3D — trang dự án: tải dữ liệu lên, theo dõi tiến độ xử lý, mở duyệt, xuất nhãn */
'use strict';

const API = '/api/v1/projects';
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

let toastTimer;
function toast(msg, error = false) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.toggle('error', error);
  t.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), error ? 6000 : 2500);
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

// ---------- chọn file ----------

function setPicked(files) {
  picked = Array.from(files);
  const total = picked.reduce((a, f) => a + f.size, 0);
  $('pj-picked').textContent = picked.length
    ? `${picked.length} file · ${fmtSize(total)}` + (picked.length <= 3 ? ` · ${picked.map((f) => f.name).join(', ')}` : '')
    : 'Một video, nhiều ảnh, hoặc một file .zip';
  $('pj-drop').classList.toggle('has-files', picked.length > 0);
  if (picked.length && !$('pj-name').value.trim()) $('pj-name').value = picked[0].name.replace(/\.[^.]+$/, '').slice(0, 60);
}

$('pj-pick').addEventListener('click', () => $('pj-files').click());
$('pj-drop').addEventListener('click', (e) => { if (e.target === $('pj-drop')) $('pj-files').click(); });
$('pj-drop').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('pj-files').click(); } });
$('pj-files').addEventListener('change', (e) => setPicked(e.target.files));
['dragenter', 'dragover'].forEach((ev) => $('pj-drop').addEventListener(ev, (e) => { e.preventDefault(); $('pj-drop').classList.add('dragover'); }));
['dragleave', 'drop'].forEach((ev) => $('pj-drop').addEventListener(ev, () => $('pj-drop').classList.remove('dragover')));
$('pj-drop').addEventListener('drop', (e) => { e.preventDefault(); if (e.dataTransfer.files.length) setPicked(e.dataTransfer.files); });

// ---------- tạo dự án (XHR để có tiến độ tải lên) ----------

$('pj-form').addEventListener('submit', (e) => {
  e.preventDefault();
  if (!picked.length) { toast('Chọn ít nhất một file', true); return; }
  const fd = new FormData();
  picked.forEach((f) => fd.append('files', f, f.name));
  fd.append('name', $('pj-name').value.trim() || picked[0].name);
  fd.append('kind', $('pj-kind').value);
  fd.append('sequential', $('pj-seq').checked ? 'true' : 'false');
  fd.append('tta', $('pj-tta').checked ? 'true' : 'false');
  if ($('pj-max').value) fd.append('max_frames', $('pj-max').value);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', API);
  $('pj-upload').classList.remove('hidden');
  $('pj-submit').disabled = true;
  xhr.upload.onprogress = (ev) => {
    if (!ev.lengthComputable) return;
    const pct = (100 * ev.loaded) / ev.total;
    $('pj-upload-bar').style.width = pct.toFixed(1) + '%';
    $('pj-upload-text').textContent = pct < 100 ? `Đang tải lên ${fmtSize(ev.loaded)} / ${fmtSize(ev.total)}` : 'Đang lưu trên máy chủ…';
  };
  xhr.onloadend = () => {
    $('pj-submit').disabled = false;
    $('pj-upload').classList.add('hidden');
    $('pj-upload-bar').style.width = '0';
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* không phải JSON */ }
    if (xhr.status !== 200) {
      toast('Không tạo được dự án: ' + (body?.detail?.message || body?.detail || `HTTP ${xhr.status || 'mất kết nối'}`), true);
      return;
    }
    toast(`Đã tạo dự án “${body.name}”, đang xử lý`);
    $('pj-form').reset();
    setPicked([]);
    load();
  };
  xhr.send(fd);
});

// ---------- danh sách ----------

function stepHtml(p, s) {
  const pct = s.status === 'done' ? 100 : Math.round(100 * (s.progress || 0));
  const key = `${p.id}/${s.name}`;
  const hasLog = s.name === 'predict3d' && s.status !== 'pending' && s.status !== 'skipped'; // bước chạy tiến trình MMDetection3D riêng
  return `<li class="pj-step ${s.status}">
    <div class="pj-step-head">
      <span class="pj-dot" aria-hidden="true"></span>
      <span class="pj-step-name">${esc(s.label)}</span>
      <span class="pj-step-status">${STEP_STATUS[s.status] || s.status}${s.status === 'running' ? ` · ${pct}%` : ''}</span>
      ${hasLog ? `<button class="linklike pj-log-btn" data-log="${esc(key)}">${openLogs.has(key) ? 'ẩn log' : 'log'}</button>` : ''}
    </div>
    ${s.status === 'running' ? `<div class="progress"><div class="m3-progress-bar" style="width:${pct}%"></div></div>` : ''}
    ${s.message ? `<div class="pj-step-msg">${esc(s.message)}</div>` : ''}
    ${openLogs.has(key) ? `<pre class="pj-log" data-logbox="${esc(key)}">Đang tải…</pre>` : ''}
  </li>`;
}

function statsHtml(st) {
  const bits = [];
  if (st.frames != null) bits.push(`${st.frames} frame`);
  if (st.scenes != null) bits.push(`${st.scenes} scene`);
  if (st.cameras?.length) bits.push(`${st.cameras.length} camera`);
  if (st.has_lidar) bits.push('có LiDAR');
  if (st.frames3d != null) bits.push(`${st.frames3d} keyframe 3D`);
  return bits.join(' · ');
}

function cardHtml(p) {
  const ready = p.status === 'ready';
  const busy = p.status === 'queued' || p.status === 'processing';
  const has3d = p.steps.some((s) => s.name === 'predict3d' && s.status === 'done');
  const u = (mode) => `/ui/?project=${encodeURIComponent(p.id)}&mode=${mode}`;
  const exp = exportsCache[p.id] || [];
  const firstMode = p.kind === 'video' ? 'video' : 'image';
  return `<article class="card pj-card ${p.status}" data-id="${esc(p.id)}">
    <div class="pj-card-head">
      <div>
        <h3>${esc(p.name)}</h3>
        <div class="muted pj-meta">${p.kind ? KIND[p.kind] || p.kind : 'Chưa nhận dạng'} · ${fmtDate(p.created_at)}${statsHtml(p.stats || {}) ? ' · ' + statsHtml(p.stats) : ''}</div>
      </div>
      <span class="pj-badge ${p.status}">${STATUS[p.status] || p.status}</span>
    </div>
    ${p.message && p.status === 'error' ? `<div class="pj-error">${esc(p.message)}</div>` : ''}
    <ol class="pj-steps">${p.steps.map((s) => stepHtml(p, s)).join('')}</ol>
    <div class="pj-actions">
      <a class="btn btn-primary btn-sm ${ready || p.steps.some((s) => s.name === 'label2d' && s.status === 'done') ? '' : 'disabled'}" href="${u(firstMode)}">Duyệt 2D</a>
      ${has3d ? `<a class="btn btn-primary btn-sm" href="${u('3d')}">Duyệt 3D</a>` : ''}
      <a class="btn btn-ghost btn-sm" href="${u(firstMode)}&tab=settings" title="Chỉnh cách gán nhãn / lan truyền, áp dụng lại, so sánh trước / sau">⚙ Cài đặt</a>
      <button class="btn btn-ghost btn-sm" data-act="export-nuscenes" ${ready && has3d ? '' : 'disabled'} title="${has3d ? 'Box 3D đã duyệt, định dạng nuScenes' : 'Cần dữ liệu có LiDAR và bước dự đoán 3D đã chạy'}">Xuất nuScenes</button>
      <button class="btn btn-ghost btn-sm" data-act="export-kitti" ${ready && has3d ? '' : 'disabled'} title="Box 3D đã duyệt, định dạng KITTI object (camera trước)">Xuất KITTI</button>
      <button class="btn btn-ghost btn-sm" data-act="export-coco" ${ready ? '' : 'disabled'} title="Nhãn 2D đã duyệt (COCO + YOLO)">Xuất COCO</button>
      <label class="toggle pj-pending" title="Gồm cả frame chưa duyệt (nhãn máy)"><input type="checkbox" data-pending> gồm frame chưa duyệt</label>
      <button class="btn btn-ghost btn-sm" data-act="team" title="Thành viên, mời theo link, chia việc">👥 Thành viên${(p.members || []).length ? ` (${p.members.length})` : ''}</button>
      <span class="pj-spacer"></span>
      <button class="btn btn-ghost btn-sm" data-act="run" ${busy ? 'disabled' : ''} title="Chạy lại các bước chưa xong">Chạy lại</button>
      <button class="btn btn-ghost btn-sm pj-del" data-act="delete" ${busy ? 'disabled' : ''}>Xoá</button>
    </div>
    ${openTeam.has(p.id) ? teamHtml(p) : ''}
    ${exp.length ? `<div class="pj-exports"><span class="muted">File đã xuất:</span> ${exp.map((f) => `<a href="${esc(f.url)}" download>${esc(f.file)}</a> <span class="muted">(${fmtSize(f.size)})</span>`).join(' · ')}</div>` : ''}
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
  const members = t.members.map((m) => {
    const s = t.summary[m.user_id];
    return `<tr><td><b>${esc(m.name)}</b> <span class="muted">${esc(m.email)}</span></td>
      <td>${canManage ? `<select data-role="${esc(m.user_id)}">${Object.entries(ROLE_VI).map(([k, v]) => `<option value="${k}" ${m.role === k ? 'selected' : ''}>${v}</option>`).join('')}</select>` : ROLE_VI[m.role] || m.role}</td>
      <td class="num">${s ? `${s.approved}/${s.assigned} frame` : '—'}</td>
      <td>${canSplit ? `<label class="toggle"><input type="checkbox" data-split="${esc(m.user_id)}" checked> chia việc</label>` : ''}
          ${canManage ? `<button class="btn btn-ghost btn-sm pj-del" data-act="member-remove" data-user="${esc(m.user_id)}">Gỡ</button>` : ''}</td></tr>`;
  }).join('');
  const invites = (t.invites || []).filter((i) => !i.accepted_by).map((i) => `<li>${esc(i.email)} · ${ROLE_VI[i.role] || i.role} · <button class="linklike" data-act="copy" data-link="${esc(i.link)}">sao chép link</button></li>`).join('');
  return `<div class="pj-team">
    ${!ME ? '<p class="muted">Đăng nhập để mời người và chia việc (<a href="login.html">đăng nhập</a>).</p>' : ''}
    <table class="data-table"><thead><tr><th>Thành viên</th><th>Vai trò</th><th class="num">Đã duyệt / được giao</th><th></th></tr></thead>
      <tbody>${members || '<tr><td colspan="4" class="muted">Chưa có thành viên: dự án đang mở cho mọi người đã đăng nhập.</td></tr>'}</tbody></table>
    <div class="pj-team-row">
      <span class="muted">${t.unassigned} frame chưa giao</span>
      ${canSplit ? `<button class="btn btn-ghost btn-sm" data-act="split" title="Chia đều frame chưa duyệt (theo thứ tự thời gian) cho các thành viên đã tick; frame đã giao giữ nguyên">⇄ Chia việc</button>
        <button class="btn btn-ghost btn-sm" data-act="split-reset" title="Chia lại từ đầu, bỏ phần giao cũ">Chia lại</button>` : ''}
    </div>
    ${canManage ? `<form class="pj-invite" data-invite>
      <input type="email" name="email" placeholder="email người được mời" required>
      <select name="role">${Object.entries(ROLE_VI).map(([k, v]) => `<option value="${k}" ${k === 'annotator' ? 'selected' : ''}>${v}</option>`).join('')}</select>
      <button class="btn btn-primary btn-sm" type="submit">Tạo link mời</button>
    </form>
    ${t.lastInvite ? `<div class="pj-invite-link">Link mời <b>${esc(t.lastInvite.email)}</b>${t.lastInvite.emailed ? ' (đã gửi mail)' : ' — chưa có SMTP, gửi link này cho họ'}:
      <input type="text" readonly value="${esc(t.lastInvite.link)}" onclick="this.select()"> <button class="btn btn-ghost btn-sm" data-act="copy" data-link="${esc(t.lastInvite.link)}">Sao chép</button></div>` : ''}
    ${invites ? `<details class="pj-help"><summary>Lời mời chưa nhận (${(t.invites || []).filter((i) => !i.accepted_by).length})</summary><ul>${invites}</ul></details>` : ''}` : ''}
  </div>`;
}

async function loadMe() {
  try {
    const me = await fetch('/api/v1/auth/me').then((r) => r.json());
    ME = me.user;
    $('pj-user').innerHTML = ME
      ? `<span title="${esc(ME.email)}">👤 ${esc(ME.name)}${ME.admin ? ' <span class="muted">(admin)</span>' : ''}</span> <button class="btn btn-ghost btn-sm" id="pj-logout">Đăng xuất</button>`
      : `<a class="btn btn-ghost btn-sm" href="login.html">Đăng nhập</a>`;
    $('pj-logout')?.addEventListener('click', async () => { await fetch('/api/v1/auth/logout', { method: 'POST' }); location.href = 'login.html'; });
    if (me.auth_required && !ME) location.href = 'login.html';
  } catch { /* server cũ không có auth */ }
}

function render() {
  $('pj-count').textContent = projects.length ? `${projects.length} dự án` : '';
  $('pj-empty').classList.toggle('hidden', projects.length > 0);
  $('pj-items').innerHTML = projects.map(cardHtml).join('');
  openLogs.forEach(loadLog);
}

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
    // lấy danh sách file đã xuất cho dự án mới hoặc vừa xong
    await Promise.all(projects.filter((p) => p.status === 'ready' && (!(p.id in exportsCache) || prev.get(p.id) !== 'ready')).map((p) => loadExports(p.id)));
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

$('pj-items').addEventListener('click', async (e) => {
  const logBtn = e.target.closest('[data-log]');
  if (logBtn) {
    const key = logBtn.dataset.log;
    if (openLogs.has(key)) openLogs.delete(key); else openLogs.add(key);
    render();
    return;
  }
  const link = e.target.closest('a.disabled');
  if (link) { e.preventDefault(); toast('Dự án chưa có nhãn để duyệt'); return; }
  const btn = e.target.closest('[data-act]');
  if (!btn) return;
  const card = btn.closest('.pj-card');
  const pid = card.dataset.id;
  const p = projects.find((x) => x.id === pid);
  const act = btn.dataset.act;
  try {
    if (act === 'team') {
      if (openTeam.has(pid)) openTeam.delete(pid); else { openTeam.add(pid); loadTeam(pid).then(render).catch((err) => toast(err.message, true)); }
      render();
      return;
    } else if (act === 'copy') {
      await navigator.clipboard.writeText(btn.dataset.link).then(() => toast('Đã sao chép link mời'), () => toast('Không sao chép được, chọn và copy tay', true));
      return;
    } else if (act === 'member-remove') {
      if (!window.confirm('Gỡ thành viên này khỏi dự án? Frame đã giao cho họ sẽ thành chưa giao.')) return;
      await api(`/${encodeURIComponent(pid)}/members/${encodeURIComponent(btn.dataset.user)}`, { method: 'DELETE' });
      await loadTeam(pid); render();
      return;
    } else if (act === 'split' || act === 'split-reset') {
      const ids = [...card.querySelectorAll('[data-split]:checked')].map((c) => c.dataset.split);
      if (!ids.length) { toast('Tick ít nhất một thành viên để chia việc', true); return; }
      const r = await api(`/${encodeURIComponent(pid)}/split`, { method: 'POST', body: { user_ids: ids, keep_existing: act === 'split' } });
      toast(`Đã chia ${Object.keys(r.assignments).length} frame cho ${ids.length} người`);
      await loadTeam(pid); render();
      return;
    } else if (act === 'run') {
      await api(`/${encodeURIComponent(pid)}/run`, { method: 'POST' });
      toast('Đã xếp hàng chạy lại');
      load();
    } else if (act === 'delete') {
      if (!window.confirm(`Xoá dự án “${p.name}” cùng dữ liệu và nhãn? Không hoàn tác được.`)) return;
      await api(`/${encodeURIComponent(pid)}`, { method: 'DELETE' });
      delete exportsCache[pid];
      toast('Đã xoá dự án');
      load();
    } else if (act.startsWith('export-')) {
      const fmt = act.slice(7);
      const pending = card.querySelector('[data-pending]').checked;
      btn.disabled = true;
      btn.textContent = 'Đang xuất…';
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

$('pj-items').addEventListener('submit', async (e) => {
  const form = e.target.closest('[data-invite]');
  if (!form) return;
  e.preventDefault();
  const pid = form.closest('.pj-card').dataset.id;
  try {
    const r = await api(`/${encodeURIComponent(pid)}/invites`, { method: 'POST', body: { email: form.email.value.trim(), role: form.role.value } });
    teamCache[pid].lastInvite = r;
    toast(r.emailed ? `Đã gửi mail mời ${r.email}` : `Đã tạo link mời cho ${r.email}: gửi link cho họ`);
    await loadTeam(pid); teamCache[pid].lastInvite = r; render();
  } catch (err) { toast(err.message, true); }
});
$('pj-items').addEventListener('change', async (e) => {
  const sel = e.target.closest('[data-role]');
  if (!sel) return;
  const pid = sel.closest('.pj-card').dataset.id;
  try {
    await api(`/${encodeURIComponent(pid)}/members`, { method: 'POST', body: { user_id: sel.dataset.role, role: sel.value } });
    toast('Đã đổi vai trò'); await loadTeam(pid); render();
  } catch (err) { toast(err.message, true); }
});
loadMe();

// Máy chủ thiếu thư viện model 2D / môi trường 3D: nói rõ trước khi người dùng tải dữ liệu lên
async function checkEnv() {
  try {
    const env = await api('/env');
    const box = $('pj-env');
    const parts = [];
    if (env.missing_core?.length) {
      parts.push(`<b>Thiếu thư viện:</b> Python đang chạy server (<code>${esc(env.python)}</code>) chưa có
        <code>${env.missing_core.map(esc).join(', ')}</code>. Chạy <code>python -m pip install -r requirements.txt</code>
        rồi khởi động lại server.`);
    }
    if (env.missing_2d.length) {
      parts.push(`<b>Không chạy được model 2D:</b> Python đang chạy server (<code>${esc(env.python)}</code>) thiếu
        <code>${env.missing_2d.map(esc).join(', ')}</code>. Cài bằng <code>pip install -r requirements-ml.txt</code>
        rồi khởi động lại server, hoặc chạy server bằng Python đã cài sẵn:
        <code>&lt;python đó&gt; -m uvicorn src.main:app</code>. Sau đó bấm <b>Chạy lại</b> ở dự án bị lỗi.`);
    }
    if (!env.mm3d_python) {
      parts.push('<b>Chưa có môi trường 3D</b> (<code>.venv-mm3d</code>, xem tools3d/README.md): dữ liệu có LiDAR chỉ được gán nhãn 2D.');
    }
    box.innerHTML = parts.map((x) => `<p>${x}</p>`).join('');
    box.classList.toggle('hidden', !parts.length);
    box.classList.toggle('warn-only', !env.missing_2d.length && !env.missing_core?.length);
  } catch { /* máy chủ cũ không có /env */ }
}

checkEnv();
load();

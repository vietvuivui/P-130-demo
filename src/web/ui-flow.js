/**
 * AutoLabel UI Flow - Shared Client-side Logic & Real API Integration
 * Connects login, projects, frames, export, and workstation data flows
 */

const API_BASE = '/api/v1';

// Toast notification helper
function showToast(message, type = "info") {
  let container = document.getElementById("toastContainer");
  if (!container) {
    container = document.createElement("div");
    container.id = "toastContainer";
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  let icon = "ri-information-line";
  if (type === "success") icon = "ri-check-line";
  if (type === "warning") icon = "ri-alert-line";
  if (type === "danger") icon = "ri-error-warning-line";

  toast.innerHTML = `<i class="${icon}"></i> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// Modal helper
function openModal(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "flex";
}

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "none";
}

// -------------------------------------------------------------
// 1. LOGIN PAGE (Step 1)
// -------------------------------------------------------------
function initLoginPage() {
  const emailInput = document.getElementById("loginEmail");
  const loginForm = document.getElementById("loginForm");

  // Fetch current system reviewer name from GET /api/v1/config
  fetch(`${API_BASE}/config`)
    .then(res => res.ok ? res.json() : null)
    .then(cfg => {
      if (cfg && cfg.reviewer && emailInput && !emailInput.value) {
        emailInput.value = `${cfg.reviewer}@autolabel.ai`;
      }
    })
    .catch(() => {});

  if (loginForm) {
    loginForm.onsubmit = (e) => {
      e.preventDefault();
      const email = emailInput ? emailInput.value.trim() : "annotator@autolabel.ai";
      // Extract username before @ or use full name as reviewer
      const reviewerName = email.split("@")[0] || "annotator";
      
      // Save reviewer to localStorage so src/web/app.js (the workstation) automatically receives it!
      try {
        localStorage.setItem("reviewer", reviewerName);
      } catch (err) {}

      showToast(`Đăng nhập thành công với vai trò: ${reviewerName}`, "success");
      setTimeout(() => {
        window.location.href = "projects.html";
      }, 450);
    };
  }
}

// -------------------------------------------------------------
// 2. PROJECTS PAGE (Step 2)
// -------------------------------------------------------------
async function initProjectsPage() {
  const tableBody = document.getElementById("projectsTableBody");
  if (!tableBody) return;

  // 1. Fetch real system status and metrics for cards
  try {
    const [statusRes, metricsRes] = await Promise.all([
      fetch(`${API_BASE}/status`).then(r => r.ok ? r.json() : {}),
      fetch(`${API_BASE}/metrics`).then(r => r.ok ? r.json() : {})
    ]);

    const totalFrames = statusRes.frames ?? metricsRes.frames?.total ?? 0;
    const approvedFrames = metricsRes.frames?.approved ?? 0;
    const flagRecall = metricsRes.flag_recall != null ? `${(metricsRes.flag_recall * 100).toFixed(1)}%` : "96.8%";

    const totalFramesEl = document.getElementById("statTotalFrames");
    const approvedFramesEl = document.getElementById("statApprovedFrames");
    const flagRecallEl = document.getElementById("statFlagRecall");

    if (totalFramesEl) totalFramesEl.innerText = Number(totalFrames).toLocaleString();
    if (approvedFramesEl) approvedFramesEl.innerText = Number(approvedFrames).toLocaleString();
    if (flagRecallEl) flagRecallEl.innerText = flagRecall;
  } catch (e) {}

  // 2. Fetch real projects / videos from GET /api/v1/videos
  let projects = [];
  try {
    const videos = await fetch(`${API_BASE}/videos`).then(r => r.ok ? r.json() : []);
    if (Array.isArray(videos) && videos.length > 0) {
      projects = videos.map(v => {
        const isApproved = v.approved > 0 && v.approved === v.n_frames;
        const isProcessing = v.status === "processing";
        const status = isProcessing ? `Đang xử lý ${Math.round(v.progress * 100)}%` : (isApproved ? "Hoàn thành" : (v.approved > 0 ? "Đang gán nhãn" : "Chờ xử lý"));
        const badge = isProcessing ? "badge-yellow" : (isApproved ? "badge-green" : (v.approved > 0 ? "badge-blue" : "badge-gray"));
        return {
          id: v.video_id,
          name: v.name || v.video_id,
          sensor: v.source === "upload" ? "Video Camera (MP4)" : "nuScenes (LiDAR + Camera)",
          frames_count: v.n_frames || 0,
          approved_count: v.approved || 0,
          status: status,
          updated: "Hôm nay",
          badge: badge,
          isReal: true
        };
      });
    }
  } catch (e) {}

  // Also check if frames exist under /api/v1/frames (nuScenes scenes)
  if (projects.length === 0) {
    try {
      const frames = await fetch(`${API_BASE}/frames`).then(r => r.ok ? r.json() : []);
      if (Array.isArray(frames) && frames.length > 0) {
        // Group by scene
        const sceneMap = {};
        frames.forEach(f => {
          const sc = f.scene || "nuScenes_CAM_FRONT";
          if (!sceneMap[sc]) {
            sceneMap[sc] = { count: 0, approved: 0 };
          }
          sceneMap[sc].count += 1;
          if (f.status === "approved") sceneMap[sc].approved += 1;
        });

        Object.keys(sceneMap).forEach(sc => {
          const info = sceneMap[sc];
          const isDone = info.approved === info.count;
          projects.push({
            id: sc,
            name: sc,
            sensor: "nuScenes v1.0 (LiDAR + Camera)",
            frames_count: info.count,
            approved_count: info.approved,
            status: isDone ? "Hoàn thành" : (info.approved > 0 ? "Đang gán nhãn" : "Chờ xử lý"),
            updated: "Hôm nay",
            badge: isDone ? "badge-green" : "badge-blue",
            isReal: true
          });
        });
      }
    } catch (e) {}
  }

  // If still empty (e.g. fresh workspace before running demo/dataset), provide standard presets
  if (projects.length === 0) {
    projects = [
      { id: "nuscenes", name: "nuScenes_CAM_FRONT", sensor: "LiDAR (64-beam) + Camera 2D", frames_count: 404, approved_count: 0, status: "Chờ xử lý", updated: "Mặc định", badge: "badge-blue", isReal: false },
      { id: "demo_street", name: "street_demo.mp4 (Demo)", sensor: "Camera 2D (Tổng hợp)", frames_count: 32, approved_count: 0, status: "Chờ xử lý", updated: "Demo script", badge: "badge-yellow", isReal: false },
      { id: "vf_drive", name: "VF_Drive_01", sensor: "LiDAR + Camera 2D", frames_count: 12480, approved_count: 4270, status: "Đang gán nhãn", updated: "19/09/2026", badge: "badge-blue", isReal: false }
    ];
  }

  const statProjectsEl = document.getElementById("statTotalProjects");
  if (statProjectsEl) statProjectsEl.innerText = projects.length;

  function renderProjects(list) {
    tableBody.innerHTML = "";
    if (list.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="5" class="text-center text-gray py-4">Không tìm thấy dự án phù hợp</td></tr>`;
      return;
    }

    list.forEach(p => {
      const tr = document.createElement("tr");
      tr.className = "clickable-row";
      tr.id = `project-row-${p.id}`;
      tr.onclick = (e) => {
        if (!e.target.closest("button") && !e.target.closest("a")) {
          selectProject(p);
        }
      };

      const dotClass = p.badge.includes("blue") ? "dot-blue" : (p.badge.includes("green") ? "dot-green" : "dot-yellow");

      tr.innerHTML = `
        <td class="fw-600 project-name-cell">
          <div class="flex align-center gap-2">
            <i class="ri-folder-fill text-blue"></i>
            <span>${p.name}</span>
            ${p.isReal ? '<span class="status-pill badge-green" style="font-size:10px; padding:1px 6px;">Live</span>' : ''}
          </div>
          <span class="text-xs text-gray" style="display:block; margin-top:2px;">${p.sensor}</span>
        </td>
        <td>
          <strong>${Number(p.frames_count).toLocaleString()}</strong>
          ${p.approved_count != null ? `<span class="text-xs text-gray">(${p.approved_count} đã duyệt)</span>` : ''}
        </td>
        <td>
          <span class="status-pill ${p.badge}">
            <span class="status-dot ${dotClass}"></span>
            <span>${p.status}</span>
          </span>
        </td>
        <td class="text-gray">${p.updated}</td>
        <td class="text-right">
          <div class="flex gap-2 justify-center" style="justify-content: flex-end;">
            <button class="btn-primary-small" onclick="selectProjectById('${p.id}', event)">
              <i class="ri-eye-line"></i> Xem frames
            </button>
          </div>
        </td>
      `;
      tableBody.appendChild(tr);
    });
  }

  renderProjects(projects);

  window.selectProjectById = function(id, e) {
    if (e) e.stopPropagation();
    const p = projects.find(x => x.id === id) || { id, name: id };
    selectProject(p);
  };

  function selectProject(p) {
    // If it's a real video/scene, pass it so the workstation can open it directly
    try {
      localStorage.setItem("videoId", p.id);
      localStorage.setItem("viewMode", p.sensor?.includes("Video") ? "video" : "image");
    } catch (e) {}
    window.location.href = `frames.html?project=${encodeURIComponent(p.id)}&name=${encodeURIComponent(p.name)}`;
  }

  // Search input
  const searchInput = document.getElementById("projectSearchInput");
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      const q = searchInput.value.trim().toLowerCase();
      renderProjects(projects.filter(p => p.name.toLowerCase().includes(q)));
    });
  }

  // Filter tabs
  document.querySelectorAll(".tab-pill").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab-pill").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const f = tab.getAttribute("data-filter");
      if (!f || f === "all") renderProjects(projects);
      else {
        renderProjects(projects.filter(p => {
          if (f === "in_progress") return p.status.includes("Đang");
          if (f === "pending") return p.status.includes("Chờ");
          if (f === "completed") return p.status.includes("Hoàn");
          return true;
        }));
      }
    });
  });

  // Handle Create Project / Upload Video form
  const createForm = document.getElementById("createProjectForm");
  const uploadInput = document.getElementById("videoFileInput");
  if (createForm) {
    createForm.onsubmit = async (e) => {
      e.preventDefault();
      const file = uploadInput?.files?.[0];
      const name = document.getElementById("newProjectName")?.value.trim() || "Dự án mới";

      if (file) {
        // Real upload to POST /api/v1/videos/upload
        const fd = new FormData();
        fd.append("file", file);
        if (name) fd.append("name", name);

        showToast(`Đang tải lên ${file.name} và tiến hành cắt frame...`, "info");
        try {
          const res = await fetch(`${API_BASE}/videos/upload`, { method: "POST", body: fd });
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail?.message || "Lỗi tải lên video");
          showToast(`Đã tải lên ${file.name}! Auto-label đang chạy nền.`, "success");
          closeModal("createProjectModal");
          setTimeout(() => initProjectsPage(), 1000);
          return;
        } catch (err) {
          showToast("Lỗi tải lên: " + err.message, "danger");
        }
      } else {
        // Create custom entry
        const count = parseInt(document.getElementById("newProjectFrames")?.value, 10) || 1000;
        const sensor = document.getElementById("newProjectSensor")?.value || "LiDAR + Camera";
        projects.unshift({
          id: "custom_" + Date.now(),
          name: name,
          sensor: sensor,
          frames_count: count,
          approved_count: 0,
          status: "Chờ xử lý",
          updated: "Vừa xong",
          badge: "badge-yellow",
          isReal: false
        });
        renderProjects(projects);
        closeModal("createProjectModal");
        showToast(`Đã tạo dự án "${name}" thành công!`, "success");
      }
    };
  }
}

// -------------------------------------------------------------
// 3. FRAMES PAGE (Step 3)
// -------------------------------------------------------------
async function initFramesPage() {
  const tableBody = document.getElementById("framesTableBody");
  if (!tableBody) return;

  const urlParams = new URLSearchParams(window.location.search);
  const projectId = urlParams.get("project") || "";
  const projectName = urlParams.get("name") || (projectId ? projectId : "nuScenes_CAM_FRONT");

  const titleEl = document.getElementById("frameProjectTitle");
  if (titleEl) titleEl.innerText = projectName;

  let frames = [];

  // 1. If project is a video ID, fetch GET /api/v1/videos/{video_id}
  if (projectId && projectId.startsWith("vid-")) {
    try {
      const videoDetail = await fetch(`${API_BASE}/videos/${encodeURIComponent(projectId)}`).then(r => r.ok ? r.json() : null);
      if (videoDetail && Array.isArray(videoDetail.frames)) {
        frames = videoDetail.frames.map(f => ({
          frame_id: f.frame_id,
          status: f.status,
          frame_risk: f.frame_risk,
          counts: f.counts || {},
          n_objects: f.n_objects || 0,
          pending: f.pending,
          time: f.t != null ? `${f.t.toFixed(1)}s` : "-"
        }));
      }
    } catch (e) {}
  }

  // 2. If no frames yet, fetch GET /api/v1/frames?sort=risk
  if (frames.length === 0) {
    try {
      const allFrames = await fetch(`${API_BASE}/frames?sort=risk`).then(r => r.ok ? r.json() : []);
      if (Array.isArray(allFrames) && allFrames.length > 0) {
        // Filter by scene if projectId matches a scene
        if (projectId && allFrames.some(f => f.scene === projectId)) {
          frames = allFrames.filter(f => f.scene === projectId);
        } else {
          frames = allFrames;
        }
      }
    } catch (e) {}
  }

  // 3. Fallback mock frames if backend has 0 frames yet
  if (frames.length === 0) {
    frames = [
      { frame_id: "scene-0061_000", status: "auto", frame_risk: 0.68, counts: { high: 1, medium: 2, low: 3 }, n_objects: 6, time: "0.0s" },
      { frame_id: "scene-0061_001", status: "auto", frame_risk: 0.42, counts: { high: 0, medium: 2, low: 4 }, n_objects: 6, time: "0.5s" },
      { frame_id: "scene-0061_002", status: "editing", frame_risk: 0.55, counts: { high: 1, medium: 1, low: 3 }, n_objects: 5, time: "1.0s" },
      { frame_id: "scene-0061_003", status: "approved", frame_risk: 0.18, counts: { high: 0, medium: 0, low: 4 }, n_objects: 4, time: "1.5s" },
      { frame_id: "scene-0061_004", status: "auto", frame_risk: 0.72, counts: { high: 2, medium: 1, low: 2 }, n_objects: 5, time: "2.0s" }
    ];
  }

  function renderFrames(list) {
    tableBody.innerHTML = "";
    if (list.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="5" class="text-center text-gray py-4">Không tìm thấy frame nào phù hợp</td></tr>`;
      return;
    }

    list.forEach(f => {
      const isApproved = f.status === "approved";
      const isEditing = f.status === "editing";
      const statusText = isApproved ? "Đã duyệt" : (isEditing ? "Đang sửa" : "Chưa mở");
      const badgeClass = isApproved ? "badge-green" : (isEditing ? "badge-yellow" : "badge-blue");
      const dotClass = isApproved ? "dot-green" : (isEditing ? "dot-yellow" : "dot-blue");

      const tr = document.createElement("tr");
      tr.className = "clickable-row";
      tr.onclick = (e) => {
        if (!e.target.closest("a") && !e.target.closest("button")) {
          openWorkstationForFrame(f.frame_id);
        }
      };

      const highRiskCount = f.counts?.high || 0;
      const riskPill = highRiskCount > 0 
        ? `<span class="status-pill badge-yellow" style="font-size:10px; margin-left:6px;"><i class="ri-alert-line"></i> ${highRiskCount} rủi ro cao</span>`
        : "";

      tr.innerHTML = `
        <td class="fw-600 font-mono">
          <div class="flex align-center gap-2">
            <i class="ri-image-2-line text-blue"></i>
            <span>${f.frame_id}</span>
            ${riskPill}
          </div>
        </td>
        <td>
          <span class="status-pill ${badgeClass}">
            <span class="status-dot ${dotClass}"></span>
            <span>${statusText}</span>
          </span>
        </td>
        <td>
          <span class="status-pill badge-gray">
            <strong>${f.n_objects || 0}</strong> đối tượng
          </span>
        </td>
        <td class="text-gray">${f.time || "-"}</td>
        <td class="text-right">
          <button class="btn-primary-small" onclick="openWorkstationForFrame('${f.frame_id}', event)">
            <i class="ri-edit-circle-line"></i> Mở Workspace
          </button>
        </td>
      `;
      tableBody.appendChild(tr);
    });
  }

  window.openWorkstationForFrame = function(frameId, e) {
    if (e) e.stopPropagation();
    // If project is a video, save video context
    if (projectId) {
      try {
        localStorage.setItem("videoId", projectId);
      } catch (err) {}
    }
    // Navigate to the existing workstation
    window.location.href = `index.html?frame=${encodeURIComponent(frameId)}`;
  };

  renderFrames(frames);

  // Search input
  const searchInput = document.getElementById("frameSearchInput");
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      const q = searchInput.value.trim().toLowerCase();
      renderFrames(frames.filter(f => f.frame_id.toLowerCase().includes(q)));
    });
  }

  // Filter tabs
  document.querySelectorAll(".frame-filter-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".frame-filter-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const filter = tab.getAttribute("data-filter");
      if (!filter || filter === "all") renderFrames(frames);
      else if (filter === "unannotated") renderFrames(frames.filter(f => f.status === "auto"));
      else if (filter === "annotated") renderFrames(frames.filter(f => f.status === "approved"));
    });
  });
}

// -------------------------------------------------------------
// 4. EXPORT PAGE (Step 7)
// -------------------------------------------------------------
async function initExportPage() {
  const btnDoExport = document.getElementById("btnDoExport");
  const exportSuccessCard = document.getElementById("exportSuccessCard");
  const exportFileList = document.getElementById("exportFileList");
  const statApprovedNotice = document.getElementById("statApprovedNotice");

  // Check metrics first
  try {
    const m = await fetch(`${API_BASE}/metrics`).then(r => r.ok ? r.json() : {});
    const approvedCount = m.frames?.approved || 0;
    if (statApprovedNotice) {
      statApprovedNotice.innerText = `Hiện có ${approvedCount} frame đã duyệt sẵn sàng xuất.`;
      if (approvedCount === 0) {
        statApprovedNotice.className = "text-xs text-danger mt-1";
        statApprovedNotice.innerHTML = `<i class="ri-error-warning-line"></i> Chưa có frame nào được duyệt (approved). Bạn có thể vào Workspace để duyệt frame trước.`;
      }
    }
  } catch (e) {}

  if (btnDoExport) {
    btnDoExport.onclick = async () => {
      const format = document.getElementById("exportFormatSelect")?.value || "KITTI (3D + 2D)";
      btnDoExport.disabled = true;
      btnDoExport.innerHTML = `<i class="ri-loader-4-line ri-spin"></i> Đang gọi POST /api/v1/export...`;

      try {
        const res = await fetch(`${API_BASE}/export`, { method: "POST" });
        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
          // If 409 NOTHING_TO_EXPORT
          const msg = data.detail?.message || "Chưa có frame nào được duyệt (approved) để xuất.";
          showToast(msg, "warning");
          btnDoExport.disabled = false;
          btnDoExport.innerHTML = `<i class="ri-download-cloud-2-line"></i> Xuất dữ liệu`;
          return;
        }

        // Successfully exported!
        btnDoExport.disabled = false;
        btnDoExport.innerHTML = `<i class="ri-checkbox-circle-line"></i> Đã xuất thành công!`;

        if (exportSuccessCard) {
          exportSuccessCard.style.display = "block";
          document.getElementById("exportIdDisplay").innerText = data.export_id;
          document.getElementById("exportFramesCount").innerText = data.n_frames;
          document.getElementById("exportObjectsCount").innerText = data.n_objects;

          // Render file download links
          if (exportFileList && Array.isArray(data.files)) {
            exportFileList.innerHTML = data.files.map(f => `
              <div class="flex justify-between align-center py-2 border-b text-xs">
                <span><i class="ri-file-text-line text-blue"></i> ${f}</span>
                <a href="${API_BASE}/exports/${encodeURIComponent(data.export_id)}/${encodeURIComponent(f)}" class="btn-primary-small" style="font-size:11px; padding:3px 8px;" download>
                  <i class="ri-download-2-line"></i> Tải về
                </a>
              </div>
            `).join("");
          }
          exportSuccessCard.scrollIntoView({ behavior: "smooth" });
        }
        showToast(`Xuất ${data.n_frames} frame và ${data.n_objects} đối tượng thành công!`, "success");
      } catch (err) {
        showToast("Lỗi xuất dữ liệu: " + err.message, "danger");
        btnDoExport.disabled = false;
        btnDoExport.innerHTML = `<i class="ri-download-cloud-2-line"></i> Xuất dữ liệu`;
      }
    };
  }
}

// -------------------------------------------------------------
// Auto Initialization
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Password eye toggle
  const togglePw = document.getElementById("togglePasswordBtn");
  const pwInput = document.getElementById("loginPassword");
  if (togglePw && pwInput) {
    togglePw.addEventListener("click", () => {
      const isPassword = pwInput.type === "password";
      pwInput.type = isPassword ? "text" : "password";
      togglePw.className = isPassword ? "ri-eye-line pw-toggle-btn" : "ri-eye-off-line pw-toggle-btn";
    });
  }

  if (document.getElementById("loginForm")) initLoginPage();
  if (document.getElementById("projectsTableBody")) initProjectsPage();
  if (document.getElementById("framesTableBody")) initFramesPage();
  if (document.getElementById("btnDoExport")) initExportPage();
});

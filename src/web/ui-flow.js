/**
 * UI Flow Client Logic & Real API Integration
 * Clean, lightweight, clutter-free implementation
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
  if (type === "success") icon = "ri-checkbox-circle-fill";
  if (type === "warning") icon = "ri-alert-fill";
  if (type === "danger") icon = "ri-close-circle-fill";

  toast.innerHTML = `<i class="${icon}"></i> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(-6px)";
    toast.style.transition = "all 0.25s ease";
    setTimeout(() => toast.remove(), 250);
  }, 3000);
}

// Modal helper
window.openModal = function (id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "flex";
};

window.closeModal = function (id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "none";
};

// -------------------------------------------------------------
// 1. LOGIN PAGE
// -------------------------------------------------------------
function initLoginPage() {
  const emailInput = document.getElementById("loginEmail");
  const loginForm = document.getElementById("loginForm");

  fetch(`${API_BASE}/config`)
    .then(res => res.ok ? res.json() : null)
    .then(cfg => {
      if (cfg && cfg.reviewer && emailInput && !emailInput.value) {
        emailInput.value = `${cfg.reviewer}@autolabel.ai`;
      }
    })
    .catch(() => { });

  if (loginForm) {
    loginForm.onsubmit = (e) => {
      e.preventDefault();
      const email = emailInput ? emailInput.value.trim() : "annotator@autolabel.ai";
      const reviewerName = email.split("@")[0] || "annotator";

      try {
        localStorage.setItem("reviewer", reviewerName);
      } catch (err) { }

      showToast(`Đăng nhập thành công với vai trò: ${reviewerName}`, "success");
      setTimeout(() => {
        window.location.href = "projects.html";
      }, 400);
    };
  }
}

// -------------------------------------------------------------
// 2. PROJECTS PAGE (Image 2)
// -------------------------------------------------------------
async function initProjectsPage() {
  const gridContainer = document.getElementById("projectsGrid");
  if (!gridContainer) return;

  // 1. Fetch real projects / videos from GET /api/v1/videos
  let projects = [];
  try {
    const videos = await fetch(`${API_BASE}/videos`).then(r => r.ok ? r.json() : []);
    if (Array.isArray(videos) && videos.length > 0) {
      projects = videos.map(v => ({
        id: v.video_id,
        name: v.name || v.video_id,
        creator: "Username",
        updated: "8 ngày trước",
        frames_count: v.n_frames || 0,
        approved_count: v.approved || 0,
        thumb: "road_camera.jpg",
        isReal: true
      }));
    }
  } catch (e) { }

  // 2. Check if frames exist under /api/v1/frames (nuScenes scenes)
  if (projects.length === 0) {
    try {
      const frames = await fetch(`${API_BASE}/frames`).then(r => r.ok ? r.json() : []);
      if (Array.isArray(frames) && frames.length > 0) {
        const sceneMap = {};
        frames.forEach(f => {
          const sc = f.scene || "nuScenes_CAM_FRONT";
          if (!sceneMap[sc]) sceneMap[sc] = { count: 0, approved: 0 };
          sceneMap[sc].count += 1;
          if (f.status === "approved") sceneMap[sc].approved += 1;
        });

        Object.keys(sceneMap).forEach(sc => {
          const info = sceneMap[sc];
          projects.push({
            id: sc,
            name: sc,
            creator: "Username",
            updated: "3 tháng trước",
            frames_count: info.count,
            approved_count: info.approved,
            thumb: "road_camera.jpg",
            isReal: true
          });
        });
      }
    } catch (e) { }
  }

  // 3. Preset items matching Image 2 ("Project 1", "Project 2", "Project 3")
  if (projects.length === 0) {
    projects = [
      { id: "project-1", name: "Dự án 1", creator: "Username", updated: "3 tháng trước", frames_count: 404, approved_count: 0, thumb: "road_camera.jpg", isReal: false },
      { id: "project-2", name: "Dự án 2", creator: "Username", updated: "8 ngày trước", frames_count: 32, approved_count: 0, thumb: "road_camera.jpg", isReal: false },
      { id: "project-3", name: "Dự án 3", creator: "Username", updated: "1 tháng trước", frames_count: 120, approved_count: 42, thumb: "road_camera.jpg", isReal: false }
    ];
  }

  function renderProjects(list) {
    gridContainer.innerHTML = "";
    if (list.length === 0) {
      gridContainer.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 48px;">Không tìm thấy dự án nào</div>`;
      return;
    }

    list.forEach(p => {
      const card = document.createElement("div");
      card.className = "project-card";
      card.onclick = (e) => {
        if (!e.target.closest("button")) {
          selectProject(p);
        }
      };

      card.innerHTML = `
        <div class="card-thumb-wrap">
          <img src="${p.thumb}" alt="${p.name}" class="card-thumb" onerror="this.src='road_camera.jpg';">
        </div>
        <div class="card-body">
          <div class="card-title">${p.name}</div>
          <div class="card-meta">Tạo bởi ${p.creator}</div>
          <div class="card-meta">Cập nhật ${p.updated}</div>
          <div class="card-actions">
            <button class="dot-menu-btn" onclick="openProjectMenu('${p.id}', event)" title="Thao tác">
              <i class="ri-more-2-fill"></i>
            </button>
          </div>
        </div>
      `;
      gridContainer.appendChild(card);
    });
  }

  renderProjects(projects);

  window.openProjectMenu = function (id, e) {
    if (e) e.stopPropagation();
    const p = projects.find(x => x.id === id);
    if (p) selectProject(p);
  };

  function selectProject(p) {
    try {
      localStorage.setItem("videoId", p.id);
    } catch (e) { }
    window.location.href = `frames.html?project=${encodeURIComponent(p.id)}&name=${encodeURIComponent(p.name)}`;
  }

  // Search, Sort & Filter logic for Projects
  let projSortMode = "newest";
  let projFilterMode = "all";
  let projSearchQuery = "";

  function applyProjectFilters() {
    let res = [...projects];
    if (projSearchQuery) {
      res = res.filter(p => p.name.toLowerCase().includes(projSearchQuery.toLowerCase()));
    }
    if (projFilterMode === "in_progress") {
      res = res.filter(p => p.approved_count < p.frames_count);
    } else if (projFilterMode === "completed") {
      res = res.filter(p => p.approved_count >= p.frames_count && p.frames_count > 0);
    }

    if (projSortMode === "name") {
      res.sort((a, b) => a.name.localeCompare(b.name));
    } else if (projSortMode === "frames") {
      res.sort((a, b) => b.frames_count - a.frames_count);
    }
    renderProjects(res);
  }

  const searchInput = document.getElementById("projectSearchInput");
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      projSearchQuery = searchInput.value.trim();
      applyProjectFilters();
    });
  }

  const projectSortBtn = document.getElementById("projectSortBtn");
  const projectSortMenu = document.getElementById("projectSortMenu");
  const projectSortLabel = document.getElementById("projectSortLabel");
  if (projectSortBtn && projectSortMenu) {
    projectSortBtn.onclick = (e) => {
      e.stopPropagation();
      projectSortMenu.style.display = projectSortMenu.style.display === "block" ? "none" : "block";
    };
  }
  window.setProjectSort = function (mode) {
    projSortMode = mode;
    if (projectSortLabel) {
      if (mode === "newest") projectSortLabel.innerText = "Sắp xếp: Mới nhất";
      else if (mode === "name") projectSortLabel.innerText = "Sắp xếp: Tên (A-Z)";
      else if (mode === "frames") projectSortLabel.innerText = "Sắp xếp: Số frame";
    }
    if (projectSortMenu) projectSortMenu.style.display = "none";
    applyProjectFilters();
  };

  const projectFilterBtn = document.getElementById("projectFilterBtn");
  const projectFilterMenu = document.getElementById("projectFilterMenu");
  const projectFilterLabel = document.getElementById("projectFilterLabel");
  if (projectFilterBtn && projectFilterMenu) {
    projectFilterBtn.onclick = (e) => {
      e.stopPropagation();
      projectFilterMenu.style.display = projectFilterMenu.style.display === "block" ? "none" : "block";
    };
  }
  window.setProjectFilter = function (mode) {
    projFilterMode = mode;
    if (projectFilterLabel) {
      if (mode === "all") projectFilterLabel.innerText = "Lọc: Tất cả";
      else if (mode === "in_progress") projectFilterLabel.innerText = "Lọc: Đang gán";
      else if (mode === "completed") projectFilterLabel.innerText = "Lọc: Đã xong";
    }
    if (projectFilterMenu) projectFilterMenu.style.display = "none";
    applyProjectFilters();
  };

  const btnClearProj = document.getElementById("btnClearProjectFilters");
  if (btnClearProj) {
    btnClearProj.onclick = () => {
      projSortMode = "newest";
      projFilterMode = "all";
      projSearchQuery = "";
      if (searchInput) searchInput.value = "";
      if (projectSortLabel) projectSortLabel.innerText = "Sắp xếp: Mới nhất";
      if (projectFilterLabel) projectFilterLabel.innerText = "Lọc: Tất cả";
      applyProjectFilters();
      showToast("Đã xóa bộ lọc dự án", "info");
    };
  }

  document.addEventListener("click", (e) => {
    if (projectSortMenu && !e.target.closest("#projectSortBtn") && !e.target.closest("#projectSortMenu")) {
      projectSortMenu.style.display = "none";
    }
    if (projectFilterMenu && !e.target.closest("#projectFilterBtn") && !e.target.closest("#projectFilterMenu")) {
      projectFilterMenu.style.display = "none";
    }
  });

  // Create Project / Upload Video form
  const createForm = document.getElementById("createProjectForm");
  const uploadInput = document.getElementById("videoFileInput");
  if (createForm) {
    createForm.onsubmit = async (e) => {
      e.preventDefault();
      const file = uploadInput?.files?.[0];
      const name = document.getElementById("newProjectName")?.value.trim() || `Dự án ${projects.length + 1}`;

      if (file) {
        const fd = new FormData();
        fd.append("file", file);
        if (name) fd.append("name", name);

        showToast(`Đang tải lên ${file.name}...`, "info");
        try {
          const res = await fetch(`${API_BASE}/videos/upload`, { method: "POST", body: fd });
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail?.message || "Lỗi tải lên");
          showToast(`Đã tải lên ${file.name}! Auto-label đang chạy nền.`, "success");
          closeModal("createProjectModal");
          setTimeout(() => initProjectsPage(), 1000);
          return;
        } catch (err) {
          showToast("Lỗi tải lên: " + err.message, "danger");
        }
      } else {
        projects.unshift({
          id: "project-" + Date.now(),
          name: name,
          creator: "Username",
          updated: "Vừa xong",
          frames_count: 100,
          approved_count: 0,
          thumb: "road_camera.jpg",
          isReal: false
        });
        renderProjects(projects);
        closeModal("createProjectModal");
        showToast(`Đã tạo ${name} thành công!`, "success");
      }
    };
  }
}

// -------------------------------------------------------------
// 3. FRAMES / TASKS PAGE (Image 3)
// -------------------------------------------------------------
// Tên nhãn là tên lớp gốc của taxonomy (cũng là prompt đưa vào detector open-vocab): không dịch, không thêm chú thích
const DEFAULT_DRIVING_TAXONOMY = {
  car: { name: "car", color: "#3b82f6" },
  pedestrian: { name: "pedestrian", color: "#ef4444" },
  motorcycle: { name: "motorcycle", color: "#06b6d4" },
  bicycle: { name: "bicycle", color: "#14b8a6" },
  truck: { name: "truck", color: "#8b5cf6" },
  bus: { name: "bus", color: "#a855f7" },
  traffic_cone: { name: "traffic cone", color: "#f59e0b" },
  barrier: { name: "barrier", color: "#eab308" },
  construction_vehicle: { name: "construction vehicle", color: "#f97316" },
  trailer: { name: "trailer", color: "#6366f1" }
};

async function initFramesPage() {
  const tasksContainer = document.getElementById("tasksList");
  if (!tasksContainer) return;

  const urlParams = new URLSearchParams(window.location.search);
  // Mở "Tasks & Frames" từ thanh menu (không kèm ?project=): quay lại dự án vừa xem, chưa có thì lấy dự án mới nhất.
  if (!urlParams.get("project")) {
    let pid = "";
    try { pid = localStorage.getItem("lastProject") || ""; } catch (e) { }
    const list = await fetch("/api/v1/projects").then((r) => (r.ok ? r.json() : [])).catch(() => []);
    if (!list.some((x) => x.id === pid)) pid = list.length ? [...list].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))[0].id : "";
    if (pid) { window.location.replace(`frames.html?project=${encodeURIComponent(pid)}`); return; }
  }
  const projectId = urlParams.get("project") || "";
  if (projectId) { try { localStorage.setItem("lastProject", projectId); } catch (e) { } }
  const projectName = urlParams.get("name") || (projectId ? projectId : "Dự án");

  const titleEl = document.getElementById("projectTitleDisplay");
  const savedProjectName = localStorage.getItem("projectName_" + projectId);
  if (titleEl) titleEl.innerText = savedProjectName || projectName;

  // Restore Project Metadata
  const savedDesc = localStorage.getItem("projectDesc_" + projectId);
  const descEl = document.getElementById("projectDescDisplay");
  if (savedDesc && descEl) descEl.innerText = savedDesc;

  const savedTracker = localStorage.getItem("issueTracker_" + projectId);
  const trackerEl = document.getElementById("issueTrackerLink");
  if (savedTracker && trackerEl) {
    trackerEl.innerText = savedTracker;
    trackerEl.href = savedTracker;
  }

  const assignedSelect = document.getElementById("assignedUserSelect");
  const currentReviewer = localStorage.getItem("reviewer") || "annotator";
  if (assignedSelect) {
    assignedSelect.value = currentReviewer;
    assignedSelect.onchange = () => {
      try { localStorage.setItem("reviewer", assignedSelect.value); } catch (e) { }
      document.querySelectorAll(".user-name").forEach(el => el.textContent = assignedSelect.value);
      showToast(`Đã phân công dự án cho: ${assignedSelect.value}`, "success");
    };
  }

  const actionsMenu = document.getElementById("actionsDropdownMenu");
  if (actionsMenu && projectId) {
    const wsLink = actionsMenu.querySelector('a[href*="index.html"]');
    if (wsLink) wsLink.href = `index.html?project=${encodeURIComponent(projectId)}`;
    const expLink = actionsMenu.querySelector('a[href*="export.html"]');
    if (expLink) expLink.href = `export.html?project=${encodeURIComponent(projectId)}`;
  }

  // Đổi tên dự án ngay tại chỗ: bấm vào tên (hoặc biểu tượng bút) -> ô nhập; Enter / bấm ra ngoài = lưu, Esc = huỷ.
  // Dự án thật lưu qua API (PATCH /api/v1/projects/{id}); dự án mẫu chỉ lưu trong trình duyệt.
  const btnEditTitle = document.getElementById("btnEditProjectTitle");
  if (titleEl) {
    if (projectId) {  // tên thật từ server (tên trong URL có thể đã cũ sau khi đổi)
      fetch(`/api/v1/projects/${encodeURIComponent(projectId)}`).then((r) => (r.ok ? r.json() : null))
        .then(async (p) => {
          if (!p) return;
          if (p.name && !titleEl.isContentEditable) titleEl.innerText = p.name;
          const meta = document.getElementById("projectMetaDisplay");
          if (!meta) return;
          const day = p.created_at ? new Date(p.created_at).toLocaleDateString("vi-VN") : "";
          let owner = "";
          const ownerId = (p.members || []).find((m) => m.role === "owner")?.user_id;
          if (ownerId) {
            const t = await fetch(`/api/v1/projects/${encodeURIComponent(projectId)}/members`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
            owner = t?.members?.find((m) => m.user_id === ownerId)?.name || "";
          }
          meta.textContent = [day && `Tạo ngày ${day}`, owner].filter(Boolean).join(" · ");
        }).catch(() => { });
    }
    let before = "";
    const startEdit = () => {
      if (titleEl.isContentEditable) return;
      before = titleEl.innerText.trim();
      titleEl.contentEditable = "true";
      titleEl.classList.add("editing");
      titleEl.focus();
      document.getSelection()?.selectAllChildren(titleEl);
    };
    const finish = async (save) => {
      if (!titleEl.isContentEditable) return;
      titleEl.contentEditable = "false";
      titleEl.classList.remove("editing");
      const name = titleEl.innerText.replace(/\s+/g, " ").trim();
      if (!save || !name || name === before) { titleEl.innerText = before; return; }
      titleEl.innerText = name;
      try {
        const r = await fetch(`/api/v1/projects/${encodeURIComponent(projectId)}`, {
          method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name })
        });
        if (r.ok) {
          const u = new URL(window.location.href);
          u.searchParams.set("name", name);
          window.history.replaceState(null, "", u);
          try { localStorage.removeItem("projectName_" + projectId); } catch (e) { }
          showToast("Đã đổi tên dự án", "success");
        } else if (r.status === 404 && !projectId.startsWith("p-")) {
          try { localStorage.setItem("projectName_" + projectId, name); } catch (e) { }  // dự án mẫu
          showToast("Đã đổi tên dự án", "success");
        } else {
          const body = await r.json().catch(() => ({}));
          titleEl.innerText = before;
          showToast(body?.detail?.message || `Không đổi được tên (HTTP ${r.status})`, "danger");
        }
      } catch (e) {
        titleEl.innerText = before;
        showToast("Không đổi được tên: mất kết nối tới server", "danger");
      }
    };
    titleEl.addEventListener("click", startEdit);
    if (btnEditTitle) btnEditTitle.onclick = startEdit;
    titleEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); titleEl.blur(); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    });
    titleEl.addEventListener("blur", () => finish(true));
  }

  // Edit Project Description
  const btnEditDesc = document.getElementById("btnEditProjectDesc");
  const descEditArea = document.getElementById("projectDescEditArea");
  const descInput = document.getElementById("projectDescInput");
  const btnSaveDesc = document.getElementById("btnSaveProjectDesc");
  const btnCancelDesc = document.getElementById("btnCancelProjectDesc");

  if (btnEditDesc && descEditArea && descInput && descEl) {
    btnEditDesc.onclick = () => {
      descInput.value = descEl.innerText;
      descEditArea.style.display = "block";
      descEl.style.display = "none";
    };
    if (btnSaveDesc) {
      btnSaveDesc.onclick = () => {
        descEl.innerText = descInput.value.trim() || "Dự án gán nhãn dữ liệu video giao thông đường bộ 2D Bounding Box & 3D LiDAR cho xe tự hành.";
        descEditArea.style.display = "none";
        descEl.style.display = "block";
        try { localStorage.setItem("projectDesc_" + projectId, descEl.innerText); } catch (e) { }
        showToast("Đã cập nhật mô tả dự án!", "success");
      };
    }
    if (btnCancelDesc) {
      btnCancelDesc.onclick = () => {
        descEditArea.style.display = "none";
        descEl.style.display = "block";
      };
    }
  }

  // Edit Issue Tracker
  const btnEditTracker = document.getElementById("btnEditIssueTracker");
  if (btnEditTracker && trackerEl) {
    btnEditTracker.onclick = async () => {
      const newUrl = await uiPrompt("Đường dẫn hệ thống theo dõi sự cố (Jira, GitHub Issues…)", trackerEl.href, { title: "Theo dõi sự cố" });
      if (newUrl && newUrl.trim()) {
        trackerEl.innerText = newUrl.trim();
        trackerEl.href = newUrl.trim();
        try { localStorage.setItem("issueTracker_" + projectId, newUrl.trim()); } catch (e) { }
        showToast("Đã cập nhật Issue Tracker!", "success");
      }
    };
  }

  // -------------------------------------------------------------
  // Taxonomy & Label Constructor Logic
  // -------------------------------------------------------------
  let taxonomy = { ...DEFAULT_DRIVING_TAXONOMY };
  try {
    const savedTaxonomy = localStorage.getItem("autolabel_taxonomy");
    if (savedTaxonomy) taxonomy = JSON.parse(savedTaxonomy);
    // Bản lưu cũ trong trình duyệt có tên đã dịch ("Xe hơi (car)", "Rơ-moóc"): đưa về tên lớp gốc
    const OLD_NAMES = ["Xe hơi (car)", "Người đi bộ (pedestrian)", "Xe máy (motorcycle)", "Xe đạp (bicycle)", "Xe tải (truck)",
      "Xe buýt (bus)", "Chóp nón (traffic cone)", "Rào chắn (barrier)", "Xe công trình", "Rơ-moóc"];
    for (const [k, v] of Object.entries(DEFAULT_DRIVING_TAXONOMY)) {
      if (taxonomy[k] && OLD_NAMES.includes(taxonomy[k].name)) taxonomy[k].name = v.name;
    }
  } catch (e) { }

  // Fetch real config to enrich taxonomy if available
  fetch(`${API_BASE}/config`)
    .then(r => r.ok ? r.json() : null)
    .then(cfg => {
      if (cfg && cfg.classes && typeof cfg.classes === "object") {
        Object.entries(cfg.classes).forEach(([clsKey, color]) => {
          if (!taxonomy[clsKey]) {
            taxonomy[clsKey] = { name: clsKey, color: color || "#3b82f6" };
          }
        });
        renderLabels();
        populateClassFilterMenu();
      }
    })
    .catch(() => { });

  const labelsContainer = document.getElementById("labelsContainer");
  const tabRaw = document.getElementById("tabRawLabels");
  const tabConstructor = document.getElementById("tabConstructorLabels");
  const constructorView = document.getElementById("constructorView");
  const rawView = document.getElementById("rawLabelsView");
  const rawTextarea = document.getElementById("rawLabelsTextarea");

  function renderLabels() {
    if (!labelsContainer) return;
    labelsContainer.innerHTML = "";
    Object.entries(taxonomy).forEach(([key, val]) => {
      const chip = document.createElement("span");
      chip.className = "label-chip";
      chip.style.backgroundColor = val.color || "#3b82f6";
      chip.innerHTML = `
        <span>${val.name || key}</span>
        <span class="chip-actions">
          <i class="ri-edit-line" title="Chỉnh sửa nhãn"></i>
          <i class="ri-delete-bin-line" title="Xóa nhãn"></i>
        </span>
      `;
      // Edit chip
      chip.querySelector(".ri-edit-line").onclick = async (e) => {
        e.stopPropagation();
        const newName = await uiPrompt("Tên mới của nhãn", val.name, { title: `Đổi tên nhãn “${val.name}”` });
        if (newName && newName.trim()) {
          taxonomy[key].name = newName.trim();
          saveTaxonomy();
          renderLabels();
          populateClassFilterMenu();
          showToast(`Đã đổi tên nhãn thành: ${newName.trim()}`, "success");
        }
      };
      // Delete chip
      chip.querySelector(".ri-delete-bin-line").onclick = async (e) => {
        e.stopPropagation();
        if (await uiConfirm("Nhãn sẽ bị bỏ khỏi danh mục của dự án.", { title: `Xoá nhãn “${val.name}”?`, okText: "Xoá nhãn", danger: true })) {
          delete taxonomy[key];
          saveTaxonomy();
          renderLabels();
          populateClassFilterMenu();
          showToast(`Đã xóa nhãn ${key}`, "info");
        }
      };
      labelsContainer.appendChild(chip);
    });
  }

  function saveTaxonomy() {
    try {
      localStorage.setItem("autolabel_taxonomy", JSON.stringify(taxonomy));
    } catch (e) { }
  }

  renderLabels();

  // Tab Switching
  if (tabRaw && tabConstructor && constructorView && rawView && rawTextarea) {
    tabRaw.onclick = () => {
      tabRaw.classList.add("active");
      tabConstructor.classList.remove("active");
      constructorView.style.display = "none";
      rawView.style.display = "block";
      rawTextarea.value = JSON.stringify(taxonomy, null, 2);
    };

    tabConstructor.onclick = () => {
      tabConstructor.classList.add("active");
      tabRaw.classList.remove("active");
      rawView.style.display = "none";
      constructorView.style.display = "block";
      renderLabels();
    };

    const btnSaveRaw = document.getElementById("btnSaveRawLabels");
    if (btnSaveRaw) {
      btnSaveRaw.onclick = () => {
        try {
          const parsed = JSON.parse(rawTextarea.value);
          if (typeof parsed === "object" && parsed !== null) {
            taxonomy = parsed;
            saveTaxonomy();
            renderLabels();
            populateClassFilterMenu();
            tabConstructor.click();
            showToast("Đã lưu cấu hình nhãn JSON thành công!", "success");
          } else {
            showToast("Dữ liệu JSON không hợp lệ!", "danger");
          }
        } catch (err) {
          showToast("Lỗi cú pháp JSON: " + err.message, "danger");
        }
      };
    }

    const btnResetRaw = document.getElementById("btnResetRawLabels");
    if (btnResetRaw) {
      btnResetRaw.onclick = async () => {
        if (await uiConfirm("Mọi nhãn đã thêm hoặc đổi tên sẽ mất.", { title: "Đặt lại danh mục nhãn về mặc định?", okText: "Đặt lại", danger: true })) {
          taxonomy = { ...DEFAULT_DRIVING_TAXONOMY };
          saveTaxonomy();
          rawTextarea.value = JSON.stringify(taxonomy, null, 2);
          renderLabels();
          populateClassFilterMenu();
          showToast("Đã khôi phục danh mục nhãn mặc định", "info");
        }
      };
    }
  }

  // Add Label Button & Modal
  const btnAddLabel = document.getElementById("btnAddLabel");
  const newLabelColor = document.getElementById("newLabelColor");
  const newLabelColorHex = document.getElementById("newLabelColorHex");
  const btnSubmitAddLabel = document.getElementById("btnSubmitAddLabel");

  if (btnAddLabel) {
    btnAddLabel.onclick = () => openModal("addLabelModal");
  }
  if (newLabelColor && newLabelColorHex) {
    newLabelColor.oninput = () => { newLabelColorHex.value = newLabelColor.value; };
    newLabelColorHex.oninput = () => { newLabelColor.value = newLabelColorHex.value; };
  }
  if (btnSubmitAddLabel) {
    btnSubmitAddLabel.onclick = () => {
      const nameInput = document.getElementById("newLabelName");
      const name = nameInput ? nameInput.value.trim() : "";
      const color = newLabelColorHex ? newLabelColorHex.value.trim() : "#3b82f6";
      if (!name) {
        showToast("Vui lòng nhập tên nhãn!", "warning");
        return;
      }
      const key = name.toLowerCase().replace(/\s+/g, "_");
      taxonomy[key] = { name: name, color: color };
      saveTaxonomy();
      renderLabels();
      populateClassFilterMenu();
      closeModal("addLabelModal");
      if (nameInput) nameInput.value = "";
      showToast(`Đã thêm nhãn "${name}" thành công!`, "success");
    };
  }


  // Sync AI Models button
  const btnSyncAI = document.getElementById("btnSyncAIModels");
  if (btnSyncAI) {
    btnSyncAI.onclick = async () => {
      btnSyncAI.disabled = true;
      try {
        const cfg = await fetch(`${API_BASE}/config`).then(r => r.ok ? r.json() : null);
        if (cfg && cfg.classes) {
          Object.entries(cfg.classes).forEach(([k, c]) => {
            if (!taxonomy[k]) taxonomy[k] = { name: k, color: c || "#3b82f6" };
          });
          saveTaxonomy();
          renderLabels();
          populateClassFilterMenu();
          showToast("Đã đồng bộ nhãn từ mô hình AI (YOLO-World / Florence-2)!", "success");
        } else {
          showToast("Đã đồng bộ cấu hình AI hoàn tất!", "success");
        }
      } catch (err) {
        showToast("Lỗi đồng bộ AI: " + err.message, "danger");
      } finally {
        btnSyncAI.disabled = false;
      }
    };
  }

  // -------------------------------------------------------------
  // Load Frame & Tasks Data
  // -------------------------------------------------------------
  let frames = [];

  // 1. If project has a specific ID (not vid-), try GET /p/{projectId}/api/v1/frames
  if (projectId && !projectId.startsWith("vid-")) {
    try {
      const pFrames = await fetch(`/p/${encodeURIComponent(projectId)}/api/v1/frames`).then(r => r.ok ? r.json() : null);
      if (Array.isArray(pFrames) && pFrames.length > 0) {
        frames = pFrames.map((f, idx) => ({
          task_id: `#${String(idx + 1).padStart(4, '0')}`,
          title: `${f.frame_id}`,
          frame_id: f.frame_id,
          created: "Dự án " + projectId,
          updated: "Hôm nay",
          status: f.status,
          frame_risk: f.frame_risk || 0,
          total: (f.counts ? (f.counts.high + f.counts.medium + f.counts.low) : f.n_objects) || 0,
          annotating: f.status === "approved" ? 0 : 1,
          thumb: `/p/${encodeURIComponent(projectId)}/api/v1/frames/${encodeURIComponent(f.frame_id)}/image`
        }));
      }
    } catch (e) { }
  }

  // 2. If project is a video ID, fetch GET /api/v1/videos/{video_id}
  if (frames.length === 0 && projectId && projectId.startsWith("vid-")) {
    try {
      const videoDetail = await fetch(`${API_BASE}/videos/${encodeURIComponent(projectId)}`).then(r => r.ok ? r.json() : null);
      if (videoDetail && Array.isArray(videoDetail.frames)) {
        frames = videoDetail.frames.map((f, idx) => ({
          task_id: `#125653${idx + 1}`,
          title: `${f.frame_id}`,
          frame_id: f.frame_id,
          created: "12 tháng 03 năm 2025",
          updated: "8 ngày trước",
          status: f.status,
          frame_risk: f.frame_risk || 0,
          total: f.n_objects || 4,
          annotating: f.status === "approved" ? (f.n_objects || 4) : 1,
          thumb: "road_camera.jpg"
        }));
      }
    } catch (e) { }
  }

  // 3. Fetch GET /api/v1/frames?sort=risk
  if (frames.length === 0) {
    try {
      const allFrames = await fetch(`${API_BASE}/frames?sort=risk`).then(r => r.ok ? r.json() : []);
      if (Array.isArray(allFrames) && allFrames.length > 0) {
        const filtered = projectId && allFrames.some(f => f.scene === projectId)
          ? allFrames.filter(f => f.scene === projectId)
          : allFrames;

        frames = filtered.slice(0, 30).map((f, idx) => ({
          task_id: `#124739${idx + 1}`,
          title: `${f.frame_id}`,
          frame_id: f.frame_id,
          created: "07 tháng 03 năm 2025",
          updated: "2 tháng trước",
          status: f.status,
          frame_risk: f.frame_risk || 0,
          total: f.n_objects || 6,
          annotating: f.status === "approved" ? (f.n_objects || 6) : 1,
          thumb: "road_camera.jpg"
        }));
      }
    } catch (e) { }
  }

  // 3. Fallback mock driving tasks if empty
  if (frames.length === 0) {
    frames = [
      {
        task_id: "#1256538",
        title: "street_demo_001.mp4",
        frame_id: "scene-0061_000",
        created: "12 tháng 03 năm 2025",
        updated: "8 ngày trước",
        status: "auto",
        frame_risk: 0.65,
        annotating: 3,
        total: 5,
        thumb: "road_camera.jpg"
      },
      {
        task_id: "#1247397",
        title: "street_demo_002.mp4",
        frame_id: "scene-0061_001",
        created: "07 tháng 03 năm 2025",
        updated: "2 tháng trước",
        status: "approved",
        frame_risk: 0.12,
        annotating: 6,
        total: 6,
        thumb: "road_camera.jpg"
      }
    ];
  }

  // -------------------------------------------------------------
  // Filtering & Sorting State
  // -------------------------------------------------------------
  let currentSort = "order"; // 'order' (mặc định) | 'risk' | 'objects'
  let currentStatusFilter = "all"; // 'all' | 'auto' | 'approved' | 'editing'
  let currentClassFilter = "all";
  let currentSearchQuery = "";

  function applyFiltersAndRender() {
    let result = [...frames];

    // Search query filter
    if (currentSearchQuery) {
      const q = currentSearchQuery.toLowerCase();
      result = result.filter(f => f.title.toLowerCase().includes(q) || f.task_id.toLowerCase().includes(q) || f.frame_id.toLowerCase().includes(q));
    }

    // Status filter
    if (currentStatusFilter !== "all") {
      result = result.filter(f => f.status === currentStatusFilter);
    }

    // Sort order
    if (currentSort === "risk") {
      result.sort((a, b) => (a.status === "approved" ? 1 : 0) - (b.status === "approved" ? 1 : 0) || (b.frame_risk || 0) - (a.frame_risk || 0));
    } else if (currentSort === "order") {
      result.sort((a, b) => a.frame_id.localeCompare(b.frame_id));
    } else if (currentSort === "objects") {
      result.sort((a, b) => (b.total || 0) - (a.total || 0));
    }

    renderTasks(result);
  }

  function renderTasks(list) {
    tasksContainer.innerHTML = "";
    if (list.length === 0) {
      tasksContainer.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 48px; background: #fff; border-radius: 6px; border: 1px solid var(--border-light);">Không tìm thấy frame nào phù hợp với bộ lọc</div>`;
      return;
    }

    list.forEach(item => {
      const card = document.createElement("div");
      card.className = "task-card";
      card.style.position = "relative";
      const pct = Math.min(100, Math.round((item.annotating / Math.max(1, item.total)) * 100));
      const statusBadge = item.status === "approved"
        ? `<span style="display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 3px; background: #e6f7ff; color: #1890ff; font-weight: 500; margin-left: 8px;">Đã duyệt</span>`
        : `<span style="display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 3px; background: #fffbe6; color: #faad14; font-weight: 500; margin-left: 8px;">Cần duyệt</span>`;

      card.innerHTML = `
        <img src="${item.thumb}" alt="${item.title}" class="task-thumb" onerror="this.src='road_camera.jpg';">
        <div class="task-info">
          <div class="task-title" style="display: flex; align-items: center;">
            <span>${item.task_id}: ${item.title}</span>
            ${statusBadge}
          </div>
          <div class="task-sub">Tạo bởi ${currentReviewer} vào ${item.created} · Cập nhật ${item.updated}</div>
          <div class="task-progress-wrap">
            <span class="task-progress-text">• ${item.annotating} đang gán • ${item.total} tổng số đối tượng</span>
            <div class="task-progress-bar">
              <div class="task-progress-fill" style="width: ${pct}%; background-color: ${item.status === 'approved' ? '#52c41a' : 'var(--primary)'};"></div>
            </div>
          </div>
        </div>
        <div class="task-right" style="position: relative;">
          <button class="btn-open" onclick="openWorkstationForFrame('${item.frame_id}', event)">Mở</button>
          <button class="dot-menu-btn" title="Thao tác"><i class="ri-more-2-fill"></i></button>
          <!-- Context menu for card -->
          <div class="task-dot-menu" style="display: none; position: absolute; right: 0; top: 100%; margin-top: 4px; background: #fff; border: 1px solid var(--border-light); border-radius: 6px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); width: 190px; z-index: 100; padding: 4px 0;">
            <a href="javascript:void(0)" class="task-menu-open" style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; font-size: 13px; color: var(--text-main); text-decoration: none;"><i class="ri-edit-line"></i> Mở Workspace</a>
            <a href="javascript:void(0)" class="task-menu-approve" style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; font-size: 13px; color: #52c41a; text-decoration: none;"><i class="ri-check-line"></i> Duyệt nhanh (Approve)</a>
            <a href="javascript:void(0)" class="task-menu-export" style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; font-size: 13px; color: var(--text-main); text-decoration: none;"><i class="ri-download-line"></i> Xuất nhãn frame</a>
          </div>
        </div>
      `;

      // Dot menu toggle & actions
      const dotBtn = card.querySelector(".dot-menu-btn");
      const dotMenu = card.querySelector(".task-dot-menu");
      if (dotBtn && dotMenu) {
        dotBtn.onclick = (e) => {
          e.stopPropagation();
          const isOpen = dotMenu.style.display === "block";
          document.querySelectorAll(".task-dot-menu").forEach(el => el.style.display = "none");
          dotMenu.style.display = isOpen ? "none" : "block";
        };
        card.querySelector(".task-menu-open").onclick = (e) => {
          e.stopPropagation();
          openWorkstationForFrame(item.frame_id, e);
        };
        card.querySelector(".task-menu-approve").onclick = async (e) => {
          e.stopPropagation();
          dotMenu.style.display = "none";
          try {
            await fetch(`${API_BASE}/frames/${encodeURIComponent(item.frame_id)}/approve`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ reviewer: currentReviewer, review_time_s: 5 })
            });
          } catch (err) { }
          item.status = "approved";
          item.annotating = item.total;
          applyFiltersAndRender();
          showToast(`Đã duyệt thành công frame: ${item.frame_id}`, "success");
        };
        card.querySelector(".task-menu-export").onclick = (e) => {
          e.stopPropagation();
          const pParam = projectId ? `&project=${encodeURIComponent(projectId)}` : '';
          window.location.href = `export.html?frame=${encodeURIComponent(item.frame_id)}${pParam}`;
        };
      }

      tasksContainer.appendChild(card);
    });
  }

  window.openWorkstationForFrame = function (frameId, e) {
    if (e) e.stopPropagation();
    if (projectId) {
      try { localStorage.setItem("videoId", projectId); } catch (err) { }
    }
    const projectParam = projectId ? `&project=${encodeURIComponent(projectId)}` : '';
    window.location.href = `index.html?frame=${encodeURIComponent(frameId)}${projectParam}`;
  };

  // Search input handler
  const searchInput = document.getElementById("taskSearchInput");
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      currentSearchQuery = searchInput.value.trim();
      applyFiltersAndRender();
    });
  }

  // Sorting Handler
  const sortBtn = document.getElementById("sortBtn");
  const sortMenu = document.getElementById("sortMenu");
  const sortBtnLabel = document.getElementById("sortBtnLabel");

  if (sortBtn && sortMenu) {
    sortBtn.onclick = (e) => {
      e.stopPropagation();
      sortMenu.style.display = sortMenu.style.display === "block" ? "none" : "block";
    };
  }
  window.setSortMode = function (mode) {
    currentSort = mode;
    if (sortBtnLabel) {
      if (mode === "risk") sortBtnLabel.innerText = "Sắp xếp: Rủi ro QA";
      else if (mode === "order") sortBtnLabel.innerText = "Sắp xếp: ID frame";
      else if (mode === "objects") sortBtnLabel.innerText = "Sắp xếp: Số đối tượng";
    }
    if (sortMenu) sortMenu.style.display = "none";
    applyFiltersAndRender();
  };

  // Quick Filter Handler
  const quickFilterBtn = document.getElementById("quickFilterBtn");
  const quickFilterMenu = document.getElementById("quickFilterMenu");
  const quickFilterLabel = document.getElementById("quickFilterLabel");

  if (quickFilterBtn && quickFilterMenu) {
    quickFilterBtn.onclick = (e) => {
      e.stopPropagation();
      quickFilterMenu.style.display = quickFilterMenu.style.display === "block" ? "none" : "block";
    };
  }
  window.setFilterStatus = function (status) {
    currentStatusFilter = status;
    if (quickFilterLabel) {
      if (status === "all") quickFilterLabel.innerText = "Lọc nhanh: Tất cả";
      else if (status === "auto") quickFilterLabel.innerText = "Lọc nhanh: Cần duyệt";
      else if (status === "approved") quickFilterLabel.innerText = "Lọc nhanh: Đã duyệt";
      else if (status === "editing") quickFilterLabel.innerText = "Lọc nhanh: Đang sửa";
    }
    if (quickFilterMenu) quickFilterMenu.style.display = "none";
    applyFiltersAndRender();
  };

  // Class Filter Handler
  const filterClassBtn = document.getElementById("filterClassBtn");
  const filterClassMenu = document.getElementById("filterClassMenu");
  const filterClassLabel = document.getElementById("filterClassLabel");

  function populateClassFilterMenu() {
    if (!filterClassMenu) return;
    filterClassMenu.innerHTML = `
      <a href="javascript:void(0)" onclick="setFilterClass('all')" style="display: block; padding: 8px 14px; font-size: 13px; color: var(--text-main); text-decoration: none; border-bottom: 1px solid var(--border-light);">Tất cả nhãn</a>
    `;
    Object.entries(taxonomy).forEach(([k, v]) => {
      const a = document.createElement("a");
      a.href = "javascript:void(0)";
      a.style.display = "flex";
      a.style.alignItems = "center";
      a.style.gap = "8px";
      a.style.padding = "6px 14px";
      a.style.fontSize = "13px";
      a.style.color = "var(--text-main)";
      a.style.textDecoration = "none";
      a.innerHTML = `<span style="width: 10px; height: 10px; border-radius: 50%; background: ${v.color || '#3b82f6'};"></span> <span>${v.name || k}</span>`;
      a.onclick = () => setFilterClass(k);
      filterClassMenu.appendChild(a);
    });
  }

  if (filterClassBtn && filterClassMenu) {
    filterClassBtn.onclick = (e) => {
      e.stopPropagation();
      filterClassMenu.style.display = filterClassMenu.style.display === "block" ? "none" : "block";
    };
  }
  window.setFilterClass = function (cls) {
    currentClassFilter = cls;
    if (filterClassLabel) {
      filterClassLabel.innerText = cls === "all" ? "Lọc theo nhãn" : `Nhãn: ${taxonomy[cls]?.name || cls}`;
    }
    if (filterClassMenu) filterClassMenu.style.display = "none";
    showToast(`Đã lọc danh sách frame theo nhãn: ${cls === 'all' ? 'Tất cả' : (taxonomy[cls]?.name || cls)}`, "info");
    applyFiltersAndRender();
  };

  populateClassFilterMenu();

  // Clear All Filters
  const btnClearFilters = document.getElementById("btnClearAllFilters");
  if (btnClearFilters) {
    btnClearFilters.onclick = () => {
      currentSort = "order";
      currentStatusFilter = "all";
      currentClassFilter = "all";
      currentSearchQuery = "";
      if (searchInput) searchInput.value = "";
      if (sortBtnLabel) sortBtnLabel.innerText = "Sắp xếp: ID frame";
      if (quickFilterLabel) quickFilterLabel.innerText = "Lọc nhanh: Tất cả";
      if (filterClassLabel) filterClassLabel.innerText = "Lọc theo nhãn";
      applyFiltersAndRender();
      showToast("Đã xóa tất cả bộ lọc", "info");
    };
  }

  // Actions Dropdown Menu (Top Right of back-bar)
  const actionsDropdownBtn = document.getElementById("actionsDropdownBtn");
  const actionsDropdownMenu = document.getElementById("actionsDropdownMenu");
  if (actionsDropdownBtn && actionsDropdownMenu) {
    actionsDropdownBtn.onclick = (e) => {
      e.stopPropagation();
      actionsDropdownMenu.style.display = actionsDropdownMenu.style.display === "block" ? "none" : "block";
    };
  }

  // Auto-Approve Low Risk action
  const btnAutoApprove = document.getElementById("btnAutoApproveLowRisk");
  if (btnAutoApprove) {
    btnAutoApprove.onclick = async () => {
      if (actionsDropdownMenu) actionsDropdownMenu.style.display = "none";
      const pending = frames.filter(f => f.status !== "approved");
      if (pending.length === 0) {
        showToast("Tất cả frame đã được duyệt hoàn tất!", "info");
        return;
      }
      showToast("Đang duyệt tự động các frame rủi ro thấp...", "info");
      for (const item of pending.slice(0, 15)) {
        try {
          await fetch(`${API_BASE}/frames/${encodeURIComponent(item.frame_id)}/approve-low-risk`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reviewer: currentReviewer })
          });
        } catch (e) { }
        item.status = "approved";
        item.annotating = item.total;
      }
      applyFiltersAndRender();
      showToast("Đã duyệt tự động các frame rủi ro thấp thành công!", "success");
    };
  }

  // Approve All Frames action
  const btnApproveAll = document.getElementById("btnApproveAllFrames");
  if (btnApproveAll) {
    btnApproveAll.onclick = async () => {
      if (actionsDropdownMenu) actionsDropdownMenu.style.display = "none";
      if (!(await uiConfirm("Mọi frame trong dự án sẽ được đánh dấu đã duyệt.", { title: "Duyệt tất cả frame?", okText: "Duyệt tất cả" }))) return;
      frames.forEach(item => {
        item.status = "approved";
        item.annotating = item.total;
      });
      applyFiltersAndRender();
      showToast(`Đã duyệt toàn bộ ${frames.length} frame thành công!`, "success");
    };
  }

  // Add Task Button & Modal
  const btnAddNewTask = document.getElementById("btnAddNewTask");
  const btnSubmitCreateTask = document.getElementById("btnSubmitCreateTask");
  if (btnAddNewTask) {
    btnAddNewTask.onclick = () => openModal("createTaskModal");
  }
  if (btnSubmitCreateTask) {
    btnSubmitCreateTask.onclick = () => {
      const taskNameInput = document.getElementById("newTaskName");
      const taskName = taskNameInput ? taskNameInput.value.trim() : "";
      const source = document.getElementById("newTaskSource")?.value || "demo";
      const finalTitle = taskName || `camera_stream_${frames.length + 1}.mp4`;
      const newTask = {
        task_id: `#125799${frames.length + 1}`,
        title: finalTitle,
        frame_id: `scene-custom_${frames.length + 1}`,
        created: "Vừa xong",
        updated: "Vừa xong",
        status: "auto",
        frame_risk: 0.5,
        annotating: 2,
        total: 5,
        thumb: "road_camera.jpg"
      };
      frames.unshift(newTask);
      applyFiltersAndRender();
      closeModal("createTaskModal");
      if (taskNameInput) taskNameInput.value = "";
      showToast(`Đã tạo Task "${finalTitle}" thành công!`, "success");
    };
  }

  // Close open dropdowns when clicking outside
  document.addEventListener("click", (e) => {
    if (actionsDropdownMenu && !e.target.closest("#actionsDropdownBtn") && !e.target.closest("#actionsDropdownMenu")) {
      actionsDropdownMenu.style.display = "none";
    }
    if (sortMenu && !e.target.closest("#sortBtn") && !e.target.closest("#sortMenu")) {
      sortMenu.style.display = "none";
    }
    if (quickFilterMenu && !e.target.closest("#quickFilterBtn") && !e.target.closest("#quickFilterMenu")) {
      quickFilterMenu.style.display = "none";
    }
    if (filterClassMenu && !e.target.closest("#filterClassBtn") && !e.target.closest("#filterClassMenu")) {
      filterClassMenu.style.display = "none";
    }
    if (!e.target.closest(".dot-menu-btn") && !e.target.closest(".task-dot-menu")) {
      document.querySelectorAll(".task-dot-menu").forEach(el => el.style.display = "none");
    }
  });

  // Initial render
  applyFiltersAndRender();
}

// -------------------------------------------------------------
// 4. EXPORT PAGE
// -------------------------------------------------------------
async function initExportPage() {
  const btnDoExport = document.getElementById("btnDoExport");
  const exportSuccessCard = document.getElementById("exportSuccessCard");
  const exportFileList = document.getElementById("exportFileList");

  if (btnDoExport) {
    btnDoExport.onclick = async () => {
      btnDoExport.disabled = true;
      btnDoExport.innerHTML = `<i class="ri-loader-4-line ri-spin"></i> Đang xuất dữ liệu...`;

      try {
        const res = await fetch(`${API_BASE}/export`, { method: "POST" });
        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
          const msg = data.detail?.message || "Chưa có frame nào được duyệt để xuất.";
          showToast(msg, "warning");
          btnDoExport.disabled = false;
          btnDoExport.innerHTML = `Xuất dữ liệu`;
          return;
        }

        btnDoExport.disabled = false;
        btnDoExport.innerHTML = `Xuất dữ liệu`;

        if (exportSuccessCard) {
          exportSuccessCard.style.display = "block";
          document.getElementById("exportIdDisplay").innerText = data.export_id;
          document.getElementById("exportFramesCount").innerText = data.n_frames;
          document.getElementById("exportObjectsCount").innerText = data.n_objects;

          if (exportFileList && Array.isArray(data.files)) {
            exportFileList.innerHTML = data.files.map(f => `
              <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border-light); font-size: 13px;">
                <span><i class="ri-file-text-line" style="color: var(--primary); margin-right: 6px;"></i> ${f}</span>
                <a href="${API_BASE}/exports/${encodeURIComponent(data.export_id)}/${encodeURIComponent(f)}" class="btn-default" style="padding: 2px 10px; font-size: 12px; height: 26px; line-height: 20px;" download>
                  Tải về
                </a>
              </div>
            `).join("");
          }
        }
        showToast(`Đã xuất ${data.n_frames} frame thành công!`, "success");
      } catch (err) {
        showToast("Lỗi xuất dữ liệu: " + err.message, "danger");
        btnDoExport.disabled = false;
        btnDoExport.innerHTML = `Xuất dữ liệu`;
      }
    };
  }
}

// -------------------------------------------------------------
// Auto Initialization
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  const togglePw = document.getElementById("togglePasswordBtn");
  const pwInput = document.getElementById("loginPassword");
  if (togglePw && pwInput) {
    togglePw.addEventListener("click", () => {
      const isPassword = pwInput.type === "password";
      pwInput.type = isPassword ? "text" : "password";
      togglePw.className = isPassword ? "ri-eye-line" : "ri-eye-off-line";
    });
  }

  if (document.getElementById("loginForm")) initLoginPage();
  if (document.getElementById("projectsGrid")) initProjectsPage();
  if (document.getElementById("tasksList")) initFramesPage();
  if (document.getElementById("btnDoExport")) initExportPage();

  const reviewer = localStorage.getItem("reviewer");
  if (reviewer) {
    document.querySelectorAll(".user-name").forEach(el => {
      el.textContent = reviewer;
    });
  }
});

/**
 * AutoLabel 3D - Main Application Controller
 */

class AutoLabelApp {
  constructor() {
    this.currentStep = 1; // 1 to 7
    this.currentMode = 'app'; // 'app' | 'flow'
    this.currentProject = APP_DATA.projects[0];
    this.currentFrame = APP_DATA.frames[0];
    this.selectedObjectId = 1;

    this.lidarView = null;
    this.cameraView = null;

    this.init();
  }

  init() {
    this.bindEvents();
    this.renderProjectsTable();
    this.renderFramesTable();
    this.renderObjectsTable();
    this.renderNearbyFrames();
    this.renderShortcutsList();

    // Show initial step
    this.goToStep(1);
  }

  bindEvents() {
    // 1. Global Stepper Clicks
    document.querySelectorAll('.step-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        const step = parseInt(pill.dataset.step, 10);
        this.goToStep(step);
      });
    });

    // 2. Mode Toggle (App vs Flow Overview)
    const modeToggleBtn = document.getElementById('btn-mode-toggle');
    if (modeToggleBtn) {
      modeToggleBtn.addEventListener('click', () => {
        this.toggleMode();
      });
    }

    // 3. Step 1: Login Form
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
      loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        this.showToast('Đăng nhập thành công! Chào mừng Nguyễn Văn A.');
        setTimeout(() => this.goToStep(2), 300);
      });
    }

    const togglePasswordBtn = document.getElementById('btn-toggle-password');
    if (togglePasswordBtn) {
      togglePasswordBtn.addEventListener('click', () => {
        const passInput = document.getElementById('login-password');
        if (passInput) {
          passInput.type = passInput.type === 'password' ? 'text' : 'password';
        }
      });
    }

    // 4. Step 2: Project search
    const projectSearch = document.getElementById('project-search-input');
    if (projectSearch) {
      projectSearch.addEventListener('input', (e) => {
        this.renderProjectsTable(e.target.value);
      });
    }

    // 5. Step 3: Frame filter tabs
    document.querySelectorAll('.filter-tab-btn').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.filter-tab-btn').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const filter = tab.dataset.filter;
        this.renderFramesTable(filter);
      });
    });

    const frameBackBtn = document.getElementById('btn-frame-back');
    if (frameBackBtn) {
      frameBackBtn.addEventListener('click', () => this.goToStep(2));
    }

    // 6. Step 4: Workstation Toolbar Actions
    const wsBackBtn = document.getElementById('btn-ws-back');
    if (wsBackBtn) {
      wsBackBtn.addEventListener('click', () => this.goToStep(3));
    }

    const autoLabelBtn = document.getElementById('btn-run-autolabel');
    if (autoLabelBtn) {
      autoLabelBtn.addEventListener('click', () => this.runAutoLabel());
    }

    const shortcutsBtn = document.getElementById('btn-shortcuts-modal');
    if (shortcutsBtn) {
      shortcutsBtn.addEventListener('click', () => this.openShortcutsModal());
    }

    // Inspector Tabs
    document.querySelectorAll('.ins-tab-btn').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.ins-tab-btn').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const tabTarget = tab.dataset.tab;
        this.switchInspectorTab(tabTarget);
      });
    });

    // Add / Delete Object buttons
    const addObjBtn = document.getElementById('btn-add-object');
    if (addObjBtn) {
      addObjBtn.addEventListener('click', () => this.addNewObject());
    }

    const delObjBtn = document.getElementById('btn-delete-object');
    if (delObjBtn) {
      delObjBtn.addEventListener('click', () => this.deleteSelectedObject());
    }

    // Finish frame button -> triggers Step 5 Confirm Modal
    const finishFrameBtn = document.getElementById('btn-finish-frame');
    if (finishFrameBtn) {
      finishFrameBtn.addEventListener('click', () => {
        this.openConfirmModal();
      });
    }

    // 7. Step 5: Confirm Modal actions
    const reviewFrameBtn = document.getElementById('btn-modal-review');
    if (reviewFrameBtn) {
      reviewFrameBtn.addEventListener('click', () => this.closeConfirmModal());
    }

    const confirmSubmitBtn = document.getElementById('btn-modal-confirm-submit');
    if (confirmSubmitBtn) {
      confirmSubmitBtn.addEventListener('click', () => {
        this.closeConfirmModal();
        this.goToStep(6);
      });
    }

    // 8. Step 6: Save Result Modal actions
    const nextFrameBtn = document.getElementById('btn-next-frame');
    if (nextFrameBtn) {
      nextFrameBtn.addEventListener('click', () => {
        this.closeSaveModal();
        this.showToast('Đã tải frame scene001_0002 để tiếp tục gán nhãn.');
        this.goToStep(4);
      });
    }

    const backToListBtn = document.getElementById('btn-back-to-list');
    if (backToListBtn) {
      backToListBtn.addEventListener('click', () => {
        this.closeSaveModal();
        this.goToStep(3);
      });
    }

    const jumpToExportBtn = document.getElementById('btn-jump-to-export');
    if (jumpToExportBtn) {
      jumpToExportBtn.addEventListener('click', () => {
        this.closeSaveModal();
        this.goToStep(7);
      });
    }

    // 9. Step 7: Export actions
    const exportBtn = document.getElementById('btn-do-export');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => this.doExportData());
    }

    const downloadZipBtn = document.getElementById('btn-download-zip');
    if (downloadZipBtn) {
      downloadZipBtn.addEventListener('click', () => this.downloadExportZip());
    }

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === ' ' && this.currentStep === 4) {
        e.preventDefault();
        this.openConfirmModal();
      } else if ((e.key === 'a' || e.key === 'A') && this.currentStep === 4) {
        this.runAutoLabel();
      } else if (e.key === 'Escape') {
        this.closeAllModals();
      }
    });
  }

  goToStep(step) {
    this.currentStep = step;

    // Update Stepper UI
    document.querySelectorAll('.step-pill').forEach(pill => {
      const s = parseInt(pill.dataset.step, 10);
      pill.classList.remove('active');
      if (s === step) {
        pill.classList.add('active');
      }
      if (s < step) {
        pill.classList.add('completed');
      } else {
        pill.classList.remove('completed');
      }
    });

    if (this.currentMode === 'flow') {
      this.currentMode = 'app';
      this.updateModeUI();
    }

    // Hide all screens
    document.querySelectorAll('.app-screen').forEach(screen => {
      screen.classList.remove('active');
    });

    // Handle screen display
    if (step === 1) {
      document.getElementById('screen-step-1').classList.add('active');
    } else if (step === 2) {
      document.getElementById('screen-step-2').classList.add('active');
    } else if (step === 3) {
      document.getElementById('screen-step-3').classList.add('active');
    } else if (step === 4) {
      document.getElementById('screen-step-4').classList.add('active');
      this.initWorkstationViewsIfNeeded();
    } else if (step === 5) {
      // Step 5 is the Confirm Modal on top of Step 4
      document.getElementById('screen-step-4').classList.add('active');
      this.initWorkstationViewsIfNeeded();
      this.openConfirmModal();
    } else if (step === 6) {
      // Step 6 is the Save Result Modal
      document.getElementById('screen-step-4').classList.add('active');
      this.initWorkstationViewsIfNeeded();
      this.openSaveModal();
    } else if (step === 7) {
      document.getElementById('screen-step-7').classList.add('active');
    }
  }

  toggleMode() {
    this.currentMode = this.currentMode === 'app' ? 'flow' : 'app';
    this.updateModeUI();
  }

  updateModeUI() {
    const btn = document.getElementById('btn-mode-toggle');
    const flowScreen = document.getElementById('screen-flow-overview');

    if (this.currentMode === 'flow') {
      if (btn) {
        btn.classList.add('active');
        btn.innerHTML = `<i data-lucide="monitor"></i> Trải nghiệm ứng dụng`;
      }
      document.querySelectorAll('.app-screen').forEach(s => s.classList.remove('active'));
      if (flowScreen) flowScreen.classList.add('active');
    } else {
      if (btn) {
        btn.classList.remove('active');
        btn.innerHTML = `<i data-lucide="layout-grid"></i> Xem toàn cảnh Flow`;
      }
      if (flowScreen) flowScreen.classList.remove('active');
      this.goToStep(this.currentStep);
    }
    if (window.lucide) window.lucide.createIcons();
  }

  initWorkstationViewsIfNeeded() {
    setTimeout(() => {
      if (!this.lidarView) {
        this.lidarView = new LidarView('lidar-canvas-container');
        this.lidarView.onSelectCallback = (id) => this.selectObject(id);
      }
      if (!this.cameraView) {
        this.cameraView = new CameraView('camera-viewport-container');
        this.cameraView.onSelectCallback = (id) => this.selectObject(id);
      }
    }, 50);
  }

  renderProjectsTable(searchQuery = '') {
    const tbody = document.getElementById('projects-table-body');
    if (!tbody) return;

    const filtered = APP_DATA.projects.filter(p =>
      p.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    tbody.innerHTML = filtered.map(p => `
      <tr>
        <td>
          <span class="project-name-link" onclick="window.app.selectProject('${p.id}')">
            📁 ${p.name}
          </span>
        </td>
        <td><strong>${p.framesCount.toLocaleString()}</strong></td>
        <td>
          <span class="status-dot ${p.statusCode}"></span>
          ${p.status}
        </td>
        <td>${p.updatedAt}</td>
        <td>
          <button class="vp-ctrl-btn" onclick="window.app.selectProject('${p.id}')">
            Mở
          </button>
        </td>
      </tr>
    `).join('');
  }

  selectProject(projectId) {
    const project = APP_DATA.projects.find(p => p.id === projectId);
    if (project) {
      this.currentProject = project;
      document.getElementById('frame-project-title').textContent = project.name;
      this.showToast(`Đã chọn dự án ${project.name}`);
      this.goToStep(3);
    }
  }

  renderFramesTable(filter = 'all') {
    const tbody = document.getElementById('frames-table-body');
    if (!tbody) return;

    let frames = APP_DATA.frames;
    if (filter === 'unannotated') {
      frames = frames.filter(f => f.statusCode === 'unannotated');
    } else if (filter === 'annotated') {
      frames = frames.filter(f => f.statusCode === 'annotated');
    }

    tbody.innerHTML = frames.map(f => `
      <tr>
        <td><strong style="font-family: var(--font-mono);">${f.id}</strong></td>
        <td>
          <span class="badge-tag ${f.statusCode}">
            ${f.status}
          </span>
        </td>
        <td style="color: var(--color-text-muted); font-size: 13px;">${f.time}</td>
        <td>
          <button class="btn-open-frame" onclick="window.app.openFrame('${f.id}')">
            Mở
          </button>
        </td>
      </tr>
    `).join('');
  }

  openFrame(frameId) {
    const frame = APP_DATA.frames.find(f => f.id === frameId);
    if (frame) {
      this.currentFrame = frame;
      document.getElementById('ws-breadcrumb-text').textContent = `${this.currentProject.name} / ${frame.id}`;
      this.showToast(`Đang mở ${frame.id}...`);
      this.goToStep(4);
    }
  }

  renderObjectsTable() {
    const tbody = document.getElementById('objects-table-body');
    if (!tbody) return;

    tbody.innerHTML = APP_DATA.objects.map(obj => {
      const isSelected = obj.id === this.selectedObjectId;
      return `
        <tr class="${isSelected ? 'selected' : ''}" onclick="window.app.selectObject(${obj.id})">
          <td class="obj-id-cell">${obj.id}</td>
          <td>
            <div class="obj-class-badge">
              <span class="obj-color-indicator" style="background: ${obj.color};"></span>
              ${obj.className}
            </div>
          </td>
          <td style="text-align: center;">
            <input type="checkbox" class="obj-sync-checkbox" ${obj.has2D ? 'checked' : ''}
              onclick="event.stopPropagation(); window.app.toggle2D(${obj.id}, this.checked)">
          </td>
          <td style="text-align: center;">
            <input type="checkbox" class="obj-sync-checkbox" ${obj.has3D ? 'checked' : ''}
              onclick="event.stopPropagation(); window.app.toggle3D(${obj.id}, this.checked)">
          </td>
          <td><span class="obj-confidence">${obj.confidence.toFixed(2)}</span></td>
          <td style="color: var(--color-text-light);">⋮</td>
        </tr>
      `;
    }).join('');
  }

  selectObject(id) {
    this.selectedObjectId = id;
    this.renderObjectsTable();

    if (this.lidarView) this.lidarView.setSelectedObject(id);
    if (this.cameraView) this.cameraView.setSelectedObject(id);

    // Update Property Tab if active
    const propDetails = document.getElementById('tab-content-properties');
    if (propDetails) {
      const obj = APP_DATA.objects.find(o => o.id === id);
      if (obj) {
        propDetails.innerHTML = `
          <div style="padding: 16px; font-size: 13px; line-height: 1.8;">
            <div style="font-weight: 700; margin-bottom: 8px; color: ${obj.color}; font-size: 15px;">
              ${obj.name} (${obj.className})
            </div>
            <div><strong>Phân loại con:</strong> ${obj.subclass}</div>
            <div><strong>Độ tin cậy AI:</strong> ${(obj.confidence * 100).toFixed(1)}%</div>
            <div><strong>Độ che khuất:</strong> ${obj.occlusion}</div>
            <div style="margin-top: 10px; font-family: var(--font-mono); font-size: 11px; background: #f8fafc; padding: 8px; border-radius: 6px;">
              <strong>Tọa độ 3D:</strong><br>
              X: ${obj.box3D?.x.toFixed(2)}m | Y: ${obj.box3D?.y.toFixed(2)}m | Z: ${obj.box3D?.z.toFixed(2)}m<br>
              Size: ${obj.box3D?.dx}m × ${obj.box3D?.dy}m × ${obj.box3D?.dz}m
            </div>
          </div>
        `;
      }
    }
  }

  toggle2D(id, checked) {
    const obj = APP_DATA.objects.find(o => o.id === id);
    if (obj) {
      obj.has2D = checked;
      if (this.cameraView) this.cameraView.toggle2DBox(id, checked);
    }
  }

  toggle3D(id, checked) {
    const obj = APP_DATA.objects.find(o => o.id === id);
    if (obj) {
      obj.has3D = checked;
      if (this.lidarView) this.lidarView.toggle3DBox(id, checked);
    }
  }

  addNewObject() {
    const newId = APP_DATA.objects.length > 0 ? Math.max(...APP_DATA.objects.map(o => o.id)) + 1 : 1;
    const newObj = {
      id: newId,
      name: `Vehicle #${newId}`,
      className: "car",
      classLabel: "car",
      color: "#06b6d4",
      box2D: { x: 350, y: 500, width: 80, height: 90 },
      box3D: { x: -1.0, y: 18.0, z: -0.3, dx: 1.9, dy: 4.6, dz: 1.5, yaw: 0.0 },
      has2D: true,
      has3D: true,
      confidence: 0.85,
      occlusion: "Mới tạo",
      truncated: 0.0,
      subclass: "Sedan"
    };
    APP_DATA.objects.push(newObj);
    this.selectObject(newId);
    this.showToast(`Đã thêm Object #${newId}`);
  }

  deleteSelectedObject() {
    if (APP_DATA.objects.length <= 1) {
      this.showToast('Cần ít nhất 1 object trong frame!');
      return;
    }
    const idx = APP_DATA.objects.findIndex(o => o.id === this.selectedObjectId);
    if (idx !== -1) {
      const removed = APP_DATA.objects.splice(idx, 1)[0];
      this.selectedObjectId = APP_DATA.objects[0].id;
      this.selectObject(this.selectedObjectId);
      this.showToast(`Đã xóa Object #${removed.id} (${removed.className})`);
    }
  }

  renderNearbyFrames() {
    const container = document.getElementById('nearby-frames-row');
    if (!container) return;

    container.innerHTML = APP_DATA.nearbyFrames.map(f => `
      <div class="thumbnail-card ${f.isCurrent ? 'active' : ''}" onclick="window.app.switchNearbyFrame('${f.id}')">
        <img src="assets/road_camera.jpg" alt="${f.id}" />
        <div class="thumb-caption">${f.id}</div>
      </div>
    `).join('') + `<button class="thumbnail-more-btn">...</button>`;
  }

  switchNearbyFrame(frameId) {
    APP_DATA.nearbyFrames.forEach(f => f.isCurrent = f.id === frameId);
    this.renderNearbyFrames();
    document.getElementById('ws-breadcrumb-text').textContent = `${this.currentProject.name} / ${frameId}`;
    this.showToast(`Đã chuyển sang ${frameId}`);
  }

  switchInspectorTab(tabKey) {
    document.querySelectorAll('.tab-content-pane').forEach(p => p.style.display = 'none');
    const target = document.getElementById(`tab-content-${tabKey}`);
    if (target) target.style.display = 'block';
  }

  runAutoLabel() {
    const btn = document.getElementById('btn-run-autolabel');
    if (!btn) return;

    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader" class="spin"></i> Đang chạy PointPillars + YOLO...`;
    if (window.lucide) window.lucide.createIcons();

    // Trigger laser sweep on 2D camera & holographic scan on 3D LiDAR
    if (this.cameraView) this.cameraView.triggerLaserScan();
    if (this.lidarView) this.lidarView.triggerAutoLabelScan();

    setTimeout(() => {
      // Simulate high confidence results
      APP_DATA.objects.forEach(obj => {
        obj.confidence = Math.min(0.99, Number((obj.confidence + 0.05).toFixed(2)));
        obj.has2D = true;
        obj.has3D = true;
      });

      this.renderObjectsTable();
      if (this.lidarView) this.lidarView.renderBoundingBoxes();
      if (this.cameraView) this.cameraView.renderBoxes();

      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="sparkles"></i> Chạy Auto-label ✨`;
      if (window.lucide) window.lucide.createIcons();

      this.showToast('✨ Auto-label AI hoàn tất! Đồng bộ 5 objects 2D ↔ 3D thành công.');
    }, 1500);
  }

  openConfirmModal() {
    const modal = document.getElementById('modal-step-5');
    if (modal) modal.classList.add('show');
  }

  closeConfirmModal() {
    const modal = document.getElementById('modal-step-5');
    if (modal) modal.classList.remove('show');
  }

  openSaveModal() {
    const modal = document.getElementById('modal-step-6');
    if (modal) modal.classList.add('show');
  }

  closeSaveModal() {
    const modal = document.getElementById('modal-step-6');
    if (modal) modal.classList.remove('show');
  }

  openShortcutsModal() {
    const modal = document.getElementById('modal-shortcuts');
    if (modal) modal.classList.add('show');
  }

  closeAllModals() {
    document.querySelectorAll('.modal-backdrop').forEach(m => m.classList.remove('show'));
  }

  renderShortcutsList() {
    const listEl = document.getElementById('shortcuts-list-container');
    if (!listEl) return;

    listEl.innerHTML = APP_DATA.shortcuts.map(s => `
      <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 13px;">
        <span style="color: var(--color-text-muted);">${s.desc}</span>
        <kbd style="background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 8px; font-family: var(--font-mono); font-weight: 600; font-size: 12px;">${s.key}</kbd>
      </div>
    `).join('');
  }

  doExportData() {
    const btn = document.getElementById('btn-do-export');
    const successBox = document.getElementById('export-success-card');
    const formatSelect = document.getElementById('export-format-select');
    const format = formatSelect ? formatSelect.value : 'KITTI (3D + 2D)';

    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `<i data-lucide="loader" class="spin"></i> Đang xuất dữ liệu (${format})...`;
      if (window.lucide) window.lucide.createIcons();
    }

    setTimeout(() => {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="upload"></i> Xuất dữ liệu`;
        if (window.lucide) window.lucide.createIcons();
      }
      if (successBox) {
        successBox.classList.add('show');
      }
      this.showToast(`Xuất dữ liệu định dạng ${format} thành công!`);
    }, 1200);
  }

  downloadExportZip() {
    const format = document.getElementById('export-format-select')?.value || 'KITTI';

    // Generate real KITTI label text
    let kittiContent = "# Values: type, truncated, occluded, alpha, bbox(4), dimensions(3), location(3), rotation_y, score\n";
    APP_DATA.objects.forEach(obj => {
      const b3 = obj.box3D || { dx: 1.8, dy: 4.5, dz: 1.5, x: 0, y: 15, z: 0, yaw: 0 };
      const b2 = obj.box2D || { x: 100, y: 100, width: 100, height: 100 };
      kittiContent += `${obj.className.toUpperCase()} 0.0 0 0.0 ${b2.x.toFixed(1)} ${b2.y.toFixed(1)} ${(b2.x + b2.width).toFixed(1)} ${(b2.y + b2.height).toFixed(1)} ${b3.dz.toFixed(2)} ${b3.dx.toFixed(2)} ${b3.dy.toFixed(2)} ${b3.x.toFixed(2)} ${b3.z.toFixed(2)} ${b3.y.toFixed(2)} ${b3.yaw.toFixed(2)} ${obj.confidence.toFixed(2)}\n`;
    });

    const metadata = {
      project: this.currentProject.name,
      frame: this.currentFrame.id,
      timestamp: new Date().toISOString(),
      format: format,
      annotator: APP_DATA.currentUser.name,
      status: "approved",
      objects_count: APP_DATA.objects.length,
      calibration_matrix: [
        [721.5, 0.0, 609.5, 44.8],
        [0.0, 721.5, 172.8, 0.2],
        [0.0, 0.0, 1.0, 0.003]
      ]
    };

    if (window.JSZip) {
      const zip = new JSZip();
      zip.file(`labels/${this.currentFrame.id}.txt`, kittiContent);
      zip.file(`metadata.json`, JSON.stringify(metadata, null, 2));
      zip.file(`calib/${this.currentFrame.id}.txt`, "P2: 7.215e+02 0.0 6.095e+02 4.485e+01 0.0 7.215e+02 1.728e+02 2.163e-01 0.0 0.0 1.0 2.745e-03\nTr_velo_to_cam: 7.533e-03 -9.999e-01 -6.166e-03 -4.069e-03 1.480e-02 6.278e-03 -9.998e-01 -7.631e-02 9.998e-01 7.441e-03 1.489e-02 -2.717e-01\n");

      zip.generateAsync({ type: "blob" }).then((content) => {
        const url = URL.createObjectURL(content);
        const a = document.createElement('a');
        a.href = url;
        a.download = `autolabel3d_${this.currentProject.name}_${this.currentFrame.id}_export.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        this.showToast('📦 Đã tải file .zip về máy thành công!');
      });
    } else {
      // Fallback text download
      const blob = new Blob([kittiContent], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${this.currentFrame.id}_kitti.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      this.showToast('Đã tải nhãn KITTI!');
    }
  }

  showToast(message) {
    const toast = document.getElementById('app-toast');
    const textEl = document.getElementById('toast-text');
    if (!toast || !textEl) return;

    textEl.textContent = message;
    toast.classList.add('show');
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => {
      toast.classList.remove('show');
    }, 3500);
  }
}

// Initialize on DOMContentLoaded
window.addEventListener('DOMContentLoaded', () => {
  window.app = new AutoLabelApp();
  if (window.lucide) {
    window.lucide.createIcons();
  }
});

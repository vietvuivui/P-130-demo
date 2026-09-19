/**
 * AutoLabel 3D - Three.js 3D LiDAR Point Cloud Viewport
 */

class LidarView {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.pointsObject = null;
    this.boxObjects = new Map();
    this.scanBeam = null;
    this.isScanning = false;
    this.selectedObjectId = 1;
    this.colorMode = 'elevation'; // 'elevation' | 'intensity' | 'monochrome'
    this.pointSize = 2.2;
    this.onSelectCallback = null;

    this.init();
  }

  init() {
    const width = this.container.clientWidth || 600;
    const height = this.container.clientHeight || 450;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0f1d); // Deep dark blue-black LiDAR background

    // 2. Camera (LiDAR perspective: looking slightly forward-down from behind ego vehicle)
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 150);
    this.resetCameraView();

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // 4. Controls
    if (window.THREE && window.THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxPolarAngle = Math.PI / 2 + 0.05; // Don't go too far below ground
      this.controls.minDistance = 3;
      this.controls.maxDistance = 80;
      this.controls.target.set(0, 16, 0); // Focus on road ahead
    }

    // 5. Environment: Ground Grid & Range Rings
    this.createEnvironment();

    // 6. Generate Synthetic LiDAR Point Cloud (nuScenes style)
    this.createPointCloud();

    // 7. Create 3D Bounding Cuboids for Objects
    this.renderBoundingBoxes();

    // 8. Scanning Beam effect
    this.createScanningBeam();

    // 9. Event Listeners
    this.setupInteractions();

    // 10. Animation Loop
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);

    // Resize observer
    const resizeObserver = new ResizeObserver(() => this.onResize());
    resizeObserver.observe(this.container);
  }

  resetCameraView() {
    // Ego vehicle position is (0,0,0). Camera sits slightly above & behind ego car
    this.camera.position.set(0, -6, 7.5); // X: 0, Y: -6m behind, Z: 7.5m above
    this.camera.up.set(0, 0, 1); // Z is UP in LiDAR coordinate system
    this.camera.lookAt(0, 16, 0);
    if (this.controls) {
      this.controls.target.set(0, 16, 0);
      this.controls.update();
    }
  }

  setBirdEyeView() {
    // Bird's Eye View (BEV)
    this.camera.position.set(0, 16, 35);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(0, 16, 0);
    if (this.controls) {
      this.controls.target.set(0, 16, 0);
      this.controls.update();
    }
  }

  createEnvironment() {
    // Range Rings: 10m, 20m, 30m, 40m
    const ringGroup = new THREE.Group();
    const distances = [10, 20, 30, 40];
    distances.forEach(d => {
      const curve = new THREE.EllipseCurve(0, 0, d, d, 0, 2 * Math.PI, false, 0);
      const points = curve.getPoints(64);
      const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p.x, p.y, -0.95)));
      const material = new THREE.LineBasicMaterial({ color: 0x1e293b, transparent: true, opacity: 0.7 });
      const ring = new THREE.Line(geometry, material);
      ringGroup.add(ring);
    });

    // Ego vehicle marker (a neat dark blue-grey footprint with red front arrow)
    const egoGeometry = new THREE.BoxGeometry(2.0, 4.8, 1.6);
    const egoEdges = new THREE.EdgesGeometry(egoGeometry);
    const egoLine = new THREE.LineSegments(egoEdges, new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 2 }));
    egoLine.position.set(0, 0, -0.1);
    ringGroup.add(egoLine);

    // Heading arrow on ego car
    const arrowDir = new THREE.Vector3(0, 1, 0);
    const arrowOrigin = new THREE.Vector3(0, 2.4, -0.1);
    const arrowHelper = new THREE.ArrowHelper(arrowDir, arrowOrigin, 1.2, 0x38bdf8, 0.4, 0.3);
    ringGroup.add(arrowHelper);

    // Coordinate Axes at Ego (X: Red/Right, Y: Green/Forward, Z: Blue/Up)
    const axes = new THREE.AxesHelper(3);
    axes.position.set(0, 0, -0.8);
    ringGroup.add(axes);

    this.scene.add(ringGroup);
  }

  createPointCloud() {
    const numPoints = 52000;
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);

    let pIdx = 0;

    // Helper to add point
    const addPt = (x, y, z, intensity, r, g, b) => {
      if (pIdx >= numPoints) return;
      positions[pIdx * 3] = x;
      positions[pIdx * 3 + 1] = y;
      positions[pIdx * 3 + 2] = z;

      if (r !== undefined && g !== undefined && b !== undefined) {
        colors[pIdx * 3] = r;
        colors[pIdx * 3 + 1] = g;
        colors[pIdx * 3 + 2] = b;
      } else {
        // Compute color based on elevation & distance
        const dist = Math.sqrt(x * x + y * y);
        const normZ = (z + 1.0) / 4.0;
        // Cyan-blue to yellow/green spectrum
        colors[pIdx * 3] = Math.min(1.0, 0.1 + normZ * 0.8 + (dist < 15 ? 0.2 : 0));
        colors[pIdx * 3 + 1] = Math.min(1.0, 0.6 + normZ * 0.4);
        colors[pIdx * 3 + 2] = Math.min(1.0, 0.9 - normZ * 0.5);
      }
      pIdx++;
    };

    // 1. Concentric LiDAR Ground Rings (64 beam simulation on asphalt, y > 0)
    for (let beam = 1; beam <= 64; beam++) {
      const radius = 2.5 + Math.pow(beam / 64, 1.8) * 45;
      const pointsInRing = Math.floor(200 + beam * 8);
      for (let i = 0; i < pointsInRing; i++) {
        const angle = (-Math.PI * 0.45) + (Math.PI * 0.9) * (i / pointsInRing);
        const x = Math.sin(angle) * radius + (Math.random() - 0.5) * 0.08;
        const y = Math.cos(angle) * radius + (Math.random() - 0.5) * 0.08;
        const z = -0.9 + (Math.random() - 0.5) * 0.04; // ground level approx -0.9m

        // Road lanes markings reflect higher intensity
        const isLaneLine = Math.abs(x + 1.8) < 0.15 || Math.abs(x - 1.8) < 0.15 || Math.abs(x - 5.4) < 0.15;
        if (isLaneLine && Math.floor(y / 4) % 2 === 0) {
          addPt(x, y, z + 0.02, 0.9, 0.2, 0.9, 0.95);
        } else {
          addPt(x, y, z, 0.3, 0.05, 0.45 + (y / 60) * 0.3, 0.65 - (y / 60) * 0.3);
        }
      }
    }

    // 2. Road Curbs & Sidewalks
    for (let y = 3; y < 45; y += 0.3) {
      // Left curb
      addPt(-7.2 + (Math.random() - 0.5) * 0.1, y, -0.75 + Math.random() * 0.1, 0.5, 0.15, 0.7, 0.8);
      // Right curb
      addPt(7.6 + (Math.random() - 0.5) * 0.1, y, -0.75 + Math.random() * 0.1, 0.5, 0.15, 0.7, 0.8);
    }

    // 3. Sidewalk trees & Poles (left & right)
    const polePositions = [
      { x: -8.0, y: 8 }, { x: -8.2, y: 20 }, { x: -8.1, y: 34 },
      { x: 8.2, y: 12 }, { x: 8.5, y: 26 }, { x: 8.3, y: 38 }
    ];
    polePositions.forEach(pos => {
      // Pole trunk
      for (let z = -0.8; z < 4.5; z += 0.15) {
        addPt(pos.x + (Math.random() - 0.5) * 0.1, pos.y + (Math.random() - 0.5) * 0.1, z, 0.8, 0.2, 0.8, 0.7);
      }
      // Tree foliage canopy
      for (let k = 0; k < 180; k++) {
        const rx = (Math.random() - 0.5) * 2.2;
        const ry = (Math.random() - 0.5) * 2.2;
        const rz = (Math.random() - 0.5) * 2.2;
        addPt(pos.x + rx, pos.y + ry, 3.2 + rz, 0.6, 0.1, 0.85, 0.4);
      }
    });

    // 4. Dense point cloud clusters on target objects
    APP_DATA.objects.forEach(obj => {
      const b = obj.box3D;
      if (!b) return;
      const count = obj.className === 'truck' ? 1200 : (obj.className === 'pedestrian' ? 250 : 700);

      for (let i = 0; i < count; i++) {
        // Generate points distributed on the surfaces of the cuboid
        const face = Math.floor(Math.random() * 6);
        let lx = 0, ly = 0, lz = 0;
        const hx = b.dx / 2;
        const hy = b.dy / 2;
        const hz = b.dz / 2;

        if (face === 0) { lx = hx; ly = (Math.random() - 0.5) * b.dy; lz = (Math.random() - 0.5) * b.dz; }
        else if (face === 1) { lx = -hx; ly = (Math.random() - 0.5) * b.dy; lz = (Math.random() - 0.5) * b.dz; }
        else if (face === 2) { lx = (Math.random() - 0.5) * b.dx; ly = -hy; lz = (Math.random() - 0.5) * b.dz; } // Rear face visible
        else if (face === 3) { lx = (Math.random() - 0.5) * b.dx; ly = hy; lz = (Math.random() - 0.5) * b.dz; }
        else if (face === 4) { lx = (Math.random() - 0.5) * b.dx; ly = (Math.random() - 0.5) * b.dy; lz = hz; } // Roof
        else { lx = (Math.random() - 0.5) * b.dx; ly = (Math.random() - 0.5) * b.dy; lz = -hz; }

        // Noise & rotation
        const cosY = Math.cos(b.yaw);
        const sinY = Math.sin(b.yaw);
        const rx = lx * cosY - ly * sinY;
        const ry = lx * sinY + ly * cosY;

        const px = b.x + rx + (Math.random() - 0.5) * 0.06;
        const py = b.y + ry + (Math.random() - 0.5) * 0.06;
        const pz = b.z + lz + (Math.random() - 0.5) * 0.06;

        // Color based on class
        const c = new THREE.Color(obj.color);
        addPt(px, py, pz, 0.85, c.r * 1.2, c.g * 1.2, c.b * 1.2);
      }
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions.subarray(0, pIdx * 3), 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors.subarray(0, pIdx * 3), 3));

    // Circular point texture
    const canvas = document.createElement('canvas');
    canvas.width = 16;
    canvas.height = 16;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(8, 8, 0, 8, 8, 8);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.7, 'rgba(255,255,255,0.8)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 16, 16);
    const texture = new THREE.CanvasTexture(canvas);

    const material = new THREE.PointsMaterial({
      size: this.pointSize,
      vertexColors: true,
      map: texture,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    this.pointsObject = new THREE.Points(geometry, material);
    this.scene.add(this.pointsObject);
  }

  renderBoundingBoxes() {
    // Clear old boxes
    this.boxObjects.forEach(group => this.scene.remove(group));
    this.boxObjects.clear();

    APP_DATA.objects.forEach(obj => {
      if (!obj.has3D || !obj.box3D) return;

      const b = obj.box3D;
      const group = new THREE.Group();
      group.userData = { objectId: obj.id, objectData: obj };

      // 1. Wireframe 3D Box
      const geom = new THREE.BoxGeometry(b.dx, b.dy, b.dz);
      const edges = new THREE.EdgesGeometry(geom);
      const isSelected = obj.id === this.selectedObjectId;
      const boxColor = new THREE.Color(obj.color);

      const lineMaterial = new THREE.LineBasicMaterial({
        color: boxColor,
        linewidth: isSelected ? 3 : 2
      });
      const wireframe = new THREE.LineSegments(edges, lineMaterial);
      group.add(wireframe);

      // 2. Translucent filled face on bottom or front
      const faceGeom = new THREE.PlaneGeometry(b.dx, b.dz);
      const faceMat = new THREE.MeshBasicMaterial({
        color: boxColor,
        transparent: true,
        opacity: isSelected ? 0.35 : 0.15,
        side: THREE.DoubleSide
      });
      const frontFace = new THREE.Mesh(faceGeom, faceMat);
      frontFace.position.set(0, b.dy / 2, 0);
      frontFace.rotation.x = Math.PI / 2;
      group.add(frontFace);

      // 3. Heading Orientation Arrow
      const arrowDir = new THREE.Vector3(0, 1, 0);
      const arrowOrigin = new THREE.Vector3(0, b.dy / 2, 0);
      const arrow = new THREE.ArrowHelper(arrowDir, arrowOrigin, 1.2, boxColor.getHex(), 0.5, 0.3);
      group.add(arrow);

      // 4. Corner bracket highlights if selected
      if (isSelected) {
        const glowGeom = new THREE.BoxGeometry(b.dx * 1.04, b.dy * 1.04, b.dz * 1.04);
        const glowEdges = new THREE.EdgesGeometry(glowGeom);
        const glowMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
        const glowLine = new THREE.LineSegments(glowEdges, glowMat);
        group.add(glowLine);
      }

      // Position & Rotate
      group.position.set(b.x, b.y, b.z);
      group.rotation.z = b.yaw;

      this.scene.add(group);
      this.boxObjects.set(obj.id, group);
    });
  }

  createScanningBeam() {
    // Holographic radar sweep plane
    const planeGeom = new THREE.PlaneGeometry(16, 2);
    const planeMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });
    this.scanBeam = new THREE.Mesh(planeGeom, planeMat);
    this.scanBeam.rotation.x = Math.PI / 2;
    this.scanBeam.position.set(0, 0, 0);
    this.scene.add(this.scanBeam);
  }

  triggerAutoLabelScan(callback) {
    if (this.isScanning) return;
    this.isScanning = true;

    let scanProgress = 0;
    this.scanBeam.material.opacity = 0.55;

    const interval = setInterval(() => {
      scanProgress += 0.025;
      const y = scanProgress * 40;
      this.scanBeam.position.set(0, y, 0.2);
      this.scanBeam.scale.set(1 + scanProgress * 1.5, 1, 1);

      if (scanProgress >= 1.0) {
        clearInterval(interval);
        this.scanBeam.material.opacity = 0.0;
        this.isScanning = false;
        if (callback) callback();
      }
    }, 30);
  }

  setupInteractions() {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    this.renderer.domElement.addEventListener('pointerdown', (e) => {
      const rect = this.renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, this.camera);
      const meshes = [];
      this.boxObjects.forEach(group => {
        group.children.forEach(child => {
          if (child.isMesh || child.isLineSegments) meshes.push({ mesh: child, group });
        });
      });

      const intersects = raycaster.intersectObjects(meshes.map(m => m.mesh));
      if (intersects.length > 0) {
        const hit = meshes.find(m => m.mesh === intersects[0].object);
        if (hit && hit.group.userData.objectId) {
          this.setSelectedObject(hit.group.userData.objectId);
          if (this.onSelectCallback) {
            this.onSelectCallback(hit.group.userData.objectId);
          }
        }
      }
    });
  }

  setSelectedObject(id) {
    this.selectedObjectId = id;
    this.renderBoundingBoxes();
  }

  toggle3DBox(id, visible) {
    const obj = APP_DATA.objects.find(o => o.id === id);
    if (obj) {
      obj.has3D = visible;
      this.renderBoundingBoxes();
    }
  }

  onResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(this.animate);
    if (this.controls) {
      this.controls.update();
    }
    this.renderer.render(this.scene, this.camera);
  }
}

window.LidarView = LidarView;

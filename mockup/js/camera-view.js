/**
 * AutoLabel 3D - 2D Camera Annotation View
 */

class CameraView {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.wrapper = this.container.querySelector('.camera-canvas-wrapper');
    this.img = this.container.querySelector('.camera-image');
    this.svg = this.container.querySelector('.camera-svg-overlay');
    this.laserScan = this.container.querySelector('.camera-laser-scan');

    this.selectedObjectId = 1;
    this.zoomLevel = 1.0;
    this.onSelectCallback = null;

    this.init();
  }

  init() {
    if (this.img.complete) {
      this.renderBoxes();
    } else {
      this.img.onload = () => this.renderBoxes();
    }

    window.addEventListener('resize', () => this.renderBoxes());
  }

  renderBoxes() {
    if (!this.svg || !this.img) return;

    // Get current rendered dimensions of image
    const rect = this.img.getBoundingClientRect();
    const naturalW = this.img.naturalWidth || 1920;
    const naturalH = this.img.naturalHeight || 1080;

    const scaleX = rect.width / naturalW;
    const scaleY = rect.height / naturalH;

    this.svg.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
    this.svg.innerHTML = '';

    APP_DATA.objects.forEach(obj => {
      if (!obj.has2D || !obj.box2D) return;

      const isSelected = obj.id === this.selectedObjectId;
      const b = obj.box2D;

      // Scaled coordinates
      const x = b.x * (rect.width / 1000);
      const y = b.y * (rect.height / 562.5);
      const w = b.width * (rect.width / 1000);
      const h = b.height * (rect.height / 562.5);

      const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      group.setAttribute('class', `box-2d-group ${isSelected ? 'selected' : ''}`);
      group.setAttribute('data-id', obj.id);

      // Main rectangle
      const rectEl = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rectEl.setAttribute('x', x);
      rectEl.setAttribute('y', y);
      rectEl.setAttribute('width', w);
      rectEl.setAttribute('height', h);
      rectEl.setAttribute('stroke', obj.color);
      rectEl.setAttribute('stroke-width', isSelected ? '3' : '2');
      rectEl.setAttribute('fill', obj.color);
      rectEl.setAttribute('fill-opacity', isSelected ? '0.22' : '0.10');
      rectEl.setAttribute('rx', '3');
      group.appendChild(rectEl);

      // Corner handles if selected
      if (isSelected) {
        const handlePositions = [
          [x, y], [x + w, y], [x, y + h], [x + w, y + h],
          [x + w / 2, y], [x + w / 2, y + h], [x, y + h / 2], [x + w, y + h / 2]
        ];
        handlePositions.forEach(([hx, hy]) => {
          const handle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          handle.setAttribute('cx', hx);
          handle.setAttribute('cy', hy);
          handle.setAttribute('r', '4.5');
          handle.setAttribute('fill', '#ffffff');
          handle.setAttribute('stroke', obj.color);
          handle.setAttribute('stroke-width', '2');
          group.appendChild(handle);
        });
      }

      // Label badge tag: Class name + Confidence
      const tagBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      const tagText = document.createElementNS('http://www.w3.org/2000/svg', 'text');

      const labelStr = `${obj.className} ${(obj.confidence * 100).toFixed(0)}%`;
      tagText.textContent = labelStr;
      tagText.setAttribute('x', x + 5);
      tagText.setAttribute('y', y - 6 > 16 ? y - 6 : y + 16);
      tagText.setAttribute('font-size', '11');
      tagText.setAttribute('font-weight', '600');
      tagText.setAttribute('fill', '#ffffff');
      tagText.setAttribute('font-family', 'Inter, system-ui, sans-serif');

      const tagW = labelStr.length * 6.8 + 10;
      const tagH = 18;
      const tagY = y - 6 > 16 ? y - 20 : y + 2;

      tagBg.setAttribute('x', x);
      tagBg.setAttribute('y', tagY);
      tagBg.setAttribute('width', tagW);
      tagBg.setAttribute('height', tagH);
      tagBg.setAttribute('fill', obj.color);
      tagBg.setAttribute('rx', '3');

      group.appendChild(tagBg);
      group.appendChild(tagText);

      // Event listeners
      group.addEventListener('click', (e) => {
        e.stopPropagation();
        this.setSelectedObject(obj.id);
        if (this.onSelectCallback) {
          this.onSelectCallback(obj.id);
        }
      });

      this.svg.appendChild(group);
    });
  }

  setSelectedObject(id) {
    this.selectedObjectId = id;
    this.renderBoxes();
  }

  toggle2DBox(id, visible) {
    const obj = APP_DATA.objects.find(o => o.id === id);
    if (obj) {
      obj.has2D = visible;
      this.renderBoxes();
    }
  }

  triggerLaserScan(callback) {
    if (!this.laserScan) return;
    this.laserScan.style.display = 'block';
    this.laserScan.style.top = '0%';

    let progress = 0;
    const interval = setInterval(() => {
      progress += 2.5;
      this.laserScan.style.top = `${progress}%`;
      if (progress >= 100) {
        clearInterval(interval);
        setTimeout(() => {
          this.laserScan.style.display = 'none';
          if (callback) callback();
        }, 150);
      }
    }, 25);
  }

  setZoom(level) {
    this.zoomLevel = level;
    if (this.wrapper) {
      this.wrapper.style.transform = `scale(${level})`;
      this.wrapper.style.transformOrigin = 'center center';
    }
    setTimeout(() => this.renderBoxes(), 100);
  }
}

window.CameraView = CameraView;

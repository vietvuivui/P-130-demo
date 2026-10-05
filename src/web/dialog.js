// Hộp thoại xác nhận / nhập liệu vẽ ngay trong giao diện, thay cho confirm() / prompt() của trình duyệt.
//   await uiConfirm('Xoá dự án?', { title: 'Xoá dự án', okText: 'Xoá', danger: true })  -> true / false
//   await uiPrompt('Tên mới:', 'giá trị cũ', { title: 'Đổi tên nhãn' })                  -> chuỗi, hoặc null nếu huỷ
// Tự chèn CSS nên dùng được ở mọi trang (projects, frames, export, index) mà không phụ thuộc file CSS nào.
(function () {
  const CSS = `
.ui-dialog-backdrop { position: fixed; inset: 0; z-index: 10000; display: flex; align-items: center; justify-content: center;
  padding: 16px; background: rgba(15, 23, 42, .45); animation: ui-dialog-fade .12s ease; }
.ui-dialog { width: min(420px, 100%); background: #fff; color: #262626; border-radius: 10px; padding: 20px 22px 16px;
  box-shadow: 0 16px 48px rgba(15, 23, 42, .25); font: 14px/1.5 Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
  animation: ui-dialog-pop .14s ease; }
.ui-dialog h3 { margin: 0 0 6px; font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.ui-dialog h3 .ui-dialog-icon { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px;
  border-radius: 50%; background: #e6f4ff; color: #1677ff; font-size: 13px; font-weight: 700; flex: none; }
.ui-dialog.danger h3 .ui-dialog-icon { background: #fff1f0; color: #cf1322; }
.ui-dialog p { margin: 0; color: #595959; white-space: pre-line; overflow-wrap: anywhere; }
.ui-dialog input { width: 100%; box-sizing: border-box; margin-top: 12px; height: 34px; padding: 4px 10px; font: inherit;
  color: #262626; background: #fff; border: 1px solid #d9d9d9; border-radius: 6px; }
.ui-dialog input:focus { outline: none; border-color: #1677ff; box-shadow: 0 0 0 2px rgba(22, 119, 255, .15); }
.ui-dialog-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 18px; }
.ui-dialog-actions button { height: 32px; padding: 0 14px; font: inherit; font-weight: 500; border-radius: 6px; cursor: pointer;
  border: 1px solid #d9d9d9; background: #fff; color: #262626; }
.ui-dialog-actions button:hover { border-color: #1677ff; color: #1677ff; }
.ui-dialog-actions button:focus-visible { outline: 2px solid rgba(22, 119, 255, .4); outline-offset: 1px; }
.ui-dialog-actions .ui-dialog-ok { background: #1677ff; border-color: #1677ff; color: #fff; }
.ui-dialog-actions .ui-dialog-ok:hover { background: #4096ff; border-color: #4096ff; color: #fff; }
.ui-dialog.danger .ui-dialog-ok { background: #cf1322; border-color: #cf1322; }
.ui-dialog.danger .ui-dialog-ok:hover { background: #e5484d; border-color: #e5484d; color: #fff; }
@keyframes ui-dialog-fade { from { opacity: 0; } }
@keyframes ui-dialog-pop { from { opacity: 0; transform: translateY(6px) scale(.98); } }
@media (prefers-reduced-motion: reduce) { .ui-dialog-backdrop, .ui-dialog { animation: none; } }`;

  function ensureCss() {
    if (document.getElementById('ui-dialog-css')) return;
    const st = document.createElement('style');
    st.id = 'ui-dialog-css';
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  function open({ title, message, okText = 'Đồng ý', cancelText = 'Huỷ', danger = false, input = null }) {
    ensureCss();
    return new Promise((resolve) => {
      const prevFocus = document.activeElement;
      const back = document.createElement('div');
      back.className = 'ui-dialog-backdrop';
      const box = document.createElement('div');
      box.className = 'ui-dialog' + (danger ? ' danger' : '');
      box.setAttribute('role', input === null ? 'alertdialog' : 'dialog');
      box.setAttribute('aria-modal', 'true');
      const h = document.createElement('h3');
      const icon = document.createElement('span');
      icon.className = 'ui-dialog-icon';
      icon.textContent = danger ? '!' : '?';
      h.append(icon, document.createTextNode(title));
      const p = document.createElement('p');
      p.textContent = message; // textContent: tên dự án / nhãn do người dùng đặt không bị hiểu thành HTML
      box.append(h, p);
      let field = null;
      if (input !== null) {
        field = document.createElement('input');
        field.type = 'text';
        field.value = input;
        box.append(field);
      }
      const actions = document.createElement('div');
      actions.className = 'ui-dialog-actions';
      const cancel = document.createElement('button');
      cancel.type = 'button';
      cancel.textContent = cancelText;
      const ok = document.createElement('button');
      ok.type = 'button';
      ok.className = 'ui-dialog-ok';
      ok.textContent = okText;
      actions.append(cancel, ok);
      box.append(actions);
      back.append(box);

      const close = (value) => {
        document.removeEventListener('keydown', onKey, true);
        back.remove();
        if (prevFocus && typeof prevFocus.focus === 'function') prevFocus.focus();
        resolve(value);
      };
      const accept = () => close(field ? field.value : true);
      const reject = () => close(field ? null : false);
      // Bắt phím ở pha capture và chặn lan xuống: phím tắt của trang duyệt (D xoá box, Enter approve…) không chạy khi hộp đang mở
      function onKey(e) {
        if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); reject(); return; }
        if (e.key === 'Enter' && e.target !== cancel) { e.preventDefault(); e.stopPropagation(); accept(); return; }
        if (e.key === 'Tab') {
          const items = [field, cancel, ok].filter(Boolean);
          const i = items.indexOf(document.activeElement);
          e.preventDefault();
          items[(i + (e.shiftKey ? items.length - 1 : 1)) % items.length].focus();
        }
        e.stopPropagation();
      }
      document.addEventListener('keydown', onKey, true);
      ok.addEventListener('click', accept);
      cancel.addEventListener('click', reject);
      back.addEventListener('mousedown', (e) => { if (e.target === back) reject(); });
      document.body.appendChild(back);
      if (field) { field.focus(); field.select(); } else (danger ? cancel : ok).focus(); // thao tác xoá: mặc định đứng ở Huỷ
    });
  }

  window.uiConfirm = (message, opts = {}) => open({ title: opts.title || 'Xác nhận', message, ...opts, input: null });
  window.uiPrompt = (message, value = '', opts = {}) =>
    open({ title: opts.title || 'Nhập thông tin', okText: 'Lưu', message, ...opts, input: String(value ?? '') });
})();

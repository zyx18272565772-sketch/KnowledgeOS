/* Shared UI primitives for the static enterprise console. */
function icon(name, size = 16, className = '') {
  return `<i data-lucide="${escapeHtml(name)}" width="${size}" height="${size}" class="${escapeHtml(className)}"></i>`;
}

function refreshIcons() {
  if (window.lucide?.createIcons) window.lucide.createIcons();
}

function statCard(label, value, iconName, note = '', tone = 'primary') {
  return `<article class="stat-card">
    <div class="stat-card-head"><span>${escapeHtml(label)}</span><span class="stat-icon stat-${tone}">${icon(iconName, 18)}</span></div>
    <div class="stat-value">${escapeHtml(String(value ?? 0))}</div>
    <div class="stat-note">${escapeHtml(note || '实时数据')}</div>
  </article>`;
}

function statusBadge(label, tone = 'neutral', dot = true) {
  return `<span class="status-badge status-${tone} ${dot ? '' : 'no-dot'}">${escapeHtml(label)}</span>`;
}

function emptyState(title, description, iconName = 'inbox', actionHtml = '') {
  return `<div class="empty-state"><span class="empty-icon">${icon(iconName, 21)}</span><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p>${actionHtml ? `<div class="empty-action">${actionHtml}</div>` : ''}</div>`;
}

function loadingState(label = '正在加载数据') {
  return `<div class="loading-state" role="status"><span class="spinner"></span><span>${escapeHtml(label)}</span></div>`;
}

function errorState(message, retryAction = '') {
  return `<div class="error-state"><span class="empty-icon">${icon('circle-alert', 21)}</span><h3>数据加载失败</h3><p>${escapeHtml(message)}</p>${retryAction ? `<button class="btn btn-secondary btn-sm" onclick="${retryAction}">${icon('refresh-cw', 15)}重新加载</button>` : ''}</div>`;
}

function showToast(message, type = 'success', title = '') {
  const region = document.getElementById('toast-region');
  if (!region) return;
  const config = {
    success: ['操作成功', 'circle-check'],
    error: ['操作失败', 'circle-alert'],
    warning: ['请注意', 'triangle-alert'],
    info: ['提示', 'info']
  }[type] || ['提示', 'info'];
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-icon">${icon(config[1], 18)}</span><div><strong>${escapeHtml(title || config[0])}</strong><p>${escapeHtml(message)}</p></div><button class="icon-button" aria-label="关闭通知">${icon('x', 14)}</button>`;
  const close = () => { toast.classList.add('toast-out'); setTimeout(() => toast.remove(), 180); };
  toast.querySelector('button').onclick = close;
  region.appendChild(toast);
  refreshIcons();
  setTimeout(close, type === 'error' ? 6000 : 3800);
}

let confirmResolver = null;
function askConfirm({ title = '确认操作', message = '', confirmText = '确认', danger = false } = {}) {
  const modal = document.getElementById('confirm-modal');
  if (!modal) return Promise.resolve(window.confirm(message));
  document.getElementById('confirm-title').textContent = title;
  document.getElementById('confirm-message').textContent = message;
  const ok = document.getElementById('confirm-ok');
  ok.textContent = confirmText;
  ok.className = `btn ${danger ? 'btn-danger' : 'btn-primary'}`;
  document.getElementById('confirm-icon').className = `modal-icon ${danger ? 'modal-icon-danger' : ''}`;
  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden', 'false');
  refreshIcons();
  return new Promise(resolve => {
    confirmResolver = resolve;
    const settle = value => {
      modal.classList.add('hidden');
      modal.setAttribute('aria-hidden', 'true');
      confirmResolver = null;
      resolve(value);
    };
    ok.onclick = () => settle(true);
    document.getElementById('confirm-cancel').onclick = () => settle(false);
    modal.querySelector('.modal-backdrop').onclick = () => settle(false);
  });
}

function closeConfirmModal() {
  if (confirmResolver) document.getElementById('confirm-cancel')?.click();
}

function setButtonLoading(button, loading, label = '处理中') {
  if (!button) return;
  if (loading) {
    button.dataset.originalHtml = button.innerHTML;
    button.disabled = true;
    button.innerHTML = `<span class="spinner spinner-light"></span>${escapeHtml(label)}`;
  } else {
    button.disabled = false;
    if (button.dataset.originalHtml) button.innerHTML = button.dataset.originalHtml;
    delete button.dataset.originalHtml;
    refreshIcons();
  }
}

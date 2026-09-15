const API = '/api';
const TASK_LABELS = { chitchat: '闲聊', knowledge_qa: '知识问答', reasoning: '深度推理', knowledge_inspection: '知识巡检', admin_copilot: '管理助手' };
const ATTACHMENT_LABELS = { summary: '附件总结', case_assist: '客服案例辅助', reasoning: '附件分析' };
const STEP_LABELS = { memory_read: '读取会话上下文', question_rewrite: '改写检索问题', knowledge_search: '检索企业知识', result_evaluation: '验证资料充分性', answer_generation: '生成回答', identity_answer: '生成回答', admin_operation: '执行管理操作', clarification: '澄清问题' };
const PAGE_META = {
  dashboard: ['工作台', '工作台', '查看知识库、Agent 与系统运行概况'],
  chat: ['工作台 / 智能问答', '智能问答', '基于企业知识库提供可追溯的智能问答'],
  kb: ['知识与 Agent / 知识库管理', '知识库管理', '管理企业知识的上传、巡检、审核与发布'],
  'agent-runs': ['知识与 Agent / Agent 执行记录', 'Agent 执行记录', '查看任务级调用质量和运行表现'],
  report: ['知识与 Agent / 自动报表', '自动报表', '查看 Agent 与知识库实时运营汇总'],
  users: ['系统管理 / 用户管理', '用户管理', '管理系统用户、角色与访问权限']
};

let state = {
  user: '', isAdmin: false, token: '', theme: 'light', sidebarCollapsed: false,
  sessions: {}, activeSid: null, currentView: 'chat', streaming: false,
  uploadingAttachments: false, attachments: [], users: [], agentRuns: [],
  kbDocs: [], kbTab: 'ACTIVE', kbBusy: {}, rejectDocId: null, drawerDocId: null,
  dashboard: { agent: null, documents: null }, report: null
};

marked.setOptions({ breaks: true, gfm: true });

async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set('Authorization', `Bearer ${state.token}`);
  return fetch(url, { ...options, headers });
}
async function responseData(response, fallback) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || fallback);
  return data;
}

async function doLogin(user, pass) {
  const response = await fetch(`${API}/auth/login`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: user, password: pass }) });
  return responseData(response, '登录失败，请检查账号和密码');
}
async function doRegister() {
  const user = document.getElementById('login-user').value.trim();
  const pass = document.getElementById('login-pass').value;
  if (user.length < 2) return showLoginErr('用户名至少需要 2 个字符');
  if (pass.length < 3) return showLoginErr('密码至少需要 3 个字符');
  const button = document.querySelector('#login-form button[type="button"]');
  setButtonLoading(button, true, '注册中');
  try {
    const response = await fetch(`${API}/auth/register`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: user, password: pass }) });
    await responseData(response, '注册失败');
    await onLoginSuccess(await doLogin(user, pass));
  } catch (error) { showLoginErr(error.message); }
  finally { setButtonLoading(button, false); }
}

document.getElementById('login-form').addEventListener('submit', async event => {
  event.preventDefault();
  const user = document.getElementById('login-user').value.trim();
  const pass = document.getElementById('login-pass').value;
  if (!user) return showLoginErr('请输入用户名');
  const button = document.getElementById('login-submit');
  setButtonLoading(button, true, '正在登录');
  document.getElementById('login-err').classList.add('hidden');
  try { await onLoginSuccess(await doLogin(user, pass)); }
  catch (error) { showLoginErr(error.message); }
  finally { setButtonLoading(button, false); }
});

async function onLoginSuccess(result) {
  state.user = result.username;
  state.isAdmin = Boolean(result.is_admin);
  state.token = result.access_token || '';
  state.sessions = {};
  document.getElementById('login-screen').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');
  document.getElementById('current-user').textContent = state.user;
  document.getElementById('current-role').textContent = state.isAdmin ? '管理员' : '普通用户';
  document.getElementById('current-user-avatar').textContent = state.user.slice(0, 1).toUpperCase();
  document.querySelectorAll('.admin-nav').forEach(element => element.classList.toggle('hidden', !state.isAdmin));
  const saved = localStorage.getItem(`agentcraft_${state.user}`);
  if (saved) { try { state.sessions = JSON.parse(saved) || {}; } catch { state.sessions = {}; } }
  try {
    const response = await apiFetch(`${API}/conversations`);
    if (response.ok) {
      const data = await response.json();
      for (const conversation of data.conversations || []) {
        const sid = conversation.conversation_id;
        if (!state.sessions[sid]) state.sessions[sid] = { title: conversation.summary || '未命名对话', messages: [], created: '', _fromServer: true };
      }
    }
  } catch { /* 本地缓存仍可使用 */ }
  if (!Object.keys(state.sessions).length) newChat(false);
  else { state.activeSid = Object.keys(state.sessions).at(-1); await loadMessages(state.activeSid); }
  renderAll();
  switchView(state.isAdmin ? 'dashboard' : 'chat');
}
function showLoginErr(message) { const element = document.getElementById('login-err'); element.textContent = message; element.classList.remove('hidden'); }
function togglePassword() {
  const input = document.getElementById('login-pass');
  input.type = input.type === 'password' ? 'text' : 'password';
  document.getElementById('password-icon').setAttribute('data-lucide', input.type === 'password' ? 'eye' : 'eye-off');
  refreshIcons();
}
function fillDemoAccount(role = 'admin') {
  const account = role === 'user'
    ? { username: 'agent', password: '123456' }
    : { username: 'admin', password: 'admin123' };
  document.getElementById('login-user').value = account.username;
  document.getElementById('login-pass').value = account.password;
  document.getElementById('login-err').classList.add('hidden');
  document.getElementById('login-submit').focus();
}
function logout() {
  saveSessions();
  if (typeof kbPollTimer !== 'undefined') clearTimeout(kbPollTimer);
  Object.assign(state, { user: '', isAdmin: false, token: '', sessions: {}, activeSid: null, attachments: [], users: [], kbDocs: [], streaming: false, currentView: 'chat' });
  document.getElementById('session-search').value = '';
  document.getElementById('session-list').innerHTML = '';
  document.getElementById('current-user').textContent = '';
  document.getElementById('current-role').textContent = '普通用户';
  document.getElementById('current-user-avatar').textContent = 'A';
  document.getElementById('chat-input').value = '';
  document.getElementById('login-screen').classList.remove('hidden');
  document.getElementById('app').classList.add('hidden');
  document.getElementById('login-pass').value = '';
  document.getElementById('login-user').focus();
}

function renderHeaderActions(view) {
  const buttons = {
    dashboard: `<button class="btn btn-secondary btn-sm" onclick="loadDashboard()">${icon('refresh-cw', 15)}刷新</button>`,
    chat: '',
    kb: `<button class="btn btn-secondary btn-sm" onclick="loadKBList()">${icon('refresh-cw', 15)}刷新</button><button class="btn btn-primary btn-sm" onclick="document.getElementById('kb-upload').click()">${icon('upload', 15)}上传知识</button>`,
    users: `<button class="btn btn-primary btn-sm" onclick="openUserModal()">${icon('user-plus', 15)}新增用户</button>`,
    'agent-runs': `<button class="btn btn-secondary btn-sm" onclick="loadAgentRuns()">${icon('refresh-cw', 15)}刷新</button>`,
    report: `<button class="btn btn-secondary btn-sm" onclick="loadReport()">${icon('refresh-cw', 15)}刷新</button><button class="btn btn-primary btn-sm" onclick="exportReport()">${icon('download', 15)}导出报表</button>`
  };
  document.getElementById('page-actions').innerHTML = buttons[view] || '';
  refreshIcons();
}
function switchView(requestedView) {
  const view = !state.isAdmin && requestedView !== 'chat' ? 'chat' : requestedView;
  state.currentView = view;
  if (view !== 'kb' && typeof kbPollTimer !== 'undefined') clearTimeout(kbPollTimer);
  document.querySelectorAll('.view-panel').forEach(element => element.classList.add('hidden'));
  document.getElementById(`${view}-view`)?.classList.remove('hidden');
  document.querySelectorAll('.nav-item[data-view]').forEach(element => element.classList.toggle('active', element.dataset.view === view));
  const meta = PAGE_META[view] || PAGE_META.chat;
  document.getElementById('page-breadcrumb').textContent = meta[0];
  document.getElementById('page-title').textContent = meta[1];
  document.getElementById('page-subtitle').textContent = meta[2];
  renderHeaderActions(view); closeMobileSidebar();
  if (view === 'dashboard') loadDashboard();
  if (view === 'users') loadUsers();
  if (view === 'kb') loadKBList();
  if (view === 'agent-runs') loadAgentRuns();
  if (view === 'report') loadReport();
  refreshIcons();
}
function applySidebarState(collapsed, persist = false) {
  state.sidebarCollapsed = Boolean(collapsed);
  document.getElementById('app').classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
  const button = document.getElementById('sidebar-toggle');
  const label = state.sidebarCollapsed ? '展开侧栏' : '收起侧栏';
  button.setAttribute('aria-label', label);
  button.setAttribute('aria-expanded', state.sidebarCollapsed ? 'false' : 'true');
  button.setAttribute('title', label);
  button.dataset.tooltip = label;
  button.innerHTML = icon(state.sidebarCollapsed ? 'panel-left-open' : 'panel-left-close');
  if (persist) localStorage.setItem('knowledgeos_sidebar_collapsed_v2', state.sidebarCollapsed ? '1' : '0');
  refreshIcons();
}
function toggleSidebar() {
  applySidebarState(!state.sidebarCollapsed, true);
}
function openMobileSidebar() { document.getElementById('app-sidebar').classList.add('mobile-open'); document.getElementById('sidebar-backdrop').classList.remove('hidden'); }
function closeMobileSidebar() { document.getElementById('app-sidebar').classList.remove('mobile-open'); document.getElementById('sidebar-backdrop').classList.add('hidden'); }

function newChat(activate = true) {
  state.attachments = []; state.uploadingAttachments = false; renderAttachments();
  const sid = `${state.user}_${Math.random().toString(36).slice(2, 10)}`;
  state.sessions[sid] = { title: '新对话', messages: [], created: new Date().toISOString() };
  state.activeSid = sid; renderAll();
  if (activate) switchView('chat');
}
function saveSessions() { if (state.user) localStorage.setItem(`agentcraft_${state.user}`, JSON.stringify(state.sessions)); }
function curSession() { return state.sessions[state.activeSid] || { messages: [] }; }
function renderSessions() {
  const list = document.getElementById('session-list'); if (!list) return;
  const query = document.getElementById('session-search')?.value.trim().toLowerCase() || '';
  const entries = Object.entries(state.sessions).reverse().filter(([, session]) => {
    const title = session.title === '新对话' && session.messages.length ? session.messages[0].content.slice(0, 30) : session.title;
    return !query || String(title).toLowerCase().includes(query);
  });
  list.innerHTML = entries.map(([sid, session]) => {
    const title = session.title === '新对话' && session.messages.length ? session.messages[0].content.slice(0, 30) : session.title;
    const displayTitle = title || '未命名对话';
    return `<div class="session-row ${sid === state.activeSid ? 'active' : ''}"><button class="session-main" onclick="switchSession(${jsArg(sid)})" title="${escapeHtml(displayTitle)}" data-tooltip="${escapeHtml(displayTitle)}" data-tooltip-side="right" aria-label="打开对话：${escapeHtml(displayTitle)}">${icon('message-square', 15)}<span>${escapeHtml(displayTitle)}</span></button><div class="session-actions"><button onclick="renameSession(${jsArg(sid)})" title="重命名" aria-label="重命名对话" data-tooltip="重命名" data-tooltip-side="top">${icon('pencil', 13)}</button><button onclick="deleteSession(${jsArg(sid)})" title="删除" aria-label="删除对话" data-tooltip="删除" data-tooltip-side="top">${icon('trash-2', 13)}</button></div></div>`;
  }).join('') || `<div class="sidebar-empty">${query ? '没有匹配的对话' : '暂无历史对话'}</div>`;
  refreshIcons();
}
async function switchSession(sid) {
  state.attachments = []; state.uploadingAttachments = false; renderAttachments();
  state.activeSid = sid; await loadMessages(sid); renderAll(); switchView('chat');
}
async function loadMessages(sid) {
  const session = state.sessions[sid]; if (!session || (session.messages.length && !session._fromServer)) return;
  try {
    const response = await apiFetch(`${API}/conversations/${encodeURIComponent(sid)}`);
    if (!response.ok) return;
    const data = await response.json();
    if (data.messages?.length) {
      session.messages = data.messages.map(message => ({ role: message.role || 'assistant', content: message.content || '', sources: message.sources || [], taskType: message.task_type || '' }));
      session._fromServer = false;
      if (['新对话', '未命名对话'].includes(session.title)) session.title = data.messages[0]?.content?.slice(0, 30) || '对话';
    }
  } catch { /* 保留本地会话 */ }
}
function renameSession(sid) {
  const name = prompt('输入新的对话名称', state.sessions[sid]?.title || '');
  if (name?.trim()) { state.sessions[sid].title = name.trim().slice(0, 50); renderAll(); }
}
async function deleteSession(sid) {
  if (!await askConfirm({ title: '删除对话', message: '这会同时删除 Redis 中保存的该会话，删除后无法恢复。', confirmText: '删除', danger: true })) return;
  try { await apiFetch(`${API}/conversations/${encodeURIComponent(sid)}`, { method: 'DELETE' }); } catch { /* 仍清理本地缓存 */ }
  delete state.sessions[sid];
  if (state.activeSid === sid) state.activeSid = Object.keys(state.sessions).at(-1) || null;
  if (!state.activeSid) newChat(false);
  renderAll(); showToast('对话已删除');
}
async function clearCurrentChat() { if (state.activeSid) await deleteSession(state.activeSid); }

function toggleTheme() {
  state.theme = state.theme === 'light' ? 'dark' : 'light';
  document.documentElement.classList.toggle('dark', state.theme === 'dark');
  const button = document.getElementById('theme-btn');
  button.innerHTML = `${icon(state.theme === 'light' ? 'moon' : 'sun', 17)}<span>${state.theme === 'light' ? '深色模式' : '浅色模式'}</span>`;
  localStorage.setItem('agentcraft_theme', state.theme); refreshIcons();
}
function renderAll() { renderSessions(); renderMessages(); saveSessions(); }

function sourceName(source) {
  if (typeof source === 'string') return source;
  return source?.doc || source?.source || source?.filename || source?.doc_name || source?.metadata?.source || '未知文档';
}
function sourceText(source) {
  if (!source || typeof source === 'string') return '';
  return source.page_content || source.content || source.chunk_text || source.preview || source.text || '';
}
function sourceChunk(source) {
  if (!source || typeof source === 'string') return null;
  return source.chunk_index ?? source.chunk_id ?? source.metadata?.chunk_index ?? source.metadata?.chunk_id ?? null;
}
function encodeSourcePayload(source) { return btoa(unescape(encodeURIComponent(JSON.stringify(source)))); }
function renderSourceList(sources) {
  if (!Array.isArray(sources) || !sources.length) return '';
  const unique = [...new Map(sources.map(source => [sourceName(source), source])).values()];
  return `<section class="answer-sources"><div class="source-heading">${icon('files', 15)}<span>参考来源</span><small>${unique.length}</small></div><div class="source-list">${unique.map(source => { const name = sourceName(source); const payload = encodeSourcePayload(source); return `<button type="button" title="${escapeHtml(name)}" aria-label="查看来源：${escapeHtml(name)}" onclick="openSourceDetail('${payload}')">${icon('file-text', 13)}<span>${escapeHtml(name)}</span></button>`; }).join('')}</div></section>`;
}
function openSourceDetail(payload) {
  let source;
  try { source = JSON.parse(decodeURIComponent(escape(atob(payload)))); } catch { source = payload; }
  const name = sourceName(source); const content = sourceText(source); const chunk = sourceChunk(source);
  if (!content && chunk == null) {
    showToast('当前接口只返回了文件名，没有引用片段可供展开。', 'info', name);
    return;
  }
  document.getElementById('detail-drawer-eyebrow').textContent = 'KNOWLEDGE SOURCE';
  document.getElementById('detail-drawer-title').textContent = '参考来源';
  document.getElementById('detail-drawer-subtitle').textContent = name;
  const metadata = [];
  if (chunk != null) metadata.push(`Chunk ${chunk}`);
  if (source?.score != null) metadata.push(`检索分数 ${Number(source.score).toFixed(3)}`);
  if (source?.similarity != null) metadata.push(`相似度 ${Math.round(Number(source.similarity) * 100)}%`);
  document.getElementById('detail-drawer-content').innerHTML = `<div class="source-detail"><div class="source-detail-head"><span class="document-icon">${icon('file-text', 18)}</span><div><strong>${escapeHtml(name)}</strong>${metadata.length ? `<small>${escapeHtml(metadata.join(' · '))}</small>` : ''}</div></div>${content ? `<div class="source-hit"><span>命中内容</span><p>${escapeHtml(content)}</p></div>` : emptyState('暂无片段内容', '当前来源只提供了 Chunk 标识。', 'file-search')}</div>`;
  const drawer = document.getElementById('detail-drawer'); drawer.classList.remove('hidden'); drawer.setAttribute('aria-hidden', 'false'); refreshIcons();
}
function renderMessages() {
  const container = document.getElementById('chat-messages');
  const welcome = document.getElementById('welcome');
  const messages = curSession().messages;
  welcome.classList.toggle('hidden', Boolean(messages.length));
  container.querySelectorAll('.msg-row').forEach(element => element.remove());
  for (const message of messages) {
    const row = document.createElement('article');
    row.className = `msg-row ${message.role === 'user' ? 'message-user' : 'message-assistant'}`;
    row.innerHTML = message.role === 'user'
      ? `<div class="message-user-bubble"><div class="prose">${marked.parse(message.content)}</div></div>`
      : `<div class="assistant-avatar">${icon('sparkles', 16)}</div><div class="assistant-content"><div class="prose">${marked.parse(message.content)}</div>${message.taskType ? `<div class="answer-meta">${statusBadge(TASK_LABELS[message.taskType] || message.taskType, 'info')}</div>` : ''}${renderSourceList(message.sources)}</div>`;
    container.appendChild(row);
  }
  container.scrollTop = container.scrollHeight; refreshIcons();
}

async function uploadFile(input) { await uploadFiles(Array.from(input.files || [])); input.value = ''; }
async function uploadFiles(files) {
  if (!files.length) return;
  state.uploadingAttachments = true; renderAttachments();
  for (const file of files) {
    const form = new FormData(); form.append('file', file);
    try {
      const data = await responseData(await apiFetch(`${API}/documents/parse-temp`, { method: 'POST', body: form }), `无法解析 ${file.name}`);
      state.attachments.push({ filename: data.filename, file_type: data.file_type || 'document', content: data.content || data.text || '', chunks_count: data.chunks_count || 0, char_count: data.char_count || 0 });
    } catch (error) { showToast(error.message, 'error', '附件解析失败'); }
  }
  state.uploadingAttachments = false; renderAttachments();
}
function removeAttachment(index) { state.attachments.splice(index, 1); renderAttachments(); }
function renderAttachments() {
  const bar = document.getElementById('attach-bar'); if (!bar) return;
  if (state.uploadingAttachments) { bar.innerHTML = `<span class="attachment-chip loading"><span class="spinner"></span>正在识别附件</span>`; return; }
  bar.innerHTML = state.attachments.map((attachment, index) => `<span class="attachment-chip">${icon(attachment.file_type === 'image' ? 'image' : 'file-text', 14)}<span>${escapeHtml(attachment.filename)}</span><small>${Number(attachment.char_count || 0)} 字</small><button onclick="removeAttachment(${index})" aria-label="移除附件">${icon('x', 12)}</button></span>`).join('');
  refreshIcons();
}
function quickAsk(text) { const input = document.getElementById('chat-input'); input.value = text; input.dispatchEvent(new Event('input')); send(); }

function agentPanel(steps) {
  if (!steps.length) return '';
  return `<div class="agent-progress">${steps.map(step => {
    const stepIndicator = step.status === 'running'
      ? '<i class="step-spinner" aria-hidden="true"></i>'
      : icon(step.status === 'completed' ? 'check' : 'x', 13);
    return `<div class="agent-step step-${step.status}"><span>${stepIndicator}</span><strong>${escapeHtml(step.label)}</strong></div>`;
  }).join('')}</div>`;
}
function renderStreamingMessage(element, steps, answer, sources, finished) {
  let content = element.querySelector('.assistant-content');
  let iconsChanged = false;
  if (!content) {
    element.innerHTML = `<div class="assistant-avatar">${icon('sparkles', 16)}</div><div class="assistant-content"><div data-stream-section="steps"></div><div data-stream-section="answer"></div><div data-stream-section="sources"></div></div>`;
    content = element.querySelector('.assistant-content');
    iconsChanged = true;
  }

  const stepsHost = content.querySelector('[data-stream-section="steps"]');
  const stepsSignature = JSON.stringify(steps.map(step => [step.id, step.label, step.status]));
  if (stepsHost.dataset.signature !== stepsSignature) {
    stepsHost.innerHTML = agentPanel(steps);
    stepsHost.dataset.signature = stepsSignature;
    iconsChanged = true;
  }

  const answerHost = content.querySelector('[data-stream-section="answer"]');
  answerHost.innerHTML = answer
    ? `<div class="prose">${marked.parse(answer)}${finished ? '' : '<span class="stream-cursor"></span>'}</div>`
    : `<div class="thinking-line"><span class="typing"><i></i><i></i><i></i></span><span>正在处理请求</span></div>`;

  const sourcesHost = content.querySelector('[data-stream-section="sources"]');
  const sourcesSignature = JSON.stringify(sources || []);
  if (sourcesHost.dataset.signature !== sourcesSignature) {
    sourcesHost.innerHTML = renderSourceList(sources);
    sourcesHost.dataset.signature = sourcesSignature;
    iconsChanged = true;
  }
  if (iconsChanged) refreshIcons();
}

async function send() {
  if (state.streaming) return;
  if (state.uploadingAttachments) return showToast('附件仍在识别，请稍后发送', 'warning');
  const input = document.getElementById('chat-input');
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = ''; input.style.height = 'auto';
  if (!state.activeSid) newChat(false);
  const messages = curSession().messages;
  messages.push({ role: 'user', content: prompt });
  if (curSession().title === '新对话') curSession().title = prompt.slice(0, 30);
  renderMessages(); renderSessions();

  const container = document.getElementById('chat-messages');
  const streamingRow = document.createElement('article');
  streamingRow.id = 'streaming-msg'; streamingRow.className = 'msg-row message-assistant';
  container.appendChild(streamingRow);
  const steps = [{ id: 'routing', label: '意图识别', status: 'running' }];
  renderStreamingMessage(streamingRow, steps, '', [], false);
  container.scrollTop = container.scrollHeight;

  state.streaming = true;
  const sendButton = document.getElementById('send-button'); setButtonLoading(sendButton, true, '生成中');
  let fullAnswer = '', sources = [], taskType = '', answerFinished = false, requestCompleted = false;
  const upsertStep = (id, label, status) => {
    const existing = steps.find(step => step.id === id);
    if (existing) { existing.status = status; if (label) existing.label = label; }
    else steps.push({ id, label, status });
  };
  const completeRouting = () => {
    const route = steps.find(step => step.id === 'routing');
    if (route?.status === 'running') route.status = 'completed';
  };
  const consumeReasoningProgress = content => {
    if (taskType !== 'reasoning' || typeof content !== 'string') return false;
    const text = content.trim();
    if (/^🔍\s*正在拆解问题/.test(text)) {
      upsertStep('reasoning_decompose', '拆解推理问题', 'running');
      return true;
    }
    const subQuestion = text.match(/^⏳\s*正在推理子问题\s*(\d+)\/(\d+)/);
    if (subQuestion) {
      upsertStep('reasoning_decompose', '拆解推理问题', 'completed');
      upsertStep('reasoning_subquestion', `分析子问题 ${subQuestion[1]}/${subQuestion[2]}`, 'running');
      return true;
    }
    if (/^✍️\s*正在汇总生成答案/.test(text)) {
      upsertStep('reasoning_subquestion', steps.find(step => step.id === 'reasoning_subquestion')?.label || '分析子问题', 'completed');
      upsertStep('reasoning_synthesis', '生成回答', 'running');
      return true;
    }
    return false;
  };

  try {
    const requestAttachments = state.attachments.map(attachment => ({ ...attachment, content: attachment.content.slice(0, 50000) }));
    const response = await apiFetch(`${API}/agent/run/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: prompt, conversation_id: state.activeSid, user_id: state.user, stream: true, context: '', attachments: requestAttachments })
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || '请求失败');
    if (!response.body) throw new Error('浏览器未收到流式响应');
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n'); buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let event; try { event = JSON.parse(line.slice(6)); } catch { continue; }
        switch (event.type) {
          case 'routing_started': upsertStep('routing', '意图识别', 'running'); break;
          case 'routed': {
            taskType = event.task_type || '';
            if (taskType === 'reasoning') {
              const genericGeneration = steps.findIndex(step => step.id === 'generation' || step.id === 'answer_generation');
              if (genericGeneration >= 0) steps.splice(genericGeneration, 1);
            }
            const routeLabel = event.attachment_action ? (ATTACHMENT_LABELS[event.attachment_action] || '附件处理') : (TASK_LABELS[taskType] || taskType || '通用处理');
            upsertStep('routing', `意图识别 · ${routeLabel}`, 'running'); break;
          }
          case 'retrieval_result':
            if (!sources.length) sources = event.documents || [];
            upsertStep('knowledge_search', `检索企业知识 · ${Number(event.count || 0)} 条`, 'completed'); break;
          case 'step_started':
            completeRouting();
            if (event.step_name !== 'memory_write' && !(taskType === 'reasoning' && event.step_name === 'answer_generation')) upsertStep(event.step_name, STEP_LABELS[event.step_name] || event.step_name || '执行步骤', 'running');
            break;
          case 'step_completed':
            if (event.step_name !== 'memory_write' && !(taskType === 'reasoning' && event.step_name === 'answer_generation')) upsertStep(event.step_name, STEP_LABELS[event.step_name] || event.step_name || '执行步骤', 'completed');
            if (event.step_name === 'answer_generation') answerFinished = true;
            break;
          case 'inspection_started': completeRouting(); upsertStep('inspection', '执行知识巡检', 'running'); break;
          case 'start': completeRouting(); if (taskType !== 'reasoning' && !steps.some(step => step.status === 'running')) upsertStep('generation', '生成回答', 'running'); break;
          case 'token': {
            const content = event.content || '';
            if (!consumeReasoningProgress(content)) fullAnswer += content;
            break;
          }
          case 'answer': fullAnswer = event.content || fullAnswer; break;
          case 'sources': sources = event.sources || []; if (!taskType) taskType = event.task_type || ''; break;
          case 'step_failed': upsertStep(event.step_name, STEP_LABELS[event.step_name] || event.step_name || '执行步骤', 'failed'); break;
          case 'end':
            if (typeof event.content === 'string') fullAnswer = event.content;
            else if (event.content?.answer) fullAnswer = event.content.answer;
            steps.filter(step => step.status === 'running').forEach(step => { step.status = 'completed'; });
            answerFinished = true; requestCompleted = true; break;
          case 'error':
            steps.filter(step => step.status === 'running').forEach(step => { step.status = 'failed'; });
            fullAnswer = event.content || '请求执行失败'; answerFinished = true; break;
        }
        renderStreamingMessage(streamingRow, steps, fullAnswer, sources, answerFinished);
        container.scrollTop = container.scrollHeight;
      }
    }
  } catch (error) {
    fullAnswer = `请求未完成：${error.message}`;
    steps.filter(step => step.status === 'running').forEach(step => { step.status = 'failed'; });
    renderStreamingMessage(streamingRow, steps, fullAnswer, sources, true);
    showToast(error.message, 'error', '问答请求失败');
  }

  if (requestCompleted) { state.attachments = []; renderAttachments(); }
  messages.push({ role: 'assistant', content: fullAnswer || '抱歉，未能生成回答。', sources, taskType });
  state.streaming = false; streamingRow.remove(); setButtonLoading(sendButton, false);
  renderMessages(); renderSessions(); saveSessions();
}

function taskLabel(type) { return TASK_LABELS[type] || type || '未分类任务'; }
function percent(value) { return `${Math.max(0, Math.min(100, Number(value || 0))).toFixed(Number(value || 0) % 1 ? 1 : 0)}%`; }

async function loadDashboard() {
  document.getElementById('stats-cards').innerHTML = Array(4).fill(statCard('加载中', '—', 'loader-circle')).join('');
  document.getElementById('agent-stats').innerHTML = loadingState('正在汇总 Agent 指标');
  document.getElementById('knowledge-distribution').innerHTML = loadingState('正在读取知识状态');
  document.getElementById('governance-overview').innerHTML = loadingState('正在读取治理结果');
  renderActiveChats(); refreshIcons();
  try {
    const [agentResponse, documentResponse] = await Promise.all([apiFetch(`${API}/agent/stats`), apiFetch(`${API}/documents?page=1&page_size=1000`)]);
    const [agentData, documentData] = await Promise.all([responseData(agentResponse, 'Agent 统计获取失败'), responseData(documentResponse, '知识库统计获取失败')]);
    state.dashboard = { agent: agentData, documents: documentData };
    const documents = documentData.documents || [];
    const stats = documentData.stats || {};
    const active = Number(stats.ACTIVE ?? documents.filter(document => document.status === 'ACTIVE').length);
    const pending = Number(stats.PENDING_REVIEW ?? documents.filter(document => document.status === 'PENDING_REVIEW').length);
    const conflicts = documents.filter(document => document.status === 'PENDING_REVIEW').reduce((sum, document) => sum + Number(document.conflict_count || 0), 0);
    document.getElementById('stats-cards').innerHTML = [
      statCard('Agent 总调用', agentData.summary?.total || 0, 'bot', '当前进程', 'primary'),
      statCard('已发布知识', active, 'book-check', '可被 RAG 检索', 'success'),
      statCard('待审核文档', pending, 'clock-3', '尚未进入线上索引', 'warning'),
      statCard('Agent 成功率', percent(agentData.summary?.overall_success_rate), 'circle-check-big', `${agentData.summary?.failed || 0} 次失败`, agentData.summary?.failed ? 'warning' : 'success')
    ].join('');
    renderAgentOverview(agentData.task_stats || []);
    renderKnowledgeDistribution(documents, stats);
    renderGovernanceOverview(documents, conflicts);
    refreshIcons();
  } catch (error) {
    document.getElementById('stats-cards').innerHTML = errorState(error.message, 'loadDashboard()');
    document.getElementById('agent-stats').innerHTML = errorState(error.message, 'loadDashboard()');
    document.getElementById('knowledge-distribution').innerHTML = errorState(error.message, 'loadDashboard()');
    document.getElementById('governance-overview').innerHTML = errorState(error.message, 'loadDashboard()');
    refreshIcons();
  }
}
function renderAgentOverview(tasks) {
  const target = document.getElementById('agent-stats');
  if (!tasks.length) { target.innerHTML = emptyState('暂无 Agent 调用', '发送一次问答后，这里会显示真实任务统计。', 'activity'); return; }
  target.innerHTML = `<div class="metric-list">${tasks.map(task => `<button class="metric-row" onclick="openAgentRunDetail('${escapeHtml(task.task_type)}')"><span class="metric-name"><span class="metric-icon">${icon('bot', 15)}</span><span><strong>${escapeHtml(taskLabel(task.task_type))}</strong><small>${Number(task.total || 0)} 次调用 · 平均 ${Number(task.avg_duration_ms || 0).toFixed(0)} ms</small></span></span><span class="metric-bar"><i style="width:${Math.max(2, Number(task.success_rate || 0))}%"></i></span><b>${percent(task.success_rate)}</b></button>`).join('')}</div>`;
}
function renderKnowledgeDistribution(documents, stats = {}) {
  const values = [
    ['已发布', Number(stats.ACTIVE ?? documents.filter(d => d.status === 'ACTIVE').length), 'success'],
    ['待审核', Number(stats.PENDING_REVIEW ?? documents.filter(d => d.status === 'PENDING_REVIEW').length), 'warning'],
    ['已驳回', Number(stats.REJECTED ?? documents.filter(d => d.status === 'REJECTED').length), 'neutral']
  ];
  const total = Math.max(1, values.reduce((sum, item) => sum + item[1], 0));
  document.getElementById('knowledge-distribution').innerHTML = `<div class="distribution-bar">${values.filter(item => item[1]).map(item => `<i class="bar-${item[2]}" style="width:${item[1] / total * 100}%"></i>`).join('')}</div><div class="distribution-list">${values.map(item => `<div><span><i class="legend-${item[2]}"></i>${item[0]}</span><strong>${item[1]}</strong></div>`).join('')}</div>`;
}
function renderActiveChats() {
  const sessions = Object.entries(state.sessions).sort((a, b) => b[1].messages.length - a[1].messages.length).slice(0, 5);
  document.getElementById('active-chats').innerHTML = sessions.length ? `<div class="chat-activity-list">${sessions.map(([sid, session]) => `<button onclick="switchSession('${escapeHtml(sid)}')"><span>${icon('message-square', 15)}</span><p><strong>${escapeHtml(session.title || '未命名对话')}</strong><small>${session.messages.length} 条消息</small></p>${icon('chevron-right', 15)}</button>`).join('')}</div>` : emptyState('暂无对话记录', '开始一次智能问答后，会话会显示在这里。', 'messages-square');
  refreshIcons();
}
function renderGovernanceOverview(documents, conflicts) {
  const pending = documents.filter(document => document.status === 'PENDING_REVIEW');
  const duplicates = pending.reduce((sum, document) => sum + Number(document.duplicate_count || 0), 0);
  const quality = pending.reduce((sum, document) => sum + Number(document.low_quality_count || 0), 0);
  document.getElementById('governance-overview').innerHTML = `<div class="governance-metrics"><button onclick="state.kbTab='PENDING_REVIEW';switchView('kb')"><span class="risk-icon risk-warning">${icon('clock-3', 17)}</span><p><strong>${pending.length}</strong><small>待审核文档</small></p></button><button onclick="state.kbTab='PENDING_REVIEW';switchView('kb')"><span class="risk-icon risk-info">${icon('copy-check', 17)}</span><p><strong>${duplicates}</strong><small>语义重复规则</small></p></button><button onclick="state.kbTab='PENDING_REVIEW';switchView('kb')"><span class="risk-icon risk-error">${icon('git-compare-arrows', 17)}</span><p><strong>${conflicts}</strong><small>潜在规则冲突</small></p></button><button onclick="state.kbTab='PENDING_REVIEW';switchView('kb')"><span class="risk-icon risk-warning">${icon('file-warning', 17)}</span><p><strong>${quality}</strong><small>Chunk 质量问题</small></p></button></div>`;
}

async function loadUsers() {
  document.getElementById('users-list').innerHTML = `<tr><td colspan="5">${loadingState('正在加载用户')}</td></tr>`;
  try { state.users = (await responseData(await apiFetch(`${API}/users`), '用户列表获取失败')).users || []; renderUsers(); }
  catch (error) { document.getElementById('users-list').innerHTML = `<tr><td colspan="5">${errorState(error.message, 'loadUsers()')}</td></tr>`; refreshIcons(); }
}
function renderUsers() {
  const query = document.getElementById('user-search')?.value.trim().toLowerCase() || '';
  const users = state.users.filter(user => !query || user.username.toLowerCase().includes(query));
  const admins = state.users.filter(user => user.is_admin).length;
  document.getElementById('user-stats').innerHTML = [
    statCard('用户总数', state.users.length, 'users-round', '全部账号'),
    statCard('管理员', admins, 'shield-check', '具备管理权限', 'success'),
    statCard('普通用户', state.users.length - admins, 'user-round', '使用智能问答', 'info'),
    statCard('角色类型', new Set(state.users.map(user => Boolean(user.is_admin))).size, 'key-round', 'Admin / User', 'neutral')
  ].join('');
  document.getElementById('users-list').innerHTML = users.length ? users.map(user => `<tr><td><div class="table-identity"><span class="avatar avatar-sm">${escapeHtml(user.username.slice(0, 1).toUpperCase())}</span><span><strong>${escapeHtml(user.username)}</strong><small>ID ${Number(user.id)}</small></span></div></td><td>${statusBadge(user.is_admin ? 'Admin' : 'User', user.is_admin ? 'info' : 'neutral', false)}</td><td>${statusBadge('正常', 'success')}</td><td>${formatTime(user.created_at)}</td><td><span class="muted">接口未记录</span></td></tr>`).join('') : `<tr><td colspan="5">${emptyState('没有匹配用户', '请调整搜索关键词。', 'search-x')}</td></tr>`;
  document.getElementById('users-pagination').innerHTML = `<span>共 ${users.length} 个用户${query ? `（全部 ${state.users.length} 个）` : ''}</span><span>当前接口未提供分页和最近登录字段</span>`;
  refreshIcons();
}
function openUserModal() { const modal = document.getElementById('user-modal'); modal.classList.remove('hidden'); modal.setAttribute('aria-hidden', 'false'); document.getElementById('new-user-name').focus(); }
function closeUserModal() { const modal = document.getElementById('user-modal'); modal.classList.add('hidden'); modal.setAttribute('aria-hidden', 'true'); document.getElementById('new-user-name').value = ''; document.getElementById('new-user-pass').value = ''; }
async function createUser() {
  const username = document.getElementById('new-user-name').value.trim(); const password = document.getElementById('new-user-pass').value;
  if (username.length < 2 || password.length < 3) return showToast('用户名至少 2 个字符，密码至少 3 个字符', 'warning');
  const button = document.getElementById('create-user-btn'); setButtonLoading(button, true, '创建中');
  try {
    await responseData(await fetch(`${API}/auth/register`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) }), '创建用户失败');
    closeUserModal(); showToast(`用户 ${username} 已创建`); await loadUsers();
  } catch (error) { showToast(error.message, 'error'); }
  finally { setButtonLoading(button, false); }
}

let kbPollTimer = null;
async function uploadToKB(input) {
  const files = Array.from(input.files || []); input.value = ''; if (!files.length) return;
  showToast(`已开始处理 ${files.length} 份文档`, 'info', '正在上传');
  for (const file of files) {
    const form = new FormData(); form.append('file', file); form.append('title', file.name);
    try {
      const data = await responseData(await apiFetch(`${API}/documents/upload`, { method: 'POST', body: form }), `${file.name} 上传失败`);
      showToast(data.message || `${file.name} 已进入待审核`, data.inspection_status === 'FAILED' ? 'warning' : 'success', data.inspection_status === 'FAILED' ? '上传完成，巡检待重试' : '上传成功');
      state.kbTab = 'PENDING_REVIEW';
    } catch (error) { showToast(error.message, 'error', file.name); }
  }
  await loadKBList(); startKBPolling();
}
function startKBPolling() {
  clearTimeout(kbPollTimer);
  if (!state.kbDocs.some(doc => doc.status === 'PENDING_REVIEW' && ['PENDING', 'RUNNING'].includes(doc.inspection_status))) return;
  kbPollTimer = setTimeout(async () => { if (state.currentView === 'kb') await loadKBList(false); startKBPolling(); }, 3000);
}
async function loadKBList(showLoading = true) {
  if (showLoading) document.getElementById('kb-content').innerHTML = loadingState('正在加载知识文档');
  try {
    const data = await responseData(await apiFetch(`${API}/documents?page=1&page_size=1000`), '知识库列表获取失败');
    state.kbDocs = data.documents || []; renderKB(); startKBPolling();
  } catch (error) { document.getElementById('kb-content').innerHTML = errorState(error.message, 'loadKBList()'); refreshIcons(); }
}
function setKBTab(tab) { state.kbTab = tab; renderKB(); }
function renderKB() {
  const docs = state.kbDocs;
  const count = status => docs.filter(doc => doc.status === status).length;
  const pending = docs.filter(doc => doc.status === 'PENDING_REVIEW');
  const conflicts = pending.reduce((sum, doc) => sum + Number(doc.conflict_count || 0), 0);
  document.getElementById('kb-stats').innerHTML = [
    statCard('已发布', count('ACTIVE'), 'book-check', '线上 RAG 可检索', 'success'),
    statCard('待审核', count('PENDING_REVIEW'), 'clock-3', '尚未发布', 'warning'),
    statCard('已驳回', count('REJECTED'), 'file-x-2', '未进入正式索引', 'neutral'),
    statCard('规则冲突', conflicts, 'git-compare-arrows', '待管理员判断', conflicts ? 'error' : 'success')
  ].join('');
  const tabs = [['ACTIVE', '已发布'], ['PENDING_REVIEW', '待审核'], ['REJECTED', '已驳回']];
  document.getElementById('kb-tabs').innerHTML = tabs.map(([value, label]) => `<button class="tab-button ${state.kbTab === value ? 'active' : ''}" onclick="setKBTab('${value}')">${label}<span>${count(value)}</span></button>`).join('');
  const filtered = docs.filter(doc => doc.status === state.kbTab);
  const content = document.getElementById('kb-content');
  if (!filtered.length) {
    const empty = { ACTIVE: ['暂无已发布知识', '审核通过的文档会显示在这里。'], PENDING_REVIEW: ['暂无待审核文档', '所有新知识都已处理完成。'], REJECTED: ['暂无已驳回文档', '被驳回且未发布的文档会显示在这里。'] }[state.kbTab];
    content.innerHTML = emptyState(empty[0], empty[1], 'library-big', state.kbTab !== 'REJECTED' ? `<button class="btn btn-primary btn-sm" onclick="document.getElementById('kb-upload').click()">${icon('upload', 15)}上传知识</button>` : '');
  } else if (state.kbTab === 'PENDING_REVIEW') {
    content.innerHTML = `<div class="pending-grid">${filtered.map(renderPendingDocument).join('')}</div>`;
  } else {
    content.innerHTML = `<article class="panel table-panel"><div class="table-scroll"><table class="data-table"><thead><tr><th>文档名称</th><th>类型</th><th>Chunk 数</th><th>更新时间</th><th>状态</th><th>操作</th></tr></thead><tbody>${filtered.map(doc => `<tr><td><div class="document-cell"><span>${icon('file-text', 16)}</span><div><strong title="${escapeHtml(doc.title || doc.filename || '未命名')}">${escapeHtml(doc.title || doc.filename || '未命名')}</strong><small>${escapeHtml(doc.filename || '')}</small></div></div></td><td><span class="file-type">${escapeHtml(String(doc.file_type || '—').toUpperCase())}</span></td><td>${Number(doc.chunks_count || 0)}</td><td>${formatTime(doc.update_time || doc.create_time)}</td><td>${documentStatusChip(doc.status)}</td><td><div class="row-actions">${doc.inspection_status ? `<button class="btn btn-ghost btn-xs" onclick="openInspectionDrawer(${Number(doc.id)})">查看</button>` : ''}<button class="btn btn-danger-ghost btn-xs" onclick="deleteKB(${Number(doc.id)})">删除</button></div></td></tr>`).join('')}</tbody></table></div></article>`;
  }
  refreshIcons();
}
function renderPendingDocument(doc) {
  const busy = Boolean(state.kbBusy[doc.id]);
  const inspecting = ['PENDING', 'RUNNING'].includes(doc.inspection_status);
  const completed = doc.inspection_status === 'COMPLETED'; const failed = doc.inspection_status === 'FAILED';
  const status = failed ? statusBadge('巡检失败', 'error') : inspecting ? statusBadge('巡检中', 'info') : completed ? statusBadge('待管理员审核', 'warning') : statusBadge('等待巡检', 'warning');
  return `<article class="pending-card"><div class="pending-card-head"><span class="document-icon">${icon('file-text', 18)}</span><div class="pending-title"><h3 title="${escapeHtml(doc.title || doc.filename || '未命名')}">${escapeHtml(doc.title || doc.filename || '未命名')}</h3><p>${formatTime(doc.create_time)} · ${Number(doc.chunks_count || 0)} Chunks</p></div>${status}</div>${failed ? `<div class="inline-alert alert-error">${icon('triangle-alert', 15)}<span>${escapeHtml(doc.inspection_error || '巡检执行失败，文档仍保持待审核')}</span></div>` : ''}<div class="risk-grid"><div><span class="risk-icon risk-warning">${icon('file-warning', 16)}</span><p><strong>${Number(doc.low_quality_count || 0)}</strong><small>质量问题</small></p></div><div><span class="risk-icon risk-info">${icon('copy-check', 16)}</span><p><strong>${Number(doc.duplicate_count || 0)}</strong><small>语义重复</small></p></div><div><span class="risk-icon risk-error">${icon('git-compare-arrows', 16)}</span><p><strong>${Number(doc.conflict_count || 0)}</strong><small>规则冲突</small></p></div></div><div class="card-actions"><button class="btn btn-ghost btn-sm" onclick="openInspectionDrawer(${Number(doc.id)})">查看报告</button><button class="btn btn-secondary btn-sm" onclick="reinspectKB(${Number(doc.id)})" ${busy || inspecting ? 'disabled' : ''}>${busy ? '<span class="spinner"></span>处理中' : '重新巡检'}</button><span class="push-right"></span><button class="btn btn-danger-ghost btn-sm" onclick="openRejectModal(${Number(doc.id)})" ${busy ? 'disabled' : ''}>驳回</button><button class="btn btn-primary btn-sm" onclick="approveKB(${Number(doc.id)})" ${busy || !completed ? 'disabled' : ''}>通过并发布</button></div></article>`;
}
function documentStatusChip(status) {
  if (status === 'ACTIVE') return statusBadge('已发布', 'success');
  if (status === 'REJECTED') return statusBadge('已驳回', 'neutral');
  return statusBadge('待审核', 'warning');
}
function formatTime(value) { if (!value) return '—'; const date = new Date(value); return Number.isNaN(date.getTime()) ? escapeHtml(String(value)) : date.toLocaleString('zh-CN', { hour12: false }); }

async function openInspectionDrawer(id) {
  state.drawerDocId = id;
  const drawer = document.getElementById('inspection-drawer'); drawer.classList.remove('hidden'); drawer.setAttribute('aria-hidden', 'false');
  document.getElementById('inspection-drawer-content').innerHTML = loadingState('正在加载巡检报告');
  document.getElementById('inspection-drawer-actions').innerHTML = '';
  try { const data = await responseData(await apiFetch(`${API}/documents/${id}/inspection`), '巡检报告加载失败'); renderInspectionDrawer(data.document, data.inspection); }
  catch (error) { document.getElementById('inspection-drawer-content').innerHTML = errorState(error.message, `openInspectionDrawer(${id})`); }
  refreshIcons();
}
function closeInspectionDrawer() { const drawer = document.getElementById('inspection-drawer'); drawer.classList.add('hidden'); drawer.setAttribute('aria-hidden', 'true'); state.drawerDocId = null; }
function inspectionLabel(status) { return ({ PENDING: '等待巡检', RUNNING: '巡检中', COMPLETED: '巡检完成', FAILED: '巡检失败' })[status] || '尚未巡检'; }

function ruleList(report, type) {
  return report?.[type === 'conflict' ? 'conflict_rules' : 'duplicate_rules'] || report?.[type === 'conflict' ? 'conflicts' : 'duplicates'] || [];
}
function renderInspectionDrawer(doc, inspection) {
  const report = inspection?.report || null; const summary = report?.summary || {};
  const conflicts = ruleList(report, 'conflict'); const duplicates = ruleList(report, 'duplicate'); const quality = report?.low_quality || [];
  document.getElementById('inspection-drawer-subtitle').textContent = `${doc.title || doc.filename || '未命名'} · ${Number(doc.chunks_count || 0)} Chunks`;
  let html = `<div class="inspection-meta"><div><small>巡检状态</small>${inspection?.inspection_status === 'COMPLETED' ? statusBadge('巡检完成', 'success') : inspection?.inspection_status === 'FAILED' ? statusBadge('巡检失败', 'error') : statusBadge(inspectionLabel(inspection?.inspection_status), 'info')}</div><div><small>巡检时间</small><strong>${formatTime(inspection?.inspection_time)}</strong></div></div>`;
  if (inspection?.inspection_status === 'FAILED') html += `<div class="inline-alert alert-error">${icon('triangle-alert', 16)}<div><strong>巡检未完整完成</strong><p>${escapeHtml(inspection.error_message || '请重新执行巡检。文档仍保持待审核，不会进入线上知识库。')}</p></div></div>`;
  if (report) {
    html += `<div class="inspection-summary"><div><strong>${Number(summary.low_quality_count ?? quality.length)}</strong><span>质量问题</span></div><div><strong>${Number(summary.duplicate_count ?? duplicates.length)}</strong><span>语义重复</span></div><div><strong>${Number(summary.conflict_count ?? conflicts.length)}</strong><span>规则冲突</span></div></div><div class="review-reminder">${icon('shield-check', 16)}<span>巡检只辅助发现风险，是否发布仍由管理员决定。</span></div>`;
    html += renderInspectionSection('规则冲突', conflicts, 'conflict');
    html += renderInspectionSection('语义重复', duplicates, 'duplicate');
    html += renderQualitySection(quality);
    if (!conflicts.length && !duplicates.length && !quality.length) html += `<div class="inspection-clear"><span>${icon('circle-check-big', 22)}</span><h3>未发现明显风险</h3><p>本次巡检未识别到质量问题、语义重复或规则冲突，仍需管理员确认后发布。</p></div>`;
  } else if (inspection?.inspection_status !== 'FAILED') {
    html += emptyState('暂无巡检结果', '文档可能仍在巡检中，请稍后刷新。', 'scan-search');
  }
  document.getElementById('inspection-drawer-content').innerHTML = html;
  document.getElementById('inspection-drawer-actions').innerHTML = doc.status === 'PENDING_REVIEW' ? `<button class="btn btn-danger-ghost" onclick="openRejectModal(${Number(doc.id)})">驳回</button><button class="btn btn-secondary" onclick="reinspectKB(${Number(doc.id)})">重新巡检</button><button class="btn btn-primary" onclick="approveKB(${Number(doc.id)})" ${inspection?.inspection_status === 'COMPLETED' ? '' : 'disabled'}>通过并发布</button>` : '';
  refreshIcons();
}
function renderInspectionSection(title, items, type) {
  if (!items.length) return '';
  return `<section class="inspection-section"><div class="section-title"><h3>${escapeHtml(title)}</h3><span>${items.length} 项</span></div><div class="issue-list">${items.map((item, index) => {
    const oldSource = item.old_document_name || item.old?.source || '未知来源';
    const newSource = item.new_document_name || item.new?.source || '当前文档';
    const oldRule = item.old_rule || item.old?.rule || item.old?.preview || '未提供旧规则原文';
    const newRule = item.new_rule || item.new?.rule || item.new?.preview || '未提供新规则原文';
    const similarity = Math.round(Number(item.cosine_similarity ?? item.similarity ?? 0) * 100);
    return `<article class="issue-card issue-${type}"><div class="issue-head"><div><small>规则 ${index + 1}</small><h4>${escapeHtml(item.topic || '未命名规则')}</h4></div>${statusBadge(type === 'conflict' ? '潜在规则冲突' : '语义重复', type === 'conflict' ? 'error' : 'info', false)}</div><div class="evidence-block"><header><span>旧知识</span><small>${escapeHtml(oldSource)} · Chunk ${escapeHtml(String(item.old_chunk_id ?? item.old_chunk_index ?? item.old?.chunk_index ?? '—'))}</small></header><p>${escapeHtml(oldRule)}</p></div><div class="evidence-block evidence-new"><header><span>新知识</span><small>${escapeHtml(newSource)} · Chunk ${escapeHtml(String(item.new_chunk_id ?? item.new_chunk_index ?? item.new?.chunk_index ?? '—'))}</small></header><p>${escapeHtml(newRule)}</p></div><div class="issue-reason"><div><span>相似度</span><strong>${similarity}%</strong></div><p><span>判断依据</span>${escapeHtml(item.reason || '未提供判断理由')}</p></div></article>`;
  }).join('')}</div></section>`;
}
function renderQualitySection(items) {
  if (!items.length) return '';
  return `<section class="inspection-section"><div class="section-title"><h3>Chunk 质量问题</h3><span>${items.length} 项</span></div><div class="issue-list">${items.map(item => `<article class="quality-card"><div><span class="risk-icon risk-warning">${icon('file-warning', 16)}</span><p><strong>Chunk ${escapeHtml(String(item.chunk_index ?? '—'))}</strong><small>${escapeHtml(item.issue_type === 'too_short' ? '内容过短' : item.issue_type === 'too_long' ? '内容过长' : item.issue_type || '内容异常')} · ${Number(item.content_length || 0)} 字</small></p></div><blockquote>${escapeHtml(item.preview || '没有可展示的内容预览')}</blockquote></article>`).join('')}</div></section>`;
}
async function reinspectKB(id) {
  if (state.kbBusy[id]) return;
  state.kbBusy[id] = true; renderKB();
  try {
    const data = await responseData(await apiFetch(`${API}/documents/${id}/inspect`, { method: 'POST' }), '巡检执行失败');
    const complete = data.inspection_status === 'COMPLETED';
    showToast(complete ? '巡检已完成，请查看报告' : '巡检未完整完成，文档仍保持待审核', complete ? 'success' : 'warning');
    await loadKBList(false); if (state.drawerDocId === id) await openInspectionDrawer(id);
  } catch (error) { showToast(error.message, 'error', '巡检失败'); await loadKBList(false); }
  finally { state.kbBusy[id] = false; renderKB(); }
}
async function approveKB(id) {
  if (state.kbBusy[id]) return;
  const doc = state.kbDocs.find(item => Number(item.id) === Number(id));
  const hasConflict = Number(doc?.conflict_count || 0) > 0;
  const confirmed = await askConfirm({ title: hasConflict ? '文档存在规则冲突' : '通过并发布文档', message: hasConflict ? `巡检发现 ${doc.conflict_count} 项潜在规则冲突。Agent 不会自动判断哪条规则正确，确认仍然发布到线上知识库吗？` : '发布后，文档将写入正式 FAISS 并可被正常问答检索。', confirmText: '确认发布', danger: hasConflict });
  if (!confirmed) return;
  state.kbBusy[id] = true; renderKB();
  try {
    await responseData(await apiFetch(`${API}/documents/${id}/approve`, { method: 'POST' }), '发布失败');
    closeInspectionDrawer(); state.kbTab = 'ACTIVE'; await loadKBList(false); showToast('文档已发布并写入正式知识索引');
  } catch (error) { showToast(error.message, 'error', '发布失败'); await loadKBList(false); }
  finally { state.kbBusy[id] = false; renderKB(); }
}
function openRejectModal(id) { state.rejectDocId = id; document.getElementById('reject-reason').value = ''; const modal = document.getElementById('reject-modal'); modal.classList.remove('hidden'); modal.setAttribute('aria-hidden', 'false'); refreshIcons(); }
function closeRejectModal() { const modal = document.getElementById('reject-modal'); modal.classList.add('hidden'); modal.setAttribute('aria-hidden', 'true'); state.rejectDocId = null; }
async function confirmReject() {
  const id = state.rejectDocId; if (!id) return;
  const button = document.getElementById('confirm-reject-btn'); setButtonLoading(button, true, '正在驳回');
  try {
    await responseData(await apiFetch(`${API}/documents/${id}/reject`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reason: document.getElementById('reject-reason').value.trim() || null }) }), '驳回失败');
    closeRejectModal(); closeInspectionDrawer(); state.kbTab = 'REJECTED'; await loadKBList(false); showToast('文档已驳回，未进入线上知识库');
  } catch (error) { showToast(error.message, 'error', '驳回失败'); }
  finally { setButtonLoading(button, false); }
}
async function deleteKB(id) {
  if (!await askConfirm({ title: '删除知识文档', message: '该操作会删除文档记录、片段与关联向量，且无法撤销。', confirmText: '删除文档', danger: true })) return;
  try { await responseData(await apiFetch(`${API}/documents/${id}`, { method: 'DELETE' }), '删除失败'); await loadKBList(false); showToast('文档已删除'); }
  catch (error) { showToast(error.message, 'error', '删除失败'); }
}

async function loadAgentRuns() {
  document.getElementById('agent-runs-list').innerHTML = `<tr><td colspan="7">${loadingState('正在加载 Agent 统计')}</td></tr>`;
  try { state.agentRuns = (await responseData(await apiFetch(`${API}/agent/stats`), 'Agent 统计获取失败')).task_stats || []; renderAgentRuns(); }
  catch (error) { document.getElementById('agent-runs-list').innerHTML = `<tr><td colspan="7">${errorState(error.message, 'loadAgentRuns()')}</td></tr>`; refreshIcons(); }
}
function renderAgentRuns() {
  const query = document.getElementById('run-user-filter')?.value.trim().toLowerCase() || '';
  const status = document.getElementById('run-status-filter')?.value || '';
  const tasks = state.agentRuns.filter(task => {
    const matchesText = !query || taskLabel(task.task_type).toLowerCase().includes(query) || String(task.task_type).toLowerCase().includes(query);
    const matchesStatus = !status || (status === 'success' ? Number(task.failed || 0) === 0 : Number(task.failed || 0) > 0);
    return matchesText && matchesStatus;
  });
  document.getElementById('agent-runs-list').innerHTML = tasks.length ? tasks.map(task => `<tr><td><div class="table-identity"><span class="metric-icon">${icon('bot', 15)}</span><span><strong>${escapeHtml(taskLabel(task.task_type))}</strong><small>${escapeHtml(task.task_type)}</small></span></div></td><td>${Number(task.total || 0)}</td><td class="success-text">${Number(task.success || 0)}</td><td class="${Number(task.failed || 0) ? 'error-text' : 'muted'}">${Number(task.failed || 0)}</td><td><div class="rate-cell"><span><i style="width:${Math.max(1, Number(task.success_rate || 0))}%"></i></span><strong>${percent(task.success_rate)}</strong></div></td><td>${Number(task.avg_duration_ms || 0).toFixed(0)} ms</td><td><button class="btn btn-ghost btn-xs" onclick="openAgentRunDetail('${escapeHtml(task.task_type)}')">详情</button></td></tr>`).join('') : `<tr><td colspan="7">${emptyState('暂无匹配统计', '当前后端没有符合筛选条件的任务统计。', 'activity')}</td></tr>`;
  refreshIcons();
}
function resetRunFilter() { document.getElementById('run-user-filter').value = ''; document.getElementById('run-status-filter').value = ''; renderAgentRuns(); }
function openAgentRunDetail(taskType) {
  const task = state.agentRuns.find(item => item.task_type === taskType) || state.dashboard.agent?.task_stats?.find(item => item.task_type === taskType);
  if (!task) return;
  document.getElementById('detail-drawer-eyebrow').textContent = 'AGENT OBSERVABILITY';
  document.getElementById('detail-drawer-title').textContent = taskLabel(task.task_type);
  document.getElementById('detail-drawer-subtitle').textContent = '任务级聚合统计';
  document.getElementById('detail-drawer-content').innerHTML = `<div class="scope-notice">${icon('info', 16)}<span>后端当前只提供任务级聚合数据，未持久化逐次请求、Router 决策和 Tool Call 明细。</span></div><div class="detail-metrics"><div><small>总调用</small><strong>${Number(task.total || 0)}</strong></div><div><small>成功</small><strong class="success-text">${Number(task.success || 0)}</strong></div><div><small>失败</small><strong class="error-text">${Number(task.failed || 0)}</strong></div><div><small>成功率</small><strong>${percent(task.success_rate)}</strong></div><div><small>平均耗时</small><strong>${Number(task.avg_duration_ms || 0).toFixed(0)} ms</strong></div></div><div class="detail-explain"><h3>数据说明</h3><p>这些指标来自当前服务进程内的真实统计。服务重启后统计会重新开始，因此页面不会将它伪装成完整的历史执行记录。</p></div>`;
  const drawer = document.getElementById('detail-drawer'); drawer.classList.remove('hidden'); drawer.setAttribute('aria-hidden', 'false'); refreshIcons();
}
function closeDetailDrawer() { const drawer = document.getElementById('detail-drawer'); drawer.classList.add('hidden'); drawer.setAttribute('aria-hidden', 'true'); }

async function loadReport() {
  document.getElementById('report-overview').innerHTML = Array(4).fill(statCard('加载中', '—', 'loader-circle')).join('');
  document.getElementById('report-agent-stats').innerHTML = loadingState('正在生成实时汇总'); refreshIcons();
  try {
    state.report = await responseData(await apiFetch(`${API}/agent/report`), '报表获取失败');
    const summary = state.report.summary || {}; const tasks = state.report.task_stats || [];
    document.getElementById('report-overview').innerHTML = [
      statCard('文档总数', summary.doc_count || 0, 'files', 'MySQL 记录'),
      statCard('线上 Chunk', summary.chunk_count || 0, 'blocks', '正式 FAISS', 'info'),
      statCard('Agent 运行', summary.agent_runs || 0, 'bot', '当前进程', 'success'),
      statCard('成功率', percent(summary.success_rate), 'circle-check-big', `${summary.failed_count || 0} 次失败`, Number(summary.failed_count) ? 'warning' : 'success')
    ].join('');
    document.getElementById('report-agent-stats').innerHTML = tasks.length ? `<div class="table-scroll"><table class="data-table compact"><thead><tr><th>任务类型</th><th>调用次数</th><th>成功</th><th>失败</th><th>成功率</th></tr></thead><tbody>${tasks.map(task => `<tr><td><strong>${escapeHtml(taskLabel(task.task_type))}</strong><small class="table-subline">${escapeHtml(task.task_type)}</small></td><td>${Number(task.total || 0)}</td><td class="success-text">${Number(task.success || 0)}</td><td>${Number(task.failed || 0)}</td><td><div class="rate-cell"><span><i style="width:${Math.max(1, Number(task.success_rate || 0))}%"></i></span><strong>${percent(task.success_rate)}</strong></div></td></tr>`).join('')}</tbody></table></div>` : emptyState('暂无 Agent 数据', '完成一次问答后，可在这里查看运行汇总。', 'chart-no-axes-combined');
    document.getElementById('report-generated-at').textContent = `生成时间：${formatTime(new Date().toISOString())}`;
    refreshIcons();
  } catch (error) { document.getElementById('report-overview').innerHTML = errorState(error.message, 'loadReport()'); document.getElementById('report-agent-stats').innerHTML = errorState(error.message, 'loadReport()'); refreshIcons(); }
}
function exportReport() {
  if (!state.report) return showToast('请先加载报表数据', 'warning');
  const rows = (state.report.task_stats || []).map(task => [taskLabel(task.task_type), task.total, task.success, task.failed, `${task.success_rate}%`]);
  const escapeCsv = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const csv = '\ufeff任务类型,次数,成功,失败,成功率\n' + rows.map(row => row.map(escapeCsv).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' }); const link = document.createElement('a');
  link.href = URL.createObjectURL(blob); link.download = `KnowledgeOS_运营报表_${new Date().toISOString().slice(0, 10)}.csv`; link.click(); URL.revokeObjectURL(link.href);
  showToast('报表已导出');
}

function escapeHtml(value) { const element = document.createElement('div'); element.textContent = value == null ? '' : String(value); return element.innerHTML; }
function jsArg(value) {
  return JSON.stringify(String(value))
    .replaceAll('&', '\\u0026')
    .replaceAll('<', '\\u003c')
    .replaceAll('>', '\\u003e')
    .replaceAll('"', '&quot;');
}

(function init() {
  const savedTheme = localStorage.getItem('agentcraft_theme');
  if (savedTheme === 'dark') { state.theme = 'dark'; document.documentElement.classList.add('dark'); }
  const savedSidebar = localStorage.getItem('knowledgeos_sidebar_collapsed_v2');
  const viewportWidth = window.innerWidth;
  const defaultCollapsed = viewportWidth > 900 && viewportWidth < 1200;
  applySidebarState(viewportWidth <= 900 ? false : (savedSidebar == null ? defaultCollapsed : savedSidebar === '1'));
  const themeButton = document.getElementById('theme-btn');
  themeButton.innerHTML = `${icon(state.theme === 'light' ? 'moon' : 'sun', 17)}<span>${state.theme === 'light' ? '深色模式' : '浅色模式'}</span>`;
  const textarea = document.getElementById('chat-input');
  textarea.addEventListener('input', () => { textarea.style.height = 'auto'; textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`; });
  textarea.addEventListener('paste', event => {
    const images = Array.from(event.clipboardData?.items || []).filter(item => item.kind === 'file' && item.type.startsWith('image/')).map((item, index) => {
      const file = item.getAsFile(); if (!file) return null;
      const extension = (file.type.split('/')[1] || 'png').replace('jpeg', 'jpg');
      return new File([file], `clipboard-${Date.now()}-${index + 1}.${extension}`, { type: file.type });
    }).filter(Boolean);
    if (images.length) { event.preventDefault(); uploadFiles(images); }
  });
  document.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'n' && state.user) { event.preventDefault(); newChat(); }
    if (event.key === 'Escape') { closeInspectionDrawer(); closeDetailDrawer(); closeRejectModal(); closeUserModal(); closeConfirmModal(); closeMobileSidebar(); }
  });
  refreshIcons();
})();

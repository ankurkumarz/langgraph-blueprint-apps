/* ══════════════════════════════════════════════════════════════════════════════
   SRE & Ops L1 Agent — Application logic
   ══════════════════════════════════════════════════════════════════════════════ */

// ── DOM references ───────────────────────────────────────────────────────────
const chatScroll     = document.getElementById('chat-scroll');
const chatInner      = document.getElementById('chat-inner');
const activityScroll = document.getElementById('activity-scroll');
const activityPane   = document.getElementById('activity-pane');
const queryEl        = document.getElementById('query');
const sendBtn        = document.getElementById('send-btn');
const liveDot        = document.getElementById('live-dot');
const actCount       = document.getElementById('act-count');
const actToggle      = document.getElementById('activity-toggle');
const actBadge       = document.getElementById('activity-badge');
const statQueries    = document.getElementById('stat-queries');
const statTools      = document.getElementById('stat-tools');
const statAvg        = document.getElementById('stat-avg');
let   placeholder    = document.getElementById('placeholder');

// new chat / history / search
const newChatBtn     = document.getElementById('new-chat-btn');
const historyBtn     = document.getElementById('history-btn');
const historyList    = document.getElementById('history-list');
const searchInput    = document.getElementById('search-input');
const searchResults  = document.getElementById('search-results');

// ── Debug mode ───────────────────────────────────────────────────────────────
const debugToggleWrap = document.getElementById('debug-toggle-wrap');
const debugToggleBtn  = document.getElementById('debug-toggle');
const debugStateEl    = document.getElementById('debug-state');
let   debugAvailable  = false;   // server has DEBUG_MODE=true
let   debugEnabled    = false;   // user toggled ON in the UI

// ── State ────────────────────────────────────────────────────────────────────
let totalQueries = 0;
let totalTools   = 0;
let totalTime    = 0;
let rowIndex     = 0;
let activeToolSteps = {};
let debugEventCount  = 0;
let _activeStream    = null;   // AbortController for the in-flight stream

// ── Dummy chat history ───────────────────────────────────────────────────────
const DUMMY_CHAT_HISTORY = [
  { id: 'chat-001', title: 'Kubernetes security hardening',   preview: 'How to resolve a security vulnerability in a Kubernetes cluster?', timestamp: new Date(Date.now() - 1*3600e3).toISOString(), messageCount: 4 },
  { id: 'chat-002', title: 'Debugging CrashLoopBackOff',      preview: 'My pod keeps crashing with CrashLoopBackOff — how do I debug it?', timestamp: new Date(Date.now() - 5*3600e3).toISOString(), messageCount: 6 },
  { id: 'chat-003', title: 'GPU scheduling issues',           preview: 'Pods requesting nvidia.com/gpu are stuck Pending…',               timestamp: new Date(Date.now() - 24*3600e3).toISOString(), messageCount: 8 },
  { id: 'chat-004', title: 'LangGraph streaming setup',       preview: 'How do I set up SSE streaming with LangGraph and FastAPI?',        timestamp: new Date(Date.now() - 48*3600e3).toISOString(), messageCount: 3 },
  { id: 'chat-005', title: 'Helm chart best practices',       preview: 'What are the best practices for structuring Helm charts?',         timestamp: new Date(Date.now() - 72*3600e3).toISOString(), messageCount: 5 },
  { id: 'chat-006', title: 'Istio service mesh debugging',    preview: 'Sidecar injection is failing silently in my namespace.',          timestamp: new Date(Date.now() - 120*3600e3).toISOString(), messageCount: 7 },
];

// ══════════════════════════════════════════════════════════════════════════════
//  Utilities
// ══════════════════════════════════════════════════════════════════════════════

function esc(val) {
  const str = Array.isArray(val)
    ? val.map(p => typeof p === 'object' ? (p.text ?? JSON.stringify(p)) : String(p)).join('')
    : String(val ?? '');
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function coerce(val) {
  return Array.isArray(val)
    ? val.map(p => typeof p === 'object' ? (p.text ?? JSON.stringify(p)) : String(p)).join('')
    : String(val ?? '');
}

function nowStr() {
  return new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' });
}

function scrollDown() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

// ── Streamable-HTTP POST helper ───────────────────────────────────────────────
// Stateless transport: no session IDs, no auto-reconnect on drop.
// Safe for multi-pod deployments — no shared stream state required.
// Returns a Promise that resolves when the stream closes or the signal fires.
function streamPost(url, body, { onEvent, signal }) {
  return (async () => {
    let res;
    try {
      res = await fetch(url, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
        body:    JSON.stringify(body),
        signal,
      });
    } catch (err) {
      if (err.name === 'AbortError') return;
      throw err;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const ct = res.headers.get('content-type') ?? '';
    if (!ct.includes('text/event-stream'))
      throw new Error(`Expected text/event-stream, got: ${ct}`);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;
          let evt;
          try { evt = JSON.parse(raw); } catch { continue; }
          onEvent(evt);
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') throw err;
    } finally {
      reader.releaseLock();
    }
  })();
}

function updateStats(elapsed) {
  totalQueries++;
  totalTime += elapsed;
  statQueries.textContent = totalQueries;
  statAvg.textContent = (totalTime / totalQueries).toFixed(1) + 's';
}

function setLive(active) {
  liveDot.classList.toggle('active', active);
}

function timeAgo(isoStr) {
  const diff = Date.now() - new Date(isoStr).getTime();
  const mins  = Math.floor(diff / 60000);
  if (mins < 1)    return 'just now';
  if (mins < 60)   return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)    return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

// ══════════════════════════════════════════════════════════════════════════════
//  Toast notifications
// ══════════════════════════════════════════════════════════════════════════════

let toastContainer = null;
function showToast(message, type = 'info') {
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container';
    document.body.appendChild(toastContainer);
  }
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
}

// ══════════════════════════════════════════════════════════════════════════════
//  Lightweight Markdown → HTML renderer
// ══════════════════════════════════════════════════════════════════════════════

function renderMarkdown(src) {
  // Escape HTML entities FIRST to prevent XSS, then apply markdown formatting.
  // Preserve fenced code blocks by extracting them before escaping.
  const codeBlocks = [];
  let html = src.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const placeholder = `%%CODEBLOCK_${codeBlocks.length}%%`;
    codeBlocks.push(`<pre><code>${esc(code.replace(/\n$/, ''))}</code></pre>`);
    return placeholder;
  });
  // Escape all remaining HTML
  html = esc(html);
  // Restore code blocks
  codeBlocks.forEach((block, i) => {
    html = html.replace(`%%CODEBLOCK_${i}%%`, block);
  });
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm,  '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm,   '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm,    '<h1>$1</h1>');
  html = html.replace(/^---$/gm, '<hr>');
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/(?<!"|>)(https?:\/\/[^\s<)]+)/g,
    '<a href="$1" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/(^[\t ]*[*\-]\s+.+(?:\n[\t ]*[*\-]\s+.+)*)/gm, (block) => {
    const items = block.split('\n').map(l =>
      `<li>${l.replace(/^[\t ]*[*\-]\s+/, '')}</li>`).join('');
    return `<ul>${items}</ul>`;
  });
  html = html.replace(/(^\d+\.\s+.+(?:\n\d+\.\s+.+)*)/gm, (block) => {
    const items = block.split('\n').map(l =>
      `<li>${l.replace(/^\d+\.\s+/, '')}</li>`).join('');
    return `<ol>${items}</ol>`;
  });
  html = html.replace(/\n{2,}/g, '\n</p><p>\n');
  html = html.replace(/(?<!<\/(li|ul|ol|pre|h[1-4]|blockquote|hr|p)>)\n(?!<)/g, '<br>\n');
  if (!/^\s*<(h[1-4]|ul|ol|pre|blockquote|hr|p|div|table)/i.test(html)) {
    html = '<p>' + html + '</p>';
  }
  html = html.replace(/<p>\s*<\/p>/g, '');
  return html;
}

// ══════════════════════════════════════════════════════════════════════════════
//  Debug Mode — Helpers
// ══════════════════════════════════════════════════════════════════════════════

/** Initialise debug mode — no server dependency.
 *  The toggle is always available; the user controls it purely client-side.
 *  If the backend sends events with a `debug` field, they will be rendered
 *  when the user has toggled debug ON.  No /api/debug call needed.
 */
function initDebugMode() {
  debugAvailable = true;
  debugToggleWrap.style.display = '';
  // Restore preference from localStorage
  const saved = localStorage.getItem('debug_mode');
  if (saved === 'true') toggleDebug(true);
}

function toggleDebug(force) {
  debugEnabled = typeof force === 'boolean' ? force : !debugEnabled;
  debugToggleBtn.classList.toggle('active', debugEnabled);
  debugStateEl.textContent = debugEnabled ? 'ON' : 'OFF';
  localStorage.setItem('debug_mode', debugEnabled);
}

/** Syntax-highlight a JSON value for the debug panel. */
function debugSyntaxHighlight(json) {
  const str = typeof json === 'string' ? json : JSON.stringify(json, null, 2);
  return str
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"([^"]+)"(?=\s*:)/g, '<span class="debug-key">"$1"</span>')
    .replace(/:\s*"([^"]*)"(?=[,\n\r\]}])/g, ': <span class="debug-str">"$1"</span>')
    .replace(/:\s*(\d+\.?\d*)(?=[,\n\r\]}])/g, ': <span class="debug-num">$1</span>')
    .replace(/:\s*(true|false)(?=[,\n\r\]}])/g, ': <span class="debug-bool">$1</span>')
    .replace(/:\s*(null)(?=[,\n\r\]}])/g, ': <span class="debug-null">$1</span>');
}

/** Render a debug block into the given container. */
function addDebugBlock(container, evt) {
  if (!debugEnabled || !debugAvailable) return;
  debugEventCount++;

  const typeMap = {
    node_response:      'node-response',
    tool_call:          'tool-call',
    tool_result:        'tool-result',
    graph_state_update: 'graph-state',
  };
  const badgeClass = typeMap[evt.type] || 'sse-event';

  const block = document.createElement('div');
  block.className = 'debug-block';

  const nodeLabel = evt.node ? ` → ${evt.node}` : (evt.agent ? ` → ${evt.agent}` : '');
  const copyId = `debug-copy-${debugEventCount}`;

  // Build the debug data to display (show full event including debug fields)
  const displayData = { ...evt };

  block.innerHTML =
    `<button class="debug-copy-btn" id="${copyId}" title="Copy JSON">📋 Copy</button>` +
    `<div class="debug-header">` +
      `<span class="debug-type-badge ${badgeClass}">${esc(evt.type)}</span>` +
      (nodeLabel ? `<span class="debug-node-label">${esc(nodeLabel)}</span>` : '') +
    `</div>` +
    `<pre>${debugSyntaxHighlight(displayData)}</pre>`;

  container.appendChild(block);

  // Wire copy button
  block.querySelector(`#${copyId}`).addEventListener('click', (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(JSON.stringify(displayData, null, 2))
      .then(() => { e.target.textContent = '✓ Copied'; setTimeout(() => e.target.textContent = '📋 Copy', 1500); })
      .catch(() => {});
  });

  scrollDown();
}

// ══════════════════════════════════════════════════════════════════════════════
//  Activity log (side panel)
// ══════════════════════════════════════════════════════════════════════════════

let _actPopover = null;   // singleton popover element
let _actPopoverHideTimer = null;

function _ensureActPopover() {
  if (_actPopover) return _actPopover;
  _actPopover = document.createElement('div');
  _actPopover.className = 'act-popover';
  _actPopover.addEventListener('mouseenter', () => clearTimeout(_actPopoverHideTimer));
  _actPopover.addEventListener('mouseleave', () => { _actPopover.classList.remove('visible'); });
  document.body.appendChild(_actPopover);
  return _actPopover;
}

function _showActPopover(row, evt) {
  if (!evt || typeof evt !== 'object') return;
  clearTimeout(_actPopoverHideTimer);
  const pop = _ensureActPopover();

  // Build structured detail sections
  let html = '';

  // Header: event type badge
  const typeLabel = (evt.type || 'event').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  const typeMap = {
    llm_start: 'pop-badge-llm', tool_call: 'pop-badge-tool', tool_result: 'pop-badge-result',
    agent_start: 'pop-badge-agent', agent_end: 'pop-badge-agent', done: 'pop-badge-done',
    error: 'pop-badge-err', node_response: 'pop-badge-llm', plan: 'pop-badge-plan',
  };
  const badgeCls = typeMap[evt.type] || 'pop-badge-default';
  html += `<div class="act-pop-header"><span class="act-pop-badge ${badgeCls}">${esc(typeLabel)}</span>`;
  if (evt.agent && evt.agent !== 'root') html += `<span class="act-pop-agent">${esc(agentLabel(evt.agent))}</span>`;
  if (evt.server) html += `<span class="act-pop-server">${esc(evt.server)}</span>`;
  html += `</div>`;

  // Key-value fields (skip type and debug — we show debug separately)
  const skipKeys = new Set(['type', 'debug']);
  const entries = Object.entries(evt).filter(([k]) => !skipKeys.has(k));
  if (entries.length) {
    html += `<div class="act-pop-fields">`;
    for (const [key, val] of entries) {
      const label = key.replace(/_/g, ' ');
      let display;
      if (typeof val === 'string') {
        display = val.length > 300 ? val.slice(0, 297) + '…' : val;
      } else if (val != null) {
        const s = JSON.stringify(val, null, 2);
        display = s.length > 300 ? s.slice(0, 297) + '…' : s;
      } else continue;
      html += `<div class="act-pop-field"><span class="act-pop-label">${esc(label)}</span><span class="act-pop-value">${esc(display)}</span></div>`;
    }
    html += `</div>`;
  }

  // Debug payload (when available)
  if (evt.debug && typeof evt.debug === 'object') {
    const debugJson = JSON.stringify(evt.debug, null, 2);
    html += `<div class="act-pop-debug-section">`;
    html += `<div class="act-pop-debug-title">🐛 Debug</div>`;
    html += `<pre class="act-pop-debug-pre">${esc(debugJson.length > 800 ? debugJson.slice(0, 797) + '…' : debugJson)}</pre>`;
    html += `</div>`;
  }

  if (!html || entries.length === 0) {
    html = `<div class="act-pop-empty">No additional details</div>`;
  }

  pop.innerHTML = html;

  // Position: to the left of the activity pane
  const rowRect = row.getBoundingClientRect();
  const paneRect = activityPane.getBoundingClientRect();
  pop.style.top  = `${Math.max(8, Math.min(rowRect.top, window.innerHeight - 320))}px`;
  pop.style.left = `${paneRect.left - 340}px`;
  pop.classList.add('visible');
}

function addActivity({ iconClass, icon, title, sub, evt }) {
  rowIndex++;
  const row = document.createElement('div');
  row.className = 'act-row';

  // Build optional MCP server badge
  let serverBadge = '';
  if (evt && evt.server) {
    const srvLabel = evt.server.replace(/[_-]/g, ' ');
    serverBadge = `<span class="act-server-badge">🔌 ${esc(srvLabel)}</span>`;
  }
  // Build optional agent badge
  let agentBadge = '';
  if (evt && evt.agent && evt.agent !== 'root') {
    agentBadge = `<span class="act-agent-badge">${esc(agentLabel(evt.agent))}</span>`;
  }
  const badges = (agentBadge || serverBadge)
    ? `<div class="act-badges">${agentBadge}${serverBadge}</div>` : '';

  row.innerHTML =
    `<div class="act-icon-sm ${iconClass}">${icon}</div>` +
    `<div class="act-body-sm">` +
      `<div class="act-title-sm">${title}${badges}</div>` +
      (sub ? `<div class="act-sub-sm">${sub}</div>` : '') +
    `</div>` +
    `<div class="act-time-sm">${nowStr()}</div>`;

  // Hover-to-inspect: show popover with full event details
  if (evt && typeof evt === 'object') {
    row.classList.add('act-row-has-detail');
    row.addEventListener('mouseenter', () => _showActPopover(row, evt));
    row.addEventListener('mouseleave', () => {
      _actPopoverHideTimer = setTimeout(() => {
        if (_actPopover) _actPopover.classList.remove('visible');
      }, 200);
    });
  }

  activityScroll.appendChild(row);
  activityScroll.scrollTop = activityScroll.scrollHeight;
  actCount.textContent = `(${rowIndex})`;
  actBadge.textContent = rowIndex;
  actBadge.classList.add('visible');
}

// ══════════════════════════════════════════════════════════════════════════════
//  Inline thinking steps
// ══════════════════════════════════════════════════════════════════════════════

/** Convert raw tool args object into a friendly, human-readable string. */
function friendlyArgs(name, args) {
  if (!args || typeof args !== 'object') return '';
  const keys = Object.keys(args);
  if (keys.length === 0) return '';
  const nl = (name || '').toLowerCase();

  // ── LangChain docs patterns (check before generic search/scrape) ────
  if (nl.includes('langchain') || nl.includes('docs')) {
    if (args.query || args.q)
      return `Searching docs for "${args.query || args.q}"`;
  }

  // ── Firecrawl / web-scraping patterns ────────────────────────────────
  if (nl.includes('firecrawl') || nl.includes('scrape') || nl.includes('crawl') ||
      (nl.includes('search') && !nl.includes('docs'))) {
    if (args.query || args.q)
      return `Searching the web for "${args.query || args.q}"${args.limit ? ` (top ${args.limit})` : ''}`;
    if (args.url) {
      try {
        const host = new URL(args.url).hostname.replace('www.', '');
        return `Scraping ${host}` + (args.formats ? ` as ${[].concat(args.formats).join(', ')}` : '');
      } catch {
        return `Scraping: ${args.url}`;
      }
    }
  }

  // ── Delegate / think_tool (orchestrator tools) ──────────────────────
  if (nl === 'delegate' && args.agent_name) {
    const agentDisplay = agentLabel(args.agent_name) || args.agent_name;
    const q = args.query || args.task || '';
    return q ? `Asking ${agentDisplay}: "${q.length > 80 ? q.slice(0, 77) + '…' : q}"` : `Delegating to ${agentDisplay}`;
  }
  if (nl === 'think_tool' && args.reflection) {
    const r = String(args.reflection);
    return r.length > 120 ? r.slice(0, 117) + '…' : r;
  }

  // ── Common single-value patterns ─────────────────────────────────────
  if (keys.length === 1) {
    const k = keys[0];
    const v = args[k];
    const kl = k.toLowerCase();
    if (['query', 'search', 'q', 'question', 'input', 'prompt'].includes(kl))
      return `Searching for: "${v}"`;
    if (['url', 'uri', 'endpoint'].includes(kl))
      return `Fetching: ${v}`;
    if (['path', 'file', 'filename', 'filepath'].includes(kl))
      return `Reading: ${v}`;
    if (['command', 'cmd'].includes(kl))
      return `Running: ${v}`;
    if (['topic', 'subject'].includes(kl))
      return `Looking up: "${v}"`;
    if (['reflection'].includes(kl))
      return typeof v === 'string' && v.length > 120 ? v.slice(0, 117) + '…' : String(v);
    return `${k.replace(/_/g, ' ')}: ${typeof v === 'string' ? v : JSON.stringify(v)}`;
  }

  // ── Multiple args — pick the most useful key for a readable summary ──
  const primaryKeys = ['query', 'q', 'url', 'path', 'command', 'topic', 'agent_name'];
  for (const pk of primaryKeys) {
    if (args[pk]) {
      const extra = keys.filter(k => k !== pk && typeof args[k] !== 'boolean').length;
      const val = String(args[pk]);
      const extraNote = extra > 0 ? ` (+${extra} option${extra > 1 ? 's' : ''})` : '';
      if (pk === 'url') {
        try { return `Fetching ${new URL(val).hostname.replace('www.', '')}${extraNote}`; } catch {}
      }
      if (pk === 'agent_name') return `Delegating to ${val}: "${args.query || args.task || ''}"`.replace(/""$/, '').replace(/: ""/, '');
      return `${val.length > 80 ? val.slice(0, 77) + '…' : val}${extraNote}`;
    }
  }

  // Fallback — build a readable list of key: value pairs
  const parts = keys.map(k => {
    const v = args[k];
    const label = k.replace(/_/g, ' ');
    const val = typeof v === 'string' ? v : JSON.stringify(v);
    return `${label}: ${val}`;
  });
  return parts.join('  ·  ');
}

/** Summarise raw tool result content into a short friendly string. */
function friendlyResult(content) {
  if (!content) return 'No results';
  const s = String(content).trim();
  if (s.length === 0) return 'No results';

  // ── Detect "Reflection recorded" from think_tool ─────────────────────
  if (s === 'Reflection recorded. Continue with your plan.') return 'Reflection noted';

  // ── Try to detect JSON arrays (lists of docs / results) ──────────────
  if (s.startsWith('[')) {
    try {
      const arr = JSON.parse(s);
      if (Array.isArray(arr)) {
        // Try to extract meaningful labels from items
        if (arr.length > 0 && arr[0].title) {
          const titles = arr.slice(0, 3).map(i => i.title);
          const more = arr.length > 3 ? ` +${arr.length - 3} more` : '';
          return `Found ${arr.length} result${arr.length === 1 ? '' : 's'}: ${titles.join(', ')}${more}`;
        }
        return `Found ${arr.length} result${arr.length === 1 ? '' : 's'}`;
      }
    } catch {}
  }

  // ── Try JSON object with known MCP / Firecrawl response shapes ───────
  if (s.startsWith('{')) {
    try {
      const obj = JSON.parse(s);

      // Firecrawl search: {success, data: {web: [...]}}
      if (obj.data && obj.data.web && Array.isArray(obj.data.web)) {
        const results = obj.data.web;
        const count = results.length;
        if (count === 0) return 'No web results found';
        const titles = results.slice(0, 3).map(r => r.title || r.url || 'untitled');
        const more = count > 3 ? ` +${count - 3} more` : '';
        return `Found ${count} web result${count === 1 ? '' : 's'}: ${titles.join(', ')}${more}`;
      }

      // Firecrawl scrape: {markdown: "...", ...} or {content: "...", ...}
      if (obj.markdown) {
        const md = String(obj.markdown).trim();
        const lines = md.split('\n').filter(l => l.trim());
        const heading = lines.find(l => l.startsWith('#'));
        if (heading) return `Scraped page: ${heading.replace(/^#+\s*/, '').slice(0, 100)}`;
        const wordCount = md.split(/\s+/).length;
        return `Scraped page content (${wordCount.toLocaleString()} words)`;
      }

      // Firecrawl crawl/map results
      if (obj.data && Array.isArray(obj.data)) {
        return `Received ${obj.data.length} page${obj.data.length === 1 ? '' : 's'}`;
      }

      // Generic: results array
      if (obj.results && Array.isArray(obj.results)) {
        const count = obj.results.length;
        return `Found ${count} result${count === 1 ? '' : 's'}`;
      }

      // Generic: success/error status
      if (obj.success === false && obj.error) return `Error: ${String(obj.error).slice(0, 100)}`;
      if (obj.success === true && obj.message) return String(obj.message).slice(0, 120);

      // Generic: answer or content field
      if (obj.answer) return String(obj.answer).slice(0, 120);
      if (obj.content && typeof obj.content === 'string') return String(obj.content).slice(0, 120);

      // Generic: title field
      if (obj.title) return String(obj.title).slice(0, 120);

      // Fallback for JSON: show top-level keys as hint
      const topKeys = Object.keys(obj).slice(0, 4);
      return `Received data (${topKeys.join(', ')}${Object.keys(obj).length > 4 ? ', …' : ''})`;
    } catch {}
  }

  // ── Plain text — truncate nicely ─────────────────────────────────────
  if (s.length > 150) {
    // Try to break at a sentence or word boundary
    const truncated = s.slice(0, 147);
    const lastSpace = truncated.lastIndexOf(' ');
    return (lastSpace > 100 ? truncated.slice(0, lastSpace) : truncated) + '…';
  }
  return s;
}

function addThinkStep(container, { iconClass, icon, label, detail, detailIsCode, subItems, active }) {
  // Auto-collapse any previously expanded step in this container
  container.querySelectorAll('.think-detail.visible').forEach(det => {
    det.classList.remove('visible');
    const prevChevron = det.closest('.think-step')?.querySelector('.think-chevron');
    if (prevChevron) prevChevron.classList.remove('open');
  });
  container.querySelectorAll('.think-sub-list.visible').forEach(sl => {
    sl.classList.remove('visible');
    const prevChevron = sl.closest('.think-step')?.querySelector('.think-chevron');
    if (prevChevron) prevChevron.classList.remove('open');
  });

  const step = document.createElement('div');
  step.className = 'think-step';
  const hasDetail = detail || (subItems && subItems.length > 0);
  const chevronHtml = hasDetail
    ? `<span class="think-chevron${active ? ' open' : ''}" title="Expand">▶</span>` : '';

  let detailHtml = '';
  if (detail) {
    detailHtml = `<div class="think-detail${active ? ' visible' : ''}">${
      detailIsCode ? `<pre>${esc(detail)}</pre>` : esc(detail)
    }</div>`;
  }

  let subListHtml = '';
  if (subItems && subItems.length > 0) {
    subListHtml = `<div class="think-sub-list${active ? ' visible' : ''}">` +
      subItems.map(s =>
        `<div class="think-sub-item">` +
          `<span class="sub-icon">${s.icon || '🔍'}</span>` +
          `<span class="sub-text">${esc(s.text)}</span>` +
          (s.source ? `<span class="sub-source">${esc(s.source)}</span>` : '') +
        `</div>`
      ).join('') + `</div>`;
  }

  step.innerHTML =
    `<div class="think-icon ${iconClass}${active ? ' active' : ''}">${icon}</div>` +
    `<div class="think-body">` +
      `<div class="think-label">${esc(label)}${chevronHtml}</div>` +
      detailHtml + subListHtml +
    `</div>`;

  if (hasDetail) {
    const chevron = step.querySelector('.think-chevron');
    chevron.addEventListener('click', () => {
      chevron.classList.toggle('open');
      const det  = step.querySelector('.think-detail');
      const subs = step.querySelector('.think-sub-list');
      if (det)  det.classList.toggle('visible');
      if (subs) subs.classList.toggle('visible');
    });
  }

  container.appendChild(step);
  scrollDown();
  return step;
}

// ══════════════════════════════════════════════════════════════════════════════
//  SSE event handler — extensible for orchestrators, sub-agents, MCP servers
//
//  Every event may carry optional context fields:
//    agent  — which agent emitted it (default: "root")
//    server — which MCP server the tool belongs to
//    ts     — ISO-8601 timestamp
//
//  The handler uses a registry pattern: unknown event types are gracefully
//  rendered as informational steps so new backend events never break the UI.
// ══════════════════════════════════════════════════════════════════════════════

/** Pretty-print an agent name for display. */
function agentLabel(name) {
  if (!name || name === 'root') return '';
  return name.replace(/[_-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/** Build a display label with optional agent/server context badge. */
function contextBadge(evt) {
  const parts = [];
  const agent = evt.agent && evt.agent !== 'root' ? agentLabel(evt.agent) : '';
  const server = evt.server ? evt.server.replace(/[_-]/g, ' ') : '';
  if (agent)  parts.push(agent);
  if (server) parts.push(server);
  return parts.length ? ` [${parts.join(' › ')}]` : '';
}

/**
 * Resolve the correct thinking-steps container for an event.
 * Sub-agent events get their own nested container so the hierarchy is visible.
 * The `ctx` object tracks per-agent containers and plan state.
 */
function resolveThinkContainer(evt, ctx) {
  const agentId = evt.agent || 'root';
  if (agentId === 'root' || !ctx.agentContainers) return ctx.thinkSteps;
  if (ctx.agentContainers[agentId]) return ctx.agentContainers[agentId];
  // Shouldn't normally reach here — agent_start creates the container
  return ctx.thinkSteps;
}

function handleEvent(evt, thinkSteps, answerBody, answerSection, group, startTime) {
  // Shared context stored on the group element for agent/plan tracking
  if (!group._ctx) {
    group._ctx = {
      thinkSteps,
      agentContainers: {},  // agentId → DOM container
      plans: {},            // agentId → plan DOM element
      activeAgents: new Set(),
    };
  }
  const ctx = group._ctx;
  const target = resolveThinkContainer(evt, ctx);
  const badge = contextBadge(evt);

  switch (evt.type) {

    // ── Core events (current protocol) ─────────────────────────────────────

    case 'llm_start': {
      target.querySelectorAll('.think-icon.active').forEach(el => el.classList.remove('active'));
      addThinkStep(target, { iconClass: 'thinking', icon: '◆', label: `Reasoning…${badge}`, active: true });
      addActivity({ iconClass: 'llm', icon: '◆', title: `Calling LLM…${badge}`, evt });
      if (evt.debug) addDebugBlock(target, evt);
      break;
    }

    case 'tool_call': {
      totalTools++;
      statTools.textContent = totalTools;
      target.querySelectorAll('.think-icon.active').forEach(el => el.classList.remove('active'));
      const prettyName = evt.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      const friendly   = friendlyArgs(evt.name, evt.args);
      // Use the friendly summary as the main label when available
      const toolLabel  = friendly ? `${prettyName}${badge}` : `${prettyName}${badge}`;
      const toolDetail = friendly || null;
      const step = addThinkStep(target, {
        iconClass: 'tool', icon: '⚙', label: toolLabel,
        detail: toolDetail, detailIsCode: false, active: true,
      });
      if (evt.id) activeToolSteps[evt.id] = step;
      addActivity({ iconClass: 'call', icon: '⚙', title: esc(prettyName), sub: esc(friendly || '') + badge, evt });
      if (evt.debug) addDebugBlock(target, evt);
      break;
    }

    case 'tool_result': {
      const content    = coerce(evt.content);
      target.querySelectorAll('.think-icon.active').forEach(el => el.classList.remove('active'));
      const prettyName = (evt.name || 'Tool').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      const summary    = friendlyResult(content);
      // Show truncated raw content as expandable detail for long results
      const rawPreview = content.length > 200 ? content.slice(0, 500) + '\n\n… (truncated)' : null;
      addThinkStep(target, { iconClass: 'result', icon: '✓', label: `${prettyName} — ${summary}${badge}`, detail: rawPreview, detailIsCode: !!rawPreview });
      addActivity({ iconClass: 'result', icon: '✓', title: esc(prettyName), sub: esc(summary) + badge, evt });
      if (evt.debug) addDebugBlock(target, evt);
      break;
    }

    case 'text_chunk': {
      const chunk = coerce(evt.content);
      if (chunk) {
        if (answerSection.style.display === 'none') {
          thinkSteps.querySelectorAll('.think-icon.active').forEach(el => el.classList.remove('active'));
          answerSection.style.display = 'flex';
        }
        if (!answerBody._rawMd) answerBody._rawMd = '';
        answerBody._rawMd += chunk;
        answerBody.innerHTML = renderMarkdown(answerBody._rawMd);
        scrollDown();
      }
      break;
    }

    case 'text': {
      const text = coerce(evt.content);
      if (text) {
        if (answerSection.style.display === 'none') {
          thinkSteps.querySelectorAll('.think-icon.active').forEach(el => el.classList.remove('active'));
          answerSection.style.display = 'flex';
        }
        if (!answerBody._rawMd) answerBody._rawMd = '';
        answerBody._rawMd += text;
        answerBody.innerHTML = renderMarkdown(answerBody._rawMd);
        scrollDown();
      }
      break;
    }

    case 'usage': {
      if (evt.usage) answerBody._tokenUsage = evt.usage;
      break;
    }

    case 'done': {
      thinkSteps.querySelectorAll('.think-icon.active').forEach(el => el.classList.remove('active'));
      answerBody.classList.remove('streaming');
      if (answerBody._rawMd) answerBody.innerHTML = renderMarkdown(answerBody._rawMd);
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

      // ── Lightbox footer ──────────────────────────────────────────────────
      {
        const u = answerBody._tokenUsage || {};
        const totalTok = u.total_tokens || (u.input_tokens || 0) + (u.output_tokens || 0);
        const fmtTokens = totalTok >= 1e6
          ? (totalTok / 1e6).toFixed(2) + 'M'
          : totalTok >= 1e3
            ? (totalTok / 1e3).toFixed(1) + 'k'
            : totalTok ? String(totalTok) : null;

        // Gemini 2.5 Flash default rates ($/1M tokens)
        const inRate  = 0.10 / 1e6;
        const outRate = 0.40 / 1e6;
        const rawCost = (u.input_tokens || 0) * inRate + (u.output_tokens || 0) * outRate;
        const fmtCost = rawCost > 0 ? `$${rawCost.toFixed(4)}` : null;

        const parts = [`${elapsed}s`];
        if (fmtTokens) parts.push(`${fmtTokens} tokens`);
        if (fmtCost)   parts.push(fmtCost);

        const footer = document.createElement('div');
        footer.className = 'steps-footer';
        footer.innerHTML = parts.map(p => `<span>${esc(p)}</span>`).join('<span class="steps-footer-sep">•</span>');
        thinkSteps.appendChild(footer);
      }

      const meta = document.createElement('div');
      meta.className = 'answer-meta';
      meta.innerHTML = `<span class="elapsed">⏱ ${elapsed}s</span>`;
      answerSection.querySelector('.answer-content').appendChild(meta);

      // Debug summary banner
      if (debugEnabled && debugEventCount > 0) {
        const summary = document.createElement('div');
        summary.className = 'debug-summary';
        summary.innerHTML =
          `<span class="debug-summary-icon">🐛</span>` +
          `<span>Debug mode</span>` +
          `<div class="debug-summary-stats">` +
            `<span class="debug-summary-stat">Events: <b>${debugEventCount}</b></span>` +
            `<span class="debug-summary-stat">Tools: <b>${totalTools}</b></span>` +
            `<span class="debug-summary-stat">Time: <b>${elapsed}s</b></span>` +
          `</div>`;
        answerSection.querySelector('.answer-content').appendChild(summary);
      }

      scrollDown();
      addActivity({ iconClass: 'done', icon: '✓', title: `Done — ${elapsed}s`, evt });
      updateStats(parseFloat(elapsed));
      break;
    }

    case 'error': {
      answerSection.style.display = 'flex';
      answerBody.classList.remove('streaming');
      answerBody.textContent = evt.detail;
      answerBody.style.color = 'var(--red)';
      addThinkStep(target, { iconClass: 'error', icon: '✕', label: `Error${badge}`, detail: evt.detail });
      addActivity({ iconClass: 'err', icon: '✕', title: `Error${badge}`, sub: esc(String(evt.detail).slice(0, 70)), evt });
      break;
    }

    // ── Sub-agent lifecycle ─────────────────────────────────────────────────

    case 'agent_start': {
      const agentId   = evt.agent || 'sub-agent';
      const agentName = agentLabel(agentId) || agentId;
      const roleSuffix = evt.role ? ` (${evt.role})` : '';
      ctx.activeAgents.add(agentId);

      // Create a nested thinking-steps container for this sub-agent
      const wrapper = document.createElement('div');
      wrapper.className = 'think-step subagent-group';
      wrapper.innerHTML =
        `<div class="think-icon thinking active">🤖</div>` +
        `<div class="think-body">` +
          `<div class="think-label">${esc(agentName)}${esc(roleSuffix)} started…` +
            `<span class="think-chevron open" title="Collapse">▶</span>` +
          `</div>` +
          `<div class="thinking-steps subagent-steps visible"></div>` +
        `</div>`;

      const nested = wrapper.querySelector('.thinking-steps');
      ctx.agentContainers[agentId] = nested;
      thinkSteps.appendChild(wrapper);

      // Wire collapse toggle
      const chevron = wrapper.querySelector('.think-chevron');
      chevron.addEventListener('click', () => {
        chevron.classList.toggle('open');
        nested.classList.toggle('visible');
      });

      scrollDown();
      addActivity({ iconClass: 'llm', icon: '🤖', title: `Sub-agent: ${esc(agentName)}${esc(roleSuffix)}`, evt });
      break;
    }

    case 'agent_end': {
      const agentId   = evt.agent || 'sub-agent';
      const agentName = agentLabel(agentId) || agentId;
      ctx.activeAgents.delete(agentId);

      // Mark sub-agent container as done
      const container = ctx.agentContainers[agentId];
      if (container) {
        const parentStep = container.closest('.subagent-group');
        if (parentStep) {
          const icon = parentStep.querySelector('.think-icon');
          if (icon) { icon.classList.remove('active', 'thinking'); icon.classList.add('done'); icon.textContent = '✓'; }
        }
        addThinkStep(container, {
          iconClass: 'done', icon: '✓',
          label: evt.summary ? `Done — ${evt.summary}` : `${agentName} completed`,
        });
      }
      addActivity({ iconClass: 'done', icon: '✓', title: `${esc(agentName)} completed`, evt });
      break;
    }

    // ── Orchestrator plan / progress ────────────────────────────────────────

    case 'plan': {
      const agentId = evt.agent || 'root';
      const steps = evt.steps || [];
      if (!steps.length) break;

      const planEl = document.createElement('div');
      planEl.className = 'think-plan';
      planEl.innerHTML =
        `<div class="plan-header">📋 Plan (${steps.length} step${steps.length > 1 ? 's' : ''})</div>` +
        `<div class="plan-steps">` +
        steps.map(s =>
          `<div class="plan-step" data-step-id="${esc(s.id)}" data-status="pending">` +
            `<span class="plan-step-icon">○</span>` +
            `<span class="plan-step-label">${esc(s.label)}</span>` +
          `</div>`
        ).join('') +
        `</div>`;

      target.appendChild(planEl);
      ctx.plans[agentId] = planEl;
      scrollDown();
      addActivity({ iconClass: 'llm', icon: '📋', title: `Plan: ${steps.length} steps`, evt });
      break;
    }

    case 'plan_step': {
      const agentId = evt.agent || 'root';
      const planEl = ctx.plans[agentId];
      if (!planEl || !evt.step_id) break;

      const stepEl = planEl.querySelector(`[data-step-id="${CSS.escape(evt.step_id)}"]`);
      if (!stepEl) break;

      const status = evt.status || 'running';
      stepEl.dataset.status = status;
      const iconEl = stepEl.querySelector('.plan-step-icon');
      const icons = { running: '◆', done: '✓', error: '✕', skipped: '—' };
      if (iconEl) iconEl.textContent = icons[status] || '○';
      break;
    }

    // ── Handoff between agents ──────────────────────────────────────────────

    case 'handoff': {
      const from = agentLabel(evt.from) || evt.from || '?';
      const to   = agentLabel(evt.to) || evt.to || '?';
      const reason = evt.reason ? ` — ${evt.reason}` : '';
      addThinkStep(target, {
        iconClass: 'thinking', icon: '🔀',
        label: `Handoff: ${from} → ${to}${reason}`,
      });
      addActivity({ iconClass: 'llm', icon: '🔀', title: `Handoff → ${esc(to)}`, sub: esc(evt.reason || ''), evt });
      break;
    }

    // ── Human-in-the-loop approval ──────────────────────────────────────────

    case 'approval_request': {
      addThinkStep(target, {
        iconClass: 'tool', icon: '🛡️',
        label: `Approval needed: ${evt.action || 'action'}${badge}`,
        detail: evt.detail || null, detailIsCode: false, active: true,
      });
      addActivity({ iconClass: 'call', icon: '🛡️', title: `Approval: ${esc(evt.action || '')}`, evt });
      break;
    }

    case 'approval_result': {
      const approved = evt.approved;
      addThinkStep(target, {
        iconClass: approved ? 'result' : 'error',
        icon: approved ? '✓' : '✕',
        label: approved ? 'Approved' : 'Rejected',
      });
      addActivity({ iconClass: approved ? 'result' : 'err', icon: approved ? '✓' : '✕', title: approved ? 'Approved' : 'Rejected', evt });
      break;
    }

    // ── Progress / status updates ───────────────────────────────────────────

    case 'status': {
      const pct = typeof evt.progress === 'number' ? ` (${evt.progress}%)` : '';
      addThinkStep(target, {
        iconClass: 'thinking', icon: 'ℹ️',
        label: `${evt.message || 'Working…'}${pct}${badge}`,
      });
      addActivity({ iconClass: 'llm', icon: 'ℹ️', title: esc(evt.message || 'Status update'), evt });
      break;
    }

    // ── MCP server connectivity ─────────────────────────────────────────────

    case 'mcp_server': {
      const srvName = (evt.server || 'unknown').replace(/[_-]/g, ' ');
      const connected = evt.status === 'connected';
      addThinkStep(target, {
        iconClass: connected ? 'result' : 'error',
        icon: connected ? '🔌' : '⚠️',
        label: `MCP: ${srvName} ${evt.status || 'unknown'}`,
      });
      addActivity({ iconClass: connected ? 'result' : 'err', icon: connected ? '🔌' : '⚠️', title: `MCP ${esc(srvName)}: ${esc(evt.status || '')}`, evt });
      break;
    }

    // ── Debug-only events: node_response and graph_state_update ────────────

    case 'node_response': {
      if (evt.debug) addDebugBlock(target, evt);
      // Also add a subtle thinking step so it's visible in the flow
      if (debugEnabled) {
        const nodeName = evt.node || 'unknown';
        addThinkStep(target, {
          iconClass: 'thinking', icon: '🐛',
          label: `Node response: ${nodeName}${badge}`,
        });
      }
      break;
    }

    case 'graph_state_update': {
      if (evt.debug) addDebugBlock(target, evt);
      if (debugEnabled) {
        const nodeName = evt.node || 'unknown';
        addThinkStep(target, {
          iconClass: 'thinking', icon: '🐛',
          label: `Graph state update: ${nodeName}`,
        });
      }
      break;
    }

    // ── Catch-all: gracefully render any unknown future event type ──────────

    default: {
      // Unknown events are shown as informational steps so the UI never breaks
      const typeLabel = (evt.type || 'unknown').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      const detail = evt.detail || evt.message || evt.content || evt.summary || null;
      addThinkStep(target, {
        iconClass: 'thinking', icon: 'ℹ️',
        label: `${typeLabel}${badge}`,
        detail: detail ? String(detail).slice(0, 500) : null,
        detailIsCode: false,
      });
      addActivity({ iconClass: 'llm', icon: 'ℹ️', title: esc(typeLabel), sub: detail ? esc(String(detail).slice(0, 70)) : '', evt });
      break;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Send query
// ══════════════════════════════════════════════════════════════════════════════

async function send() {
  const query = queryEl.value.trim();
  if (!query) return;
  if (_activeStream) { _activeStream.abort(); _activeStream = null; }
  if (placeholder) { placeholder.remove(); placeholder = null; }

  const userMsg = document.createElement('div');
  userMsg.className = 'user-msg';
  userMsg.innerHTML = `<div class="user-avatar">U</div><div class="user-text">${esc(query)}</div>`;
  chatInner.appendChild(userMsg);
  scrollDown();

  queryEl.value = '';
  queryEl.style.height = 'auto';
  sendBtn.disabled = true;
  sendBtn.innerHTML = '<div class="spinner"></div>Thinking…';
  setLive(true);

  const group = document.createElement('div');
  group.className = 'agent-group';
  const thinkSteps = document.createElement('div');
  thinkSteps.className = 'thinking-steps';
  group.appendChild(thinkSteps);

  // Show an immediate "Thinking…" placeholder so the user never sees an empty bar
  const thinkingPlaceholder = document.createElement('div');
  thinkingPlaceholder.className = 'think-step thinking-placeholder';
  thinkingPlaceholder.innerHTML =
    `<div class="think-icon thinking active">◆</div>` +
    `<div class="think-body">` +
      `<div class="think-label">Thinking<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span></div>` +
    `</div>`;
  thinkSteps.appendChild(thinkingPlaceholder);
  scrollDown();

  const answerSection = document.createElement('div');
  answerSection.className = 'answer-section';
  answerSection.innerHTML = `<div class="agent-avatar">🤖</div><div class="answer-content"><div class="answer-body streaming"></div></div>`;
  answerSection.style.display = 'none';
  group.appendChild(answerSection);
  chatInner.appendChild(group);

  const answerBody = answerSection.querySelector('.answer-body');
  const startTime  = Date.now();
  debugEventCount  = 0;

  const ctrl = new AbortController();
  _activeStream = ctrl;

  try {
    await streamPost('/agent/conversation', { query, debug: debugEnabled }, {
      signal:  ctrl.signal,
      onEvent(evt) {
        if (thinkingPlaceholder.parentNode) thinkingPlaceholder.remove();
        handleEvent(evt, thinkSteps, answerBody, answerSection, group, startTime);
      },
    });
  } catch (err) {
    answerSection.style.display = 'flex';
    answerBody.classList.remove('streaming');
    answerBody.textContent = err.message;
    answerBody.style.color = 'var(--red)';
    addThinkStep(thinkSteps, { iconClass: 'error', icon: '✕', label: 'Error — ' + err.message.slice(0, 60) });
    addActivity({ iconClass: 'err', icon: '✕', title: 'Error', sub: err.message.slice(0, 60) });
  } finally {
    _activeStream = null;
    sendBtn.disabled = false;
    sendBtn.innerHTML = 'Send ↑';
    setLive(false);
    queryEl.focus();
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  New Chat
// ══════════════════════════════════════════════════════════════════════════════

const PLACEHOLDER_HTML =
  `<div class="placeholder" id="placeholder">` +
    `<span class="icon">💬</span>` +
    `<p>Ask anything — watch the agent think, search, and reason before answering.</p>` +
    `<div class="sample-cards">` +
      `<div class="sample-card" data-query="How to resolve a security vulnerability in a Kubernetes cluster?"><div class="card-icon security">🔒</div><div class="card-text">How to resolve a Security Issue</div></div>` +
      `<div class="sample-card" data-query="How to debug a LangGraph agent?"><div class="card-icon debug">🐛</div><div class="card-text">How to Debug LangGraph Agent</div></div>` +
      `<div class="sample-card" data-query="How to design a multi-agent production architecture?"><div class="card-icon gpu">🖥️</div><div class="card-text">How to design guardrails for agents</div></div>` +
    `</div>` +
  `</div>`;

function newChat() {
  chatInner.innerHTML = PLACEHOLDER_HTML;
  placeholder = document.getElementById('placeholder');
  bindSampleCards();
  activityScroll.innerHTML = '';
  rowIndex = 0;
  actCount.textContent = '';
  actBadge.textContent = '';
  actBadge.classList.remove('visible');
  totalQueries = 0; totalTools = 0; totalTime = 0;
  statQueries.textContent = '0';
  statTools.textContent = '0';
  statAvg.textContent = '—';
  sendBtn.disabled = false;
  sendBtn.innerHTML = 'Send ↑';
  setLive(false);
  queryEl.value = '';
  queryEl.focus();
}

// ══════════════════════════════════════════════════════════════════════════════
//  Top-bar Tab Navigation
// ══════════════════════════════════════════════════════════════════════════════

function initTabs() {
  const tabs = document.querySelectorAll('.top-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
      const target = document.getElementById('tab-' + tab.dataset.tab);
      if (target) target.classList.add('active');

      // Initialize dashboard data on first visit
      if (tab.dataset.tab === 'insights' && !insightsInitialized) {
        initInsightsDashboard();
        insightsInitialized = true;
      }
      if (tab.dataset.tab === 'monitoring' && !monitoringInitialized) {
        initMonitoringDashboard();
        monitoringInitialized = true;
      }
    });
  });
}

let insightsInitialized = false;
let monitoringInitialized = false;

// ══════════════════════════════════════════════════════════════════════════════
//  ███  AI INSIGHTS (INCIDENTS) DASHBOARD  ███
// ══════════════════════════════════════════════════════════════════════════════

const INCIDENTS = [
  {
    id: 'INC-4821', severity: 'P1', title: 'Payment Gateway 5xx Spike',
    service: 'payment-svc', status: 'investigating', duration: '18m',
    aiRootCause: 'Connection pool exhaustion in payment-svc due to upstream Stripe API latency spike (p99 from 200ms → 4.2s). Thread pool saturated at 200/200 connections.',
    confidence: 94,
    blastRadius: ['checkout-flow', 'order-svc', 'notification-svc', '~12K users/min'],
    timeline: [
      { type: 'trigger', time: '14:23:07', title: 'Stripe API latency spike detected', desc: 'p99 latency increased from 200ms to 4.2s' },
      { type: 'propagation', time: '14:23:42', title: 'Connection pool exhaustion', desc: 'payment-svc thread pool: 200/200 active connections' },
      { type: 'impact', time: '14:24:15', title: '5xx error rate exceeded threshold', desc: 'Error rate jumped from 0.02% to 34.7%' },
      { type: 'fix', time: '—', title: 'Suggested: Enable circuit breaker', desc: 'Set timeout=2s, retry=2, circuit-break after 5 failures' },
    ],
    suggestion: { title: '🛠️ AI Recommended Fix', text: 'Enable circuit breaker pattern with <code>resilience4j</code>. Set connection timeout to 2s with max 2 retries. Add fallback to cached payment token validation. Run: <code>kubectl set env deploy/payment-svc CIRCUIT_BREAKER_ENABLED=true TIMEOUT_MS=2000</code>' },
    similar: [
      { id: 'INC-3901', title: 'Payment timeout cascade', match: '92%', resolution: 'Circuit breaker enabled, connection pool increased to 300', date: '2025-04-12', ttm: '34m' },
      { id: 'INC-2847', title: 'Stripe webhook backpressure', match: '78%', resolution: 'Added async webhook processing queue', date: '2025-02-18', ttm: '52m' },
      { id: 'INC-4102', title: 'Payment DB connection leak', match: '65%', resolution: 'Fixed unclosed connections in retry handler', date: '2025-03-27', ttm: '1h 12m' },
    ],
  },
  {
    id: 'INC-4822', severity: 'P1', title: 'Redis Cluster Failover — Data Loss Risk',
    service: 'redis-cluster', status: 'escalated', duration: '42m',
    aiRootCause: 'Redis primary node redis-prod-03 OOM killed (RSS 31.2GB / limit 32GB). Failover promoted a replica with 4min replication lag. Risk of stale session data.',
    confidence: 97,
    blastRadius: ['auth-svc', 'session-mgr', 'rate-limiter', 'cache-layer', '~45K sessions'],
    timeline: [
      { type: 'trigger', time: '13:41:22', title: 'Redis OOM kill on redis-prod-03', desc: 'RSS: 31.2GB exceeded 32GB limit, kernel killed process' },
      { type: 'propagation', time: '13:41:28', title: 'Sentinel initiated failover', desc: 'Promoted redis-prod-03-replica-1 (4min replication lag)' },
      { type: 'impact', time: '13:42:01', title: 'Auth failures spike', desc: 'Session lookup failures: 2,847 in 60s, users forced to re-login' },
      { type: 'fix', time: '—', title: 'Suggested: Restore + key migration', desc: 'Restore from RDB snapshot (13:40), reconcile missing keys from AOF' },
    ],
    suggestion: { title: '🛠️ AI Recommended Fix', text: 'Immediately restore from latest RDB snapshot at 13:40. Reconcile with AOF log for delta. Increase maxmemory to 48GB: <code>redis-cli CONFIG SET maxmemory 48gb</code>. Enable maxmemory-policy allkeys-lru. Set up memory usage alerting at 80% threshold.' },
    similar: [
      { id: 'INC-3244', title: 'Redis memory spike from session bloat', match: '88%', resolution: 'Added TTL to session keys, enabled LRU eviction', date: '2025-01-15', ttm: '1h 5m' },
      { id: 'INC-4010', title: 'Redis replication lag during backup', match: '71%', resolution: 'Switched to BGSAVE with fork optimization', date: '2025-03-20', ttm: '28m' },
    ],
  },
  {
    id: 'INC-4823', severity: 'P1', title: 'Kubernetes Node NotReady — GPU Pool',
    service: 'k8s-gpu-pool', status: 'mitigating', duration: '55m',
    aiRootCause: 'NVIDIA driver crash on nodes gpu-node-07 and gpu-node-08 after automatic kernel update (5.15.0-91 → 5.15.0-92). Driver version mismatch with CUDA 12.2.',
    confidence: 89,
    blastRadius: ['ml-inference-svc', 'model-serving', 'batch-training', '8 GPU pods evicted'],
    timeline: [
      { type: 'trigger', time: '12:18:00', title: 'Automatic kernel update applied', desc: 'Kernel: 5.15.0-91-generic → 5.15.0-92-generic on gpu-node-07,08' },
      { type: 'propagation', time: '12:18:34', title: 'NVIDIA driver module failed to load', desc: 'nvidia-smi: "NVIDIA-SMI has failed... driver not loaded"' },
      { type: 'impact', time: '12:19:01', title: 'Nodes transitioned to NotReady', desc: '8 GPU pods evicted, ML inference latency p99 → timeout' },
      { type: 'fix', time: '—', title: 'Suggested: Pin kernel + reinstall driver', desc: 'apt-mark hold linux-image-5.15.0-91, reinstall nvidia-driver-535' },
    ],
    suggestion: { title: '🛠️ AI Recommended Fix', text: 'Pin kernel version: <code>apt-mark hold linux-image-5.15.0-91-generic</code>. Reinstall NVIDIA driver 535 compatible with CUDA 12.2. Add preStop hook to cordon GPU nodes before kernel updates. Add kured DaemonSet for controlled reboots.' },
    similar: [
      { id: 'INC-3677', title: 'GPU driver failure post-update', match: '95%', resolution: 'Pinned kernel, added pre-update cordon script', date: '2025-02-28', ttm: '2h 15m' },
    ],
  },
  {
    id: 'INC-4824', severity: 'P2', title: 'API Gateway Latency Degradation',
    service: 'api-gateway', status: 'investigating', duration: '12m',
    aiRootCause: 'Kong rate-limiting plugin hitting PostgreSQL with excessive counter updates. DB connection pool at 85% capacity during traffic peak.',
    confidence: 82,
    blastRadius: ['all-apis', 'mobile-app', 'partner-integrations'],
    timeline: [
      { type: 'trigger', time: '14:35:00', title: 'Traffic spike: 3x normal volume', desc: 'Flash sale triggered 45K req/s vs normal 15K req/s' },
      { type: 'propagation', time: '14:35:22', title: 'Rate limiter DB queries spiked', desc: 'PostgreSQL CPU: 92%, query latency: 340ms avg' },
      { type: 'impact', time: '14:36:00', title: 'API p95 latency exceeded 2s SLO', desc: 'p95: 2.4s (SLO: 500ms), affecting all downstream services' },
      { type: 'fix', time: '—', title: 'Suggested: Switch to Redis rate limiting', desc: 'Move rate limiter counters from PostgreSQL to Redis' },
    ],
    suggestion: { title: '🛠️ AI Recommended Fix', text: 'Switch rate-limiting plugin from <code>postgres</code> to <code>redis</code> strategy. Redis handles 100K+ counter ops/s vs PostgreSQL\'s 5K. Apply: <code>kong config set rate-limiting.policy=redis rate-limiting.redis_host=redis-prod</code>' },
    similar: [
      { id: 'INC-4201', title: 'API gateway timeout during promo', match: '85%', resolution: 'Migrated rate limiter to Redis, added request queuing', date: '2025-04-01', ttm: '45m' },
    ],
  },
  {
    id: 'INC-4825', severity: 'P2', title: 'Elasticsearch Cluster Yellow — Unassigned Shards',
    service: 'elasticsearch', status: 'investigating', duration: '28m',
    aiRootCause: 'Disk watermark exceeded on es-data-04 (92.3% used). Shard relocation blocked. 47 unassigned replica shards across 12 indices.',
    confidence: 91,
    blastRadius: ['log-aggregation', 'search-svc', 'audit-trail'],
    timeline: [
      { type: 'trigger', time: '14:05:33', title: 'Disk watermark high exceeded', desc: 'es-data-04: 92.3% disk used (watermark: 90%)' },
      { type: 'propagation', time: '14:06:01', title: 'Shard allocation blocked', desc: '47 replica shards cannot be assigned to any node' },
      { type: 'impact', time: '14:06:45', title: 'Cluster status: YELLOW', desc: 'Search latency +40%, log ingestion throttled to 60%' },
      { type: 'fix', time: '—', title: 'Suggested: Cleanup + expand storage', desc: 'Delete indices older than 30d, expand PV to 2TB' },
    ],
    suggestion: { title: '🛠️ AI Recommended Fix', text: 'Delete old indices: <code>curator_cli --host es-master delete indices --older-than 30 --time-unit days</code>. Expand PersistentVolume to 2TB. Set ILM rollover at 50GB per shard. Add hot-warm-cold architecture for cost optimization.' },
    similar: [],
  },
  {
    id: 'INC-4826', severity: 'P2', title: 'Certificate Expiry — TLS Handshake Failures',
    service: 'ingress-nginx', status: 'mitigating', duration: '8m',
    aiRootCause: 'Wildcard cert *.api.prod.internal expired 2h ago. cert-manager renewal failed: ACME challenge timeout (DNS-01 propagation delay).',
    confidence: 98,
    blastRadius: ['all-internal-apis', 'service-mesh-mTLS', 'partner-webhook-receivers'],
    timeline: [
      { type: 'trigger', time: '14:00:00', title: 'Certificate expired', desc: '*.api.prod.internal cert expired at 12:00 UTC' },
      { type: 'propagation', time: '14:00:05', title: 'TLS handshake failures', desc: 'All HTTPS connections returning ERR_CERT_DATE_INVALID' },
      { type: 'impact', time: '14:00:12', title: 'Internal API calls failing', desc: '100% failure rate on internal API mesh' },
      { type: 'fix', time: '14:08:00', title: 'Manual cert renewal triggered', desc: 'kubectl cert-manager renew --namespace=istio-system' },
    ],
    suggestion: { title: '🛠️ AI Recommended Fix', text: 'Manual renewal: <code>kubectl cert-manager renew wildcard-api-prod -n istio-system</code>. Root cause: DNS-01 challenge failed due to Route53 propagation delay. Fix: Switch to HTTP-01 challenge or reduce DNS TTL to 60s. Add 30-day expiry alert.' },
    similar: [
      { id: 'INC-2991', title: 'mTLS cert expiry cascade', match: '90%', resolution: 'Implemented 30d pre-expiry alerts, switched to HTTP-01', date: '2025-01-22', ttm: '15m' },
    ],
  },
  {
    id: 'INC-4827', severity: 'P3', title: 'Kafka Consumer Lag — Order Processing',
    service: 'order-processor', status: 'investigating', duration: '35m',
    aiRootCause: 'Consumer group order-processor-v2 experiencing rebalancing storm. 3 consumers crashed due to OOM on processing large order batches (>50MB payloads).',
    confidence: 76,
    blastRadius: ['order-fulfillment', 'inventory-svc', 'shipping-svc'],
    timeline: [
      { type: 'trigger', time: '13:50:00', title: 'Large batch orders submitted', desc: 'Batch API received 3 orders with 50MB+ payload each' },
      { type: 'propagation', time: '13:50:45', title: 'Consumer OOM crashes', desc: '3 of 8 consumers OOM killed, triggering rebalance' },
      { type: 'impact', time: '13:52:00', title: 'Consumer lag growing', desc: 'Lag: 847K messages and growing at ~15K/min' },
      { type: 'fix', time: '—', title: 'Suggested: Increase memory + chunk processing', desc: 'Set -Xmx4g, implement chunked deserialization' },
    ],
    suggestion: { title: '🛠️ AI Recommended Fix', text: 'Increase consumer memory: <code>kubectl set resources deploy/order-processor --limits=memory=4Gi</code>. Implement chunked message processing for payloads >10MB. Add max.poll.records=100 to prevent batch overload.' },
    similar: [],
  },
];

const CORRELATION_CLUSTERS = [
  { name: 'Payment + Checkout cascade', color: '#dc2626', services: 'payment-svc, checkout, order-svc', count: '124 alerts → 1 incident' },
  { name: 'Redis failover ripple', color: '#d97706', services: 'redis, auth-svc, session-mgr', count: '89 alerts → 1 incident' },
  { name: 'GPU node drain effects', color: '#4f46e5', services: 'gpu-pool, ml-inference, model-serving', count: '56 alerts → 1 incident' },
  { name: 'API latency propagation', color: '#16a34a', services: 'api-gw, rate-limiter, postgres', count: '42 alerts → 1 incident' },
  { name: 'ES disk + logging pipeline', color: '#7c3aed', services: 'elasticsearch, fluentd, kibana', count: '18 alerts → 1 incident' },
  { name: 'TLS cert chain failure', color: '#0891b2', services: 'ingress, cert-manager, istio', count: '11 alerts → 1 incident' },
  { name: 'Kafka consumer health', color: '#ea580c', services: 'kafka, order-processor, inventory', count: '8 alerts → 1 incident' },
  { name: 'DNS resolution latency', color: '#64748b', services: 'coredns, kube-proxy, calico', count: '4 alerts → 1 incident' },
];

const REMEDIATION_QUEUE = [
  { icon: '🔄', title: 'Restart payment-svc pods (3 replicas)', desc: 'Auto-approved: matches runbook RB-042', status: 'Ready' },
  { icon: '📈', title: 'Scale order-processor to 12 replicas', desc: 'Kafka lag > 500K threshold', status: 'Ready' },
  { icon: '🗑️', title: 'Delete ES indices older than 30 days', desc: 'Disk watermark exceeded on es-data-04', status: 'Pending Approval' },
  { icon: '🔐', title: 'Renew wildcard TLS certificate', desc: 'cert-manager renewal command queued', status: 'Executing' },
];

const RUNBOOKS = [
  { icon: 'auto', title: 'Pod CrashLoopBackOff Recovery', desc: 'Auto-detect crash reason, check OOM/liveness probe failure, collect logs, restart with resource adjustment.', type: 'auto', time: '~2 min' },
  { icon: 'auto', title: 'Certificate Auto-Renewal', desc: 'Detect expired/expiring certs, trigger cert-manager renewal, verify TLS handshake, update Ingress.', type: 'auto', time: '~30 sec' },
  { icon: 'semi', title: 'Database Connection Pool Tuning', desc: 'Analyze connection metrics, recommend pool size based on QPS and latency targets, apply with approval.', type: 'semi', time: '~5 min' },
  { icon: 'semi', title: 'Kafka Consumer Lag Remediation', desc: 'Identify slow consumers, check for rebalancing storms, scale consumer group, adjust poll settings.', type: 'semi', time: '~8 min' },
  { icon: 'manual', title: 'Redis Cluster Failover Playbook', desc: 'Verify replication state, check data consistency, promote replica if needed, update Sentinel config.', type: 'manual', time: '~15 min' },
  { icon: 'auto', title: 'Disk Space Cleanup', desc: 'Identify large log files, old Docker images, and unused PVCs. Clean up based on retention policies.', type: 'auto', time: '~1 min' },
  { icon: 'semi', title: 'Network Policy Audit', desc: 'Scan all namespaces for missing NetworkPolicies, suggest deny-all base + allow-list rules.', type: 'semi', time: '~3 min' },
  { icon: 'manual', title: 'Multi-cluster DR Failover', desc: 'Verify secondary cluster health, DNS failover, stateful workload sync, traffic cutover procedure.', type: 'manual', time: '~30 min' },
  { icon: 'auto', title: 'Node Drain & Cordon', desc: 'Gracefully drain workloads, respect PDBs, cordon node for maintenance, verify pod rescheduling.', type: 'auto', time: '~5 min' },
];

const TREND_DATA = [
  { day: 'Mon', p1: 1, p2: 3, p3: 5 },
  { day: 'Tue', p1: 0, p2: 4, p3: 7 },
  { day: 'Wed', p1: 2, p2: 2, p3: 4 },
  { day: 'Thu', p1: 1, p2: 5, p3: 8 },
  { day: 'Fri', p1: 0, p2: 3, p3: 6 },
  { day: 'Sat', p1: 0, p2: 1, p3: 2 },
  { day: 'Today', p1: 3, p2: 4, p3: 3 },
];

function initInsightsDashboard() {
  renderIncidentTable();
  renderCorrelationClusters();
  renderRemediationQueue();
  renderTrendChart();
  renderRunbooks();

  // Filter handler
  document.getElementById('incident-filter').addEventListener('change', renderIncidentTable);
  document.getElementById('refresh-incidents-btn').addEventListener('click', () => {
    showToast('🔄 Refreshing incident data from ITSM...', 'info');
    setTimeout(() => { renderIncidentTable(); showToast('✅ Incidents refreshed', 'success'); }, 800);
  });
}

function renderIncidentTable() {
  const filter = document.getElementById('incident-filter').value;
  const filtered = filter === 'all' ? INCIDENTS : INCIDENTS.filter(i => i.severity === filter);
  const tbody = document.getElementById('incident-tbody');
  tbody.innerHTML = filtered.map(inc => {
    const confClass = inc.confidence >= 90 ? 'conf-high' : inc.confidence >= 70 ? 'conf-medium' : 'conf-low';
    return `
      <tr data-id="${esc(inc.id)}">
        <td><span class="sev-badge sev-${esc(inc.severity)}">${esc(inc.severity)}</span></td>
        <td>
          <div style="font-weight:600;color:var(--text);font-size:.78rem">${esc(inc.id)}</div>
          <div style="font-size:.73rem;color:var(--text-sec);margin-top:2px">${esc(inc.title)}</div>
          <div style="font-size:.65rem;color:var(--text-muted);margin-top:2px">Duration: ${esc(inc.duration)}</div>
        </td>
        <td><code style="font-size:.73rem">${esc(inc.service)}</code></td>
        <td style="max-width:220px;font-size:.73rem;color:var(--text-sec);line-height:1.5">${esc(inc.aiRootCause).slice(0, 120)}…</td>
        <td><div class="blast-radius">${inc.blastRadius.map(b => `<span class="blast-chip">${esc(b)}</span>`).join('')}</div></td>
        <td>
          <div style="font-size:.78rem;font-weight:700">${esc(String(inc.confidence))}%</div>
          <div class="confidence-bar"><div class="confidence-fill ${confClass}" style="width:${Number(inc.confidence)}%"></div></div>
        </td>
        <td><span class="status-badge status-${esc(inc.status)}">${esc(inc.status)}</span></td>
        <td><button class="dash-btn primary" onclick="event.stopPropagation();showIncidentRCA('${esc(inc.id)}')">View RCA</button></td>
      </tr>`;
  }).join('');

  tbody.querySelectorAll('tr').forEach(tr => {
    tr.addEventListener('click', () => showIncidentRCA(tr.dataset.id));
  });
}

function showIncidentRCA(id) {
  const inc = INCIDENTS.find(i => i.id === id);
  if (!inc) return;

  // Highlight selected row
  document.querySelectorAll('#incident-tbody tr').forEach(r => r.classList.toggle('selected', r.dataset.id === id));

  // RCA Panel
  const rcaPanel = document.getElementById('rca-detail');
  rcaPanel.innerHTML = `
    <div class="rca-content">
      <h3><span class="sev-badge sev-${esc(inc.severity)}">${esc(inc.severity)}</span> ${esc(inc.id)} — ${esc(inc.title)}</h3>
      <p>${esc(inc.aiRootCause)}</p>
      <h3>📍 Event Timeline</h3>
      <div class="rca-timeline">
        ${inc.timeline.map(t => `
          <div class="rca-tl-item">
            <div class="rca-tl-dot ${esc(t.type)}">
              ${t.type === 'trigger' ? '⚡' : t.type === 'propagation' ? '🔀' : t.type === 'impact' ? '💥' : '🔧'}
            </div>
            <div class="rca-tl-body">
              <div class="rca-tl-title">${esc(t.title)}</div>
              <div class="rca-tl-desc">${esc(t.desc)}</div>
            </div>
            <div class="rca-tl-time">${esc(t.time)}</div>
          </div>
        `).join('')}
      </div>
      <div class="rca-suggestion">
        <h4>${esc(inc.suggestion.title)}</h4>
        <p>${esc(inc.suggestion.text)}</p>
      </div>
      <div style="display:flex;gap:8px;margin-top:14px">
        <button class="dash-btn success" onclick="showToast('✅ Remediation applied for ${esc(inc.id)}','success')">Apply Fix</button>
        <button class="dash-btn primary" onclick="showToast('📋 Runbook generated for ${esc(inc.id)}','info')">Generate Runbook</button>
        <button class="dash-btn" onclick="showToast('📤 Escalated ${esc(inc.id)} to P1 on-call','warning')">Escalate</button>
      </div>
    </div>`;

  // Similar Incidents
  const simPanel = document.getElementById('similar-incidents');
  if (inc.similar.length === 0) {
    simPanel.innerHTML = '<div class="rca-empty">No similar incidents found in the knowledge base.</div>';
  } else {
    simPanel.innerHTML = inc.similar.map(s => `
      <div class="similar-card">
        <div class="similar-card-header">
          <span class="similar-card-title">${esc(s.id)} — ${esc(s.title)}</span>
          <span class="similar-match">${esc(s.match)} match</span>
        </div>
        <div class="similar-card-body"><strong>Resolution:</strong> ${esc(s.resolution)}</div>
        <div class="similar-card-meta">
          <span>📅 ${esc(s.date)}</span>
          <span>⏱ TTR: ${esc(s.ttm)}</span>
        </div>
      </div>
    `).join('');
  }

  // Smooth scroll to RCA section
  rcaPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderCorrelationClusters() {
  const container = document.getElementById('correlation-clusters');
  container.innerHTML = CORRELATION_CLUSTERS.map(c => `
    <div class="ti-cluster-item">
      <div class="ti-cluster-dot" style="background:${esc(c.color)}"></div>
      <span class="ti-cluster-name">${esc(c.name)}</span>
      <span class="ti-cluster-count">${esc(c.count)}</span>
    </div>
  `).join('');
}

function renderRemediationQueue() {
  const container = document.getElementById('remediation-queue');
  container.innerHTML = REMEDIATION_QUEUE.map(r => {
    const statusColor = r.status === 'Ready' ? 'var(--green)' : r.status === 'Executing' ? 'var(--accent)' : 'var(--amber)';
    return `
    <div class="ti-rem-item">
      <span class="ti-rem-icon">${r.icon}</span>
      <div class="ti-rem-body">
        <div class="ti-rem-title">${esc(r.title)}</div>
        <div class="ti-rem-desc">${esc(r.desc)}</div>
      </div>
      <button class="ti-rem-action" style="background:${statusColor}" onclick="showToast('⚡ Executing: ${esc(r.title)}','success')">${r.status === 'Executing' ? '⏳ Running' : r.status}</button>
    </div>`;
  }).join('');
}

function renderTrendChart() {
  const container = document.getElementById('incident-trend-chart');
  const maxTotal = Math.max(...TREND_DATA.map(d => d.p1 + d.p2 + d.p3));
  container.innerHTML = TREND_DATA.map(d => {
    const total = d.p1 + d.p2 + d.p3;
    const pct = (total / maxTotal) * 100;
    const p1h = d.p1 ? (d.p1 / total) * pct : 0;
    const p2h = d.p2 ? (d.p2 / total) * pct : 0;
    const p3h = d.p3 ? (d.p3 / total) * pct : 0;
    return `
      <div class="ti-chart-bar">
        <div style="width:100%;display:flex;flex-direction:column;gap:1px;align-items:stretch;justify-content:flex-end;height:80px">
          ${d.p1 ? `<div class="ti-chart-fill" style="height:${p1h}%;background:var(--red)"></div>` : ''}
          ${d.p2 ? `<div class="ti-chart-fill" style="height:${p2h}%;background:var(--amber)"></div>` : ''}
          ${d.p3 ? `<div class="ti-chart-fill" style="height:${p3h}%;background:var(--accent)"></div>` : ''}
        </div>
        <div class="ti-chart-label">${d.day}</div>
      </div>`;
  }).join('');
}

function renderRunbooks() {
  const grid = document.getElementById('runbook-grid');
  grid.innerHTML = RUNBOOKS.map(rb => `
    <div class="runbook-card">
      <div class="rb-header">
        <div class="rb-icon ${rb.icon}">${rb.icon === 'auto' ? '⚡' : rb.icon === 'semi' ? '🤝' : '📋'}</div>
        <div class="rb-title">${esc(rb.title)}</div>
      </div>
      <div class="rb-desc">${esc(rb.desc)}</div>
      <div class="rb-meta">
        <span class="rb-tag ${rb.type}">${rb.type === 'auto' ? 'Fully Automated' : rb.type === 'semi' ? 'Semi-Automated' : 'Manual Runbook'}</span>
        <span class="rb-time">⏱ ${esc(rb.time)}</span>
      </div>
    </div>
  `).join('');
}


// ══════════════════════════════════════════════════════════════════════════════
//  ███  AI MONITORING (PROACTIVE) DASHBOARD  ███
// ══════════════════════════════════════════════════════════════════════════════

const SERVICES_HEALTH = [
  { name: 'api-gateway', status: 'healthy', score: 98.5, cpu: '34%', mem: '62%', latency: '12ms', errRate: '0.01%', pods: '6/6' },
  { name: 'payment-svc', status: 'critical', score: 42.1, cpu: '94%', mem: '87%', latency: '4200ms', errRate: '34.7%', pods: '3/5' },
  { name: 'auth-svc', status: 'degraded', score: 71.3, cpu: '67%', mem: '78%', latency: '340ms', errRate: '2.1%', pods: '4/4' },
  { name: 'order-svc', status: 'healthy', score: 96.2, cpu: '28%', mem: '55%', latency: '45ms', errRate: '0.05%', pods: '8/8' },
  { name: 'inventory-svc', status: 'healthy', score: 99.1, cpu: '15%', mem: '42%', latency: '8ms', errRate: '0.00%', pods: '4/4' },
  { name: 'notification-svc', status: 'healthy', score: 94.8, cpu: '22%', mem: '48%', latency: '23ms', errRate: '0.08%', pods: '3/3' },
  { name: 'search-svc', status: 'degraded', score: 73.9, cpu: '82%', mem: '71%', latency: '520ms', errRate: '1.4%', pods: '5/6' },
  { name: 'ml-inference', status: 'critical', score: 31.2, cpu: '12%', mem: '34%', latency: 'timeout', errRate: '89.2%', pods: '0/4' },
  { name: 'user-profile-svc', status: 'healthy', score: 97.4, cpu: '19%', mem: '45%', latency: '15ms', errRate: '0.02%', pods: '4/4' },
  { name: 'redis-cluster', status: 'degraded', score: 68.5, cpu: '45%', mem: '92%', latency: '8ms', errRate: '0.5%', pods: '5/6' },
  { name: 'kafka-brokers', status: 'healthy', score: 95.7, cpu: '41%', mem: '63%', latency: '3ms', errRate: '0.00%', pods: '3/3' },
  { name: 'postgres-primary', status: 'healthy', score: 93.2, cpu: '55%', mem: '68%', latency: '5ms', errRate: '0.01%', pods: '2/2' },
];

const ANOMALIES = [
  {
    type: 'critical', icon: '🔴', title: 'Memory leak detected — payment-svc',
    desc: 'Heap usage growing linearly at 2.3MB/min without GC recovery. Projected OOM in ~47 minutes at current rate. Container memory: 3.6GB/4GB.',
    metric: 'Memory: 87% → 91% (4min)', service: 'payment-svc', detected: '3m ago',
  },
  {
    type: 'warning', icon: '🟡', title: 'Abnormal error rate spike — search-svc',
    desc: 'Error rate jumped from baseline 0.1% to 1.4% in last 8 minutes. Correlated with Elasticsearch cluster yellow status. AI predicts further degradation.',
    metric: 'Error Rate: 0.1% → 1.4%', service: 'search-svc', detected: '8m ago',
  },
  {
    type: 'warning', icon: '🟡', title: 'CPU utilization anomaly — api-gateway',
    desc: 'CPU pattern deviates from historical norm by 2.4 standard deviations. Unusual sustained load without corresponding traffic increase. Possible crypto-mining or runaway goroutine.',
    metric: 'CPU: 34% (expected: 18-22%)', service: 'api-gateway', detected: '15m ago',
  },
  {
    type: 'info', icon: '🔵', title: 'Network I/O pattern shift — kafka-brokers',
    desc: 'Inter-broker replication traffic increased 3.2x. Likely cause: partition reassignment in progress. No immediate impact but monitor for completion.',
    metric: 'Network: 450MB/s → 1.4GB/s', service: 'kafka-brokers', detected: '22m ago',
  },
  {
    type: 'warning', icon: '🟡', title: 'Disk I/O latency spike — postgres-primary',
    desc: 'Write latency increased from 2ms to 18ms. Correlated with VACUUM FULL running on large orders table (420GB). Auto-analyze scheduled.',
    metric: 'Disk Write: 2ms → 18ms', service: 'postgres-primary', detected: '31m ago',
  },
];

const PREDICTIVE_ALERTS = [
  {
    type: 'capacity', icon: '📊', title: 'Redis memory exhaustion in ~2.1 hours',
    desc: 'Based on current growth rate (12MB/min) and eviction pattern, redis-cluster will hit maxmemory (32GB) at approximately 16:35 UTC. Recommend increasing limit or enabling LRU eviction now.',
    confidence: '91%', eta: '2.1h', action: 'Scale Memory',
  },
  {
    type: 'failure', icon: '⚠️', title: 'payment-svc OOM crash predicted in ~47 min',
    desc: 'Memory leak pattern matches heap fragmentation issue (similar to INC-3901). Heap growing at 2.3MB/min. Recommend proactive restart with rolling deployment to avoid downtime.',
    confidence: '87%', eta: '47min', action: 'Rolling Restart',
  },
  {
    type: 'scaling', icon: '📈', title: 'Traffic surge expected — API gateway',
    desc: 'Historical pattern analysis: Tuesday 16:00-18:00 UTC typically sees 2.8x traffic spike (flash sale window). Current capacity handles 1.5x. Recommend pre-scaling to 12 replicas.',
    confidence: '82%', eta: '1.5h', action: 'Pre-Scale',
  },
];

const RESOURCES = [
  { name: 'Cluster CPU', icon: '🔲', used: 67, total: '480 cores', available: '158 cores', forecast: '🔮 Projected to hit 85% by Friday based on workload trends. Consider adding 2 nodes.' },
  { name: 'Cluster Memory', icon: '🧠', used: 74, total: '1.92 TB', available: '499 GB', forecast: '⚠️ Growing 1.2% daily. Will exceed 80% threshold in 5 days without intervention.' },
  { name: 'Persistent Storage', icon: '💾', used: 82, total: '24 TB', available: '4.32 TB', forecast: '🔴 Critical: Elasticsearch consuming 67% of storage. Enable ILM and delete old indices.' },
  { name: 'GPU Allocation', icon: '🎮', used: 45, total: '32 GPUs', available: '18 GPUs', forecast: '✅ Stable. ML training batch jobs scheduled for 02:00 UTC will use 12 additional GPUs.' },
  { name: 'Network Bandwidth', icon: '🌐', used: 38, total: '40 Gbps', available: '24.8 Gbps', forecast: '✅ Healthy. Inter-AZ traffic within normal ranges.' },
  { name: 'Load Balancer Connections', icon: '⚖️', used: 56, total: '100K', available: '44K', forecast: '📈 Expect 80% during Tuesday peak. Connection pooling optimization available.' },
];

const COST_OPTIMIZATIONS = [
  {
    icon: 'save', title: 'Right-size user-profile-svc', subtitle: 'Over-provisioned',
    body: 'CPU request 2000m but avg usage is 380m (19%). Memory request 4Gi but avg usage is 1.8Gi (45%). Recommend reducing to 500m CPU / 2Gi memory.',
    savings: '$2,840/month',
  },
  {
    icon: 'save', title: 'Convert dev/staging to spot instances', subtitle: 'Non-production workloads',
    body: '14 non-production nodes running on-demand. Historical interruption rate for c5.2xlarge spot: 3.2%. Implement with graceful drain.',
    savings: '$5,200/month',
  },
  {
    icon: 'warn', title: 'Orphaned PVCs detected', subtitle: '12 unused volumes',
    body: '12 PersistentVolumeClaims (totaling 2.4TB) are not mounted to any pod. Last accessed 30+ days ago. Safe to delete after verification.',
    savings: '$380/month',
  },
  {
    icon: 'opt', title: 'Enable cluster autoscaler tuning', subtitle: 'Scale-down optimization',
    body: 'Cluster has 3 underutilized nodes (<20% CPU/mem for 4+ hours daily). Enable aggressive scale-down: --scale-down-delay-after-add=5m.',
    savings: '$1,920/month',
  },
  {
    icon: 'save', title: 'Migrate logs to S3 tiered storage', subtitle: 'Log retention optimization',
    body: 'Elasticsearch retaining 90 days of logs (18TB). Move >7-day logs to S3 with Athena queries. Reduce ES cluster to 3 hot nodes.',
    savings: '$2,100/month',
  },
  {
    icon: 'opt', title: 'Schedule GPU nodes off-peak shutdown', subtitle: 'GPU cost optimization',
    body: 'GPU nodes idle 02:00-06:00 UTC daily (no training jobs). Shut down 8 GPU nodes during idle window. On-demand spin-up time: ~3min.',
    savings: '$960/month',
  },
];

const SCALING_RECOMMENDATIONS = [
  { service: 'payment-svc', current: 5, recommended: 8, reason: 'Memory pressure + traffic growth. p99 latency SLO at risk.', confidence: '94%', impact: 'Latency -60%, Error rate -90%', urgency: 'high' },
  { service: 'api-gateway', current: 6, recommended: 12, reason: 'Predicted Tuesday 16:00 traffic surge (2.8x historical pattern).', confidence: '82%', impact: 'Prevent SLO breach during peak', urgency: 'medium' },
  { service: 'search-svc', current: 6, recommended: 4, reason: 'Over-provisioned. p95 latency target met with 4 pods even at peak.', confidence: '88%', impact: 'Save $340/mo, no SLO impact', urgency: 'low' },
  { service: 'order-processor', current: 8, recommended: 14, reason: 'Kafka consumer lag 847K messages. Need more consumers to drain backlog.', confidence: '91%', impact: 'Lag cleared in ~12 min', urgency: 'high' },
  { service: 'user-profile-svc', current: 4, recommended: 2, reason: 'CPU avg 19%, memory avg 45%. 2 pods handle current load comfortably.', confidence: '86%', impact: 'Save $480/mo, maintain SLO', urgency: 'low' },
];

function initMonitoringDashboard() {
  renderHealthMap();
  renderAnomalies();
  renderPredictiveAlerts();
  renderResources();
  renderCostOptimizations();
  renderScalingTable();

  document.getElementById('refresh-health-btn').addEventListener('click', () => {
    showToast('🔄 Refreshing health metrics...', 'info');
    setTimeout(() => {
      // Simulate small score changes
      SERVICES_HEALTH.forEach(s => {
        if (s.status === 'healthy') s.score = Math.min(100, s.score + (Math.random() * 0.5 - 0.2));
      });
      renderHealthMap();
      showToast('✅ Health map refreshed', 'success');
    }, 600);
  });

  // Simulate live updates
  setInterval(() => {
    if (!monitoringInitialized) return;
    SERVICES_HEALTH.forEach(s => {
      if (s.status === 'healthy') s.score = Math.min(100, Math.max(80, s.score + (Math.random() * 1 - 0.5)));
    });
    // Only re-render if monitoring tab is active
    if (document.querySelector('.top-tab[data-tab="monitoring"]').classList.contains('active')) {
      renderHealthMap();
    }
  }, 10000);
}

function renderHealthMap() {
  const container = document.getElementById('health-map');
  container.innerHTML = SERVICES_HEALTH.map(s => `
    <div class="health-tile ${s.status}">
      <div class="ht-name">
        <span class="ht-status-dot ${s.status}"></span>
        ${esc(s.name)}
      </div>
      <div class="ht-score ${s.status}">${s.score.toFixed(1)}</div>
      <div class="ht-metrics">
        <div class="ht-metric"><span class="ht-metric-label">CPU</span><span class="ht-metric-value">${esc(s.cpu)}</span></div>
        <div class="ht-metric"><span class="ht-metric-label">Memory</span><span class="ht-metric-value">${esc(s.mem)}</span></div>
        <div class="ht-metric"><span class="ht-metric-label">Latency</span><span class="ht-metric-value">${esc(s.latency)}</span></div>
        <div class="ht-metric"><span class="ht-metric-label">Errors</span><span class="ht-metric-value">${esc(s.errRate)}</span></div>
        <div class="ht-metric"><span class="ht-metric-label">Pods</span><span class="ht-metric-value">${esc(s.pods)}</span></div>
      </div>
    </div>
  `).join('');
}

function renderAnomalies() {
  const container = document.getElementById('anomaly-list');
  container.innerHTML = ANOMALIES.map(a => `
    <div class="anomaly-item">
      <div class="anomaly-icon ${a.type}">${a.icon}</div>
      <div class="anomaly-body">
        <div class="anomaly-title">${esc(a.title)}</div>
        <div class="anomaly-desc">${esc(a.desc)}</div>
        <div class="anomaly-meta">
          <span>📊 ${esc(a.metric)}</span>
          <span>🏷️ ${esc(a.service)}</span>
          <span>⏱ ${esc(a.detected)}</span>
        </div>
      </div>
      <button class="anomaly-action" onclick="showToast('🔍 Investigating anomaly: ${esc(a.title).slice(0,40)}','info')">Investigate</button>
    </div>
  `).join('');
}

function renderPredictiveAlerts() {
  const container = document.getElementById('predictive-list');
  container.innerHTML = PREDICTIVE_ALERTS.map(p => `
    <div class="predictive-item">
      <div class="predict-icon ${p.type}">${p.icon}</div>
      <div class="predict-body">
        <div class="predict-title">${esc(p.title)}</div>
        <div class="predict-desc">${esc(p.desc)}</div>
        <div class="predict-meta">
          <span>🎯 Confidence: ${esc(p.confidence)}</span>
          <span>⏰ ETA: ${esc(p.eta)}</span>
        </div>
      </div>
      <button class="dash-btn primary" onclick="showToast('⚡ ${esc(p.action)} initiated','success')" style="align-self:center;white-space:nowrap">${esc(p.action)}</button>
    </div>
  `).join('');
}

function renderResources() {
  const grid = document.getElementById('resource-grid');
  grid.innerHTML = RESOURCES.map(r => {
    const level = r.used >= 80 ? 'critical' : r.used >= 60 ? 'warning' : 'safe';
    return `
      <div class="resource-card">
        <div class="rc-header">
          <div class="rc-name"><span class="rc-icon">${r.icon}</span>${esc(r.name)}</div>
          <div class="rc-pct ${level}">${r.used}%</div>
        </div>
        <div class="rc-bar-wrap"><div class="rc-bar ${level}" style="width:${r.used}%"></div></div>
        <div class="rc-details">
          <span>Total: ${esc(r.total)}</span>
          <span>Available: ${esc(r.available)}</span>
        </div>
        <div class="rc-forecast"><span class="rc-forecast-icon">🔮</span>${esc(r.forecast)}</div>
      </div>`;
  }).join('');
}

function renderCostOptimizations() {
  const grid = document.getElementById('cost-opt-grid');
  grid.innerHTML = COST_OPTIMIZATIONS.map(c => `
    <div class="cost-card">
      <div class="cc-header">
        <div class="cc-icon ${c.icon}">${c.icon === 'save' ? '💰' : c.icon === 'warn' ? '⚠️' : '⚡'}</div>
        <div>
          <div class="cc-title">${esc(c.title)}</div>
          <div class="cc-subtitle">${esc(c.subtitle)}</div>
        </div>
      </div>
      <div class="cc-body">${esc(c.body)}</div>
      <div class="cc-savings"><span class="cc-savings-icon">💵</span> Est. saving: ${esc(c.savings)}</div>
    </div>
  `).join('');
}

function renderScalingTable() {
  const tbody = document.getElementById('scaling-tbody');
  tbody.innerHTML = SCALING_RECOMMENDATIONS.map(s => {
    const direction = s.recommended > s.current ? '↑' : '↓';
    const dirClass = s.recommended > s.current ? 'scale-up' : 'scale-down';
    const urgencyBg = s.urgency === 'high' ? 'var(--red-bg)' : s.urgency === 'medium' ? 'var(--amber-bg)' : 'var(--surface2)';
    const urgencyColor = s.urgency === 'high' ? 'var(--red)' : s.urgency === 'medium' ? 'var(--amber)' : 'var(--text-muted)';
    return `
      <tr>
        <td><code style="font-size:.78rem">${esc(s.service)}</code></td>
        <td style="text-align:center;font-weight:700">${Number(s.current)}</td>
        <td style="text-align:center"><span style="font-weight:800;color:${s.recommended > s.current ? 'var(--red)' : 'var(--green)'}">${direction} ${s.recommended}</span></td>
        <td style="font-size:.73rem;color:var(--text-sec);max-width:200px">${esc(s.reason)}</td>
        <td>
          <div style="font-size:.78rem;font-weight:700">${esc(s.confidence)}</div>
          <div class="confidence-bar"><div class="confidence-fill conf-high" style="width:${parseInt(s.confidence)}%"></div></div>
        </td>
        <td style="font-size:.73rem;color:var(--text-sec)">${esc(s.impact)}</td>
        <td>
          <button class="dash-btn ${s.urgency === 'high' ? 'danger' : s.urgency === 'medium' ? 'primary' : ''}" onclick="showToast('⚖️ Scaling ${esc(s.service)}: ${Number(s.current)} → ${Number(s.recommended)} replicas','success')">
            ${s.urgency === 'high' ? '🚨 Apply Now' : s.urgency === 'medium' ? 'Schedule' : 'Review'}
          </button>
        </td>
      </tr>`;
  }).join('');
}


// ══════════════════════════════════════════════════════════════════════════════
//  Sidebar History & Search
// ══════════════════════════════════════════════════════════════════════════════

function loadSidebarHistory() {
  const chats = DUMMY_CHAT_HISTORY;
  if (!chats.length) {
    historyList.innerHTML = '<div class="sidebar-empty">No chat history yet.</div>';
    return;
  }
  historyList.innerHTML = chats.map(c => `
    <div class="history-item" data-id="${esc(c.id)}">
      <div class="history-icon">💬</div>
      <div class="history-body">
        <div class="history-title">${esc(c.title)}</div>
        <div class="history-preview">${esc(c.preview)}</div>
      </div>
      <div class="history-meta">
        <span class="history-time">${timeAgo(c.timestamp)}</span>
        <span class="history-count">${esc(String(c.messageCount))} msgs</span>
      </div>
    </div>
  `).join('');

  historyList.querySelectorAll('.history-item').forEach(item => {
    item.addEventListener('click', () => {
      const preview = item.querySelector('.history-preview').textContent;
      // Switch to chat tab
      document.querySelector('.top-tab[data-tab="chat"]').click();
      queryEl.value = preview;
      send();
    });
  });
}

let searchDebounce = null;

function highlightMatch(text, query) {
  if (!query) return esc(text);
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(${escaped})`, 'gi');
  return esc(text).replace(re, '<mark>$1</mark>');
}

function doSidebarSearch(query) {
  if (!query.trim()) {
    searchResults.style.display = 'none';
    historyList.style.display = '';
    return;
  }
  const needle = query.toLowerCase();
  const results = DUMMY_CHAT_HISTORY.filter(c =>
    c.title.toLowerCase().includes(needle) || c.preview.toLowerCase().includes(needle)
  );
  historyList.style.display = 'none';
  searchResults.style.display = '';
  if (!results.length) {
    searchResults.innerHTML = '<div class="sidebar-empty">No matches found.</div>';
    return;
  }
  searchResults.innerHTML = results.map(c => `
    <div class="history-item" data-id="${esc(c.id)}">
      <div class="history-icon">🔍</div>
      <div class="history-body">
        <div class="history-title">${highlightMatch(c.title, query)}</div>
        <div class="history-preview">${highlightMatch(c.preview, query)}</div>
      </div>
      <div class="history-meta">
        <span class="history-time">${timeAgo(c.timestamp)}</span>
      </div>
    </div>
  `).join('');

  searchResults.querySelectorAll('.history-item').forEach(item => {
    item.addEventListener('click', () => {
      const preview = item.querySelector('.history-preview').textContent;
      document.querySelector('.top-tab[data-tab="chat"]').click();
      queryEl.value = preview;
      queryEl.focus();
    });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
//  Event bindings
// ══════════════════════════════════════════════════════════════════════════════

function bindSampleCards() {
  document.querySelectorAll('.sample-card').forEach(card => {
    card.addEventListener('click', () => {
      queryEl.value = card.dataset.query;
      send();
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  bindSampleCards();
  loadSidebarHistory();
  initTabs();
  initDebugMode();

  // Debug toggle
  debugToggleBtn.addEventListener('click', () => toggleDebug());

  // textarea auto-resize
  queryEl.addEventListener('input', () => {
    queryEl.style.height = 'auto';
    queryEl.style.height = Math.min(queryEl.scrollHeight, 130) + 'px';
  });

  // Enter to send, Shift+Enter for newline
  queryEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  sendBtn.addEventListener('click', send);

  // activity panel toggle
  actToggle.addEventListener('click', () => activityPane.classList.toggle('open'));
  document.getElementById('close-activity').addEventListener('click', () =>
    activityPane.classList.remove('open'));

  // new chat
  newChatBtn.addEventListener('click', newChat);

  // sidebar search
  searchInput.addEventListener('input', () => {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => doSidebarSearch(searchInput.value), 200);
  });
});

// mnemo observability dashboard — app.js
// Vanilla JS, no framework. D3.js for graph only.

(function () {
  'use strict';

  // ── Domain Colors ──
  const DOMAIN_COLORS = {
    architecture: '#4fc3f7',
    decisions:    '#ab47bc',
    patterns:     '#66bb6a',
    tasks:        '#ffa726',
    issues:       '#ef5350',
    dependencies: '#78909c',
    history:      '#8d6e63',
    context:      '#bdbdbd',
  };

  // ── State & Cache ──
  const cache = {
    status: null,
    nodes: null,
    nodesParams: null, // "{domain}|{sort}" to detect stale
    graph: null,
    nodeDetails: {},   // addr → detail
    provenance: {},    // addr → provenance chain
  };

  let logWs = null;
  let logPaused = false;
  let logEntries = [];
  const LOG_MAX = 500;
  let graphSim = null;
  let currentTab = 'overview';
  let expandedNodeAddr = null;

  // ── Helpers ──
  function $(sel, ctx) { return (ctx || document).querySelector(sel); }
  function $$(sel, ctx) { return (ctx || document).querySelectorAll(sel); }

  function domainColor(d) { return DOMAIN_COLORS[d] || '#888'; }

  function fmtAge(days) {
    if (days == null) return '--';
    if (days < 1) return '<1d';
    if (days < 30) return Math.round(days) + 'd';
    return Math.round(days / 30) + 'mo';
  }

  function fmtPct(v) {
    if (v == null) return '--';
    return (v * 100).toFixed(0) + '%';
  }

  function fmtCtx(chars) {
    if (chars == null) return '--';
    if (chars < 1000) return chars + ' chars';
    return (chars / 1000).toFixed(1) + 'k';
  }

  function truncAddr(addr) {
    return addr ? addr.slice(0, 8) : '--';
  }

  function escHtml(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function fmtTime(ts) {
    if (!ts) return '--';
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch { return '--'; }
  }

  async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`API ${path}: ${r.status}`);
    return r.json();
  }

  // ── Tab Switching ──
  function initTabs() {
    $$('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        if (tab === currentTab) return;
        currentTab = tab;

        $$('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
        $$('.panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + tab));

        onTabActivate(tab);
      });
    });
  }

  function onTabActivate(tab) {
    if (tab === 'overview') loadOverview();
    else if (tab === 'nodes') loadNodes();
    else if (tab === 'graph') loadGraph();
    else if (tab === 'log') ensureLogWs();
  }

  // ── Status Bar ──
  async function loadStatus() {
    try {
      const s = await api('/api/status');
      cache.status = s;
      renderStatus(s);
      return s;
    } catch (e) {
      console.error('Status fetch failed:', e);
    }
  }

  function renderStatus(s) {
    const dot = $('#status-health .chip-dot');
    const lbl = $('#status-health .chip-label');
    const h = (s.pressure || 'unknown').toLowerCase();
    dot.className = 'chip-dot ' + (h === 'low' ? 'ok' : h === 'high' ? 'warn' : '');
    lbl.textContent = s.pressure || '--';
    $('#status-nodes').textContent = s.active_count != null ? s.active_count + ' nodes' : '--';
    $('#status-context').textContent = fmtCtx(s.context_chars);
    $('#status-roots').textContent = s.root_count != null ? s.root_count + ' roots' : '--';
  }

  // ── Overview Panel ──
  async function loadOverview() {
    const s = cache.status || await loadStatus();
    if (!s) return;

    renderDomainChart(s.domains || {});
    renderHealthDetail(s);
    loadAttentionMetrics();
  }

  function renderDomainChart(domains) {
    const container = $('#domain-chart');
    const entries = Object.entries(domains).sort((a, b) => b[1] - a[1]);
    const maxCount = Math.max(...entries.map(e => e[1]), 1);

    container.innerHTML = entries.map(([name, count]) => `
      <div class="domain-bar-row">
        <span class="domain-bar-label">${escHtml(name)}</span>
        <div class="domain-bar-track">
          <div class="domain-bar-fill" style="width:${(count / maxCount * 100).toFixed(1)}%;background:${domainColor(name)}"></div>
        </div>
        <span class="domain-bar-count">${count}</span>
      </div>
    `).join('');

    if (entries.length === 0) {
      container.innerHTML = '<div class="attention-none">No domains found</div>';
    }
  }

  function renderHealthDetail(s) {
    const el = $('#health-detail');
    const rows = [
      ['Active Nodes', s.active_count],
      ['Context Size', fmtCtx(s.context_chars)],
      ['Root Count', s.root_count],
      ['Pressure', s.pressure],
    ];
    el.innerHTML = rows.map(([k, v]) => `
      <div class="health-row">
        <span class="health-key">${escHtml(k)}</span>
        <span class="health-val">${escHtml(String(v != null ? v : '--'))}</span>
      </div>
    `).join('');
  }

  async function loadAttentionMetrics() {
    const el = $('#attention-metrics');
    try {
      if (!cache.nodes) {
        const resp = await api('/api/nodes');
        cache.nodes = resp.nodes || resp;
      }
      const nodes = cache.nodes;

      const items = [];

      // Low hit rate nodes (with at least some recalls)
      const lowHitRate = nodes
        .filter(n => n.recall_count >= 5 && n.recall_hit_rate != null && n.recall_hit_rate < 0.2)
        .sort((a, b) => a.recall_hit_rate - b.recall_hit_rate)
        .slice(0, 5);

      lowHitRate.forEach(n => {
        items.push({
          critical: n.recall_hit_rate < 0.05,
          text: `Low hit rate: <span class="att-addr">${truncAddr(n.addr)}</span> — ${fmtPct(n.recall_hit_rate)} (${n.recall_count} recalls)`,
        });
      });

      // Low coverage compressions
      const lowCoverage = nodes
        .filter(n => n.type === 'compress' && n.coverage_score != null && n.coverage_score < 0.5)
        .sort((a, b) => a.coverage_score - b.coverage_score)
        .slice(0, 5);

      lowCoverage.forEach(n => {
        items.push({
          critical: n.coverage_score < 0.2,
          text: `Low coverage compression: <span class="att-addr">${truncAddr(n.addr)}</span> — ${fmtPct(n.coverage_score)}`,
        });
      });

      if (items.length === 0) {
        el.innerHTML = '<div class="attention-none">All metrics within normal range</div>';
      } else {
        el.innerHTML = items.map(it =>
          `<div class="attention-item ${it.critical ? 'critical' : ''}">${it.text}</div>`
        ).join('');
      }
    } catch (e) {
      el.innerHTML = '<div class="attention-none">Could not load metrics</div>';
    }
  }

  // ── Nodes Panel ──
  function initNodesControls() {
    $('#filter-domain').addEventListener('change', loadNodes);
    $('#sort-field').addEventListener('change', loadNodes);
    $('#nodes-refresh').addEventListener('click', () => {
      cache.nodes = null;
      cache.nodesParams = null;
      loadNodes();
    });
  }

  async function loadNodes() {
    const domain = $('#filter-domain').value;
    const sort = $('#sort-field').value;
    const paramKey = `${domain}|${sort}`;

    // Use cache if same params
    if (cache.nodesParams === paramKey && cache.nodes) {
      renderNodesTable(cache.nodes);
      return;
    }

    try {
      let url = '/api/nodes?';
      if (domain) url += 'domain=' + encodeURIComponent(domain) + '&';
      if (sort) url += 'sort=' + encodeURIComponent(sort);
      const resp = await api(url);
      const nodes = resp.nodes || resp;
      cache.nodes = nodes;
      cache.nodesParams = paramKey;
      renderNodesTable(nodes);
    } catch (e) {
      console.error('Nodes fetch failed:', e);
      $('#nodes-tbody').innerHTML = '<tr><td colspan="9" style="color:var(--text-muted);padding:20px">Failed to load nodes</td></tr>';
    }
  }

  function renderNodesTable(nodes) {
    const tbody = $('#nodes-tbody');
    expandedNodeAddr = null;

    tbody.innerHTML = nodes.map(n => `
      <tr data-addr="${escHtml(n.addr)}">
        <td class="addr-cell">${truncAddr(n.addr)}</td>
        <td><span class="domain-badge" style="color:${domainColor(n.domain)};border:1px solid ${domainColor(n.domain)}33">${escHtml(n.domain || '--')}</span></td>
        <td><span class="type-badge ${escHtml(n.type || '')}">${escHtml(n.type || '--')}</span></td>
        <td>${fmtAge(n.age_days)}</td>
        <td>${n.confidence != null ? n.confidence.toFixed(2) : '--'}</td>
        <td>${n.recall_count != null ? n.recall_count : '--'}</td>
        <td>${n.recall_hits != null ? n.recall_hits : '--'}</td>
        <td>${fmtPct(n.recall_hit_rate)}</td>
        <td>${escHtml(n.content_preview || '')}</td>
      </tr>
    `).join('');

    // Row click → expand detail
    tbody.querySelectorAll('tr[data-addr]').forEach(tr => {
      tr.addEventListener('click', () => toggleNodeDetail(tr));
    });
  }

  async function toggleNodeDetail(tr) {
    const addr = tr.dataset.addr;

    // Collapse if already expanded
    if (expandedNodeAddr === addr) {
      const detailRow = tr.nextElementSibling;
      if (detailRow && detailRow.classList.contains('node-detail-row')) {
        detailRow.remove();
      }
      expandedNodeAddr = null;
      return;
    }

    // Collapse any existing expansion
    const existing = $('#nodes-tbody .node-detail-row');
    if (existing) existing.remove();

    expandedNodeAddr = addr;

    // Fetch detail and provenance in parallel
    const [detail, prov] = await Promise.all([
      cache.nodeDetails[addr] || api('/api/nodes/' + addr).then(d => { cache.nodeDetails[addr] = d; return d; }).catch(() => null),
      cache.provenance[addr] || api('/api/provenance/' + addr).then(p => { const chain = p.chain || p; cache.provenance[addr] = chain; return chain; }).catch(() => null),
    ]);

    // Build detail row
    const detailTr = document.createElement('tr');
    detailTr.className = 'node-detail-row';
    const td = document.createElement('td');
    td.colSpan = 9;

    let html = '<div class="node-detail">';

    // Full content
    if (detail && detail.content) {
      html += `
        <div class="node-detail-section">
          <div class="node-detail-label">Content</div>
          <div class="node-detail-content">${escHtml(detail.content)}</div>
        </div>`;
    }

    // Meta
    if (detail && detail.meta && Object.keys(detail.meta).length > 0) {
      html += `<div class="node-detail-section">
        <div class="node-detail-label">Metadata</div>
        <div class="node-detail-meta">
          ${Object.entries(detail.meta).map(([k, v]) =>
            `<div class="meta-pair"><span class="meta-key">${escHtml(k)}:</span><span class="meta-val">${escHtml(String(v))}</span></div>`
          ).join('')}
        </div>
      </div>`;
    }

    // Inputs
    if (detail && detail.inputs && detail.inputs.length > 0) {
      html += `<div class="node-detail-section">
        <div class="node-detail-label">Inputs (${detail.inputs.length})</div>
        <div class="node-detail-meta">
          ${detail.inputs.map(i => `<span class="addr-cell">${truncAddr(i)}</span>`).join(' ')}
        </div>
      </div>`;
    }

    // Provenance
    if (prov && prov.length > 0) {
      html += `<div class="node-detail-section">
        <div class="node-detail-label">Provenance Chain</div>
        <div class="prov-chain">
          ${prov.map((p, i) => `
            <div class="prov-step">
              ${i > 0 ? '<span class="prov-arrow">&#x2190;</span>' : ''}
              <span class="prov-addr">${truncAddr(p.addr)}</span>
              <span class="prov-type type-badge ${escHtml(p.type || '')}">${escHtml(p.type || '')}</span>
              <span class="prov-preview">${escHtml(p.content_preview || '')}</span>
              ${p.coverage_score != null ? `<span class="meta-val">(cov: ${fmtPct(p.coverage_score)})</span>` : ''}
            </div>
          `).join('')}
        </div>
      </div>`;
    }

    html += '</div>';
    td.innerHTML = html;
    detailTr.appendChild(td);
    tr.after(detailTr);
  }

  // ── Graph Panel ──
  async function loadGraph() {
    if (typeof d3 === 'undefined') {
      console.warn('D3.js not loaded yet');
      return;
    }

    try {
      const data = cache.graph || await api('/api/graph');
      cache.graph = data;
      renderGraph(data);
    } catch (e) {
      console.error('Graph fetch failed:', e);
    }
  }

  function renderGraph(data) {
    const svg = d3.select('#graph-svg');
    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Clear previous
    svg.selectAll('*').remove();
    if (graphSim) graphSim.stop();

    if (!data.nodes || data.nodes.length === 0) {
      svg.append('text')
        .attr('x', width / 2).attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#8b949e')
        .attr('font-size', '14px')
        .text('No graph data available');
      return;
    }

    const tooltip = document.getElementById('graph-tooltip');

    // Scale for node size
    const maxRecall = Math.max(...data.nodes.map(n => n.recall_count || 0), 1);
    const rScale = d3.scaleSqrt().domain([0, maxRecall]).range([4, 18]);

    // Simulation
    const sim = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => rScale(d.recall_count || 0) + 4));

    graphSim = sim;

    // Zoom
    const g = svg.append('g');
    svg.call(d3.zoom()
      .scaleExtent([0.2, 4])
      .on('zoom', (e) => g.attr('transform', e.transform)));

    // Links
    const link = g.append('g')
      .selectAll('line')
      .data(data.links)
      .join('line')
      .attr('class', d => 'graph-link ' + (d.rel || ''));

    // Nodes
    const node = g.append('g')
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('class', 'graph-node')
      .attr('r', d => rScale(d.recall_count || 0))
      .attr('fill', d => domainColor(d.domain))
      .attr('stroke', '#0d1117')
      .attr('stroke-width', 1.5)
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    // Labels (only for larger nodes)
    const label = g.append('g')
      .selectAll('text')
      .data(data.nodes.filter(n => (n.recall_count || 0) > maxRecall * 0.2))
      .join('text')
      .attr('class', 'graph-label')
      .attr('dy', d => rScale(d.recall_count || 0) + 12)
      .text(d => d.label || truncAddr(d.id));

    // Tooltip on hover
    node.on('mouseover', (e, d) => {
      tooltip.classList.remove('hidden');
      tooltip.innerHTML = `
        <div class="tooltip-addr">${escHtml(d.id)}</div>
        <div style="margin-bottom:4px">
          <span class="domain-badge" style="color:${domainColor(d.domain)};border:1px solid ${domainColor(d.domain)}33">${escHtml(d.domain || '--')}</span>
          <span class="type-badge ${escHtml(d.type || '')}" style="margin-left:6px">${escHtml(d.type || '')}</span>
        </div>
        <div class="tooltip-content">${escHtml(d.label || '')}</div>
        <div style="margin-top:4px;color:#8b949e;font-size:11px">
          Recalls: ${d.recall_count || 0} | Hits: ${d.recall_hits || 0} | Conf: ${d.confidence != null ? d.confidence.toFixed(2) : '--'}
        </div>
      `;
    }).on('mousemove', (e) => {
      const rect = container.getBoundingClientRect();
      tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
      tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
    }).on('mouseout', () => {
      tooltip.classList.add('hidden');
    });

    // Tick
    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);
      label
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    });
  }

  // ── Log Panel ──
  function initLogControls() {
    $('#log-pause').addEventListener('click', () => {
      logPaused = !logPaused;
      $('#log-pause').textContent = logPaused ? 'Resume' : 'Pause';
      $('#log-pause').classList.toggle('active', logPaused);
    });

    $('#log-clear').addEventListener('click', () => {
      logEntries = [];
      $('#log-feed').innerHTML = '';
    });
  }

  function ensureLogWs() {
    if (logWs && logWs.readyState <= 1) return; // CONNECTING or OPEN

    // Load initial logs via REST
    loadInitialLogs();

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/logs`;
    $('#log-status').textContent = 'Connecting...';

    try {
      logWs = new WebSocket(url);
    } catch (e) {
      $('#log-status').textContent = 'WebSocket not available — using polling';
      startLogPolling();
      return;
    }

    logWs.onopen = () => {
      $('#log-status').textContent = 'Live';
    };

    logWs.onmessage = (evt) => {
      try {
        const entry = JSON.parse(evt.data);
        appendLogEntry(entry);
      } catch (e) { /* ignore malformed */ }
    };

    logWs.onclose = () => {
      $('#log-status').textContent = 'Disconnected — retrying in 5s';
      setTimeout(ensureLogWs, 5000);
    };

    logWs.onerror = () => {
      logWs.close();
    };
  }

  async function loadInitialLogs() {
    try {
      const resp = await api('/api/logs?limit=100');
      const logs = resp.entries || resp;
      if (Array.isArray(logs)) {
        const feed = $('#log-feed');
        feed.innerHTML = '';
        logEntries = [];
        logs.forEach(entry => appendLogEntry(entry));
      }
    } catch (e) { /* ignore */ }
  }

  let pollTimer = null;
  function startLogPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const resp = await api('/api/logs?limit=20');
        const logs = resp.entries || resp;
        if (Array.isArray(logs)) {
          // Only add new entries
          const existing = new Set(logEntries.map(e => e.ts));
          logs.forEach(entry => {
            if (!existing.has(entry.ts)) appendLogEntry(entry);
          });
        }
      } catch (e) { /* ignore */ }
    }, 3000);
  }

  function appendLogEntry(entry) {
    if (logPaused) return;

    logEntries.push(entry);
    if (logEntries.length > LOG_MAX) logEntries.shift();

    const feed = $('#log-feed');
    const div = document.createElement('div');
    div.className = 'log-entry';

    const layer = entry.layer || 'system';
    const domColor = domainColor(entry.domain);

    div.innerHTML = `
      <span class="log-ts">${fmtTime(entry.ts)}</span>
      <span class="log-layer ${escHtml(layer)}">${escHtml(layer)}</span>
      <span class="log-event">${escHtml(entry.event || '')}</span>
      <span class="log-summary">${escHtml(entry.summary || '')}</span>
      ${entry.domain ? `<span class="log-domain" style="color:${domColor}">${escHtml(entry.domain)}</span>` : ''}
    `;

    feed.appendChild(div);

    // Trim DOM if too many
    while (feed.children.length > LOG_MAX) {
      feed.removeChild(feed.firstChild);
    }

    // Auto-scroll
    feed.scrollTop = feed.scrollHeight;
  }

  // ── Modal (unused for now, available for future use) ──
  function initModal() {
    const modal = $('#node-modal');
    modal.querySelector('.modal-backdrop').addEventListener('click', closeModal);
    modal.querySelector('.modal-close').addEventListener('click', closeModal);
  }

  function closeModal() {
    $('#node-modal').classList.add('hidden');
  }

  // ── Window resize for graph ──
  function initResize() {
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (currentTab === 'graph' && cache.graph) {
          renderGraph(cache.graph);
        }
      }, 250);
    });
  }

  // ── Keyboard shortcut ──
  function initKeys() {
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal();
      // Tab shortcuts: 1-4
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      const tabs = ['overview', 'nodes', 'graph', 'log'];
      const idx = parseInt(e.key) - 1;
      if (idx >= 0 && idx < tabs.length) {
        $$('.tab-btn')[idx].click();
      }
    });
  }

  // ── Init ──
  function init() {
    initTabs();
    initNodesControls();
    initLogControls();
    initModal();
    initResize();
    initKeys();

    // Initial load
    loadStatus().then(() => loadOverview());

    // Refresh status every 30s
    setInterval(loadStatus, 30000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

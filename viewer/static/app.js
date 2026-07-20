const state = {
  runs: [],
  selectedRun: null,
  selectedEpisode: null,
};

const elements = {
  cards: document.querySelector('#benchmark-cards'),
  rows: document.querySelector('#run-rows'),
  empty: document.querySelector('#empty-runs'),
  updatedAt: document.querySelector('#updated-at'),
  benchmarkFilter: document.querySelector('#benchmark-filter'),
  statusFilter: document.querySelector('#status-filter'),
  taskFilter: document.querySelector('#task-filter'),
  connectionDot: document.querySelector('#connection-dot'),
  connectionText: document.querySelector('#connection-text'),
  panel: document.querySelector('#detail-panel'),
  scrim: document.querySelector('#panel-scrim'),
  close: document.querySelector('#detail-close'),
  detailBenchmark: document.querySelector('#detail-benchmark'),
  detailTitle: document.querySelector('#detail-title'),
  detailMeta: document.querySelector('#detail-meta'),
  detailBody: document.querySelector('#detail-body'),
};

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined && text !== null) item.textContent = String(text);
  return item;
}

function formatRate(value) {
  return value === null || value === undefined ? '—' : `${Math.round(Number(value) * 100)}%`;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function formatDate(value) {
  if (!value) return 'Unknown time';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function badge(value, labels = { true: 'Yes', false: 'No', null: 'N/A' }) {
  const normal = value === true ? true : value === false ? false : null;
  const className = normal === true ? 'ok' : normal === false ? 'bad' : 'na';
  return node('span', `badge ${className}`, labels[String(normal)]);
}

function statusBadge(status) {
  const className = status === 'complete' ? 'ok' : status === 'failed' ? 'bad' : 'warn';
  return node('span', `badge ${className}`, status || 'unknown');
}

function rateBadge(value) {
  if (value === null || value === undefined) return node('span', 'badge na', 'N/A');
  const numeric = Number(value);
  const className = numeric === 1 ? 'ok' : numeric === 0 ? 'bad' : 'warn';
  return node('span', `badge ${className}`, formatRate(numeric));
}

function metric(label, value) {
  const box = node('div', 'metric');
  box.append(node('span', 'metric-label', label), node('span', 'metric-value', value));
  return box;
}

function renderSummary(summary) {
  elements.cards.replaceChildren();
  const byName = new Map((summary.benchmarks || []).map(item => [item.benchmark, item]));
  for (const benchmark of ['robotwin', 'libero']) {
    const item = byName.get(benchmark) || { benchmark, metrics: {} };
    const metrics = item.metrics || {};
    const card = node('article', 'benchmark-card');
    const header = node('div', 'benchmark-card-header');
    header.append(
      node('span', 'benchmark-name', benchmark === 'robotwin' ? 'RobotTwin' : 'LIBERO'),
      node('span', 'muted', `${metrics.runs || 0} run${metrics.runs === 1 ? '' : 's'}`),
    );
    const grid = node('div', 'metrics');
    grid.append(
      metric('Task success', formatRate(metrics.task_success_rate)),
      metric('Code execution', formatRate(metrics.code_execution_rate)),
      metric('Plan success', formatRate(metrics.plan_success_rate)),
      metric('Mean reward', formatNumber(metrics.mean_reward)),
      metric('Episodes', `${metrics.episodes_finished || 0}/${metrics.episodes_total || 0}`),
    );
    card.append(header, grid);
    elements.cards.append(card);
  }
  elements.updatedAt.textContent = summary.updated_at ? `Updated ${formatDate(summary.updated_at)}` : 'No results yet';
}

function taskLabel(run) {
  const config = run.config || {};
  return config.task || config.suite || '—';
}

function renderRuns(runs) {
  state.runs = runs;
  elements.rows.replaceChildren();
  elements.empty.hidden = runs.length > 0;
  for (const run of runs) {
    const row = document.createElement('tr');
    const runCell = document.createElement('td');
    const button = node('button', 'run-link run-id', run.run_id);
    button.type = 'button';
    button.title = run.run_id;
    button.addEventListener('click', () => openRun(run.run_id));
    runCell.append(button);
    const metrics = run.metrics || {};
    const cells = [
      node('td', '', run.benchmark === 'robotwin' ? 'RobotTwin' : 'LIBERO'),
      node('td', '', taskLabel(run)),
      node('td', '', run.policy || '—'),
      document.createElement('td'),
      document.createElement('td'),
      document.createElement('td'),
      document.createElement('td'),
      node('td', '', formatNumber(metrics.mean_reward)),
    ];
    cells[3].append(statusBadge(run.status));
    cells[4].append(rateBadge(metrics.code_execution_rate));
    cells[5].append(rateBadge(metrics.plan_success_rate));
    cells[6].append(rateBadge(metrics.task_success_rate));
    row.append(runCell, ...cells);
    row.addEventListener('dblclick', () => openRun(run.run_id));
    elements.rows.append(row);
  }
}

function filtersQuery() {
  const params = new URLSearchParams();
  if (elements.benchmarkFilter.value) params.set('benchmark', elements.benchmarkFilter.value);
  if (elements.statusFilter.value) params.set('status', elements.statusFilter.value);
  if (elements.taskFilter.value.trim()) params.set('task', elements.taskFilter.value.trim());
  const query = params.toString();
  return query ? `?${query}` : '';
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function refresh() {
  try {
    const [summary, runPayload] = await Promise.all([
      fetchJSON('/api/summary'),
      fetchJSON(`/api/runs${filtersQuery()}`),
    ]);
    renderSummary(summary);
    renderRuns(runPayload.runs || []);
    elements.connectionDot.className = 'connection-dot online';
    elements.connectionText.textContent = 'Live';
  } catch (error) {
    elements.connectionDot.className = 'connection-dot offline';
    elements.connectionText.textContent = `Unavailable: ${error.message}`;
  }
}

function closePanel() {
  elements.panel.classList.remove('open');
  elements.panel.setAttribute('aria-hidden', 'true');
  elements.scrim.hidden = true;
  state.selectedRun = null;
  state.selectedEpisode = null;
}

async function openRun(runId) {
  try {
    const run = await fetchJSON(`/api/runs/${encodeURIComponent(runId)}`);
    state.selectedRun = run;
    state.selectedEpisode = run.episodes && run.episodes.length ? run.episodes[0].episode_id : null;
    elements.panel.classList.add('open');
    elements.panel.setAttribute('aria-hidden', 'false');
    elements.scrim.hidden = false;
    renderDetail();
  } catch (error) {
    elements.connectionText.textContent = `Could not load run: ${error.message}`;
  }
}

function stage(label, value) {
  const box = node('div', 'stage');
  box.append(node('span', 'metric-label', label));
  const text = value === true ? 'Passed' : value === false ? 'Failed' : 'Not available';
  const strong = node('strong', value === true ? 'ok-text' : value === false ? 'bad-text' : '', text);
  box.append(strong);
  return box;
}

async function loadArtifact(url, output) {
  output.textContent = 'Loading…';
  try {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    output.textContent = await response.text();
  } catch (error) {
    output.textContent = `Could not load artifact: ${error.message}`;
  }
}

function renderEpisode(run, episode, container) {
  const metrics = episode.metrics || {};
  const stages = node('div', 'stage-grid');
  stages.append(
    stage('Code execution', metrics.code_execution_success),
    stage('Plan', metrics.plan_success),
    stage('Task', metrics.task_success),
    stage('Reward', null),
  );
  stages.lastChild.querySelector('strong').textContent = formatNumber(metrics.reward);
  container.append(stages);

  const videoSection = node('section', 'detail-section');
  videoSection.append(node('h3', '', 'Trajectory'));
  const videoWrap = node('div', 'video-wrap');
  if (episode.artifacts && episode.artifacts.video) {
    const video = document.createElement('video');
    video.controls = true;
    video.preload = 'metadata';
    video.src = episode.artifacts.video;
    videoWrap.append(video);
  } else {
    videoWrap.append(node('p', 'no-video', 'No trajectory video was recorded.'));
  }
  videoSection.append(videoWrap);
  container.append(videoSection);

  if (episode.error) {
    const errorSection = node('section', 'detail-section');
    errorSection.append(node('h3', '', `Error · ${episode.error.stage || 'unknown stage'}`));
    const text = episode.error.traceback || episode.error.message || JSON.stringify(episode.error, null, 2);
    errorSection.append(node('pre', 'error-box', text));
    container.append(errorSection);
  }

  if (episode.details) {
    const detailsSection = node('details', 'detail-section');
    detailsSection.append(node('summary', '', 'Environment details'));
    detailsSection.append(node('pre', 'details-box', JSON.stringify(episode.details, null, 2)));
    container.append(detailsSection);
  }

  const artifacts = episode.artifacts || {};
  const available = [
    ['Generated code', artifacts.generated_code],
    ['Summary / logs', artifacts.summary],
    ['Events', artifacts.events],
    ['Responses', artifacts.responses],
  ].filter(([, url]) => Boolean(url));
  if (available.length) {
    const artifactSection = node('section', 'detail-section');
    artifactSection.append(node('h3', '', 'Artifacts'));
    const actions = node('div', 'artifact-actions');
    const output = node('pre', 'artifact-text', 'Choose an artifact to inspect.');
    for (const [label, url] of available) {
      const button = node('button', 'button secondary', label);
      button.type = 'button';
      button.addEventListener('click', () => loadArtifact(url, output));
      actions.append(button);
    }
    artifactSection.append(actions, output);
    container.append(artifactSection);
  }
}

function renderDetail() {
  const run = state.selectedRun;
  if (!run) return;
  elements.detailBenchmark.textContent = `${run.benchmark} · ${run.policy}`;
  elements.detailTitle.textContent = run.run_id;
  elements.detailMeta.textContent = `${run.status} · ${formatDate(run.updated_at)} · ${taskLabel(run)}`;
  elements.detailBody.replaceChildren();

  const episodes = run.episodes || [];
  if (!episodes.length) {
    elements.detailBody.append(node('p', 'empty', 'No episodes have been written yet.'));
    return;
  }
  if (!episodes.some(item => item.episode_id === state.selectedEpisode)) {
    state.selectedEpisode = episodes[0].episode_id;
  }
  const tabs = node('div', 'episode-tabs');
  for (const episode of episodes) {
    const button = node(
      'button',
      `episode-tab${episode.episode_id === state.selectedEpisode ? ' active' : ''}`,
      episode.episode_id,
    );
    button.type = 'button';
    button.addEventListener('click', () => {
      state.selectedEpisode = episode.episode_id;
      renderDetail();
    });
    tabs.append(button);
  }
  elements.detailBody.append(tabs);
  const episode = episodes.find(item => item.episode_id === state.selectedEpisode);
  renderEpisode(run, episode, elements.detailBody);
}

let taskTimer = null;
elements.benchmarkFilter.addEventListener('change', refresh);
elements.statusFilter.addEventListener('change', refresh);
elements.taskFilter.addEventListener('input', () => {
  clearTimeout(taskTimer);
  taskTimer = setTimeout(refresh, 250);
});
elements.close.addEventListener('click', closePanel);
elements.scrim.addEventListener('click', closePanel);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closePanel();
});

refresh();
setInterval(refresh, 5000);

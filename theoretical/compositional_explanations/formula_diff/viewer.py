from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

from theoretical.compositional_explanations.formula_diff import (
    analysis as formula_diff,
)


DEFAULT_FORMULA_DIFF_DIR = Path("14th") / "results" / "formula_diff"
SAME_NEURON_SUFFIX = "_same_neuron_structure_diff.csv"
HEATMAP_SUFFIX = "_formula_diff_score_heatmap.png"
DEFAULT_OUTPUT_FILENAME = "formula_diff_viewer.html"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def parse_optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def task_from_name(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)]


def diff_payload(row: dict[str, str]) -> dict[str, object]:
    return json.loads(row["formula_diff"])


def normalize_same_rows(formula_diff_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(formula_diff_dir.glob(f"*{SAME_NEURON_SUFFIX}")):
        for row in load_csv(path):
            rows.append(
                {
                    "task": row["task"],
                    "neuron": parse_int(row["neuron"]),
                    "diff_score": parse_optional_int(row["diff_score"]),
                    "base_formula": row["base_formula"],
                    "finetuned_formula": row["finetuned_formula"],
                    "diff": diff_payload(row),
                }
            )
    return rows


def normalize_closest_buckets(formula_diff_dir: Path) -> list[dict[str, object]]:
    path = formula_diff_dir / formula_diff.CLOSEST_BUCKETS_FILENAME
    if not path.exists():
        return []

    rows: list[dict[str, object]] = []
    buckets_by_task = json.loads(path.read_text(encoding="utf-8"))
    for task, buckets in sorted(buckets_by_task.items()):
        for bucket in buckets:
            rows.append(
                {
                    "task": task,
                    "base_neuron": parse_int(bucket["neuron_id"]),
                    "base_formula": bucket["base_formula"],
                    "lowest_diff_score": parse_optional_int(
                        bucket["lowest_diff_score"]
                    ),
                    "candidates": bucket.get("candidates", []),
                }
            )
    return rows


def discover_heatmaps(formula_diff_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "task": task_from_name(path, HEATMAP_SUFFIX),
            "src": path.name,
        }
        for path in sorted(formula_diff_dir.glob(f"*{HEATMAP_SUFFIX}"))
    ]


def build_summary(
    same_rows: list[dict[str, object]],
    closest_buckets: list[dict[str, object]],
) -> dict[str, object]:
    same_by_task = Counter(row["task"] for row in same_rows)
    closest_by_task = Counter(row["task"] for row in closest_buckets)
    candidate_counts = [
        len(bucket.get("candidates", []))
        for bucket in closest_buckets
    ]
    return {
        "sameRows": len(same_rows),
        "sameRowsByTask": dict(sorted(same_by_task.items())),
        "closestBuckets": len(closest_buckets),
        "closestBucketsByTask": dict(sorted(closest_by_task.items())),
        "closestCandidates": sum(candidate_counts),
        "maxClosestSplit": max(candidate_counts, default=0),
        "closestSameNeuronBuckets": sum(
            1
            for bucket in closest_buckets
            if any(
                parse_int(candidate["neuron_id"]) == bucket["base_neuron"]
                for candidate in bucket.get("candidates", [])
            )
        ),
    }


def render_html(
    same_rows: list[dict[str, object]],
    closest_buckets: list[dict[str, object]],
    heatmaps: list[dict[str, str]],
    summary: dict[str, object],
) -> str:
    payload = json.dumps(
        {
            "sameRows": same_rows,
            "closestBuckets": closest_buckets,
            "heatmaps": heatmaps,
            "summary": summary,
        },
        ensure_ascii=False,
    )
    return HTML_TEMPLATE.replace("__FORMULA_PAYLOAD__", html.escape(payload, quote=False))


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Formula Diff Viewer</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --panel-soft: #f9fafb;
  --line: #d8dee8;
  --ink: #172033;
  --muted: #667085;
  --base: #b42318;
  --fine: #067647;
  --shared: #925a00;
  --op: #1d4ed8;
  --focus: #2457c5;
  --shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
  font-size: 14px;
}
header {
  position: sticky;
  top: 0;
  z-index: 5;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.97);
  backdrop-filter: blur(8px);
}
.topbar {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 18px;
  align-items: center;
  padding: 14px 20px 10px;
}
h1 {
  margin: 0;
  font-size: 20px;
  letter-spacing: 0;
}
.metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}
.metric {
  min-width: 112px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  background: var(--panel-soft);
}
.metric span {
  display: block;
  color: var(--muted);
  font-size: 11px;
}
.metric strong {
  display: block;
  font-size: 16px;
  font-variant-numeric: tabular-nums;
}
.toolbar {
  display: grid;
  grid-template-columns: 130px 150px 140px 1fr 140px 120px 120px;
  gap: 8px;
  align-items: end;
  padding: 0 20px 12px;
}
label {
  display: grid;
  gap: 4px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}
select, input, button {
  min-height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  font: inherit;
  padding: 6px 8px;
}
button {
  cursor: pointer;
  font-weight: 700;
}
button.active {
  color: #fff;
  border-color: var(--focus);
  background: var(--focus);
}
.toggle {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
}
.toggle button { width: 100%; }
main {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 14px;
  padding: 14px 20px 28px;
}
.heatmaps {
  display: none;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 10px;
}
.heatmaps.open { display: grid; }
.heatmap {
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  padding: 8px;
}
.heatmap h2 {
  margin: 0 0 6px;
  font-size: 14px;
}
.heatmap img {
  width: 100%;
  max-height: 320px;
  object-fit: contain;
  background: #fff;
}
.status {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--muted);
}
.cards {
  display: grid;
  gap: 12px;
}
.card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.card-head {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: start;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  background: var(--panel-soft);
}
.title {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  font-weight: 700;
}
.pill {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 8px;
  background: #fff;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
.pill.score { color: var(--op); }
.pill.split { color: #7c2d12; }
.card-body {
  display: grid;
  grid-template-columns: minmax(260px, 0.32fr) minmax(0, 0.68fr);
  min-height: 360px;
}
.formula-pane {
  border-right: 1px solid var(--line);
  padding: 12px;
  overflow: auto;
}
.formula-block {
  margin-bottom: 12px;
}
.formula-label {
  margin-bottom: 4px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}
.formula {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
  line-height: 1.4;
}
.candidate-branches {
  display: grid;
  gap: 6px;
  max-height: 310px;
  overflow: auto;
  padding-right: 2px;
}
.candidate-button {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 8px;
  align-items: center;
  width: 100%;
  min-height: 32px;
  text-align: left;
  font-weight: 600;
}
.candidate-button .branch {
  color: var(--muted);
  font-family: Consolas, "Courier New", monospace;
}
.candidate-button.active {
  color: #fff;
  background: var(--focus);
}
.candidate-button.active .branch,
.candidate-button.active .candidate-formula {
  color: #fff;
}
.candidate-formula {
  min-width: 0;
  overflow: hidden;
  color: var(--ink);
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tree-pane {
  display: grid;
  grid-template-rows: auto minmax(260px, 1fr) auto;
  min-width: 0;
  padding: 12px;
}
.tree-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.tree-title {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
.tree-actions {
  display: flex;
  gap: 6px;
}
.tree-actions button {
  min-height: 28px;
  padding: 3px 8px;
  font-size: 12px;
}
.tree-wrap {
  min-height: 280px;
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: auto;
  background: #fff;
}
.tree-svg {
  display: block;
  min-width: 100%;
}
.tree-link {
  fill: none;
  stroke: #b8c0cc;
  stroke-width: 1.5;
}
.tree-node rect {
  fill: #fff;
  stroke: #aeb8c7;
  stroke-width: 1.2;
}
.tree-node text {
  fill: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
  font-size: 11px;
}
.tree-node.shared rect { stroke: var(--shared); fill: #fffbeb; }
.tree-node.added rect { stroke: var(--fine); fill: #ecfdf3; }
.tree-node.removed rect { stroke: var(--base); fill: #fff1f0; }
.tree-node.operator rect { stroke: var(--op); fill: #eff6ff; }
.diff-text {
  max-height: 170px;
  overflow: auto;
  margin: 10px 0 0;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: var(--panel-soft);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
  line-height: 1.4;
}
.minus { color: var(--base); }
.plus { color: var(--fine); }
.same { color: var(--shared); }
.pager {
  display: flex;
  gap: 8px;
  align-items: center;
  justify-content: flex-end;
}
.pager button { min-width: 90px; }
@media (max-width: 1150px) {
  .topbar { grid-template-columns: 1fr; }
  .metrics { justify-content: flex-start; }
  .toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .card-body { grid-template-columns: 1fr; }
  .formula-pane { border-right: 0; border-bottom: 1px solid var(--line); }
}
</style>
</head>
<body>
<header>
  <section class="topbar">
    <h1>Formula Diff Viewer</h1>
    <div class="metrics" id="metrics"></div>
  </section>
  <section class="toolbar">
    <div class="toggle">
      <button id="sameTab" class="active" type="button">Same</button>
      <button id="closestTab" type="button">Closest</button>
    </div>
    <label>Task
      <select id="taskFilter"><option value="">All</option></select>
    </label>
    <label>Max Score
      <input id="scoreFilter" type="number" min="0" value="">
    </label>
    <label>Search
      <input id="searchFilter" type="search" placeholder="formula or neuron">
    </label>
    <label>Sort
      <select id="sortSelect">
        <option value="score">Score</option>
        <option value="neuron">Neuron</option>
        <option value="split">Split</option>
      </select>
    </label>
    <label>Page Size
      <select id="pageSize">
        <option>10</option>
        <option selected>20</option>
        <option>50</option>
      </select>
    </label>
    <button id="heatmapToggle" type="button">Heatmaps</button>
  </section>
</header>
<main>
  <section class="heatmaps" id="heatmaps"></section>
  <div class="status">
    <span id="status"></span>
    <span id="pageStatus"></span>
  </div>
  <section class="cards" id="cards"></section>
  <div class="pager">
    <button id="prevPage" type="button">Previous</button>
    <button id="nextPage" type="button">Next</button>
  </div>
</main>
<script id="formula-data" type="application/json">__FORMULA_PAYLOAD__</script>
<script>
const data = JSON.parse(document.getElementById('formula-data').textContent);
let activeView = 'same';
let page = 0;
const selectedCandidate = new Map();
const treeZoom = new Map();

const controls = {
  sameTab: document.getElementById('sameTab'),
  closestTab: document.getElementById('closestTab'),
  taskFilter: document.getElementById('taskFilter'),
  scoreFilter: document.getElementById('scoreFilter'),
  searchFilter: document.getElementById('searchFilter'),
  sortSelect: document.getElementById('sortSelect'),
  pageSize: document.getElementById('pageSize'),
  heatmapToggle: document.getElementById('heatmapToggle'),
  prevPage: document.getElementById('prevPage'),
  nextPage: document.getElementById('nextPage'),
};

function escapeText(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]));
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderMetrics() {
  const s = data.summary;
  document.getElementById('metrics').innerHTML = [
    metric('same rows', s.sameRows),
    metric('closest buckets', s.closestBuckets),
    metric('candidate branches', s.closestCandidates),
    metric('max split', s.maxClosestSplit),
    metric('same-neuron buckets', s.closestSameNeuronBuckets),
  ].join('');
}

function renderHeatmaps() {
  document.getElementById('heatmaps').innerHTML = data.heatmaps.map(item => `
    <article class="heatmap">
      <h2>${escapeText(item.task)} score heatmap</h2>
      <img src="${escapeText(item.src)}" alt="${escapeText(item.task)} formula diff score heatmap">
    </article>
  `).join('');
}

function diffSearchText(diff) {
  if (!diff) return '';
  if (diff.kind === 'shared') return diff.formula ?? '';
  if (diff.kind === 'base_only' || diff.kind === 'finetuned_only') return diff.formula ?? '';
  if (diff.kind === 'changed') return `${diff.base ?? ''} ${diff.finetuned ?? ''}`;
  return `${diff.operator ?? ''} ${(diff.children ?? []).map(diffSearchText).join(' ')}`;
}

function rowSearchText(row) {
  if (activeView === 'same') {
    return `${row.task} ${row.neuron} ${row.base_formula} ${row.finetuned_formula} ${diffSearchText(row.diff)}`.toLowerCase();
  }
  const candidates = row.candidates ?? [];
  return `${row.task} ${row.base_neuron} ${row.base_formula} ${candidates.map(candidate => `${candidate.neuron_id} ${candidate.formula} ${diffSearchText(candidate.diff)}`).join(' ')}`.toLowerCase();
}

function rowScore(row) {
  return activeView === 'same' ? row.diff_score : row.lowest_diff_score;
}

function scoreLabel(value) {
  return value ?? 'null';
}

function sortScore(row) {
  return rowScore(row) ?? Number.POSITIVE_INFINITY;
}

function rowNeuron(row) {
  return activeView === 'same' ? row.neuron : row.base_neuron;
}

function rowSplit(row) {
  return activeView === 'same' ? 1 : (row.candidates?.length ?? 0);
}

function rowsForView() {
  const rows = activeView === 'same' ? data.sameRows : data.closestBuckets;
  const task = controls.taskFilter.value;
  const maxScoreText = controls.scoreFilter.value.trim();
  const maxScore = maxScoreText === '' ? null : Number(maxScoreText);
  const search = controls.searchFilter.value.trim().toLowerCase();
  return rows
    .filter(row => {
      if (task && row.task !== task) return false;
      if (maxScore !== null && (rowScore(row) === null || rowScore(row) > maxScore)) return false;
      if (search && !rowSearchText(row).includes(search)) return false;
      return true;
    })
    .sort(sortRows);
}

function sortRows(left, right) {
  const sort = controls.sortSelect.value;
  if (sort === 'neuron') return rowNeuron(left) - rowNeuron(right);
  if (sort === 'split') return rowSplit(right) - rowSplit(left) || sortScore(left) - sortScore(right);
  return sortScore(left) - sortScore(right) || rowSplit(left) - rowSplit(right) || rowNeuron(left) - rowNeuron(right);
}

function renderCards() {
  const rows = rowsForView();
  const pageSize = Number(controls.pageSize.value);
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  page = Math.min(page, pageCount - 1);
  const start = page * pageSize;
  const pageRows = rows.slice(start, start + pageSize);
  document.getElementById('status').textContent = `${rows.length} ${activeView === 'same' ? 'same-neuron rows' : 'closest buckets'} after filters`;
  document.getElementById('pageStatus').textContent = `Page ${page + 1} / ${pageCount}`;
  controls.prevPage.disabled = page === 0;
  controls.nextPage.disabled = page >= pageCount - 1;
  document.getElementById('cards').innerHTML = pageRows
    .map((row, index) => activeView === 'same' ? renderSameCard(row, index) : renderClosestCard(row, index))
    .join('');
  mountVisibleTrees(pageRows);
}

function renderSameCard(row, index) {
  const treeId = `same-${page}-${index}`;
  return `<article class="card">
    <div class="card-head">
      <div class="title">
        <span>${escapeText(row.task)}</span>
        <span class="pill">neuron ${row.neuron}</span>
        <span class="pill score">score ${scoreLabel(row.diff_score)}</span>
      </div>
    </div>
    <div class="card-body">
      <aside class="formula-pane">
        <div class="formula-block">
          <div class="formula-label">Base formula</div>
          <pre class="formula">${escapeText(row.base_formula)}</pre>
        </div>
        <div class="formula-block">
          <div class="formula-label">Fine-tuned formula</div>
          <pre class="formula">${escapeText(row.finetuned_formula)}</pre>
        </div>
      </aside>
      ${renderTreePane(treeId, 'Same-neuron structural diff')}
    </div>
  </article>`;
}

function renderClosestCard(bucket, index) {
  const bucketKey = `${bucket.task}:${bucket.base_neuron}`;
  const candidates = bucket.candidates ?? [];
  const selected = Math.min(selectedCandidate.get(bucketKey) ?? 0, Math.max(0, candidates.length - 1));
  const candidate = candidates[selected] ?? null;
  const treeId = `closest-${page}-${index}`;
  return `<article class="card">
    <div class="card-head">
      <div class="title">
        <span>${escapeText(bucket.task)}</span>
        <span class="pill">base ${bucket.base_neuron}</span>
        <span class="pill score">lowest score ${scoreLabel(bucket.lowest_diff_score)}</span>
        <span class="pill split">${candidates.length} candidate branches</span>
      </div>
    </div>
    <div class="card-body">
      <aside class="formula-pane">
        <div class="formula-block">
          <div class="formula-label">Base formula</div>
          <pre class="formula">${escapeText(bucket.base_formula)}</pre>
        </div>
        <div class="formula-block">
          <div class="formula-label">Candidate branches</div>
          <div class="candidate-branches">
            ${candidates.map((item, candidateIndex) => renderCandidateButton(bucketKey, item, candidateIndex, candidateIndex === selected)).join('')}
          </div>
        </div>
        <div class="formula-block">
          <div class="formula-label">Selected closest formula</div>
          <pre class="formula">${escapeText(candidate?.formula ?? '')}</pre>
        </div>
      </aside>
      ${renderTreePane(treeId, candidate ? `Base formula to fine-tuned candidate ${candidate.neuron_id}` : 'No candidate')}
    </div>
  </article>`;
}

function renderCandidateButton(bucketKey, candidate, index, isActive) {
  return `<button class="candidate-button ${isActive ? 'active' : ''}" type="button" data-bucket="${escapeText(bucketKey)}" data-candidate="${index}">
    <span class="branch">└→ ${escapeText(candidate.neuron_id)}</span>
    <span class="candidate-formula">${escapeText(candidate.formula)}</span>
    <span>${index + 1}</span>
  </button>`;
}

function renderTreePane(treeId, title) {
  return `<section class="tree-pane">
    <div class="tree-toolbar">
      <div class="tree-title">${escapeText(title)}</div>
      <div class="tree-actions">
        <button type="button" data-zoom="${treeId}" data-delta="-0.15">-</button>
        <button type="button" data-zoom="${treeId}" data-delta="0.15">+</button>
      </div>
    </div>
    <div class="tree-wrap" id="${treeId}"></div>
    <pre class="diff-text" id="${treeId}-text"></pre>
  </section>`;
}

function mountVisibleTrees(pageRows) {
  pageRows.forEach((row, index) => {
    const treeId = activeView === 'same' ? `same-${page}-${index}` : `closest-${page}-${index}`;
    let diff = row.diff;
    if (activeView === 'closest') {
      const bucketKey = `${row.task}:${row.base_neuron}`;
      const selected = selectedCandidate.get(bucketKey) ?? 0;
      diff = row.candidates?.[selected]?.diff ?? row.candidates?.[0]?.diff;
    }
    renderDiffTree(document.getElementById(treeId), diff, treeZoom.get(treeId) ?? 1);
    document.getElementById(`${treeId}-text`).innerHTML = renderDiffText(diff);
  });
}

function diffToNode(diff) {
  if (!diff) return {label: 'empty', detail: '', cls: 'operator', children: []};
  if (diff.kind === 'shared') return {label: '= shared', detail: diff.formula ?? '', cls: 'shared', children: []};
  if (diff.kind === 'base_only') return {label: '- base', detail: diff.formula ?? '', cls: 'removed', children: []};
  if (diff.kind === 'finetuned_only') return {label: '+ fine-tuned', detail: diff.formula ?? '', cls: 'added', children: []};
  if (diff.kind === 'changed') return [
    {label: '- base', detail: diff.base ?? '', cls: 'removed', children: []},
    {label: '+ fine-tuned', detail: diff.finetuned ?? '', cls: 'added', children: []},
  ];
  return {
    label: diff.operator ?? diff.kind ?? 'operator',
    detail: '',
    cls: 'operator',
    children: (diff.children ?? []).flatMap(diffToNode),
  };
}

function diffToTree(diff) {
  const nodes = diffToNode(diff);
  if (Array.isArray(nodes)) {
    return {virtual: true, label: '', detail: '', cls: 'operator', children: nodes};
  }
  return nodes;
}

function layoutTree(root) {
  let leaf = 0;
  let maxDepth = 0;
  function visit(node, depth) {
    node.depth = depth;
    if (!node.virtual) maxDepth = Math.max(maxDepth, depth);
    if (!node.children.length) {
      node.ySlot = leaf++;
    } else {
      const childDepth = node.virtual ? depth : depth + 1;
      node.children.forEach(child => visit(child, childDepth));
      node.ySlot = (node.children[0].ySlot + node.children[node.children.length - 1].ySlot) / 2;
    }
  }
  visit(root, 0);
  return {leafCount: Math.max(leaf, 1), maxDepth};
}

function collectNodes(root) {
  const nodes = [];
  const links = [];
  function visit(node) {
    if (!node.virtual) nodes.push(node);
    node.children.forEach(child => {
      if (!node.virtual) links.push([node, child]);
      visit(child);
    });
  }
  visit(root);
  return {nodes, links};
}

function truncate(value, max) {
  const text = String(value ?? '');
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function renderDiffTree(container, diff, zoom) {
  if (!container) return;
  const root = diffToTree(diff);
  const layout = layoutTree(root);
  const {nodes, links} = collectNodes(root);
  const xGap = 230 * zoom;
  const yGap = 74 * zoom;
  const nodeWidth = 170 * zoom;
  const nodeHeight = 46 * zoom;
  const margin = 28 * zoom;
  const width = Math.max(container.clientWidth || 700, margin * 2 + nodeWidth + layout.maxDepth * xGap);
  const height = Math.max(260, margin * 2 + layout.leafCount * yGap);
  for (const node of nodes) {
    node.x = margin + node.depth * xGap;
    node.y = margin + node.ySlot * yGap;
  }
  const linkMarkup = links.map(([source, target]) => {
    const x1 = source.x + nodeWidth;
    const y1 = source.y + nodeHeight / 2;
    const x2 = target.x;
    const y2 = target.y + nodeHeight / 2;
    const mid = (x1 + x2) / 2;
    return `<path class="tree-link" d="M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}"></path>`;
  }).join('');
  const nodeMarkup = nodes.map(node => `
    <g class="tree-node ${escapeText(node.cls)}" transform="translate(${node.x}, ${node.y})">
      <rect width="${nodeWidth}" height="${nodeHeight}" rx="6"></rect>
      <text x="${10 * zoom}" y="${17 * zoom}" font-weight="700">${escapeText(truncate(node.label, 26))}</text>
      <text x="${10 * zoom}" y="${34 * zoom}">${escapeText(truncate(node.detail, 34))}</text>
      <title>${escapeText(node.detail || node.label)}</title>
    </g>
  `).join('');
  container.innerHTML = `<svg class="tree-svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${linkMarkup}${nodeMarkup}</svg>`;
}

function diffTextLines(diff, depth = 0) {
  const pad = '  '.repeat(depth);
  if (!diff) return [];
  if (diff.kind === 'shared') return [`${pad}= ${diff.formula ?? ''}`];
  if (diff.kind === 'base_only') return [`${pad}- base ${diff.formula ?? ''}`];
  if (diff.kind === 'finetuned_only') return [`${pad}+ fine-tuned ${diff.formula ?? ''}`];
  if (diff.kind === 'changed') return [
    `${pad}- base ${diff.base ?? ''}`,
    `${pad}+ fine-tuned ${diff.finetuned ?? ''}`,
  ];
  return [
    `${pad}${diff.operator ?? diff.kind ?? 'operator'}`,
    ...(diff.children ?? []).flatMap(child => diffTextLines(child, depth + 1)),
  ];
}

function renderDiffText(diff) {
  return escapeText(diffTextLines(diff).join('\\n'))
    .split('\\n')
    .map(line => {
      const trimmed = line.trimStart();
      const cls = trimmed.startsWith('-') ? 'minus'
        : trimmed.startsWith('+') ? 'plus'
        : trimmed.startsWith('=') ? 'same'
        : '';
      return cls ? `<span class="${cls}">${line}</span>` : line;
    })
    .join('\\n');
}

function setView(view) {
  activeView = view;
  page = 0;
  controls.sameTab.classList.toggle('active', view === 'same');
  controls.closestTab.classList.toggle('active', view === 'closest');
  renderCards();
}

function populateTasks() {
  const tasks = [...new Set([...data.sameRows, ...data.closestBuckets].map(row => row.task))].sort();
  for (const task of tasks) {
    const option = document.createElement('option');
    option.value = task;
    option.textContent = task;
    controls.taskFilter.appendChild(option);
  }
}

controls.sameTab.addEventListener('click', () => setView('same'));
controls.closestTab.addEventListener('click', () => setView('closest'));
controls.prevPage.addEventListener('click', () => { page -= 1; renderCards(); });
controls.nextPage.addEventListener('click', () => { page += 1; renderCards(); });
controls.heatmapToggle.addEventListener('click', () => {
  document.getElementById('heatmaps').classList.toggle('open');
  controls.heatmapToggle.classList.toggle('active');
});
for (const control of [controls.taskFilter, controls.scoreFilter, controls.searchFilter, controls.sortSelect, controls.pageSize]) {
  control.addEventListener('input', () => { page = 0; renderCards(); });
}
document.getElementById('cards').addEventListener('click', event => {
  const candidateButton = event.target.closest('[data-candidate]');
  if (candidateButton) {
    selectedCandidate.set(candidateButton.dataset.bucket, Number(candidateButton.dataset.candidate));
    renderCards();
    return;
  }
  const zoomButton = event.target.closest('[data-zoom]');
  if (zoomButton) {
    const treeId = zoomButton.dataset.zoom;
    const next = Math.min(1.8, Math.max(0.65, (treeZoom.get(treeId) ?? 1) + Number(zoomButton.dataset.delta)));
    treeZoom.set(treeId, next);
    mountVisibleTrees(rowsForView().slice(page * Number(controls.pageSize.value), (page + 1) * Number(controls.pageSize.value)));
  }
});
window.addEventListener('resize', () => {
  mountVisibleTrees(rowsForView().slice(page * Number(controls.pageSize.value), (page + 1) * Number(controls.pageSize.value)));
});

renderMetrics();
renderHeatmaps();
populateTasks();
renderCards();
</script>
</body>
</html>
"""


def build_viewer(formula_diff_dir: Path, output_path: Path) -> None:
    same_rows = normalize_same_rows(formula_diff_dir)
    closest_buckets = normalize_closest_buckets(formula_diff_dir)
    heatmaps = discover_heatmaps(formula_diff_dir)
    summary = build_summary(same_rows, closest_buckets)
    output_path.write_text(
        render_html(same_rows, closest_buckets, heatmaps, summary),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a static formula-diff inspection viewer."
    )
    parser.add_argument(
        "--formula-diff-dir",
        type=Path,
        default=DEFAULT_FORMULA_DIFF_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to formula_diff_viewer.html inside --formula-diff-dir.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output or args.formula_diff_dir / DEFAULT_OUTPUT_FILENAME
    build_viewer(args.formula_diff_dir, output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

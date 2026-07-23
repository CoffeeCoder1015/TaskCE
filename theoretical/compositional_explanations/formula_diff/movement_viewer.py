from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


DEFAULT_FORMULA_DIFF_DIR = Path("14th") / "results" / "formula_diff"
CLOSEST_BUCKETS_FILENAME = "closest_finetuned_formula_match_buckets.json"
DEFAULT_2D_OUTPUT_FILENAME = "formula_movement_arrows.html"
DEFAULT_3D_OUTPUT_FILENAME = "formula_movement_arrows_3d.html"


def load_movement_buckets(formula_diff_dir: Path) -> dict[str, list[dict[str, object]]]:
    path = formula_diff_dir / CLOSEST_BUCKETS_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def movement_payload(buckets: dict[str, list[dict[str, object]]]) -> str:
    payload = json.dumps(
        {
            "buckets": buckets,
        },
        ensure_ascii=False,
    )
    return html.escape(payload, quote=False)


def render_2d_html(buckets: dict[str, list[dict[str, object]]]) -> str:
    escaped_payload = movement_payload(buckets)
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Formula Movement Arrows 2D</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: rgba(255, 255, 255, 0.92);
  --line: rgba(120, 130, 150, 0.28);
  --ink: #172033;
  --muted: #667085;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  overflow: hidden;
  background: var(--bg);
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
  font-size: 14px;
}
.toolbar {
  position: fixed;
  top: 16px;
  left: 16px;
  z-index: 3;
  display: grid;
  grid-template-columns: 160px 120px;
  gap: 8px;
  padding: 10px;
  width: min(312px, calc(100vw - 32px));
  align-items: end;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}
label {
  display: grid;
  gap: 4px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
select, input {
  min-height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  font: inherit;
  color: var(--ink);
  background: #fff;
}
.status {
  position: fixed;
  right: 18px;
  bottom: 14px;
  z-index: 3;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
}
.canvas-shell {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
}
canvas {
  display: block;
  width: 100vw;
  height: 100vh;
}
.legend {
  position: fixed;
  left: 16px;
  bottom: 14px;
  z-index: 3;
  width: min(320px, calc(100vw - 32px));
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 10px;
  color: var(--muted);
  font-size: 12px;
  backdrop-filter: blur(8px);
}
.legend-title {
  margin-bottom: 6px;
  color: var(--ink);
  font-weight: 700;
}
.legend-bar {
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, #440154, #3b528b, #21918c, #5ec962, #fde725);
  margin-bottom: 5px;
}
.legend-labels {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}
@media (max-width: 760px) {
  .toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .status { left: 16px; right: auto; bottom: 86px; }
}
</style>
</head>
<body>
<section class="toolbar">
  <label>Task
    <select id="taskFilter"></select>
  </label>
  <label>Best %
    <input id="percentFilter" type="number" min="1" max="100" step="1" inputmode="numeric" placeholder="all" value="">
  </label>
</section>
<section class="canvas-shell">
  <canvas id="movementCanvas"></canvas>
</section>
<section class="legend">
  <div class="legend-title">diff_score color scale</div>
  <div class="legend-bar"></div>
  <div class="legend-labels">
    <span id="legendMinScore">score -</span>
    <span id="legendMaxScore">score -</span>
  </div>
</section>
<div class="status" id="status"></div>
<script id="movement-data" type="application/json">__MOVEMENT_PAYLOAD__</script>
<script>
const data = JSON.parse(document.getElementById('movement-data').textContent);
const allRows = expandBuckets(data.buckets);
const scoreDomain = allRows.map(row => row.diff_score);
const SCORE_MIN = scoreDomain.length ? Math.min(...scoreDomain) : 0;
const SCORE_MAX = scoreDomain.length ? Math.max(...scoreDomain) : 1;
const controls = {
  taskFilter: document.getElementById('taskFilter'),
  percentFilter: document.getElementById('percentFilter'),
};
const canvas = document.getElementById('movementCanvas');
const context = canvas.getContext('2d');
const legend = {
  minScore: document.getElementById('legendMinScore'),
  maxScore: document.getElementById('legendMaxScore'),
};
const VIRIDIS_STOPS = [
  [0.00, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.50, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1.00, [253, 231, 37]],
];
const MAX_NEURON = Math.max(2047, ...allRows.flatMap(row => [row.base_neuron, row.finetuned_neuron]));

function expandBuckets(buckets) {
  const rows = [];
  for (const [task, taskBuckets] of Object.entries(buckets)) {
    for (const bucket of taskBuckets) {
      const candidates = Array.isArray(bucket.candidates) ? bucket.candidates : [];
      const candidateCount = Math.max(1, candidates.length);
      for (const candidate of candidates) {
        rows.push({
          task,
          base_neuron: bucket.neuron_id,
          finetuned_neuron: candidate.neuron_id,
          diff_score: bucket.lowest_diff_score,
          candidate_count: candidateCount,
          base_formula: bucket.base_formula,
          finetuned_formula: candidate.formula,
        });
      }
    }
  }
  return rows;
}

function populateTasks() {
  const tasks = Object.keys(data.buckets).sort();
  for (const task of tasks) {
    const option = document.createElement('option');
    option.value = task;
    option.textContent = task;
    controls.taskFilter.appendChild(option);
  }
}

function rowsForTask() {
  return allRows.filter(row => row.task === controls.taskFilter.value);
}

function edgeConfidence(row) {
  return 1 / Math.sqrt(Math.max(1, row.candidate_count));
}

function uncertaintyAlpha(row) {
  return 0.04 + 0.96 * edgeConfidence(row);
}

function visualDistance(row) {
  return row.diff_score + (1 - edgeConfidence(row));
}

function movementSort(left, right) {
  return visualDistance(left) - visualDistance(right)
    || left.base_neuron - right.base_neuron
    || left.finetuned_neuron - right.finetuned_neuron;
}

function filteredRows() {
  const rows = rowsForTask();
  const requestedPercent = Number(controls.percentFilter.value);
  if (!Number.isFinite(requestedPercent) || requestedPercent < 1) {
    return rows.slice().sort((left, right) => left.base_neuron - right.base_neuron);
  }
  const boundedPercent = Math.min(100, requestedPercent);
  const k = Math.max(1, Math.ceil(rows.length * (boundedPercent / 100)));
  return rows
    .slice()
    .sort(movementSort)
    .slice(0, k)
    .sort((left, right) => left.base_neuron - right.base_neuron);
}

function formatScore(score) {
  return Number.isFinite(score) ? String(score) : '-';
}

function renderStatus(rows) {
  const uniqueTargets = new Set(rows.map(row => row.finetuned_neuron)).size;
  const bestScore = rows.length ? Math.min(...rows.map(row => row.diff_score)) : 0;
  const worstScore = rows.length ? Math.max(...rows.map(row => row.diff_score)) : 0;
  const maxSplit = rows.length ? Math.max(...rows.map(row => row.candidate_count)) : 0;
  document.getElementById('status').textContent =
    `${rows.length} arrows | ${uniqueTargets} targets | selected score ${bestScore}-${worstScore} | max split ${maxSplit}`;
}

function renderLegend() {
  legend.minScore.textContent = `score ${formatScore(SCORE_MIN)}`;
  legend.maxScore.textContent = `score ${formatScore(SCORE_MAX)}`;
}

function scoreRatio(score) {
  if (SCORE_MAX <= SCORE_MIN) return 0;
  return (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN);
}

function viridisColor(value) {
  const bounded = Math.max(0, Math.min(1, value));
  for (let index = 1; index < VIRIDIS_STOPS.length; index += 1) {
    const [stop, color] = VIRIDIS_STOPS[index];
    const [previousStop, previousColor] = VIRIDIS_STOPS[index - 1];
    if (bounded <= stop) {
      const span = stop - previousStop || 1;
      const local = (bounded - previousStop) / span;
      const mixed = color.map((channel, channelIndex) =>
        Math.round(previousColor[channelIndex] + (channel - previousColor[channelIndex]) * local)
      );
      return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
    }
  }
  const last = VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1][1];
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}

function styleForRow(row) {
  const ratio = Math.max(0, Math.min(1, scoreRatio(row.diff_score)));
  const similarity = 1 - ratio;
  const confidence = edgeConfidence(row);
  return {
    color: viridisColor(ratio),
    width: 0.25 + similarity * 1.7 * confidence,
    alpha: uncertaintyAlpha(row),
  };
}

function drawArrowhead(ctx, x, y, angle, size, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(-size, -size * 0.45);
  ctx.moveTo(0, 0);
  ctx.lineTo(-size, size * 0.45);
  ctx.stroke();
  ctx.restore();
}

function drawCurve(ctx, x1, y1, x2, y2, color, width, alpha) {
  const controlOffset = Math.max(120, Math.abs(x2 - x1) * 0.18);
  const c1x = x1;
  const c1y = y1 + controlOffset;
  const c2x = x2;
  const c2y = y2 - controlOffset;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.bezierCurveTo(c1x, c1y, c2x, c2y, x2, y2);
  ctx.stroke();
  const angle = Math.atan2(y2 - c2y, x2 - c2x);
  drawArrowhead(ctx, x2, y2, angle, Math.max(3.2, width * 3.2), color);
  ctx.globalAlpha = 1;
}

function renderPlot() {
  const rows = filteredRows();
  renderStatus(rows);
  renderLegend();
  const paddingX = 42;
  const viewportWidth = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
  const viewportHeight = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
  const cssWidth = viewportWidth;
  const cssHeight = viewportHeight;
  const topY = Math.min(132, Math.max(72, cssHeight * 0.22));
  const bottomY = Math.min(cssHeight - 72, Math.max(topY + 160, cssHeight - 96));
  const usableWidth = Math.max(1, cssWidth - paddingX * 2);
  const ratio = window.devicePixelRatio || 1;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.floor(cssWidth * ratio);
  canvas.height = Math.floor(cssHeight * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);

  function xForNeuron(neuron) {
    return paddingX + (neuron / MAX_NEURON) * usableWidth;
  }

  context.strokeStyle = 'rgba(100, 116, 139, 0.16)';
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(paddingX, topY);
  context.lineTo(cssWidth - paddingX, topY);
  context.moveTo(paddingX, bottomY);
  context.lineTo(cssWidth - paddingX, bottomY);
  context.stroke();

  context.font = '700 13px Arial, Helvetica, sans-serif';
  context.textBaseline = 'middle';
  context.fillStyle = 'rgba(124, 45, 18, 0.82)';
  context.textAlign = 'left';
  context.fillText('base', paddingX, topY - 16);
  context.fillStyle = 'rgba(21, 94, 117, 0.82)';
  context.fillText('fine-tuned', paddingX, bottomY + 16);
  context.font = '700 11px Arial, Helvetica, sans-serif';
  context.fillStyle = 'rgba(102, 112, 133, 0.9)';
  context.textAlign = 'left';
  context.fillText('neuron 0', paddingX, topY + 16);
  context.fillText('neuron 0', paddingX, bottomY - 16);
  context.textAlign = 'right';
  context.fillText(`neuron ${MAX_NEURON}`, cssWidth - paddingX, topY + 16);
  context.fillText(`neuron ${MAX_NEURON}`, cssWidth - paddingX, bottomY - 16);

  const drawRows = rows
    .slice()
    .sort((left, right) => visualDistance(right) - visualDistance(left));
  for (const row of drawRows) {
    const style = styleForRow(row);
    drawCurve(context, xForNeuron(row.base_neuron), topY, xForNeuron(row.finetuned_neuron), bottomY, style.color, style.width, style.alpha);
  }

  context.fillStyle = 'rgba(124, 45, 18, 0.55)';
  for (const neuron of new Set(rows.map(row => row.base_neuron))) {
    context.fillRect(xForNeuron(neuron) - 1, topY - 4, 2, 8);
  }
  context.fillStyle = 'rgba(21, 94, 117, 0.55)';
  for (const neuron of new Set(rows.map(row => row.finetuned_neuron))) {
    context.fillRect(xForNeuron(neuron) - 1, bottomY - 4, 2, 8);
  }
}

for (const control of Object.values(controls)) {
  control.addEventListener('input', renderPlot);
}
window.addEventListener('resize', renderPlot);
populateTasks();
renderPlot();
</script>
</body>
</html>
""".replace("__MOVEMENT_PAYLOAD__", escaped_payload)


def render_3d_html(buckets: dict[str, list[dict[str, object]]]) -> str:
    escaped_payload = movement_payload(buckets)
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Formula Movement Arrows</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: rgba(255, 255, 255, 0.92);
  --line: rgba(120, 130, 150, 0.28);
  --ink: #172033;
  --muted: #667085;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  overflow: hidden;
  background: var(--bg);
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
  font-size: 14px;
}
#scene {
  position: fixed;
  inset: 0;
  background: var(--bg);
}
#scene canvas {
  display: block;
  width: 100vw;
  height: 100vh;
}
.toolbar {
  position: fixed;
  top: 16px;
  left: 16px;
  z-index: 3;
  display: grid;
  grid-template-columns: 160px 150px;
  gap: 8px;
  padding: 10px;
  width: min(342px, calc(100vw - 32px));
  align-items: end;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}
label {
  display: grid;
  gap: 4px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
select, input {
  min-height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  font: inherit;
  color: var(--ink);
  background: #fff;
}
.status {
  position: fixed;
  right: 18px;
  bottom: 14px;
  z-index: 3;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
}
.legend {
  position: fixed;
  left: 16px;
  bottom: 14px;
  z-index: 3;
  width: min(320px, calc(100vw - 32px));
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 10px;
  color: var(--muted);
  font-size: 12px;
  backdrop-filter: blur(8px);
}
.legend-title {
  margin-bottom: 6px;
  color: var(--ink);
  font-weight: 700;
}
.legend-bar {
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, #440154, #3b528b, #21918c, #5ec962, #fde725);
  margin-bottom: 5px;
}
.legend-labels {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}
@media (max-width: 760px) {
  .toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .status { left: 16px; right: auto; bottom: 86px; }
}
</style>
</head>
<body>
<section class="toolbar">
  <label>Task
    <select id="taskFilter"></select>
  </label>
  <label>Percentile gap
    <input id="percentileGap" type="number" min="1" max="100" step="1" inputmode="numeric" value="10">
  </label>
</section>
<div id="scene"></div>
<section class="legend">
  <div class="legend-title">diff_score color scale</div>
  <div class="legend-bar"></div>
  <div class="legend-labels">
    <span id="legendMinScore">score -</span>
    <span id="legendMaxScore">score -</span>
  </div>
</section>
<div class="status" id="status"></div>
<script id="movement-data" type="application/json">__MOVEMENT_PAYLOAD__</script>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

const data = JSON.parse(document.getElementById('movement-data').textContent);
const allRows = expandBuckets(data.buckets);
const scoreDomain = allRows.map(row => row.diff_score);
const SCORE_MIN = scoreDomain.length ? Math.min(...scoreDomain) : 0;
const SCORE_MAX = scoreDomain.length ? Math.max(...scoreDomain) : 1;
const controls = {
  taskFilter: document.getElementById('taskFilter'),
  percentileGap: document.getElementById('percentileGap'),
};
const legend = {
  minScore: document.getElementById('legendMinScore'),
  maxScore: document.getElementById('legendMaxScore'),
};
const status = document.getElementById('status');
const sceneHost = document.getElementById('scene');
const VIRIDIS_STOPS = [
  [0.00, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.50, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1.00, [253, 231, 37]],
];
const SCENE_WIDTH = 2200;
const MODEL_GAP = 420;
const LAYER_GAP = 132;
const MAX_NEURON = Math.max(2047, ...allRows.flatMap(row => [row.base_neuron, row.finetuned_neuron]));

const scene = new THREE.Scene();
scene.background = new THREE.Color('#f6f7f9');
const camera = new THREE.PerspectiveCamera(44, 1, 1, 20000);
camera.up.set(0, 1, 0);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
sceneHost.appendChild(renderer.domElement);
const cameraControls = new OrbitControls(camera, renderer.domElement);
cameraControls.enableDamping = true;
cameraControls.dampingFactor = 0.08;
cameraControls.enablePan = true;
cameraControls.screenSpacePanning = true;
cameraControls.rotateSpeed = 0.85;
cameraControls.zoomSpeed = 1.25;
cameraControls.panSpeed = 1.35;

let movementGroup = new THREE.Group();
scene.add(movementGroup);

function expandBuckets(buckets) {
  const rows = [];
  for (const [task, taskBuckets] of Object.entries(buckets)) {
    for (const bucket of taskBuckets) {
      const candidates = Array.isArray(bucket.candidates) ? bucket.candidates : [];
      const candidateCount = Math.max(1, candidates.length);
      for (const candidate of candidates) {
        rows.push({
          task,
          base_neuron: bucket.neuron_id,
          finetuned_neuron: candidate.neuron_id,
          diff_score: bucket.lowest_diff_score,
          candidate_count: candidateCount,
          base_formula: bucket.base_formula,
          finetuned_formula: candidate.formula,
        });
      }
    }
  }
  return rows;
}

function populateTasks() {
  const tasks = Object.keys(data.buckets).sort();
  for (const task of tasks) {
    const option = document.createElement('option');
    option.value = task;
    option.textContent = task;
    controls.taskFilter.appendChild(option);
  }
}

function rowsForTask() {
  return allRows.filter(row => row.task === controls.taskFilter.value);
}

function edgeConfidence(row) {
  return 1 / Math.sqrt(Math.max(1, row.candidate_count));
}

function uncertaintyAlpha(row) {
  return 0.04 + 0.96 * edgeConfidence(row);
}

function visualDistance(row) {
  return row.diff_score + (1 - edgeConfidence(row));
}

function movementSort(left, right) {
  return visualDistance(left) - visualDistance(right)
    || left.base_neuron - right.base_neuron
    || left.finetuned_neuron - right.finetuned_neuron;
}

function percentileGap() {
  const value = Number(controls.percentileGap.value);
  if (!Number.isFinite(value) || value < 1) return 10;
  return Math.min(100, Math.floor(value));
}

function percentileBands(rows, gap) {
  const ordered = rows
    .slice()
    .sort(movementSort);
  const bands = [];
  let startIndex = 0;
  for (let start = 0; start < 100 && startIndex < ordered.length; start += gap) {
    const end = Math.min(100, start + gap);
    const endIndex = end >= 100
      ? ordered.length
      : Math.max(startIndex + 1, Math.ceil(ordered.length * (end / 100)));
    bands.push({
      start,
      end,
      rows: ordered.slice(startIndex, endIndex),
    });
    startIndex = endIndex;
  }
  return bands.filter(band => band.rows.length > 0);
}

function formatScore(score) {
  return Number.isFinite(score) ? String(score) : '-';
}

function renderLegend() {
  legend.minScore.textContent = `score ${formatScore(SCORE_MIN)}`;
  legend.maxScore.textContent = `score ${formatScore(SCORE_MAX)}`;
}

function scoreRatio(score) {
  if (SCORE_MAX <= SCORE_MIN) return 0;
  return (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN);
}

function viridisColor(value) {
  const bounded = Math.max(0, Math.min(1, value));
  for (let index = 1; index < VIRIDIS_STOPS.length; index += 1) {
    const [stop, color] = VIRIDIS_STOPS[index];
    const [previousStop, previousColor] = VIRIDIS_STOPS[index - 1];
    if (bounded <= stop) {
      const span = stop - previousStop || 1;
      const local = (bounded - previousStop) / span;
      const mixed = color.map((channel, channelIndex) =>
        Math.round(previousColor[channelIndex] + (channel - previousColor[channelIndex]) * local)
      );
      return new THREE.Color(mixed[0] / 255, mixed[1] / 255, mixed[2] / 255);
    }
  }
  const last = VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1][1];
  return new THREE.Color(last[0] / 255, last[1] / 255, last[2] / 255);
}

function xForNeuron(neuron) {
  return (neuron / MAX_NEURON) * SCENE_WIDTH - SCENE_WIDTH / 2;
}

function disposeObject(object) {
  object.traverse(child => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) material.dispose();
    }
  });
}

function lineSegments(points, color, name) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  const material = new THREE.LineBasicMaterial({ color });
  const line = new THREE.LineSegments(geometry, material);
  line.name = name;
  return line;
}

function textSprite(text, color) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = 512;
  canvas.height = 160;
  context.font = '700 52px Arial, Helvetica, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.lineWidth = 10;
  context.strokeStyle = 'rgba(246, 247, 249, 0.92)';
  context.fillStyle = color;
  context.strokeText(text, canvas.width / 2, canvas.height / 2);
  context.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.renderOrder = 10;
  sprite.scale.set(190, 60, 1);
  sprite.name = `${text}-axis-label`;
  return sprite;
}

function buildBandGeometry(bands) {
  const rowCount = bands.reduce((total, band) => total + band.rows.length, 0);
  const segmentsPerRow = 3;
  const positions = new Float32Array(rowCount * segmentsPerRow * 2 * 3);
  const colors = new Float32Array(rowCount * segmentsPerRow * 2 * 3);
  const yBase = MODEL_GAP / 2;
  const yFine = -MODEL_GAP / 2;
  const arrowLength = 18;
  const arrowWidth = 8;
  let cursor = 0;

  bands.forEach((band, bandIndex) => {
    const z = ((bands.length - 1) / 2 - bandIndex) * LAYER_GAP;
    for (const row of band.rows) {
      const x1 = xForNeuron(row.base_neuron);
      const x2 = xForNeuron(row.finetuned_neuron);
      const color = viridisColor(scoreRatio(row.diff_score));
      color.multiplyScalar(uncertaintyAlpha(row));
      const tangent = new THREE.Vector2(
        x2 - x1,
        yFine - yBase,
      ).normalize();
      const perpendicular = new THREE.Vector2(-tangent.y, tangent.x);
      const leftX = x2 - tangent.x * arrowLength + perpendicular.x * arrowWidth;
      const leftY = yFine - tangent.y * arrowLength + perpendicular.y * arrowWidth;
      const rightX = x2 - tangent.x * arrowLength - perpendicular.x * arrowWidth;
      const rightY = yFine - tangent.y * arrowLength - perpendicular.y * arrowWidth;
      const vertexOffset = cursor * segmentsPerRow * 6;
      const rowPositions = [
        x1, yBase, z, x2, yFine, z,
        x2, yFine, z, leftX, leftY, z,
        x2, yFine, z, rightX, rightY, z,
      ];
      positions.set(rowPositions, vertexOffset);
      for (let colorOffset = vertexOffset; colorOffset < vertexOffset + rowPositions.length; colorOffset += 3) {
        colors[colorOffset] = color.r;
        colors[colorOffset + 1] = color.g;
        colors[colorOffset + 2] = color.b;
      }
      cursor += 1;
    }
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({ vertexColors: true });
  const lines = new THREE.LineSegments(geometry, material);
  movementGroup.add(lines);
}

function buildLayerGuides(bands) {
  const yBase = MODEL_GAP / 2;
  const yFine = -MODEL_GAP / 2;
  const xLeft = -SCENE_WIDTH / 2;
  const xRight = SCENE_WIDTH / 2;
  const guidePoints = [];
  const baseTickPoints = [];
  const fineTickPoints = [];
  const tickHalfLength = 10;
  bands.forEach((band, bandIndex) => {
    const z = ((bands.length - 1) / 2 - bandIndex) * LAYER_GAP;
    guidePoints.push(xLeft, yBase, z, xRight, yBase, z);
    guidePoints.push(xLeft, yFine, z, xRight, yFine, z);
    guidePoints.push(xLeft, yBase, z, xLeft, yFine, z);
    guidePoints.push(xRight, yBase, z, xRight, yFine, z);

    for (const neuron of new Set(band.rows.map(row => row.base_neuron))) {
      const x = xForNeuron(neuron);
      baseTickPoints.push(x, yBase - tickHalfLength, z, x, yBase + tickHalfLength, z);
    }
    for (const neuron of new Set(band.rows.map(row => row.finetuned_neuron))) {
      const x = xForNeuron(neuron);
      fineTickPoints.push(x, yFine - tickHalfLength, z, x, yFine + tickHalfLength, z);
    }
  });
  movementGroup.add(lineSegments(guidePoints, 0x9aa4b2, 'percentile-band-guides'));
  movementGroup.add(lineSegments(baseTickPoints, 0x7c2d12, 'base-neuron-ticks'));
  movementGroup.add(lineSegments(fineTickPoints, 0x155e75, 'finetuned-neuron-ticks'));

  const zFront = ((bands.length - 1) / 2) * LAYER_GAP + 56;
  const xLabel = xLeft + 110;
  const baseLabel = textSprite('base', '#7c2d12');
  baseLabel.position.set(xLabel, yBase + 46, zFront);
  const fineLabel = textSprite('fine-tuned', '#155e75');
  fineLabel.position.set(xLabel, yFine - 46, zFront);
  const lowNeuronLabel = textSprite('neuron 0', '#667085');
  lowNeuronLabel.position.set(xLeft, 0, zFront + 64);
  const highNeuronLabel = textSprite(`neuron ${MAX_NEURON}`, '#667085');
  highNeuronLabel.position.set(xRight, 0, zFront + 64);
  movementGroup.add(baseLabel, fineLabel, lowNeuronLabel, highNeuronLabel);
}

function fitCamera() {
  const box = new THREE.Box3().setFromObject(movementGroup);
  if (box.isEmpty()) {
    camera.position.set(0, -1100, 900);
    cameraControls.target.set(0, 0, 0);
    cameraControls.update();
    return;
  }
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const aspect = Math.max(0.1, camera.aspect || 1);
  const framedSize = Math.max(size.x / aspect, size.y, size.z, 1);
  const distance = (framedSize / (2 * Math.tan(fov / 2))) * 1.35;
  const viewDirection = new THREE.Vector3(0, -0.28, 1).normalize();
  camera.near = Math.max(1, distance / 200);
  camera.far = distance * 6;
  camera.position.copy(center).addScaledVector(viewDirection, distance);
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  cameraControls.target.copy(center);
  cameraControls.update();
}

function renderScene() {
  disposeObject(movementGroup);
  scene.remove(movementGroup);
  movementGroup = new THREE.Group();
  scene.add(movementGroup);

  const rows = rowsForTask();
  const gap = percentileGap();
  const bands = percentileBands(rows, gap);
  buildLayerGuides(bands);
  buildBandGeometry(bands);
  fitCamera();
  renderLegend();

  const minTaskScore = rows.length ? Math.min(...rows.map(row => row.diff_score)) : 0;
  const maxTaskScore = rows.length ? Math.max(...rows.map(row => row.diff_score)) : 0;
  const maxSplit = rows.length ? Math.max(...rows.map(row => row.candidate_count)) : 0;
  status.textContent =
    `${controls.taskFilter.value} | ${rows.length} paths | ${bands.length} percentile layers | gap ${gap}% | task score ${minTaskScore}-${maxTaskScore} | max split ${maxSplit}`;
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
}

function animate() {
  cameraControls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

for (const control of Object.values(controls)) {
  control.addEventListener('input', renderScene);
}
window.addEventListener('resize', resize);
populateTasks();
resize();
renderScene();
animate();
</script>
</body>
</html>
""".replace("__MOVEMENT_PAYLOAD__", escaped_payload)


def build_viewers(
    formula_diff_dir: Path,
    output_2d_path: Path,
    output_3d_path: Path,
) -> None:
    buckets = load_movement_buckets(formula_diff_dir)
    output_2d_path.write_text(
        render_2d_html(buckets),
        encoding="utf-8",
    )
    output_3d_path.write_text(
        render_3d_html(buckets),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render 2D and 3D base-to-fine-tuned formula movement plots."
    )
    parser.add_argument(
        "--formula-diff-dir",
        type=Path,
        default=DEFAULT_FORMULA_DIFF_DIR,
    )
    parser.add_argument(
        "--output-2d",
        type=Path,
        default=None,
        help="Defaults to formula_movement_arrows.html inside --formula-diff-dir.",
    )
    parser.add_argument(
        "--output-3d",
        type=Path,
        default=None,
        help="Defaults to formula_movement_arrows_3d.html inside --formula-diff-dir.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_2d = args.output_2d or args.formula_diff_dir / DEFAULT_2D_OUTPUT_FILENAME
    output_3d = args.output_3d or args.formula_diff_dir / DEFAULT_3D_OUTPUT_FILENAME
    build_viewers(args.formula_diff_dir, output_2d, output_3d)
    print(f"Wrote {output_2d}")
    print(f"Wrote {output_3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np


RESULTS_DIR = Path("results")
CAPTURED_RESULTS_PATH = RESULTS_DIR / "captured_activations" / "captured_results.pt"
GRAPH_DIR = RESULTS_DIR / "graph"
OUTPUT_2D_FILENAME = "activation_cross_movements.html"
OUTPUT_3D_FILENAME = "activation_cross_movements_3d.html"
OUTPUT_DATA_FILENAME = "activation_cross_movements_data.js"
SCORE_SCALE = 32767


def load_captured_results():
    import torch

    return torch.load(CAPTURED_RESULTS_PATH, map_location="cpu", weights_only=False)


def encoded_score_matrix(matrix) -> str:
    values = np.asarray(matrix, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
    values = np.clip(values, -1.0, 1.0)
    scaled_values = np.rint(values * SCORE_SCALE).astype("<i2", copy=False)
    return base64.b64encode(scaled_values.tobytes(order="C")).decode("ascii")


def activation_cross_payload(captured_results) -> dict[str, dict[str, object]]:
    import torch

    from analysis.activation_diagnostics import (
        raw_activation_correlation_matrix,
        raw_activation_cosine_similarity_matrix,
        validate_same_activation_shape,
    )

    tasks: dict[str, dict[str, object]] = {}
    for task_name, task_results in captured_results.items():
        if "base" not in task_results or "finetuned" not in task_results:
            continue

        base_states = task_results["base"].states
        finetuned_states = task_results["finetuned"].states
        validate_same_activation_shape(base_states, finetuned_states, "base", "finetuned")

        neuron_count = int(base_states.shape[1])
        vectors = torch.cat((base_states, finetuned_states), dim=1)
        pearson_full = raw_activation_correlation_matrix(vectors)
        cosine_full = raw_activation_cosine_similarity_matrix(vectors)
        pearson_cross = pearson_full[:neuron_count, neuron_count:]
        cosine_cross = cosine_full[:neuron_count, neuron_count:]

        tasks[str(task_name)] = {
            "neuron_count": neuron_count,
            "cell_count": neuron_count * neuron_count,
            "encoding": "row-major-i16-base64",
            "pearson": encoded_score_matrix(pearson_cross),
            "cosine": encoded_score_matrix(cosine_cross),
        }

    return {
        "score_scale": SCORE_SCALE,
        "tasks": tasks,
    }


def render_data_js(payload: dict[str, dict[str, object]]) -> str:
    serialized_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"window.ACTIVATION_CROSS_MOVEMENTS={serialized_payload};\n"


def render_2d_html(data_filename: str) -> str:
    escaped_data_filename = html.escape(data_filename, quote=True)
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activation Cross Movements 2D</title>
<style>
:root {
  --bg: #f7f8fb;
  --panel: rgba(255, 255, 255, 0.93);
  --line: rgba(112, 122, 142, 0.28);
  --ink: #182033;
  --muted: #697386;
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
  grid-template-columns: 150px 132px 132px;
  gap: 8px;
  padding: 10px;
  width: min(438px, calc(100vw - 32px));
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
.canvas-shell {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}
canvas {
  display: block;
  width: 100vw;
  height: 100vh;
}
.legend, .status {
  position: fixed;
  z-index: 3;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--muted);
  backdrop-filter: blur(8px);
}
.legend {
  left: 16px;
  bottom: 14px;
  width: min(330px, calc(100vw - 32px));
  padding: 8px 10px;
  font-size: 12px;
}
.status {
  right: 18px;
  bottom: 14px;
  padding: 6px 8px;
  font-size: 12px;
}
.legend-title {
  margin-bottom: 6px;
  color: var(--ink);
  font-weight: 700;
}
.legend-bar {
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, #b40426, #f7f7f7, #3b4cc0);
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
  <label>Metric
    <select id="metricFilter">
      <option value="pearson">Pearson</option>
      <option value="cosine">Cosine</option>
    </select>
  </label>
  <label>Percentile gap
    <input id="percentileGap" type="number" min="1" max="100" step="1" inputmode="numeric" value="10">
  </label>
</section>
<section class="canvas-shell">
  <canvas id="movementCanvas"></canvas>
</section>
<section class="legend">
  <div class="legend-title">coolwarm relationship scale</div>
  <div class="legend-bar"></div>
  <div class="legend-labels">
    <span>strong negative</span>
    <span>zero</span>
    <span>strong positive</span>
  </div>
</section>
<div class="status" id="status"></div>
<script src="__DATA_FILENAME__"></script>
<script>
const payload = window.ACTIVATION_CROSS_MOVEMENTS;
const controls = {
  taskFilter: document.getElementById('taskFilter'),
  metricFilter: document.getElementById('metricFilter'),
  percentileGap: document.getElementById('percentileGap'),
};
const canvas = document.getElementById('movementCanvas');
const context = canvas.getContext('2d');
const status = document.getElementById('status');
const SCORE_SCALE = payload.score_scale || 32767;
const COOLWARM_STOPS = [
  [0.00, [180, 4, 38]],
  [0.25, [244, 165, 130]],
  [0.50, [247, 247, 247]],
  [0.75, [146, 197, 222]],
  [1.00, [59, 76, 192]],
];

function populateTasks() {
  const tasks = Object.keys(payload.tasks).sort();
  for (const task of tasks) {
    const option = document.createElement('option');
    option.value = task;
    option.textContent = task;
    controls.taskFilter.appendChild(option);
  }
}

function selectedTask() {
  return payload.tasks[controls.taskFilter.value] || { neuron_count: 0, cell_count: 0 };
}

function decodeInt16Scores(encoded, expectedLength) {
  const binary = atob(encoded || '');
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  const scores = new Int16Array(bytes.buffer);
  if (scores.length !== expectedLength) {
    throw new Error(`decoded ${scores.length} scores, expected ${expectedLength}`);
  }
  return scores;
}

function metricData(task) {
  const metric = controls.metricFilter.value;
  const cacheKey = `_${metric}_data`;
  if (!task[cacheKey]) {
    const cellCount = task.cell_count || task.neuron_count * task.neuron_count;
    const scores = decodeInt16Scores(task[metric], cellCount);
    let maxAbsInt = 0;
    let nonZeroCount = 0;
    for (const value of scores) {
      const absValue = Math.abs(value);
      if (absValue > maxAbsInt) maxAbsInt = absValue;
      if (value !== 0) nonZeroCount += 1;
    }
    task[cacheKey] = {
      scores,
      maxAbsInt: Math.max(1, maxAbsInt),
      nonZeroCount,
      orderedIndexes: null,
    };
  }
  return task[cacheKey];
}

function baseNeuron(task, index) {
  return Math.floor(index / task.neuron_count);
}

function finetunedNeuron(task, index) {
  return index % task.neuron_count;
}

function scoreValue(data, index) {
  return data.scores[index] / SCORE_SCALE;
}

function relationshipStrength(data, index) {
  return Math.abs(data.scores[index]);
}

function sortedIndexes(task, data) {
  if (!data.orderedIndexes) {
    const ordered = new Uint32Array(data.scores.length);
    for (let index = 0; index < ordered.length; index += 1) {
      ordered[index] = index;
    }
    ordered.sort((left, right) =>
      relationshipStrength(data, right) - relationshipStrength(data, left)
      || baseNeuron(task, left) - baseNeuron(task, right)
      || finetunedNeuron(task, left) - finetunedNeuron(task, right)
    );
    data.orderedIndexes = ordered;
  }
  return data.orderedIndexes;
}

function percentileGap() {
  const value = Number(controls.percentileGap.value);
  if (!Number.isFinite(value) || value < 1) return 10;
  return Math.min(100, Math.floor(value));
}

function percentileBands(task, data, gap) {
  const ordered = sortedIndexes(task, data);
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
      startIndex,
      endIndex,
    });
    startIndex = endIndex;
  }
  return bands.filter(band => band.endIndex > band.startIndex);
}

function mixColor(left, right, local) {
  return left.map((channel, index) =>
    Math.round(channel + (right[index] - channel) * local)
  );
}

function coolwarm(scoreValue, maxAbs) {
  const ratio = 0.5 + 0.5 * (scoreValue / maxAbs);
  const bounded = Math.max(0, Math.min(1, ratio));
  for (let index = 1; index < COOLWARM_STOPS.length; index += 1) {
    const [stop, color] = COOLWARM_STOPS[index];
    const [previousStop, previousColor] = COOLWARM_STOPS[index - 1];
    if (bounded <= stop) {
      const span = stop - previousStop || 1;
      const local = (bounded - previousStop) / span;
      const mixed = mixColor(previousColor, color, local);
      return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
    }
  }
  const last = COOLWARM_STOPS[COOLWARM_STOPS.length - 1][1];
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
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
  const controlOffset = Math.max(48, Math.abs(y2 - y1) * 0.75);
  const c1y = y1 + (y2 > y1 ? controlOffset : -controlOffset);
  const c2y = y2 - (y2 > y1 ? controlOffset : -controlOffset);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.bezierCurveTo(x1, c1y, x2, c2y, x2, y2);
  ctx.stroke();
  const angle = Math.atan2(y2 - c2y, 0);
  drawArrowhead(ctx, x2, y2, angle, Math.max(3.5, width * 3.2), color);
  ctx.restore();
}

function renderStatus(task, data, bands) {
  const nonZero = data.nonZeroCount;
  const zero = data.scores.length - nonZero;
  status.textContent =
    `${controls.taskFilter.value} | ${task.neuron_count} neurons | ${nonZero} arrows | ${zero} zero marks | ${bands.length} percentile layers`;
}

function renderPlot() {
  const task = selectedTask();
  const data = metricData(task);
  const bands = percentileBands(task, data, percentileGap());
  renderStatus(task, data, bands);

  const viewportWidth = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
  const viewportHeight = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
  const ratio = window.devicePixelRatio || 1;
  const cssWidth = viewportWidth;
  const cssHeight = viewportHeight;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.floor(cssWidth * ratio);
  canvas.height = Math.floor(cssHeight * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);

  const paddingX = 44;
  const usableWidth = Math.max(1, cssWidth - paddingX * 2);
  const centerY = cssHeight * 0.5;
  const layerSpacing = Math.min(38, Math.max(16, cssHeight * 0.28 / Math.max(1, bands.length)));
  const maxAbs = data.maxAbsInt / SCORE_SCALE;
  const maxNeuron = Math.max(1, task.neuron_count - 1);
  const xForNeuron = neuron => paddingX + (neuron / maxNeuron) * usableWidth;

  context.strokeStyle = 'rgba(93, 105, 126, 0.36)';
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(paddingX, centerY);
  context.lineTo(cssWidth - paddingX, centerY);
  context.stroke();

  context.font = '700 13px Arial, Helvetica, sans-serif';
  context.textBaseline = 'middle';
  context.fillStyle = 'rgba(24, 32, 51, 0.82)';
  context.textAlign = 'left';
  context.fillText('base rail', paddingX, centerY - 18);
  context.fillStyle = 'rgba(59, 76, 192, 0.82)';
  context.fillText('positive fine-tuned direction', paddingX, centerY - layerSpacing * Math.max(1, bands.length) - 18);
  context.fillStyle = 'rgba(180, 4, 38, 0.82)';
  context.fillText('negative fine-tuned direction', paddingX, centerY + layerSpacing * Math.max(1, bands.length) + 18);

  const ordered = sortedIndexes(task, data);
  for (let bandIndex = bands.length - 1; bandIndex >= 0; bandIndex -= 1) {
    const band = bands[bandIndex];
    for (let orderIndex = band.endIndex - 1; orderIndex >= band.startIndex; orderIndex -= 1) {
      const index = ordered[orderIndex];
      const rowScore = scoreValue(data, index);
      const x1 = xForNeuron(baseNeuron(task, index));
      const x2 = xForNeuron(finetunedNeuron(task, index));
      if (rowScore === 0) {
        context.save();
        context.globalAlpha = 0.32;
        context.fillStyle = 'rgb(148, 154, 166)';
        context.beginPath();
        context.arc((x1 + x2) / 2, centerY, 2.4, 0, Math.PI * 2);
        context.fill();
        context.restore();
        continue;
      }
      const magnitude = Math.abs(rowScore);
      const normalized = magnitude / maxAbs;
      const direction = rowScore > 0 ? -1 : 1;
      const y2 = centerY + direction * layerSpacing * (bandIndex + 1);
      const color = coolwarm(rowScore, maxAbs);
      const width = 0.25 + normalized * 2.1;
      const alpha = 0.05 + normalized * 0.68;
      drawCurve(context, x1, centerY, x2, y2, color, width, alpha);
    }
  }

  context.fillStyle = 'rgba(24, 32, 51, 0.55)';
  for (let neuron = 0; neuron < task.neuron_count; neuron += Math.max(1, Math.ceil(task.neuron_count / 300))) {
    const x = xForNeuron(neuron);
    context.fillRect(x - 0.5, centerY - 3, 1, 6);
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
""".replace("__DATA_FILENAME__", escaped_data_filename)


def render_3d_html(data_filename: str) -> str:
    escaped_data_filename = html.escape(data_filename, quote=True)
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activation Cross Movements 3D</title>
<style>
:root {
  --bg: #f7f8fb;
  --panel: rgba(255, 255, 255, 0.93);
  --line: rgba(112, 122, 142, 0.28);
  --ink: #182033;
  --muted: #697386;
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
  grid-template-columns: 150px 132px 132px;
  gap: 8px;
  padding: 10px;
  width: min(438px, calc(100vw - 32px));
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
.legend, .status {
  position: fixed;
  z-index: 3;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--muted);
  backdrop-filter: blur(8px);
}
.legend {
  left: 16px;
  bottom: 14px;
  width: min(330px, calc(100vw - 32px));
  padding: 8px 10px;
  font-size: 12px;
}
.status {
  right: 18px;
  bottom: 14px;
  padding: 6px 8px;
  font-size: 12px;
}
.legend-title {
  margin-bottom: 6px;
  color: var(--ink);
  font-weight: 700;
}
.legend-bar {
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, #b40426, #f7f7f7, #3b4cc0);
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
  <label>Metric
    <select id="metricFilter">
      <option value="pearson">Pearson</option>
      <option value="cosine">Cosine</option>
    </select>
  </label>
  <label>Percentile gap
    <input id="percentileGap" type="number" min="1" max="100" step="1" inputmode="numeric" value="10">
  </label>
</section>
<div id="scene"></div>
<section class="legend">
  <div class="legend-title">coolwarm relationship scale</div>
  <div class="legend-bar"></div>
  <div class="legend-labels">
    <span>strong negative</span>
    <span>zero</span>
    <span>strong positive</span>
  </div>
</section>
<div class="status" id="status"></div>
<script src="__DATA_FILENAME__"></script>
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

const payload = window.ACTIVATION_CROSS_MOVEMENTS;
const controls = {
  taskFilter: document.getElementById('taskFilter'),
  metricFilter: document.getElementById('metricFilter'),
  percentileGap: document.getElementById('percentileGap'),
};
const sceneHost = document.getElementById('scene');
const status = document.getElementById('status');
const SCORE_SCALE = payload.score_scale || 32767;
const COOLWARM_STOPS = [
  [0.00, [180, 4, 38]],
  [0.25, [244, 165, 130]],
  [0.50, [247, 247, 247]],
  [0.75, [146, 197, 222]],
  [1.00, [59, 76, 192]],
];
const SCENE_WIDTH = 2200;
const DIRECTION_GAP = 420;
const LAYER_GAP = 132;

const scene = new THREE.Scene();
scene.background = new THREE.Color('#f7f8fb');
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

function populateTasks() {
  const tasks = Object.keys(payload.tasks).sort();
  for (const task of tasks) {
    const option = document.createElement('option');
    option.value = task;
    option.textContent = task;
    controls.taskFilter.appendChild(option);
  }
}

function selectedTask() {
  return payload.tasks[controls.taskFilter.value] || { neuron_count: 0, cell_count: 0 };
}

function decodeInt16Scores(encoded, expectedLength) {
  const binary = atob(encoded || '');
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  const scores = new Int16Array(bytes.buffer);
  if (scores.length !== expectedLength) {
    throw new Error(`decoded ${scores.length} scores, expected ${expectedLength}`);
  }
  return scores;
}

function metricData(task) {
  const metric = controls.metricFilter.value;
  const cacheKey = `_${metric}_data`;
  if (!task[cacheKey]) {
    const cellCount = task.cell_count || task.neuron_count * task.neuron_count;
    const scores = decodeInt16Scores(task[metric], cellCount);
    let maxAbsInt = 0;
    let nonZeroCount = 0;
    for (const value of scores) {
      const absValue = Math.abs(value);
      if (absValue > maxAbsInt) maxAbsInt = absValue;
      if (value !== 0) nonZeroCount += 1;
    }
    task[cacheKey] = {
      scores,
      maxAbsInt: Math.max(1, maxAbsInt),
      nonZeroCount,
      orderedIndexes: null,
    };
  }
  return task[cacheKey];
}

function baseNeuron(task, index) {
  return Math.floor(index / task.neuron_count);
}

function finetunedNeuron(task, index) {
  return index % task.neuron_count;
}

function scoreValue(data, index) {
  return data.scores[index] / SCORE_SCALE;
}

function relationshipStrength(data, index) {
  return Math.abs(data.scores[index]);
}

function sortedIndexes(task, data) {
  if (!data.orderedIndexes) {
    const ordered = new Uint32Array(data.scores.length);
    for (let index = 0; index < ordered.length; index += 1) {
      ordered[index] = index;
    }
    ordered.sort((left, right) =>
      relationshipStrength(data, right) - relationshipStrength(data, left)
      || baseNeuron(task, left) - baseNeuron(task, right)
      || finetunedNeuron(task, left) - finetunedNeuron(task, right)
    );
    data.orderedIndexes = ordered;
  }
  return data.orderedIndexes;
}

function percentileGap() {
  const value = Number(controls.percentileGap.value);
  if (!Number.isFinite(value) || value < 1) return 10;
  return Math.min(100, Math.floor(value));
}

function percentileBands(task, data, gap) {
  const ordered = sortedIndexes(task, data);
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
      startIndex,
      endIndex,
    });
    startIndex = endIndex;
  }
  return bands.filter(band => band.endIndex > band.startIndex);
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

function mixColor(left, right, local) {
  return left.map((channel, index) =>
    Math.round(channel + (right[index] - channel) * local)
  );
}

function coolwarm(scoreValue, maxAbs) {
  const ratio = 0.5 + 0.5 * (scoreValue / maxAbs);
  const bounded = Math.max(0, Math.min(1, ratio));
  for (let index = 1; index < COOLWARM_STOPS.length; index += 1) {
    const [stop, color] = COOLWARM_STOPS[index];
    const [previousStop, previousColor] = COOLWARM_STOPS[index - 1];
    if (bounded <= stop) {
      const span = stop - previousStop || 1;
      const local = (bounded - previousStop) / span;
      const mixed = mixColor(previousColor, color, local);
      return new THREE.Color(mixed[0] / 255, mixed[1] / 255, mixed[2] / 255);
    }
  }
  const last = COOLWARM_STOPS[COOLWARM_STOPS.length - 1][1];
  return new THREE.Color(last[0] / 255, last[1] / 255, last[2] / 255);
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
  context.strokeStyle = 'rgba(247, 248, 251, 0.92)';
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
  return sprite;
}

function xForNeuron(neuron, neuronCount) {
  const maxNeuron = Math.max(1, neuronCount - 1);
  return (neuron / maxNeuron) * SCENE_WIDTH - SCENE_WIDTH / 2;
}

function buildMovementGeometry(bands, task, data) {
  const rowCount = bands.reduce((total, band) => total + band.endIndex - band.startIndex, 0);
  const nonZeroCount = data.nonZeroCount;
  const zeroCount = rowCount - nonZeroCount;
  const segmentsPerArrow = 3;
  const positions = new Float32Array((nonZeroCount * segmentsPerArrow + zeroCount * 2) * 2 * 3);
  const colors = new Float32Array(positions.length);
  const maxAbs = data.maxAbsInt / SCORE_SCALE;
  const arrowLength = 18;
  const arrowWidth = 8;
  let cursor = 0;

  function pushSegment(x1, y1, z1, x2, y2, z2, color, alpha) {
    positions.set([x1, y1, z1, x2, y2, z2], cursor);
    const shaded = color.clone().multiplyScalar(alpha);
    for (let offset = cursor; offset < cursor + 6; offset += 3) {
      colors[offset] = shaded.r;
      colors[offset + 1] = shaded.g;
      colors[offset + 2] = shaded.b;
    }
    cursor += 6;
  }

  bands.forEach((band, bandIndex) => {
    const z = ((bands.length - 1) / 2 - bandIndex) * LAYER_GAP;
    const ordered = sortedIndexes(task, data);
    for (let orderIndex = band.startIndex; orderIndex < band.endIndex; orderIndex += 1) {
      const index = ordered[orderIndex];
      const rowScore = scoreValue(data, index);
      const x1 = xForNeuron(baseNeuron(task, index), task.neuron_count);
      const x2 = xForNeuron(finetunedNeuron(task, index), task.neuron_count);
      if (rowScore === 0) {
        const color = new THREE.Color('#949aa6');
        pushSegment(x2 - 7, 0, z, x2 + 7, 0, z, color, 0.48);
        pushSegment(x2, -7, z, x2, 7, z, color, 0.48);
        continue;
      }

      const direction = rowScore > 0 ? 1 : -1;
      const y1 = 0;
      const y2 = direction * DIRECTION_GAP;
      const color = coolwarm(rowScore, maxAbs);
      const alpha = 0.08 + 0.92 * (Math.abs(rowScore) / maxAbs);
      const tangent = new THREE.Vector2(x2 - x1, y2 - y1).normalize();
      const perpendicular = new THREE.Vector2(-tangent.y, tangent.x);
      const leftX = x2 - tangent.x * arrowLength + perpendicular.x * arrowWidth;
      const leftY = y2 - tangent.y * arrowLength + perpendicular.y * arrowWidth;
      const rightX = x2 - tangent.x * arrowLength - perpendicular.x * arrowWidth;
      const rightY = y2 - tangent.y * arrowLength - perpendicular.y * arrowWidth;

      pushSegment(x1, y1, z, x2, y2, z, color, alpha);
      pushSegment(x2, y2, z, leftX, leftY, z, color, alpha);
      pushSegment(x2, y2, z, rightX, rightY, z, color, alpha);
    }
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions.subarray(0, cursor), 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors.subarray(0, cursor), 3));
  const material = new THREE.LineBasicMaterial({ vertexColors: true });
  movementGroup.add(new THREE.LineSegments(geometry, material));
}

function buildLayerGuides(bands, task) {
  const xLeft = -SCENE_WIDTH / 2;
  const xRight = SCENE_WIDTH / 2;
  const guidePoints = [];
  const tickPoints = [];
  const tickHalfLength = 8;
  bands.forEach((band, bandIndex) => {
    const z = ((bands.length - 1) / 2 - bandIndex) * LAYER_GAP;
    guidePoints.push(xLeft, 0, z, xRight, 0, z);
    guidePoints.push(xLeft, DIRECTION_GAP, z, xRight, DIRECTION_GAP, z);
    guidePoints.push(xLeft, -DIRECTION_GAP, z, xRight, -DIRECTION_GAP, z);
    guidePoints.push(xLeft, -DIRECTION_GAP, z, xLeft, DIRECTION_GAP, z);
    guidePoints.push(xRight, -DIRECTION_GAP, z, xRight, DIRECTION_GAP, z);

    const neuronStep = Math.max(1, Math.ceil(task.neuron_count / 300));
    for (let neuron = 0; neuron < task.neuron_count; neuron += neuronStep) {
      const x = xForNeuron(neuron, task.neuron_count);
      tickPoints.push(x, -tickHalfLength, z, x, tickHalfLength, z);
    }
  });
  movementGroup.add(lineSegments(guidePoints, 0x9aa4b2, 'percentile-layer-guides'));
  movementGroup.add(lineSegments(tickPoints, 0x182033, 'base-neuron-ticks'));

  const zFront = ((bands.length - 1) / 2) * LAYER_GAP + 56;
  const xLabel = xLeft + 120;
  const baseLabel = textSprite('base', '#182033');
  baseLabel.position.set(xLabel, 0, zFront);
  const positiveLabel = textSprite('positive', '#3b4cc0');
  positiveLabel.position.set(xLabel, DIRECTION_GAP + 46, zFront);
  const negativeLabel = textSprite('negative', '#b40426');
  negativeLabel.position.set(xLabel, -DIRECTION_GAP - 46, zFront);
  const lowNeuronLabel = textSprite('neuron 0', '#697386');
  lowNeuronLabel.position.set(xLeft, 0, zFront + 64);
  const highNeuronLabel = textSprite(`neuron ${Math.max(0, task.neuron_count - 1)}`, '#697386');
  highNeuronLabel.position.set(xRight, 0, zFront + 64);
  movementGroup.add(baseLabel, positiveLabel, negativeLabel, lowNeuronLabel, highNeuronLabel);
}

function fitCamera() {
  const box = new THREE.Box3().setFromObject(movementGroup);
  if (box.isEmpty()) {
    camera.position.set(0, -1200, 950);
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
  const viewDirection = new THREE.Vector3(0, -0.35, 1).normalize();
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

  const task = selectedTask();
  const data = metricData(task);
  const gap = percentileGap();
  const bands = percentileBands(task, data, gap);
  buildLayerGuides(bands, task);
  buildMovementGeometry(bands, task, data);
  fitCamera();

  const nonZero = data.nonZeroCount;
  const zero = data.scores.length - nonZero;
  status.textContent =
    `${controls.taskFilter.value} | ${task.neuron_count} neurons | ${nonZero} arrows | ${zero} zero marks | ${bands.length} percentile layers | gap ${gap}%`;
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
""".replace("__DATA_FILENAME__", escaped_data_filename)


def build_viewers() -> tuple[Path, Path, Path]:
    captured_results = load_captured_results()
    payload = activation_cross_payload(captured_results)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    output_2d_path = GRAPH_DIR / OUTPUT_2D_FILENAME
    output_3d_path = GRAPH_DIR / OUTPUT_3D_FILENAME
    output_data_path = GRAPH_DIR / OUTPUT_DATA_FILENAME
    output_data_path.write_text(render_data_js(payload), encoding="utf-8")
    output_2d_path.write_text(render_2d_html(OUTPUT_DATA_FILENAME), encoding="utf-8")
    output_3d_path.write_text(render_3d_html(OUTPUT_DATA_FILENAME), encoding="utf-8")
    return output_2d_path, output_3d_path, output_data_path


def main() -> int:
    output_2d_path, output_3d_path, output_data_path = build_viewers()
    print(f"Wrote {output_2d_path}")
    print(f"Wrote {output_3d_path}")
    print(f"Wrote {output_data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

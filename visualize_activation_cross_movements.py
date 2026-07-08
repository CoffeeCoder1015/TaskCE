from __future__ import annotations

import html
import json
from pathlib import Path


RESULTS_DIR = Path("results")
CAPTURED_RESULTS_PATH = RESULTS_DIR / "captured_activations" / "captured_results.pt"
GRAPH_DIR = RESULTS_DIR / "graph"
OUTPUT_TENSOR_FILENAME = "activation_cross_movements.pt"
OUTPUT_2D_HTML_FILENAME = "activation_cross_movements.html"
OUTPUT_3D_HTML_FILENAME = "activation_cross_movements_3d.html"
DEFAULT_TOP_BAND_EPSILON = 0.01
DEFAULT_MAX_BUCKET_ARROWS = 8
METRICS = ("pearson", "cosine")
RELATIONSHIPS = ("positive", "negative")


def load_captured_results():
    import torch

    return torch.load(CAPTURED_RESULTS_PATH, map_location="cpu", weights_only=False)


def activation_cross_task_matrices(captured_results) -> dict[str, dict[str, object]]:
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

        tasks[str(task_name)] = {
            "neuron_count": neuron_count,
            "pearson": pearson_full[:neuron_count, neuron_count:],
            "cosine": cosine_full[:neuron_count, neuron_count:],
        }

    return tasks


def write_activation_cross_tensors(
    task_matrices: dict[str, dict[str, object]],
    tensor_path: Path,
) -> None:
    import torch

    payload = {
        "format": "activation_cross_movements",
        "dtype": "float32",
        "tasks": {},
    }

    for task_name in sorted(task_matrices):
        task_payload = task_matrices[task_name]
        payload["tasks"][task_name] = {
            "neuron_count": int(task_payload["neuron_count"]),
            "pearson": torch.as_tensor(task_payload["pearson"], dtype=torch.float32).contiguous(),
            "cosine": torch.as_tensor(task_payload["cosine"], dtype=torch.float32).contiguous(),
        }

    torch.save(payload, tensor_path)


def representative_indices(indices, limit: int) -> list[int]:
    indices = sorted(int(index) for index in indices)
    if limit < 1:
        return []
    if len(indices) <= limit:
        return indices
    if limit == 1:
        return [indices[0]]

    return [
        indices[round(position * (len(indices) - 1) / (limit - 1))]
        for position in range(limit)
    ]


def activation_correspondence_payload(
    task_matrices: dict[str, dict[str, object]],
    *,
    top_band_epsilon: float = DEFAULT_TOP_BAND_EPSILON,
    max_bucket_arrows: int = DEFAULT_MAX_BUCKET_ARROWS,
) -> dict[str, object]:
    import numpy as np

    payload: dict[str, object] = {
        "format": "activation_cross_correspondence_viewer",
        "top_band_epsilon": float(top_band_epsilon),
        "max_bucket_arrows": int(max_bucket_arrows),
        "metrics": list(METRICS),
        "relationships": list(RELATIONSHIPS),
        "tasks": {},
    }

    for task_name in sorted(task_matrices):
        task_payload = task_matrices[task_name]
        task_rows: list[dict[str, object]] = []

        for metric in METRICS:
            matrix = np.asarray(task_payload[metric], dtype=float)
            if matrix.ndim != 2:
                raise ValueError(f"{task_name} {metric} matrix must be 2D, got shape {matrix.shape}")

            for base_neuron, scores in enumerate(matrix):
                finite_mask = np.isfinite(scores)
                if not finite_mask.any():
                    continue

                finite_scores = scores[finite_mask]
                best_positive = float(finite_scores.max())
                if best_positive > 0.0:
                    bucket_mask = finite_mask & (scores >= best_positive - float(top_band_epsilon))
                    add_correspondence_rows(
                        task_rows,
                        metric=metric,
                        relationship="positive",
                        base_neuron=base_neuron,
                        scores=scores,
                        bucket_indices=np.flatnonzero(bucket_mask),
                        bucket_strength=best_positive,
                        max_bucket_arrows=max_bucket_arrows,
                    )

                best_negative = float(finite_scores.min())
                if best_negative < 0.0:
                    bucket_mask = finite_mask & (scores <= best_negative + float(top_band_epsilon))
                    add_correspondence_rows(
                        task_rows,
                        metric=metric,
                        relationship="negative",
                        base_neuron=base_neuron,
                        scores=scores,
                        bucket_indices=np.flatnonzero(bucket_mask),
                        bucket_strength=abs(best_negative),
                        max_bucket_arrows=max_bucket_arrows,
                    )

        payload["tasks"][task_name] = {
            "neuron_count": int(task_payload["neuron_count"]),
            "rows": task_rows,
        }

    return payload


def add_correspondence_rows(
    rows: list[dict[str, object]],
    *,
    metric: str,
    relationship: str,
    base_neuron: int,
    scores,
    bucket_indices,
    bucket_strength: float,
    max_bucket_arrows: int,
) -> None:
    bucket_count = int(len(bucket_indices))
    if bucket_count == 0:
        return

    rendered_indices = representative_indices(bucket_indices, max_bucket_arrows)
    bucket_scores = [float(scores[index]) for index in bucket_indices]
    bucket_mean_strength = sum(abs(score) for score in bucket_scores) / bucket_count

    for finetuned_neuron in rendered_indices:
        score = float(scores[finetuned_neuron])
        rows.append(
            {
                "metric": metric,
                "relationship": relationship,
                "base_neuron": int(base_neuron),
                "finetuned_neuron": int(finetuned_neuron),
                "score": score,
                "strength": abs(score),
                "bucket_strength": float(bucket_strength),
                "bucket_mean_strength": float(bucket_mean_strength),
                "bucket_count": bucket_count,
                "rendered_count": len(rendered_indices),
            }
        )


def correspondence_payload_json(payload: dict[str, object]) -> str:
    return html.escape(json.dumps(payload, ensure_ascii=False), quote=False)


def render_2d_html(payload: dict[str, object]) -> str:
    escaped_payload = correspondence_payload_json(payload)
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activation Cross-Movement Arrows</title>
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
  grid-template-columns: 150px 130px 150px 110px;
  gap: 8px;
  padding: 10px;
  width: min(572px, calc(100vw - 32px));
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
  max-width: min(760px, calc(100vw - 36px));
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
  width: min(340px, calc(100vw - 32px));
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
  margin-bottom: 5px;
}
.legend-bar.positive { background: linear-gradient(90deg, #dbeafe, #60a5fa, #1d4ed8, #172554); }
.legend-bar.negative { background: linear-gradient(90deg, #ffedd5, #fb923c, #dc2626, #7f1d1d); }
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
    <select id="metricFilter"></select>
  </label>
  <label>Relationship
    <select id="relationshipFilter">
      <option value="positive">Similar activation</option>
      <option value="negative">Opposite activation</option>
    </select>
  </label>
  <label>Best %
    <input id="percentFilter" type="number" min="1" max="100" step="1" inputmode="numeric" value="10">
  </label>
</section>
<section class="canvas-shell">
  <canvas id="movementCanvas"></canvas>
</section>
<section class="legend">
  <div class="legend-title" id="legendTitle">strength color scale</div>
  <div class="legend-bar" id="legendBar"></div>
  <div class="legend-labels">
    <span>weak</span>
    <span>strong</span>
  </div>
</section>
<div class="status" id="status"></div>
<script id="correspondence-data" type="application/json">__CORRESPONDENCE_PAYLOAD__</script>
<script>
const data = JSON.parse(document.getElementById('correspondence-data').textContent);
const controls = {
  taskFilter: document.getElementById('taskFilter'),
  metricFilter: document.getElementById('metricFilter'),
  relationshipFilter: document.getElementById('relationshipFilter'),
  percentFilter: document.getElementById('percentFilter'),
};
const canvas = document.getElementById('movementCanvas');
const context = canvas.getContext('2d');
const status = document.getElementById('status');
const legendTitle = document.getElementById('legendTitle');
const legendBar = document.getElementById('legendBar');

function populateControls() {
  for (const task of Object.keys(data.tasks).sort()) {
    const option = document.createElement('option');
    option.value = task;
    option.textContent = task;
    controls.taskFilter.appendChild(option);
  }
  for (const metric of data.metrics) {
    const option = document.createElement('option');
    option.value = metric;
    option.textContent = metric;
    controls.metricFilter.appendChild(option);
  }
}

function rowsForSelection() {
  const task = data.tasks[controls.taskFilter.value];
  if (!task) return [];
  return task.rows.filter(row =>
    row.metric === controls.metricFilter.value &&
    row.relationship === controls.relationshipFilter.value
  );
}

function edgeConfidence(row) {
  return 1 / Math.sqrt(Math.max(1, row.bucket_count));
}

function uncertaintyAlpha(row) {
  return 0.04 + 0.96 * edgeConfidence(row);
}

function visualDistance(row) {
  return (1 - row.bucket_strength) + (1 - edgeConfidence(row));
}

function movementSort(left, right) {
  return visualDistance(left) - visualDistance(right)
    || left.base_neuron - right.base_neuron
    || left.finetuned_neuron - right.finetuned_neuron;
}

function filteredRows() {
  const rows = rowsForSelection();
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

function mixColor(left, right, value) {
  const bounded = Math.max(0, Math.min(1, value));
  const mixed = left.map((channel, index) => Math.round(channel + (right[index] - channel) * bounded));
  return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
}

function strengthColor(strength) {
  const relationship = controls.relationshipFilter.value;
  if (relationship === 'negative') {
    return mixColor([251, 146, 60], [127, 29, 29], strength);
  }
  return mixColor([96, 165, 250], [23, 37, 84], strength);
}

function styleForRow(row) {
  const confidence = edgeConfidence(row);
  return {
    color: strengthColor(row.strength),
    width: 0.25 + row.strength * 2.1 * confidence,
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

function renderStatus(rows) {
  const selectedTask = data.tasks[controls.taskFilter.value];
  const neuronCount = selectedTask ? selectedTask.neuron_count : 0;
  const uniqueTargets = new Set(rows.map(row => row.finetuned_neuron)).size;
  const minStrength = rows.length ? Math.min(...rows.map(row => row.strength)) : 0;
  const maxStrength = rows.length ? Math.max(...rows.map(row => row.strength)) : 0;
  const maxBucket = rows.length ? Math.max(...rows.map(row => row.bucket_count)) : 0;
  status.textContent =
    `${controls.taskFilter.value || '-'} | ${controls.metricFilter.value || '-'} | ${controls.relationshipFilter.value} | ` +
    `${rows.length} arrows | ${uniqueTargets} targets | neurons ${neuronCount} | ` +
    `strength ${minStrength.toFixed(4)}-${maxStrength.toFixed(4)} | ` +
    `top-band epsilon ${data.top_band_epsilon} | max bucket ${maxBucket}`;
}

function renderLegend() {
  const negative = controls.relationshipFilter.value === 'negative';
  legendTitle.textContent = negative ? 'anticorrelation strength' : 'correlation strength';
  legendBar.className = `legend-bar ${negative ? 'negative' : 'positive'}`;
}

function renderPlot() {
  const rows = filteredRows();
  renderStatus(rows);
  renderLegend();
  const selectedTask = data.tasks[controls.taskFilter.value];
  const maxNeuron = Math.max(1, selectedTask ? selectedTask.neuron_count - 1 : 1);
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
    return paddingX + (neuron / maxNeuron) * usableWidth;
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
  context.fillText('neuron 0', paddingX, topY + 16);
  context.fillText('neuron 0', paddingX, bottomY - 16);
  context.textAlign = 'right';
  context.fillText(`neuron ${maxNeuron}`, cssWidth - paddingX, topY + 16);
  context.fillText(`neuron ${maxNeuron}`, cssWidth - paddingX, bottomY - 16);

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
populateControls();
renderPlot();
</script>
</body>
</html>
""".replace("__CORRESPONDENCE_PAYLOAD__", escaped_payload)


def render_3d_html(payload: dict[str, object]) -> str:
    escaped_payload = correspondence_payload_json(payload)
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activation Cross-Movement Arrows 3D</title>
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
  grid-template-columns: 150px 120px 150px 140px;
  gap: 8px;
  padding: 10px;
  width: min(590px, calc(100vw - 32px));
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
  max-width: min(820px, calc(100vw - 36px));
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
}
@media (max-width: 760px) {
  .toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .status { left: 16px; right: auto; }
}
</style>
</head>
<body>
<section class="toolbar">
  <label>Task
    <select id="taskFilter"></select>
  </label>
  <label>Metric
    <select id="metricFilter"></select>
  </label>
  <label>Relationship
    <select id="relationshipFilter">
      <option value="positive">Similar activation</option>
      <option value="negative">Opposite activation</option>
    </select>
  </label>
  <label>Percentile gap
    <input id="percentileGap" type="number" min="1" max="100" step="1" inputmode="numeric" value="10">
  </label>
</section>
<div id="scene"></div>
<div class="status" id="status"></div>
<script id="correspondence-data" type="application/json">__CORRESPONDENCE_PAYLOAD__</script>
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

const data = JSON.parse(document.getElementById('correspondence-data').textContent);
const controls = {
  taskFilter: document.getElementById('taskFilter'),
  metricFilter: document.getElementById('metricFilter'),
  relationshipFilter: document.getElementById('relationshipFilter'),
  percentileGap: document.getElementById('percentileGap'),
};
const status = document.getElementById('status');
const sceneHost = document.getElementById('scene');
const SCENE_WIDTH = 2200;
const MODEL_GAP = 420;
const LAYER_GAP = 132;

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

function populateControls() {
  for (const task of Object.keys(data.tasks).sort()) {
    const option = document.createElement('option');
    option.value = task;
    option.textContent = task;
    controls.taskFilter.appendChild(option);
  }
  for (const metric of data.metrics) {
    const option = document.createElement('option');
    option.value = metric;
    option.textContent = metric;
    controls.metricFilter.appendChild(option);
  }
}

function rowsForSelection() {
  const task = data.tasks[controls.taskFilter.value];
  if (!task) return [];
  return task.rows.filter(row =>
    row.metric === controls.metricFilter.value &&
    row.relationship === controls.relationshipFilter.value
  );
}

function edgeConfidence(row) {
  return 1 / Math.sqrt(Math.max(1, row.bucket_count));
}

function visualDistance(row) {
  return (1 - row.bucket_strength) + (1 - edgeConfidence(row));
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
  const ordered = rows.slice().sort(movementSort);
  const bands = [];
  let startIndex = 0;
  for (let start = 0; start < 100 && startIndex < ordered.length; start += gap) {
    const end = Math.min(100, start + gap);
    const endIndex = end >= 100
      ? ordered.length
      : Math.max(startIndex + 1, Math.ceil(ordered.length * (end / 100)));
    bands.push({ start, end, rows: ordered.slice(startIndex, endIndex) });
    startIndex = endIndex;
  }
  return bands.filter(band => band.rows.length > 0);
}

function xForNeuron(neuron) {
  const selectedTask = data.tasks[controls.taskFilter.value];
  const maxNeuron = Math.max(1, selectedTask ? selectedTask.neuron_count - 1 : 1);
  return (neuron / maxNeuron) * SCENE_WIDTH - SCENE_WIDTH / 2;
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

function colorForRow(row) {
  const color = controls.relationshipFilter.value === 'negative'
    ? new THREE.Color(0.86, 0.15, 0.15)
    : new THREE.Color(0.12, 0.33, 0.76);
  color.multiplyScalar(0.25 + row.strength * 0.75);
  color.multiplyScalar(0.04 + 0.96 * edgeConfidence(row));
  return color;
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
      const color = colorForRow(row);
      const tangent = new THREE.Vector2(x2 - x1, yFine - yBase).normalize();
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
  movementGroup.add(new THREE.LineSegments(geometry, material));
}

function buildLayerGuides(bands) {
  const yBase = MODEL_GAP / 2;
  const yFine = -MODEL_GAP / 2;
  const xLeft = -SCENE_WIDTH / 2;
  const xRight = SCENE_WIDTH / 2;
  const guidePoints = [];
  bands.forEach((band, bandIndex) => {
    const z = ((bands.length - 1) / 2 - bandIndex) * LAYER_GAP;
    guidePoints.push(xLeft, yBase, z, xRight, yBase, z);
    guidePoints.push(xLeft, yFine, z, xRight, yFine, z);
    guidePoints.push(xLeft, yBase, z, xLeft, yFine, z);
    guidePoints.push(xRight, yBase, z, xRight, yFine, z);
  });
  movementGroup.add(lineSegments(guidePoints, 0x9aa4b2, 'percentile-band-guides'));
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

  const rows = rowsForSelection();
  const gap = percentileGap();
  const bands = percentileBands(rows, gap);
  buildLayerGuides(bands);
  buildBandGeometry(bands);
  fitCamera();

  const maxBucket = rows.length ? Math.max(...rows.map(row => row.bucket_count)) : 0;
  status.textContent =
    `${controls.taskFilter.value || '-'} | ${controls.metricFilter.value || '-'} | ${controls.relationshipFilter.value} | ` +
    `${rows.length} paths | ${bands.length} percentile layers | gap ${gap}% | ` +
    `top-band epsilon ${data.top_band_epsilon} | max bucket ${maxBucket}`;
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
populateControls();
resize();
renderScene();
animate();
</script>
</body>
</html>
""".replace("__CORRESPONDENCE_PAYLOAD__", escaped_payload)


def write_activation_cross_viewers(
    task_matrices: dict[str, dict[str, object]],
    output_2d_path: Path,
    output_3d_path: Path,
) -> None:
    payload = activation_correspondence_payload(task_matrices)
    output_2d_path.write_text(render_2d_html(payload), encoding="utf-8")
    output_3d_path.write_text(render_3d_html(payload), encoding="utf-8")


def build_viewers() -> dict[str, Path]:
    captured_results = load_captured_results()
    task_matrices = activation_cross_task_matrices(captured_results)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    output_tensor_path = GRAPH_DIR / OUTPUT_TENSOR_FILENAME
    output_2d_path = GRAPH_DIR / OUTPUT_2D_HTML_FILENAME
    output_3d_path = GRAPH_DIR / OUTPUT_3D_HTML_FILENAME
    write_activation_cross_tensors(task_matrices, output_tensor_path)
    write_activation_cross_viewers(task_matrices, output_2d_path, output_3d_path)

    return {
        "tensor": output_tensor_path,
        "viewer_2d": output_2d_path,
        "viewer_3d": output_3d_path,
    }


def main() -> int:
    output_paths = build_viewers()
    for output_path in output_paths.values():
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

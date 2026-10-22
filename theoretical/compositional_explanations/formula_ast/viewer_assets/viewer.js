'use strict';

// Each record owns a display copy of its Z3 AST. Repeated concept leaves share
// one truth value by name; selection identifies a particular tree occurrence.
const payload = JSON.parse(document.getElementById('records').textContent);
const records = payload.records;
const trueConcepts = new Set();
const neuronIndices = new Map(records.map((record, index) => [record.neuronId, index]));
const sortedNeurons = records.map((record, index) => ({ record, index }))
  .sort((a, b) => a.record.neuronId - b.record.neuronId);
const svgNamespace = 'http://www.w3.org/2000/svg';
const byId = id => document.getElementById(id);
let currentIndex = 0;
let selectedNodeId = '0';
let nodes = [];
let trueNeurons = [];
let fitTree = true;
let treeWidth = 0;
let treeHeight = 0;

function evaluate(node) {
  switch (node.type) {
    case 'concept': return trueConcepts.has(node.name);
    case 'constant': return node.value;
    case 'and': return node.children.every(evaluate);
    case 'or': return node.children.some(evaluate);
    case 'not': return !evaluate(node.children[0]);
    default: throw new Error('Unsupported AST node: ' + node.type);
  }
}

function expression(node) {
  if (node.type === 'concept') return node.name;
  if (node.type === 'constant') return String(node.value).toUpperCase();
  if (node.type === 'not') return 'NOT ' + expression(node.children[0]);
  return '(' + node.children.map(expression).join(' ' + node.type.toUpperCase() + ' ') + ')';
}

function measureTree(node, id = '0', depth = 0) {
  const children = (node.children ?? []).map((child, i) => measureTree(child, id + '.' + i, depth + 1));
  const label = node.name?.slice(node.name.indexOf(':') + 1) ?? '';
  const width = node.type === 'concept' ? Math.max(150, label.length * 8.5 + 48) : 90;
  const childSpan = children.reduce((sum, child) => sum + child.span, 0)
    + Math.max(0, children.length - 1) * 26;
  return { node, id, depth, children, width, span: Math.max(width, childSpan) };
}

function positionTree(item, left) {
  item.x = left + item.span / 2;
  item.y = 54 + item.depth * 104;
  nodes.push(item);
  const childSpan = item.children.reduce((sum, child) => sum + child.span, 0)
    + Math.max(0, item.children.length - 1) * 26;
  let cursor = left + (item.span - childSpan) / 2;
  for (const child of item.children) {
    positionTree(child, cursor);
    cursor += child.span + 26;
  }
}

function svgElement(tag, attributes = {}, text) {
  const element = document.createElementNS(svgNamespace, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function sizeTree() {
  const stage = byId('stage');
  const scale = fitTree
    ? Math.min(1, (stage.clientWidth - 24) / treeWidth, (stage.clientHeight - 24) / treeHeight)
    : 1;
  byId('canvas').style.width = treeWidth * scale + 'px';
  byId('canvas').style.minHeight = treeHeight * scale + 'px';
  byId('tree').style.height = treeHeight * scale + 'px';
}

function inspectSelection() {
  const node = nodes.find(item => item.id === selectedNodeId).node;
  const outcome = String(evaluate(node)).toUpperCase();
  byId('inspection-title').textContent = node.type === 'concept' ? 'Input' : node.type.toUpperCase();
  if (node.type === 'concept') {
    byId('calculation').textContent = node.name + ' = ' + outcome;
    byId('expression').textContent = '';
  } else if (node.type === 'constant') {
    byId('calculation').textContent = outcome;
    byId('expression').textContent = '';
  } else {
    const operands = node.children.map(child => String(evaluate(child)).toUpperCase());
    const calculation = node.type === 'not'
      ? 'NOT ' + operands[0]
      : operands.join(' ' + node.type.toUpperCase() + ' ');
    byId('calculation').textContent = calculation + ' = ' + outcome;
    byId('expression').textContent = selectedNodeId === '0' ? '' : expression(node);
  }
}

function refreshTree() {
  for (const item of nodes) {
    const element = byId('node-' + item.id);
    const on = evaluate(item.node);
    const concept = item.node.type === 'concept';
    element.classList.toggle('true', on);
    element.classList.toggle('selected', item.id === selectedNodeId);
    element.querySelector('.bit').textContent = String(on).toUpperCase();
    if (concept) element.setAttribute('aria-pressed', String(on));
    element.setAttribute('aria-label', (item.node.name ?? item.node.type.toUpperCase())
      + ', ' + on + ', ' + (concept ? 'toggle input' : 'inspect computation'));
    byId('edge-' + item.id)?.classList.toggle('on', on);
  }
  const output = evaluate(records[currentIndex].ast);
  byId('output').textContent = String(output).toUpperCase();
  byId('output').classList.toggle('true', output);
  inspectSelection();
}

function activateNode(item) {
  selectedNodeId = item.id;
  if (item.node.type !== 'concept') {
    refreshTree();
    return;
  }
  if (trueConcepts.has(item.node.name)) trueConcepts.delete(item.node.name);
  else trueConcepts.add(item.node.name);
  refreshInputs();
}

function drawTree() {
  nodes = [];
  const root = measureTree(records[currentIndex].ast);
  positionTree(root, 30);
  treeWidth = root.span + 60;
  treeHeight = Math.max(...nodes.map(item => item.y)) + 66;
  const tree = byId('tree');
  tree.replaceChildren();
  tree.setAttribute('viewBox', `0 0 ${treeWidth} ${treeHeight}`);
  for (const parent of nodes) {
    for (const child of parent.children) {
      tree.append(svgElement('path', {
        id: 'edge-' + child.id,
        class: 'edge',
        d: `M ${parent.x} ${parent.y + 28} C ${parent.x} ${parent.y + 65},${child.x} ${child.y - 65},${child.x} ${child.y - 28}`,
      }));
    }
  }
  for (const item of nodes) {
    const { node } = item;
    const concept = node.type === 'concept';
    const group = svgElement('g', {
      id: 'node-' + item.id,
      class: 'node ' + (concept ? 'concept' : 'operator'),
      role: 'button',
      tabindex: '0',
      transform: `translate(${item.x},${item.y})`,
    });
    group.append(svgElement('title', {}, concept ? 'Toggle ' + node.name : expression(node)));
    group.append(svgElement('rect', {
      x: -item.width / 2, y: -29, width: item.width, height: 58, rx: concept ? 11 : 20,
    }));
    if (concept) {
      const colon = node.name.indexOf(':');
      group.append(svgElement('text', { y: -11, class: 'prefix' }, colon < 0 ? 'concept' : node.name.slice(0, colon)));
      group.append(svgElement('text', { y: 8 }, colon < 0 ? node.name : node.name.slice(colon + 1)));
    } else {
      group.append(svgElement('text', { y: -3 }, node.type === 'constant' ? 'CONST' : node.type.toUpperCase()));
    }
    group.append(svgElement('text', { y: 23, class: 'bit' }));
    group.addEventListener('click', () => activateNode(item));
    group.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activateNode(item);
      }
    });
    tree.append(group);
  }
  sizeTree();
  requestAnimationFrame(() => {
    const stage = byId('stage');
    stage.scrollLeft = (stage.scrollWidth - stage.clientWidth) / 2;
    stage.scrollTop = 0;
  });
}

function renderCoactivation() {
  const currentId = records[currentIndex].neuronId;
  const others = trueNeurons.filter(({ record }) => record.neuronId !== currentId);
  byId('coactivation-count').textContent = others.length.toLocaleString()
    + ' / ' + (records.length - 1).toLocaleString();
  const buttons = others.map(({ record, index }) => {
    const button = document.createElement('button');
    button.textContent = 'Neuron ' + record.neuronId;
    button.dataset.neuronId = record.neuronId;
    button.title = record.formula;
    button.addEventListener('click', () => showNeuron(index, true));
    return button;
  });
  byId('true-neurons').replaceChildren(...buttons);
  if (!others.length) {
    const empty = document.createElement('span');
    empty.className = 'empty';
    empty.textContent = 'No other formulas are true.';
    byId('true-neurons').append(empty);
  }
}

function refreshInputs() {
  trueNeurons = sortedNeurons.filter(({ record }) => evaluate(record.ast));
  const chips = [...trueConcepts].sort().map(name => {
    const button = document.createElement('button');
    button.textContent = name + ' ×';
    button.setAttribute('aria-label', 'Set ' + name + ' false');
    button.addEventListener('click', () => {
      trueConcepts.delete(name);
      refreshInputs();
    });
    return button;
  });
  byId('true-inputs').replaceChildren(...chips);
  if (!chips.length) byId('true-inputs').textContent = 'None';
  refreshTree();
  renderCoactivation();
}

function showNeuron(index, focusHeading = false) {
  currentIndex = index;
  selectedNodeId = '0';
  const record = records[index];
  byId('neuron-title').textContent = 'Neuron ' + record.neuronId;
  byId('neuron-meta').textContent = `${index + 1} / ${records.length.toLocaleString()} · IoU ${record.iou.toFixed(3)}`;
  byId('neuron-meta').title = 'CSV order · formula fit (intersection over union)';
  byId('neuron-id').value = record.neuronId;
  byId('jump-status').textContent = '';
  byId('original-formula').textContent = record.formula;
  drawTree();
  refreshTree();
  renderCoactivation();
  if (focusHeading) {
    byId('neuron-title').focus({ preventScroll: true });
    byId('neuron-title').scrollIntoView({ block: 'center' });
  }
}

function navigate(delta) {
  showNeuron((currentIndex + delta + records.length) % records.length);
}

byId('prev').addEventListener('click', () => navigate(-1));
byId('next').addEventListener('click', () => navigate(1));
byId('fit').addEventListener('click', () => {
  fitTree = !fitTree;
  byId('fit').setAttribute('aria-pressed', String(fitTree));
  byId('fit').textContent = fitTree ? 'Full size' : 'Fit tree';
  sizeTree();
});
byId('reset').addEventListener('click', () => {
  trueConcepts.clear();
  refreshInputs();
});
byId('jump-form').addEventListener('submit', event => {
  event.preventDefault();
  const index = neuronIndices.get(Number(byId('neuron-id').value));
  if (index === undefined) {
    byId('jump-status').textContent = 'No formula for that neuron.';
    return;
  }
  showNeuron(index);
});
document.addEventListener('keydown', event => {
  if (event.target.matches('input') || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    navigate(event.key === 'ArrowLeft' ? -1 : 1);
  }
});

byId('source').textContent = payload.source + ' · formula_ast/analysis.py → Z3 AST';
byId('fingerprint').textContent = 'CSV SHA-256 ' + payload.sha256;
showNeuron(0);
refreshInputs();
new ResizeObserver(sizeTree).observe(byId('stage'));

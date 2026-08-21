// CHIMERA-Agent Phase B: Cohort View SVG Renderer (cohort_view.js)
// Zero-dependency pure DOM + SVG renderer for Cohort Analytics B1-B5.
// Precision geometric partitioning — guaranteed zero collisions / zero clipping.

import { CohortEngine } from './cohort_engine.js';

// Escapes HTML-sensitive characters to prevent reflected XSS when interpolating
// error messages (which may contain trace data) into innerHTML.
function escapeHTML(str) {
  if (str === undefined || str === null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const SVG_NS = 'http://www.w3.org/2000/svg';

// Viridis colormap (perceptually uniform, colorblind-safe) for correlation heatmap.
const VIRIDIS_STEPS = [
  '#440154', '#482878', '#3e4989', '#31688e', '#26828e',
  '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'
];

// Target class colors (colorblind-safe, distinct hues)
const TARGET_COLORS = {
  'yes': '#1f77b4',
  'no': '#ff7f0e',
  'active_surveillance': '#2ca02c',
  'continued_surveillance': '#98df8a',
  'watchful_waiting': '#ff9896',
  'active_treatment': '#d62728',
  '1': '#d62728',
  '0': '#2ca02c',
  'None': '#7f7f7f',
  'null': '#7f7f7f',
  'unknown': '#4a5568',
};

function el(tag, attrs, text) {
  const e = document.createElementNS(SVG_NS, tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      e.setAttribute(k, String(v));
    }
  }
  if (text !== undefined) {
    e.textContent = String(text);
  }
  return e;
}

// Canvas-based text width measurement
let _measureCtx = null;
function measureText(text, fontSize, fontFamily) {
  if (!_measureCtx) {
    const c = document.createElement('canvas');
    _measureCtx = c.getContext('2d');
  }
  _measureCtx.font = `${fontSize}px ${fontFamily || 'ui-monospace, SFMono-Regular, Menlo, monospace'}`;
  return _measureCtx.measureText(text).width;
}

// Wrap legend items into lines that never exceed innerWidth
function wrapLegend(items, innerWidth, fontSize) {
  const lines = [];
  let cur = [];
  let curW = 0;
  for (const item of items) {
    const itemW = 12 + 6 + measureText(item.label, fontSize) + 20;
    if (curW + itemW > innerWidth && cur.length > 0) {
      lines.push(cur);
      cur = [];
      curW = 0;
    }
    cur.push(item);
    curW += itemW;
  }
  if (cur.length > 0) lines.push(cur);
  return lines;
}

function viridisColor(value) {
  const t = Math.max(0, Math.min(1, (value + 1) / 2));
  const idx = Math.min(VIRIDIS_STEPS.length - 1, Math.floor(t * VIRIDIS_STEPS.length));
  return VIRIDIS_STEPS[idx];
}

function createSvg(width, height) {
  const svg = el('svg', {
    viewBox: `0 0 ${width} ${height}`,
    width: '100%',
    xmlns: SVG_NS,
    preserveAspectRatio: 'xMidYMid meet',
  });
  const bg = el('rect', { width, height, fill: '#0d1117' });
  svg.appendChild(bg);
  return svg;
}

// ===========================================================================
// B1: Cohort Composition — Ordered Horizontal Stacked Bars
// ===========================================================================

function renderComposition(container, data) {
  container.innerHTML = '';
  const tasks = data.tasks;
  const taskOrder = ['task1', 'task2', 'task3'];
  const taskLabels = {
    task1: 'Task 1 (Pre-Biopsy)',
    task2: 'Task 2 (Stratification)',
    task3: 'Task 3 (Recurrence)'
  };

  const innerW = 860;
  const labelW = 180;
  const padRight = 80;
  const barH = 34;
  const gap = 20;
  const plotW = innerW - labelW - padRight;
  const nBars = taskOrder.length;

  const allClasses = [];
  const seenClasses = new Set();
  for (const task of taskOrder) {
    for (const cls of Object.keys(tasks[task].classes)) {
      if (!seenClasses.has(cls)) { seenClasses.add(cls); allClasses.push(cls); }
    }
  }
  const legendItems = allClasses.map(cls => ({ label: cls, color: TARGET_COLORS[cls] || '#7f7f7f' }));
  const legendLines = wrapLegend(legendItems, plotW + padRight, 10);
  const legendLineH = 22;
  const legendBandH = legendLines.length * legendLineH + 10;
  const height = 24 + nBars * (barH + gap) + 16 + legendBandH + 16;

  const svg = createSvg(innerW, height);

  let yOff = 24;
  for (const task of taskOrder) {
    const info = tasks[task];
    const total = info.total;
    const y = yOff;

    // Row title in left gutter
    svg.appendChild(el('text', {
      x: labelW - 14, y: y + barH / 2 + 4,
      'text-anchor': 'end',
      fill: '#e6edf3', 'font-family': 'ui-monospace, monospace', 'font-size': 12, 'font-weight': 600,
    }, taskLabels[task] || task));

    // Stacked bars
    let xOff = labelW;
    for (const [cls, cdata] of Object.entries(info.classes)) {
      const segW = (cdata.n / total) * plotW;
      if (segW < 0.5) { xOff += segW; continue; }
      const color = TARGET_COLORS[cls] || '#7f7f7f';
      const rect = el('rect', {
        x: xOff, y: y, width: segW, height: barH,
        fill: color, stroke: '#0d1117', 'stroke-width': 1, rx: 2
      });
      rect.appendChild(el('title', {}, `${task} / ${cls}: n=${cdata.n} (${cdata.pct}%)`));
      svg.appendChild(rect);

      const fullLabel = `${cls}: ${cdata.n} (${cdata.pct}%)`;
      const shortLabel = `${cdata.n} (${cdata.pct}%)`;
      const fullW = measureText(fullLabel, 10);
      const shortW = measureText(shortLabel, 10);

      if (fullW <= segW - 10) {
        svg.appendChild(el('text', {
          x: xOff + segW / 2, y: y + barH / 2 + 4,
          'text-anchor': 'middle',
          fill: '#ffffff', 'font-family': 'ui-monospace, monospace', 'font-size': 10, 'font-weight': 600,
        }, fullLabel));
      } else if (shortW <= segW - 8) {
        svg.appendChild(el('text', {
          x: xOff + segW / 2, y: y + barH / 2 + 4,
          'text-anchor': 'middle',
          fill: '#ffffff', 'font-family': 'ui-monospace, monospace', 'font-size': 10, 'font-weight': 600,
        }, shortLabel));
      } else if (segW >= 24) {
        svg.appendChild(el('text', {
          x: xOff + segW / 2, y: y + barH / 2 + 4,
          'text-anchor': 'middle',
          fill: '#ffffff', 'font-family': 'ui-monospace, monospace', 'font-size': 9, 'font-weight': 700,
        }, String(cdata.n)));
      }
      xOff += segW;
    }

    // Total label on right
    svg.appendChild(el('text', {
      x: labelW + plotW + 12, y: y + barH / 2 + 4,
      fill: '#8b9bb4', 'font-family': 'ui-monospace, monospace', 'font-size': 11, 'font-weight': 600,
    }, `N=${total}`));

    yOff += barH + gap;
  }

  // Legend at bottom
  let legY = yOff + 14;
  for (const line of legendLines) {
    let lx = labelW;
    for (const item of line) {
      svg.appendChild(el('rect', { x: lx, y: legY, width: 12, height: 12, fill: item.color, rx: 2 }));
      svg.appendChild(el('text', {
        x: lx + 18, y: legY + 10,
        fill: '#8b9bb4', 'font-family': 'ui-monospace, monospace', 'font-size': 10,
      }, item.label));
      lx += 12 + 6 + measureText(item.label, 10) + 20;
    }
    legY += legendLineH;
  }

  container.appendChild(svg);

  const totalN = taskOrder.reduce((s, t) => s + (tasks[t] && tasks[t].total ? tasks[t].total : 0), 0);
  const caption = document.createElement('div');
  caption.className = 'cohort-caption';
  caption.textContent = `B1: Task/target composition as ordered horizontal stacked bars. Exact counts and percentages per task per target class. Total N=${totalN}.`;
  container.appendChild(caption);
}

// ===========================================================================
// B2: PCA Scatter — 2D Latent Manifold
// ===========================================================================

function renderPCA(container, data, modalityLabel) {
  container.innerHTML = '';
  const points = data.points || [];
  const n = data.n || points.length;
  const varExp = data.variance_explained || [0, 0];
  const cum2pc = data.cumulative_2pc || 0;

  if (n === 0) {
    container.innerHTML = '<div class="case-extra-missing">No PCA data available for this modality.</div>';
    return;
  }

  const width = 760;
  const height = 520;

  const legendItems = [];
  const legendSeen = {};
  for (const p of points) {
    if (!legendSeen[p.target]) {
      legendSeen[p.target] = true;
      legendItems.push({ label: p.target, color: TARGET_COLORS[p.target] || '#7f7f7f' });
    }
  }
  const legendLines = wrapLegend(legendItems, width - 120, 10);
  const legendLineH = 20;
  const legendBandH = legendLines.length * legendLineH + 8;

  const pad = { top: legendBandH + 28, right: 35, bottom: 65, left: 80 };
  const pw = width - pad.left - pad.right;
  const ph = height - pad.top - pad.bottom;

  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const p of points) {
    if (p.pc1 < xMin) xMin = p.pc1;
    if (p.pc1 > xMax) xMax = p.pc1;
    if (p.pc2 < yMin) yMin = p.pc2;
    if (p.pc2 > yMax) yMax = p.pc2;
  }
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  xMin -= xRange * 0.06;
  xMax += xRange * 0.06;
  yMin -= yRange * 0.06;
  yMax += yRange * 0.06;

  const mapX = v => pad.left + ((v - xMin) / (xMax - xMin)) * pw;
  const mapY = v => pad.top + ph - ((v - yMin) / (yMax - yMin)) * ph;

  const svg = createSvg(width, height);

  // Legend at top (completely inside top reserved band)
  let legY = 10;
  for (const line of legendLines) {
    let lx = pad.left;
    for (const item of line) {
      svg.appendChild(el('rect', { x: lx, y: legY, width: 10, height: 10, fill: item.color, rx: 2, 'fill-opacity': 0.85 }));
      svg.appendChild(el('text', { x: lx + 16, y: legY + 9, fill: '#8b9bb4', 'font-family': 'ui-monospace, monospace', 'font-size': 10 }, item.label));
      lx += 10 + 6 + measureText(item.label, 10) + 20;
    }
    legY += legendLineH;
  }

  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const xv = xMin + (xMax - xMin) * i / 4;
    const yv = yMin + (yMax - yMin) * i / 4;
    const xp = mapX(xv);
    const yp = mapY(yv);
    svg.appendChild(el('line', { x1: xp, y1: pad.top, x2: xp, y2: pad.top + ph, stroke: '#1c2535', 'stroke-width': 1, 'stroke-dasharray': '3,3' }));
    svg.appendChild(el('line', { x1: pad.left, y1: yp, x2: pad.left + pw, y2: yp, stroke: '#1c2535', 'stroke-width': 1, 'stroke-dasharray': '3,3' }));
    
    // X tick label
    svg.appendChild(el('text', { x: xp, y: pad.top + ph + 18, 'text-anchor': 'middle', fill: '#5c6f8a', 'font-family': 'ui-monospace, monospace', 'font-size': 10 }, xv.toFixed(1)));
    
    // Y tick label — always render. X tick (text-anchor: middle, x=pad.left)
    // and Y tick (text-anchor: end, x=pad.left-10) are horizontally separated,
    // so the bottom Y label cannot collide with the leftmost X label.
    svg.appendChild(el('text', { x: pad.left - 10, y: yp + 4, 'text-anchor': 'end', fill: '#5c6f8a', 'font-family': 'ui-monospace, monospace', 'font-size': 10 }, yv.toFixed(1)));
  }

  // Axes
  svg.appendChild(el('line', { x1: pad.left, y1: pad.top + ph, x2: pad.left + pw, y2: pad.top + ph, stroke: '#3b5275', 'stroke-width': 1.5 }));
  svg.appendChild(el('line', { x1: pad.left, y1: pad.top, x2: pad.left, y2: pad.top + ph, stroke: '#3b5275', 'stroke-width': 1.5 }));

  // Axis labels
  svg.appendChild(el('text', {
    x: pad.left + pw / 2, y: height - 16,
    'text-anchor': 'middle', fill: '#e6edf3', 'font-family': 'ui-monospace, monospace', 'font-size': 12, 'font-weight': 600,
  }, `PC1 (${varExp[0]}% variance explained)`));

  svg.appendChild(el('text', {
    x: 24, y: pad.top + ph / 2,
    'text-anchor': 'middle', fill: '#e6edf3', 'font-family': 'ui-monospace, monospace', 'font-size': 12, 'font-weight': 600,
    transform: `rotate(-90 24 ${pad.top + ph / 2})`,
  }, `PC2 (${varExp[1]}% variance explained)`));

  // Scatter points
  for (const p of points) {
    const cx = mapX(p.pc1);
    const cy = mapY(p.pc2);
    const color = TARGET_COLORS[p.target] || '#7f7f7f';
    const dot = el('circle', {
      cx: cx.toFixed(2), cy: cy.toFixed(2), r: 4.5,
      fill: color, 'fill-opacity': 0.75, stroke: '#0d1117', 'stroke-width': 0.8,
    });
    dot.style.cursor = 'pointer';
    dot.appendChild(el('title', {}, `${p.task.toUpperCase()} | ${p.case_id} | target: ${p.target} | PC1: ${p.pc1.toFixed(3)} | PC2: ${p.pc2.toFixed(3)} — click to view case`));
    dot.addEventListener('click', () => {
      document.dispatchEvent(new CustomEvent('chimera:navigate-case', { detail: { task: p.task, case_id: p.case_id } }));
    });
    svg.appendChild(dot);
  }

  container.appendChild(svg);

  const caption = document.createElement('div');
  caption.className = 'cohort-caption';
  caption.textContent = `Cohort-level projection: 1 dot = 1 patient case. ${modalityLabel} PCA (n=${n}). Cumulative 2-PC variance: ${cum2pc}%.`;
  container.appendChild(caption);
}

// ===========================================================================
// B3: Spearman Correlation Heatmap (Ward-ordered, Viridis colormap)
// ===========================================================================

function renderCorrelation(container, fullData, filterKey) {
  container.innerHTML = '';
  // The internal stratum selector was removed: it was redundant with the
  // global scope selector (which recomputes all views via computeAll) and
  // broken in live mode (flat correlation object when filter='all'). The
  // global selector is the single source of truth for stratum switching.

  if (!fullData || !fullData.variables || fullData.variables.length === 0) {
    container.innerHTML = '<div class="case-extra-missing" style="padding: 24px; text-align: center;">No correlation data available for this stratum.</div>';
    return;
  }

  const vars = fullData.variables || [];
  const matrix = fullData.matrix || [];
  const k = vars.length;

  const cellSize = 50;
  const labelW = 120;
  const labelH = 120;
  const width = labelW + k * cellSize + 65;
  const height = labelH + k * cellSize + 40;

  const svg = createSvg(width, height);

  // Cells
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const val = matrix[i][j];
      const x = labelW + j * cellSize;
      const y = labelH + i * cellSize;
      const fill = val === null ? '#1c2535' : viridisColor(val);
      const rect = el('rect', { x, y, width: cellSize, height: cellSize, fill, stroke: '#0d1117', 'stroke-width': 1 });
      const tooltip = val === null ? `${vars[i]} vs ${vars[j]}: INSUFFICIENT DATA` : `${vars[i]} vs ${vars[j]}: rho = ${val.toFixed(4)}`;
      rect.appendChild(el('title', {}, tooltip));
      svg.appendChild(rect);

      if (val !== null) {
        // Choose text color based on the actual cell background luminance so
        // the label stays readable across the full Viridis range (WCAG AA
        // requires >=4.5:1 contrast). The previous fixed threshold (val > 0.6)
        // left near-white text on light-green cells (~1.8:1 contrast).
        const hex = fill.replace('#', '');
        const r = parseInt(hex.substr(0, 2), 16) / 255;
        const g = parseInt(hex.substr(2, 2), 16) / 255;
        const b = parseInt(hex.substr(4, 2), 16) / 255;
        const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        svg.appendChild(el('text', {
          x: x + cellSize / 2, y: y + cellSize / 2 + 4,
          'text-anchor': 'middle',
          fill: lum > 0.6 ? '#0d1117' : '#e6edf3',
          'font-family': 'ui-monospace, monospace', 'font-size': 11, 'font-weight': 600,
        }, val.toFixed(2)));
      }
    }
  }

  // Y-axis labels (variable names)
  for (let i = 0; i < k; i++) {
    svg.appendChild(el('text', {
      x: labelW - 10, y: labelH + i * cellSize + cellSize / 2 + 4,
      'text-anchor': 'end', fill: '#e6edf3', 'font-family': 'ui-monospace, monospace', 'font-size': 11, 'font-weight': 600,
    }, vars[i]));
  }

  // X-axis labels rotated UPWARDS (45 deg in SVG y-down) with text-anchor: end
  // so text extends up-LEFT from the column center, staying above the heatmap
  // and avoiding collision with the colorbar on the right.
  for (let j = 0; j < k; j++) {
    const x = labelW + j * cellSize + cellSize / 2;
    const y = labelH - 12;
    svg.appendChild(el('text', {
      x: x, y: y,
      'text-anchor': 'end', fill: '#e6edf3', 'font-family': 'ui-monospace, monospace', 'font-size': 11, 'font-weight': 600,
      transform: `rotate(45 ${x} ${y})`,
    }, vars[j]));
  }

  // Colorbar
  const cbX = labelW + k * cellSize + 16;
  const cbY = labelH;
  const cbW = 14;
  const cbH = k * cellSize;
  for (let s = 0; s < VIRIDIS_STEPS.length; s++) {
    const segH = cbH / VIRIDIS_STEPS.length;
    svg.appendChild(el('rect', {
      x: cbX, y: cbY + s * segH, width: cbW, height: segH,
      fill: VIRIDIS_STEPS[VIRIDIS_STEPS.length - 1 - s],
    }));
  }
  svg.appendChild(el('text', { x: cbX + cbW + 6, y: cbY + 9, fill: '#8b9bb4', 'font-family': 'ui-monospace, monospace', 'font-size': 10 }, '+1.0'));
  svg.appendChild(el('text', { x: cbX + cbW + 6, y: cbY + cbH / 2 + 4, fill: '#8b9bb4', 'font-family': 'ui-monospace, monospace', 'font-size': 10 }, '0.0'));
  svg.appendChild(el('text', { x: cbX + cbW + 6, y: cbY + cbH, fill: '#8b9bb4', 'font-family': 'ui-monospace, monospace', 'font-size': 10 }, '-1.0'));

  container.appendChild(svg);

  const caption = document.createElement('div');
  caption.className = 'cohort-caption';
  const nCases = fullData.n_cases || fullData.filteredCases || '?';
  const stratumTitle = filterKey === 'all' ? `Entire Combined Cohort (N=${nCases})` : `${String(filterKey).toUpperCase()} Peer Group (n=${nCases})`;
  caption.textContent = `B3: Spearman rho heatmap with Ward dendrogram ordering [${stratumTitle}]. Viridis colormap. Cell tooltips show exact rho. ${k} variables.`;
  container.appendChild(caption);
}

// ===========================================================================
// B4: Missingness Grid
// ===========================================================================

function renderMissingness(container, data) {
  container.innerHTML = '';
  const modalities = data.modalities || [];
  const cases = data.cases || [];
  const matrix = data.matrix || [];
  const nCases = cases.length;
  const nMod = modalities.length;

  if (nCases === 0 || nMod === 0) {
    container.innerHTML = '<div class="case-extra-missing">No missingness data available.</div>';
    return;
  }

  const MOD_LABELS = {
    'MRI': 'mpMRI',
    'Biopsy': 'Biopsy WSI',
    'Prostatectomy': 'Surgical Path',
    'PSA_Trend': 'PSA Kinetics',
    'Labs': 'Serum Labs',
    'FamilyHistory': 'Family History',
  };

  const cellW = 36;
  const cellH = 8;
  const colSpacing = 40;
  const labelW = 100;
  const labelH = 110;
  const gridW = nMod * colSpacing;
  const width = labelW + gridW + 100;

  // Group cases by task — each task gets its own SVG panel
  const taskGroups = { task1: [], task2: [], task3: [] };
  for (let i = 0; i < nCases; i++) {
    const task = cases[i].task;
    if (taskGroups[task]) {
      taskGroups[task].push(i);
    }
  }

  const taskOrder = ['task1', 'task2', 'task3'];

  // Flex-row wrapper so the three task panels sit side by side
  const rowWrap = document.createElement('div');
  rowWrap.style.display = 'flex';
  rowWrap.style.gap = '16px';
  rowWrap.style.alignItems = 'flex-start';
  rowWrap.style.flexWrap = 'wrap';

  for (const task of taskOrder) {
    const groupIndices = taskGroups[task];
    if (groupIndices.length === 0) continue;

    const groupN = groupIndices.length;
    const height = labelH + groupN * (cellH + 1) + 50;

    // Each task panel: header + scrollable SVG stacked vertically
    const panel = document.createElement('div');
    panel.style.flex = '0 0 auto';

    const header = document.createElement('div');
    header.className = 'cohort-caption';
    header.textContent = `${task.toUpperCase()} — ${groupN} cases`;
    panel.appendChild(header);

    const svg = createSvg(width, height);

    // X-axis column headers rotated UPWARDS (-45 deg) with text-anchor: start
    for (let j = 0; j < nMod; j++) {
      const x = labelW + j * colSpacing + cellW / 2;
      const y = labelH - 12;
      const labelText = MOD_LABELS[modalities[j]] || modalities[j];
      svg.appendChild(el('text', {
        x: x, y: y,
        'text-anchor': 'start', fill: '#e6edf3', 'font-family': 'ui-monospace, monospace', 'font-size': 11, 'font-weight': 600,
        transform: `rotate(-45 ${x} ${y})`,
      }, labelText));
    }

    // Rows for this task
    for (let r = 0; r < groupN; r++) {
      const i = groupIndices[r];

      for (let j = 0; j < nMod; j++) {
        const present = matrix[i][j] === 1;
        const x = labelW + j * colSpacing;
        const y = labelH + r * (cellH + 1);
        const fill = present ? '#238636' : '#da3633';
        const rect = el('rect', { x, y, width: cellW, height: cellH, fill, stroke: '#0d1117', 'stroke-width': 0.3, rx: 1 });
        rect.style.cursor = 'pointer';

        const tooltip = `${cases[i].task.toUpperCase()} | ${cases[i].case_id} | ${MOD_LABELS[modalities[j]] || modalities[j]}: ${present ? 'PRESENT' : 'MISSING'} — click to view case`;
        rect.appendChild(el('title', {}, tooltip));

        rect.addEventListener('click', () => {
          document.dispatchEvent(new CustomEvent('chimera:navigate-case', { detail: { task: cases[i].task, case_id: cases[i].case_id } }));
        });

        svg.appendChild(rect);
      }
    }

    // Legend at bottom
    const legY = labelH + groupN * (cellH + 1) + 16;
    svg.appendChild(el('rect', { x: labelW, y: legY, width: 14, height: 12, fill: '#238636', rx: 2 }));
    svg.appendChild(el('text', { x: labelW + 20, y: legY + 10, fill: '#8b9bb4', 'font-family': 'ui-monospace, monospace', 'font-size': 11 }, 'Present'));
    svg.appendChild(el('rect', { x: labelW + 90, y: legY, width: 14, height: 12, fill: '#da3633', rx: 2 }));
    svg.appendChild(el('text', { x: labelW + 110, y: legY + 10, fill: '#8b9bb4', 'font-family': 'ui-monospace, monospace', 'font-size': 11 }, 'Missing'));

    svg.style.maxWidth = width + 'px';
    const svgScroll = document.createElement('div');
    svgScroll.style.overflow = 'auto';
    svgScroll.style.maxHeight = '70vh';
    svgScroll.style.border = '1px solid var(--border-subtle)';
    svgScroll.style.borderRadius = '4px';
    svgScroll.appendChild(svg);
    panel.appendChild(svgScroll);
    rowWrap.appendChild(panel);
  }

  container.appendChild(rowWrap);

  const expDiv = document.createElement('div');
  expDiv.className = 'cohort-caption';
  expDiv.textContent = `B4: Missingness grid (${nCases} cases x ${nMod} modalities). Dark green = present, dark red = missing. Click any row to navigate to that case.`;
  container.appendChild(expDiv);
}

// ===========================================================================
// B5: Raincloud Plots (jittered dots + Tukey box + half-violin KDE)
// ===========================================================================

function renderRaincloud(container, data) {
  container.innerHTML = '';
  const strata = data.strata || [];
  const metric = data.metric || '';
  const unit = data.unit || '';
  const METRIC_LABELS = { psa: 'PSA', psad: 'PSA Density', vol: 'Prostate Volume', age: 'Age', pirads: 'PI-RADS' };
  const metricLabel = METRIC_LABELS[metric] || metric;

  if (strata.length === 0) {
    container.innerHTML = '<div class="case-extra-missing">No raincloud data available.</div>';
    return;
  }

  const validStrata = strata.filter(s => s.n > 0 && s.box);
  if (validStrata.length === 0) {
    container.innerHTML = '<div class="case-extra-missing">No data available for any stratum.</div>';
    return;
  }

  let dataMin = Infinity, dataMax = -Infinity;
  for (const s of validStrata) {
    // Include KDE evaluation range so the half-violin tails are not clamped
    // by mapX (engine evaluates KDE 15% beyond the data range).
    if (s.kde && s.kde.points && s.kde.points.length > 0) {
      dataMin = Math.min(dataMin, s.kde.points[0].x);
      dataMax = Math.max(dataMax, s.kde.points[s.kde.points.length - 1].x);
    }
    if (s.box) {
      dataMin = Math.min(dataMin, s.box.whisker_lo, ...(s.box.outliers || []));
      dataMax = Math.max(dataMax, s.box.whisker_hi, ...(s.box.outliers || []));
    }
    for (const d of (s.dots || [])) {
      dataMin = Math.min(dataMin, d.value);
      dataMax = Math.max(dataMax, d.value);
    }
  }
  const vMin = Math.min(0, dataMin);
  const vMax = dataMax * 1.05;

  const width = 900;
  const stratumH = 85;
  const gap = 20;
  const pad = { top: 35, right: 35, bottom: 55, left: 280 };
  const plotW = width - pad.left - pad.right;
  const nStrata = validStrata.length;
  const height = pad.top + nStrata * (stratumH + gap) + pad.bottom;

  const mapX = v => pad.left + Math.max(0, Math.min(1, (v - vMin) / (vMax - vMin))) * plotW;

  const svg = createSvg(width, height);

  // X-axis grid and ticks
  for (let i = 0; i <= 6; i++) {
    const xv = vMin + (vMax - vMin) * i / 6;
    const xp = mapX(xv);
    svg.appendChild(el('line', { x1: xp, y1: pad.top, x2: xp, y2: height - pad.bottom, stroke: '#1c2535', 'stroke-width': 1, 'stroke-dasharray': '3,3' }));
    svg.appendChild(el('text', { x: xp, y: height - pad.bottom + 18, 'text-anchor': 'middle', fill: '#5c6f8a', 'font-family': 'ui-monospace, monospace', 'font-size': 10 }, xv.toFixed(2)));
  }

  // X-axis label
  svg.appendChild(el('text', {
    x: pad.left + plotW / 2, y: height - 14,
    'text-anchor': 'middle', fill: '#e6edf3', 'font-family': 'ui-monospace, monospace', 'font-size': 12, 'font-weight': 600,
  }, `${metricLabel}${unit ? ' (' + unit + ')' : ''}`));

  // Render each stratum
  let yOff = pad.top;
  for (const s of validStrata) {
    const cy = yOff + stratumH / 2;
    const color = TARGET_COLORS[s.target] || '#7f7f7f';
    const labelX = pad.left - 18;

    // Two-line split on left gutter: line 1 = task & target, line 2 = count
    const taskName = (s.task || '').toUpperCase();
    const targetName = String(s.target || '').replace(/_/g, ' ').toUpperCase();
    
    svg.appendChild(el('text', {
      x: labelX, y: cy - 4,
      'text-anchor': 'end', fill: '#e6edf3', 'font-family': 'ui-monospace, monospace', 'font-size': 11, 'font-weight': 600,
    }, `${taskName} / ${targetName}`));

    svg.appendChild(el('text', {
      x: labelX, y: cy + 12,
      'text-anchor': 'end', fill: '#8b9bb4', 'font-family': 'ui-monospace, monospace', 'font-size': 10,
    }, `(n = ${s.n})`));

    // Half-violin KDE (upper half)
    if (s.kde && s.kde.points && s.kde.points.length > 1) {
      const kdePoints = s.kde.points;
      const maxDensity = Math.max(...kdePoints.map(p => p.y)) || 1;
      const violinH = stratumH * 0.36;
      let pathD = `M ${mapX(kdePoints[0].x).toFixed(2)} ${cy.toFixed(2)}`;
      for (const p of kdePoints) {
        const xp = mapX(p.x);
        const yp = cy - (p.y / maxDensity) * violinH;
        pathD += ` L ${xp.toFixed(2)} ${yp.toFixed(2)}`;
      }
      pathD += ` L ${mapX(kdePoints[kdePoints.length - 1].x).toFixed(2)} ${cy.toFixed(2)} Z`;
      const path = el('path', { d: pathD, fill: color, 'fill-opacity': 0.3, stroke: color, 'stroke-width': 1.5 });
      path.appendChild(el('title', {}, `KDE (Silverman bw=${s.kde.bandwidth})`));
      svg.appendChild(path);
    }

    // Tukey box (lower half)
    if (s.box) {
      const boxH = stratumH * 0.24;
      const boxYTop = cy + 4;
      const boxYBot = cy + 4 + boxH;
      const boxYMid = cy + 4 + boxH / 2;

      const q1x = mapX(s.box.q1);
      const q3x = mapX(s.box.q3);
      const medx = mapX(s.box.median);
      const wlox = mapX(s.box.whisker_lo);
      const whix = mapX(s.box.whisker_hi);

      // Whiskers (strictly within plot domain)
      svg.appendChild(el('line', { x1: wlox, y1: boxYMid, x2: q1x, y2: boxYMid, stroke: color, 'stroke-width': 1.5 }));
      svg.appendChild(el('line', { x1: q3x, y1: boxYMid, x2: whix, y2: boxYMid, stroke: color, 'stroke-width': 1.5 }));
      svg.appendChild(el('line', { x1: wlox, y1: boxYTop, x2: wlox, y2: boxYBot, stroke: color, 'stroke-width': 1.5 }));
      svg.appendChild(el('line', { x1: whix, y1: boxYTop, x2: whix, y2: boxYBot, stroke: color, 'stroke-width': 1.5 }));

      // Box
      const boxRect = el('rect', { x: q1x, y: boxYTop, width: Math.max(1, q3x - q1x), height: boxH, fill: color, 'fill-opacity': 0.45, stroke: color, 'stroke-width': 1.5, rx: 1 });
      boxRect.appendChild(el('title', {}, `Q1=${s.box.q1} | Median=${s.box.median} | Q3=${s.box.q3} | IQR=${s.box.iqr}`));
      svg.appendChild(boxRect);

      // Median line
      svg.appendChild(el('line', { x1: medx, y1: boxYTop, x2: medx, y2: boxYBot, stroke: '#ffffff', 'stroke-width': 2 }));

      // Outliers
      for (const ov of (s.box.outliers || [])) {
        const ox = mapX(ov);
        const oc = el('circle', { cx: ox.toFixed(2), cy: boxYMid.toFixed(2), r: 2.5, fill: 'none', stroke: color, 'stroke-width': 1.5 });
        oc.appendChild(el('title', {}, `Outlier: ${ov}`));
        svg.appendChild(oc);
      }
    }

    // Jittered raw dots
    for (const d of (s.dots || [])) {
      const dx = mapX(d.value);
      const dy = cy + stratumH * 0.38 + (d.jitter || 0) * stratumH * 0.12;
      const dot = el('circle', { cx: dx.toFixed(2), cy: dy.toFixed(2), r: 2, fill: color, 'fill-opacity': 0.6 });
      dot.appendChild(el('title', {}, `${metricLabel}=${d.value}`));
      svg.appendChild(dot);
    }

    yOff += stratumH + gap;
  }

  container.appendChild(svg);

  const caption = document.createElement('div');
  caption.className = 'cohort-caption';
  caption.textContent = `B5: Raincloud plot for ${metric} (${unit}). Half-violin KDE (Silverman bandwidth) + Tukey box (Q1/median/Q3/1.5*IQR whiskers) + jittered raw dots. n shown per stratum.`;
      container.appendChild(caption);
}

// ===========================================================================
// Main CohortView Export
// ===========================================================================

export const CohortView = {
  _data: {},
  _rawTraces: null,

  async loadArtifacts(loadedTraces = null) {
    // Live-only mode: all cohort analytics calculated in-browser from raw traces.
    // Static artifact loading has been removed — no pre-calculated JSON files needed.
    if (loadedTraces && Array.isArray(loadedTraces) && loadedTraces.length > 0) {
      this._rawTraces = loadedTraces;
      try {
        this._data = CohortEngine.computeAll(loadedTraces, 'all');
      } catch (e) {
        console.error('[CohortView] Cohort computation failed:', e);
        this._data = { _loaded: true, _error: 'Cohort computation failed: ' + e.message };
      }
      this._data._loaded = true;
      return this._data;
    }

    // No traces provided — return empty state
    this._data._loaded = true;
    this._data._error = 'No traces loaded. Upload a folder to compute cohort analytics.';
    return this._data;
  },

  async renderCohortTab(loadedTraces = null) {
    const root = document.getElementById('cohort-view');
    if (!root) return;

    // Cache raw traces for per-task recomputation on filter change
    if (loadedTraces && loadedTraces.length > 0) this._rawTraces = loadedTraces;

    root.innerHTML = '<div style="padding: 20px; color: var(--text-muted); font-family: var(--font-mono);">Computing cohort analytics live in browser...</div>';

    // Initial full-cohort computation (live mode only)
    let initialData;
    try {
      if (!loadedTraces || loadedTraces.length === 0) {
        root.innerHTML = '<div style="padding: 20px; color: var(--text-muted); font-family: var(--font-mono);">No data loaded — upload a folder to compute cohort analytics.</div>';
        return;
      }
      initialData = CohortEngine.computeAll(loadedTraces, 'all');
    } catch (e) {
      console.error('[CohortView] Cohort computation failed:', e);
      root.innerHTML = `<div style="padding:20px;color:#da3633;font-family:var(--font-mono);">Cohort computation failed: ${escapeHTML(e.message)}</div>`;
      return;
    }


    root.innerHTML = '';

    // ------------------------------------------------------------------
    // Provenance Banner — all cohort visualizations are calculated in-browser
    // ------------------------------------------------------------------
    const provBanner = document.createElement('div');
    provBanner.className = 'provenance-banner';
    provBanner.innerHTML = '<span class="provenance-badge calculated" style="cursor:default">CALCULATED</span> All cohort visualizations are calculated in-browser from uploaded traces. <a href="computations.html#cohort-analytics" target="_blank">View computation documentation &rarr;</a>';
    root.appendChild(provBanner);

    // ------------------------------------------------------------------
    // Global Task-Scope Selector Banner
    // All 5 B-views recompute dynamically when the filter changes.
    // ------------------------------------------------------------------
    const TASK_OPTIONS = [
      { key: 'all',   label: 'All Cohort' },
      { key: 'task1', label: 'Task 1: Pre-Biopsy' },
      { key: 'task2', label: 'Task 2: Stratification' },
      { key: 'task3', label: 'Task 3: Recurrence' },
    ];

    // Populate per-task case counts from initial computation.
    // Engine emits totalCases/filteredCases (top-level) and composition.tasks.{task}.total.
    // n_cases/n aliases are also present at top-level; we read canonical names first for robustness.
    if (initialData && initialData.composition && initialData.composition.tasks) {
      const t = initialData.composition.tasks;
      TASK_OPTIONS[0].n = initialData.totalCases || initialData.n_cases || initialData.filteredCases || initialData.n || (loadedTraces ? loadedTraces.length : '?');
      TASK_OPTIONS[1].n = (t.task1 && t.task1.total != null) ? t.task1.total : null;
      TASK_OPTIONS[2].n = (t.task2 && t.task2.total != null) ? t.task2.total : null;
      TASK_OPTIONS[3].n = (t.task3 && t.task3.total != null) ? t.task3.total : null;
    }

    const scopeBar = document.createElement('div');
    scopeBar.style.cssText = [
      'display:flex', 'gap:8px', 'align-items:center', 'flex-wrap:wrap',
      'margin-bottom:18px', 'padding:10px 14px',
      'background:rgba(255,255,255,0.03)',
      'border:1px solid rgba(255,255,255,0.08)', 'border-radius:6px',
    ].join(';');

    const scopeLabel = document.createElement('span');
    scopeLabel.style.cssText = 'font-size:11px;font-weight:700;font-family:var(--font-mono);color:var(--text-muted);margin-right:4px;';
    scopeLabel.textContent = 'COHORT SCOPE:';
    scopeBar.appendChild(scopeLabel);

    let activeFilter = 'all';
    const btnRefs = {};

    // The content area that gets fully re-rendered on filter change
    const contentArea = document.createElement('div');

    const renderAllViews = (data, filterKey) => {
      contentArea.innerHTML = '';

      // B1: Composition
      const b1Card = this._makeCard('B1: Cohort Composition (Task × Target)', 'ORDERED STACKED BARS');
      contentArea.appendChild(b1Card.card);
      if (data.composition) renderComposition(b1Card.body, data.composition);
      else b1Card.body.innerHTML = '<div class="case-extra-missing">Composition data unavailable for this stratum.</div>';

      // B2a: PCA MRI manifold
      const b2aCard = this._makeCard('B2: Cohort 2D Latent Manifold — MRI PCA (R-16)', '1 DOT = 1 PATIENT');
      contentArea.appendChild(b2aCard.card);
      if (data.pca_mri) renderPCA(b2aCard.body, data.pca_mri, 'MRI (1024-d)');
      else b2aCard.body.innerHTML = '<div class="case-extra-missing">MRI PCA unavailable for this stratum (insufficient cases).</div>';

      // B2b: PCA Biopsy manifold
      const b2bCard = this._makeCard('B2: Cohort 2D Latent Manifold — Biopsy PCA (R-16)', '1 DOT = 1 PATIENT');
      contentArea.appendChild(b2bCard.card);
      if (data.pca_biopsy) renderPCA(b2bCard.body, data.pca_biopsy, 'Biopsy (960-d)');
      else b2bCard.body.innerHTML = '<div class="case-extra-missing">Biopsy PCA unavailable for this stratum (insufficient cases).</div>';

      // B3: Spearman correlation heatmap
      const b3Card = this._makeCard('B3: Hierarchical Spearman Correlation Heatmap (R-17)', 'WARD-ORDERED // VIRIDIS');
      contentArea.appendChild(b3Card.card);
      if (data.correlation) renderCorrelation(b3Card.body, data.correlation, filterKey);
      else b3Card.body.innerHTML = '<div class="case-extra-missing">Correlation data unavailable for this stratum.</div>';

      // B4: Missingness grid
      const b4Card = this._makeCard('B4: Missingness Grid (R-18)', 'CASES × MODALITIES');
      contentArea.appendChild(b4Card.card);
      if (data.missingness) renderMissingness(b4Card.body, data.missingness);
      else b4Card.body.innerHTML = '<div class="case-extra-missing">Missingness data unavailable for this stratum.</div>';

      // B5: Raincloud plots (4 metrics)
      const raincloudKeys = [
        { key: 'raincloud_psa',  label: 'PSA (ng/mL)' },
        { key: 'raincloud_psad', label: 'PSA Density (ng/mL/cc)' },
        { key: 'raincloud_vol',  label: 'Prostate Volume (cc)' },
        { key: 'raincloud_age',  label: 'Age (years)' },
        { key: 'raincloud_pirads', label: 'PI-RADS (score)' },
      ];
      for (const m of raincloudKeys) {
        const card = this._makeCard(`B5: Raincloud — ${m.label} (R-19)`, 'KDE + BOX + DOTS');
        contentArea.appendChild(card.card);
        if (data[m.key]) renderRaincloud(card.body, data[m.key]);
        else card.body.innerHTML = `<div class="case-extra-missing">${m.label} raincloud unavailable for this stratum.</div>`;
      }
    };

    // Build scope-selector buttons
    TASK_OPTIONS.forEach(opt => {
      const btn = document.createElement('button');
      const nSuffix = (opt.n !== null && opt.n !== undefined) ? ` (N=${opt.n})` : '';
      btn.textContent = `${opt.label}${nSuffix}`;
      btn.className = 'cohort-tab-btn' + (opt.key === 'all' ? ' active' : '');
      btn.style.cssText = 'padding:5px 12px;font-size:11px;font-family:var(--font-mono);cursor:pointer;';
      btnRefs[opt.key] = btn;

      btn.addEventListener('click', async () => {
        if (activeFilter === opt.key) return;
        activeFilter = opt.key;

        // Update active button state
        Object.values(btnRefs).forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // If live traces available, recompute all statistics for this scope
        if (this._rawTraces && this._rawTraces.length > 0) {
          contentArea.innerHTML = `<div style="padding:20px;color:var(--text-muted);font-family:var(--font-mono);">Recomputing ${opt.label} statistics (${opt.n !== null ? 'n=' + opt.n + ' cases' : 'subset'})...</div>`;
          // Double-rAF: guarantees one paint cycle between setting the loading
          // message and starting the blocking compute, so the "Recomputing..."
          // state actually renders before the main thread is monopolized.
          await new Promise(resolve => {
            requestAnimationFrame(() => {
              requestAnimationFrame(() => resolve());
            });
          });
          let filteredData;
          try {
            filteredData = CohortEngine.computeAll(this._rawTraces, opt.key);
          } catch (e) {
            console.error('[CohortView] Cohort computation failed:', e);
            contentArea.innerHTML = `<div style="padding:20px;color:#da3633;font-family:var(--font-mono);">Cohort computation failed: ${escapeHTML(e.message)}</div>`;
            return;
          }
          renderAllViews(filteredData, opt.key);
        } else {
          // Static JSON fallback: re-render with same data (no per-task re-slicing in static mode)
          renderAllViews(initialData, opt.key);
        }
      });

      scopeBar.appendChild(btn);
    });

    root.appendChild(scopeBar);
    root.appendChild(contentArea);

    // Initial render: full cohort
    renderAllViews(initialData, 'all');
  },

  _makeCard(title, subtitle) {
    const card = document.createElement('section');
    card.className = 'cohort-card panel-full';
    const header = document.createElement('div');
    header.className = 'cohort-card-header';
    const titleSpan = document.createElement('span');
    titleSpan.textContent = title;
    const subSpan = document.createElement('span');
    subSpan.style.fontSize = '10px';
    subSpan.style.color = 'var(--text-muted)';
    subSpan.textContent = subtitle;
    header.appendChild(titleSpan);
    header.appendChild(subSpan);
    const body = document.createElement('div');
    body.className = 'cohort-card-body';
    card.appendChild(header);
    card.appendChild(body);
    return { card, body };
  },
};

// standard_components.js — Pure Vanilla JS/SVG Clinical Renderers
//
// Zero-dependency clinical visualisation module. Pure native browser APIs:
// document.createElementNS for SVG, plain HTML elements for grids/tables.
// No 3D, no GPU-accelerated graphics, no external charting libraries. All axes start at 0
// and use strict linear scales (Cleveland-McGill perceptual rigour).

import { EAU_SENTINEL, isSentinel } from './constants.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function svgEl(name, attrs) {
  const el = document.createElementNS(SVG_NS, name);
  if (attrs) {
    for (const k in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) {
        el.setAttribute(k, attrs[k]);
      }
    }
  }
  return el;
}

function svgText(x, y, content, opts) {
  opts = opts || {};
  const t = svgEl('text', {
    x: x,
    y: y,
    'font-family': 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
    'font-size': opts.size || 10,
    fill: opts.fill || '#2c3e50',
    'text-anchor': opts.anchor || 'start'
  });
  if (opts.weight) t.setAttribute('font-weight', opts.weight);
  if (opts.transform) t.setAttribute('transform', opts.transform);
  t.textContent = content;
  return t;
}

function fmtNum(n, digits) {
  if (digits === undefined) digits = 1;
  if (n === null || n === undefined || (typeof n === 'number' && isNaN(n))) return '';
  return Number(n).toFixed(digits);
}

// Shared missing-value predicate for clinical scalars. Centralises the
// null/undefined/empty-string/NaN guard so the demographics status badges
// (standard_view.js) and the bullet-strip renderer never disagree on what
// counts as "missing" — the root cause of M-5 (empty-string PSA/PSAD/VOL
// rendered MISSING in the value cell but NORMAL/LOW RISK in the status badge,
// because Number("") === 0 slips past a bare isNaN guard).
export function isMissingClinicalValue(val) {
  return val === null || val === undefined || val === '' ||
    (typeof val === 'number' && isNaN(val));
}

// ---------------------------------------------------------------------------
// 1. renderClevelandBulletStrip
//    Horizontal bullet strip: a scalar value positioned against clinical
//    threshold bands. Position-on-a-common-scale (Cleveland-McGell #1, the
//    most accurate perceptual task). Strict linear scale, baseline at domain
//    minimum. No 3D, no GPU, no external libraries.
//
//    LAYOUT INVARIANT (deterministic, collision-free text placement):
//      The SVG viewport is partitioned into four non-overlapping horizontal
//      bands — chip strip (y 2..18), band-label zone (y 22..30), strip/dot
//      zone (y 32..44), tick zone (y 52..66). No two rendered text boxes can
//      intersect because:
//        (a) chip vs band-label vs tick-label occupy disjoint y-ranges;
//        (b) band labels are emitted only when band width >= 70px, and adjacent
//            bands are non-overlapping intervals so their centered labels are
//            separated by >= 70px > max label width (~54px);
//        (c) tick labels are filtered by an interval-occupancy sweep that
//            rejects any label whose box intersects an already-kept label;
//        (d) the value chip is clamped to [2, W-2] so it never leaves the SVG.
// ---------------------------------------------------------------------------

export function renderClevelandBulletStrip(container, value, thresholds) {
  if (!container) return;
  while (container.firstChild) container.removeChild(container.firstChild);

  const W = 400, H = 70;
  const svg = svgEl('svg', {
    viewBox: '0 0 ' + W + ' ' + H,
    width: '100%',
    xmlns: SVG_NS,
    preserveAspectRatio: 'xMidYMid meet'
  });

  // Missing-value guard: render a grey MISSING row, no strip. Uses the shared
  // isMissingClinicalValue predicate so empty-string values (Number("") === 0)
  // are treated as missing here too, matching the demographics status badges.
  if (isMissingClinicalValue(value) ||
      (typeof value === 'string' && String(value).trim().toUpperCase() === 'N/A')) {
    svg.appendChild(svgEl('rect', {
      x: 10, y: 22, width: W - 20, height: 30,
      fill: '#21262d', stroke: '#30363d', 'stroke-width': 1, rx: 3
    }));
    svg.appendChild(svgText(W / 2, 42, 'MISSING', {
      anchor: 'middle', fill: '#8b9bb4', size: 12, weight: '700'
    }));
    container.appendChild(svg);
    return;
  }

  // Guard against missing/invalid thresholds object — render MISSING fallback
  // rather than throwing TypeError on property access.
  if (!thresholds || typeof thresholds !== 'object') {
    svg.appendChild(svgEl('rect', {
      x: 10, y: 22, width: W - 20, height: 30,
      fill: '#21262d', stroke: '#30363d', 'stroke-width': 1, rx: 3
    }));
    svg.appendChild(svgText(W / 2, 42, 'MISSING', {
      anchor: 'middle', fill: '#8b9bb4', size: 12, weight: '700'
    }));
    container.appendChild(svg);
    return;
  }

  const v = Number(value);
  const nm = Number(thresholds.normal_max);
  const bm = Number(thresholds.borderline_max);

  // Domain: explicit per-metric config, else auto-compute (backward compatible).
  let dMin, dMax;
  if (Array.isArray(thresholds.domain) && thresholds.domain.length === 2 &&
      typeof thresholds.domain[0] === 'number' && typeof thresholds.domain[1] === 'number') {
    dMin = thresholds.domain[0];
    dMax = thresholds.domain[1];
  } else {
    dMin = 0;
    dMax = Math.max(v * 1.2, bm * 1.5, 20);
  }

  // Strict linear scale, baseline at dMin. No log, no truncation.
  const padX = 10;
  const plotW = W - 2 * padX;
  const span = Math.max(1e-6, dMax - dMin);
  const xOf = function (val) { return padX + ((val - dMin) / span) * plotW; };

  // Clamp detection for values outside the domain (e.g. PSAD > 0.6 or < 0).
  const upperClamped = v > dMax;
  const lowerClamped = v < dMin;
  const dotVal = Math.max(dMin, Math.min(v, dMax));

  // Background bands — clinical semantic fills, opaque for label contrast.
  // Band label color chosen per band: dark on grey/amber, white on red.
  const bandDefs = [];
  if (typeof thresholds.normal_min === 'number') {
    bandDefs.push({ from: dMin, to: thresholds.normal_min, fill: '#6c757d', label: 'Atrophic', labelFill: '#ffffff' });
    bandDefs.push({ from: thresholds.normal_min, to: nm, fill: '#e0e0e0', label: 'Normal', labelFill: '#1f2328' });
  } else {
    bandDefs.push({ from: dMin, to: nm, fill: '#e0e0e0', label: 'Normal', labelFill: '#1f2328' });
  }
  bandDefs.push({ from: nm, to: bm, fill: '#f0ad4e', label: 'Elevated', labelFill: '#1f2328' });
  bandDefs.push({ from: bm, to: dMax, fill: '#d9534f', label: thresholds.pathological_label || 'Pathological', labelFill: '#ffffff' });

  // Layout constants (see LAYOUT INVARIANT above).
  const bandY = 22, bandH = 30, stripY = 38, bandLabelY = 30;
  const tickTop = 52, tickBottom = 58, tickLabelY = 66;

  bandDefs.forEach(function (b) {
    const x0 = xOf(b.from);
    const x1 = xOf(b.to);
    svg.appendChild(svgEl('rect', {
      x: x0, y: bandY, width: Math.max(0, x1 - x0), height: bandH,
      fill: b.fill, 'fill-opacity': '0.9'
    }));
    // Band label only if band width >= 70px (collision-free guarantee).
    if ((x1 - x0) >= 70) {
      svg.appendChild(svgText((x0 + x1) / 2, bandLabelY, b.label, {
        anchor: 'middle', fill: b.labelFill, size: 9, weight: '700'
      }));
    }
  });

  // Bullet line spanning the band domain at stripY.
  svg.appendChild(svgEl('line', {
    x1: xOf(dMin), y1: stripY, x2: xOf(dMax), y2: stripY,
    stroke: '#1f2328', 'stroke-width': 2
  }));

  // Tick set: explicit per-metric, else auto [dMin, nm, bm, dMax].
  let tickVals;
  if (Array.isArray(thresholds.ticks) && thresholds.ticks.length > 0) {
    tickVals = thresholds.ticks.slice();
  } else {
    tickVals = [dMin, nm, bm, dMax];
  }
  // De-duplicate & sort ascending.
  tickVals = tickVals.filter(function (val, idx, arr) {
    return arr.indexOf(val) === idx;
  }).sort(function (a, b) { return a - b; });

  // Priority: clinical-threshold ticks rank above the 0 and domain-max endpoints.
  const thresholdSet = {};
  thresholdSet[String(nm)] = true;
  thresholdSet[String(bm)] = true;
  if (Array.isArray(thresholds.ticks)) {
    thresholds.ticks.forEach(function (t) { thresholdSet[String(t)] = true; });
  }

  const tickDefs = tickVals.map(function (val) {
    const x = xOf(val);
    const digits = val < 1 ? 2 : (val >= 10 ? 0 : 1);
    const label = fmtNum(val, digits);
    const isEndpoint = (Math.abs(val - dMin) < 1e-9) || (Math.abs(val - dMax) < 1e-9);
    const pri = isEndpoint ? 0 : (thresholdSet[String(val)] ? 1 : 0);
    const boxW = label.length * 5.4 + 4; // monospace 9px ~5.4px/char + padding
    return { val: val, x: x, label: label, pri: pri,
             lo: x - boxW / 2, hi: x + boxW / 2 };
  });

  // Collision avoidance: process by priority desc then x asc; reject any tick
  // whose label box intersects an already-kept tick's box. Guarantees no two
  // rendered tick labels overlap (deterministic placement invariant).
  tickDefs.sort(function (a, b) { return b.pri - a.pri || a.x - b.x; });
  const keptTicks = [];
  const occupied = [];
  tickDefs.forEach(function (tk) {
    let conflict = false;
    for (let i = 0; i < occupied.length; i++) {
      if (tk.lo < occupied[i].hi && tk.hi > occupied[i].lo) { conflict = true; break; }
    }
    if (!conflict) { keptTicks.push(tk); occupied.push({ lo: tk.lo, hi: tk.hi }); }
  });
  keptTicks.sort(function (a, b) { return a.x - b.x; });

  keptTicks.forEach(function (tk) {
    svg.appendChild(svgEl('line', {
      x1: tk.x, y1: tickTop, x2: tk.x, y2: tickBottom,
      stroke: '#1f2328', 'stroke-width': 1
    }));
    svg.appendChild(svgText(tk.x, tickLabelY, tk.label, {
      anchor: 'middle', fill: '#8b9bb4', size: 9
    }));
  });

  // Value dot (clamped to domain) with native tooltip.
  const vx = xOf(dotVal);
  const dot = svgEl('circle', {
    cx: vx, cy: stripY, r: 6, fill: '#1f2328',
    stroke: '#ffffff', 'stroke-width': 1.5
  });
  const dotTitle = svgEl('title', {});
  dotTitle.textContent = 'Value: ' + fmtNum(v, 2) +
                         (thresholds.unit ? ' ' + thresholds.unit : '');
  dot.appendChild(dotTitle);
  svg.appendChild(dot);

  // ">" arrow when value is clamped beyond domain max; "<" when below min.
  if (upperClamped) {
    svg.appendChild(svgEl('path', {
      d: 'M ' + (vx - 4) + ',' + (stripY - 5) + ' L ' + (vx + 4) + ',' + stripY +
         ' L ' + (vx - 4) + ',' + (stripY + 5),
      fill: 'none', stroke: '#1f2328', 'stroke-width': 2
    }));
  }
  if (lowerClamped) {
    svg.appendChild(svgEl('path', {
      d: 'M ' + (vx + 4) + ',' + (stripY - 5) + ' L ' + (vx - 4) + ',' + stripY +
         ' L ' + (vx + 4) + ',' + (stripY + 5),
      fill: 'none', stroke: '#1f2328', 'stroke-width': 2
    }));
  }

  // Value chip ABOVE the strip, anchored to dot x, clamped to SVG bounds.
  const chipUnit = thresholds.unit || '';
  const chipText = (upperClamped ? '>' : (lowerClamped ? '<' : '')) +
                   fmtNum(v, v < 10 ? 2 : 1) +
                   (chipUnit ? ' ' + chipUnit : '');
  const chipFont = 12;
  const chipCharW = chipFont * 0.6; // monospace char width approx
  const chipPadX = 6, chipPadY = 3;
  const chipW = chipText.length * chipCharW + 2 * chipPadX;
  const chipH = chipFont + 2 * chipPadY;
  const chipY = 2;
  let chipX = vx - chipW / 2;
  // Clamp so the chip never leaves the SVG viewBox.
  if (chipX < 2) chipX = 2;
  if (chipX + chipW > W - 2) chipX = W - 2 - chipW;
  // If the chip is wider than the SVG plot area, pin to left edge to avoid
  // a negative chipX from the right-edge clamp overriding the left clamp.
  if (chipW > W - 4) chipX = 2;

  svg.appendChild(svgEl('rect', {
    x: chipX, y: chipY, width: chipW, height: chipH,
    fill: '#0d1117', stroke: '#30363d', 'stroke-width': 1, rx: 3
  }));
  svg.appendChild(svgText(chipX + chipW / 2, chipY + chipH / 2 + 4, chipText, {
    anchor: 'middle', fill: '#e6edf3', size: chipFont, weight: '700'
  }));

  container.appendChild(svg);
}


// ---------------------------------------------------------------------------
// 2. renderEAUScorecard
//    HTML grid of the 5 EAU risk tiers with the active tier highlighted.
// ---------------------------------------------------------------------------

export function renderEAUScorecard(container, tier, criteria_matched) {
  if (!container) return;
  while (container.firstChild) container.removeChild(container.firstChild);

  const tiers = [
    { name: 'Low', color: '#28a745', criteria: 'PSA<10, ISUP 1, cT1-cT2a' },
    { name: 'Favorable Intermediate', color: '#5cb85c', criteria: 'PSA<10, ISUP 2, cT≤cT2a, no high-risk patterns' },
    { name: 'Unfavorable Intermediate', color: '#f0ad4e', criteria: 'ISUP 3, PSA 10-20 + ISUP 2, cT2b, or ISUP 2 + high-risk' },
    { name: 'High', color: '#e67e22', criteria: 'ISUP 4/5, PSA>20, or cT2c' },
    { name: 'Locally Advanced', color: '#d9534f', criteria: 'cT3-4' }
  ];

  // Normalized missing-sentinel check via shared constants module.
  const isMissing = isSentinel(tier);
  const isUnclassified = tier === 'Intermediate (unclassified)';

  if (isMissing) {
    const warn = document.createElement('div');
    warn.style.color = '#d9534f';
    warn.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    warn.style.fontSize = '12px';
    warn.style.fontWeight = '700';
    warn.style.marginBottom = '10px';
    warn.textContent = 'Insufficient data for EAU stratification';
    container.appendChild(warn);
  } else if (isUnclassified) {
    const warn = document.createElement('div');
    warn.style.color = '#8a6d3b';
    warn.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    warn.style.fontSize = '12px';
    warn.style.fontWeight = '700';
    warn.style.marginBottom = '10px';
    warn.textContent = 'Intermediate (unclassified) — data incomplete for Favorable/Unfavorable stratification';
    container.appendChild(warn);
  }

  const grid = document.createElement('div');
  grid.style.display = 'grid';
  grid.style.gridTemplateColumns = 'repeat(5, 1fr)';
  grid.style.gap = '10px';

  tiers.forEach(function (tdef) {
    const isActive = !isMissing && !isUnclassified && tier === tdef.name;
    const isPartial = isUnclassified && (tdef.name === 'Favorable Intermediate' || tdef.name === 'Unfavorable Intermediate');
    const card = document.createElement('div');
    card.style.border = '1px solid #cccccc';
    card.style.borderRadius = '6px';
    card.style.padding = '10px';
    card.style.textAlign = 'center';
    card.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    card.style.background = '#ffffff';
    card.style.borderTopColor = tdef.color;
    card.style.borderTopWidth = '4px';
    if (isActive) {
      card.style.border = '3px solid #2c3e50';
      card.style.borderTopColor = tdef.color;
      card.style.borderTopWidth = '4px';
      card.style.transform = 'scale(1.05)';
      card.style.position = 'relative';
    } else if (isPartial) {
      card.style.border = '2px dashed #8a6d3b';
      card.style.borderTopColor = tdef.color;
      card.style.borderTopWidth = '4px';
      card.style.opacity = '0.75';
    } else {
      card.style.opacity = '0.5';
    }

    const name = document.createElement('div');
    name.style.fontSize = '12px';
    name.style.fontWeight = '700';
    name.style.marginBottom = '6px';
    name.style.color = tdef.color;
    name.textContent = tdef.name;
    card.appendChild(name);

    const crit = document.createElement('div');
    crit.style.fontSize = '10px';
    crit.style.color = '#555555';
    crit.style.lineHeight = '1.3';
    crit.textContent = tdef.criteria;
    card.appendChild(crit);

    if (isActive) {
      const badge = document.createElement('span');
      badge.style.display = 'inline-block';
      badge.style.marginTop = '6px';
      badge.style.background = '#2c3e50';
      badge.style.color = '#ffffff';
      badge.style.fontSize = '9px';
      badge.style.fontWeight = '700';
      badge.style.padding = '2px 6px';
      badge.style.borderRadius = '3px';
      badge.style.letterSpacing = '0.5px';
      badge.textContent = 'ACTIVE';
      card.appendChild(badge);
    }
    grid.appendChild(card);
  });
  container.appendChild(grid);

  const line = document.createElement('div');
  line.style.marginTop = '10px';
  line.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  line.style.fontSize = '11px';
  line.style.color = '#e6edf3';
  line.textContent = 'Criteria Matched: ' +
    (isMissing ? EAU_SENTINEL : (isUnclassified ? (criteria_matched || 'Intermediate (unclassified)') : (criteria_matched || '—')));
  container.appendChild(line);
}

// ---------------------------------------------------------------------------
// 3. renderConcordanceMatrix
//    HTML table: PI-RADS (cols) vs ISUP (rows) agreement matrix.
// ---------------------------------------------------------------------------

export function renderConcordanceMatrix(container, pirads, isup) {
  if (!container) return;
  while (container.firstChild) container.removeChild(container.firstChild);

  const wrap = document.createElement('div');
  wrap.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

  const pNum = (pirads === null || pirads === undefined) ? null : Number(pirads);
  const iNum = (isup === null || isup === undefined) ? null : Number(isup);
  // Treat out-of-range values (PI-RADS/ISUP must be 1-5) as missing so a
  // verdict is never rendered without a corresponding patient cell anchor.
  const outOfRange = (pNum !== null && !isNaN(pNum) && (pNum < 1 || pNum > 5)) ||
                     (iNum !== null && !isNaN(iNum) && (iNum < 1 || iNum > 5));
  const missing = (pNum === null || isNaN(pNum)) || (iNum === null || isNaN(iNum)) || outOfRange;

  // Concordance classification for a (pirads, isup) cell
  function classify(p, i) {
    if (p === 3) return 'equivocal';
    if ((p >= 4 && i >= 2) || (p <= 2 && i === 1)) return 'concordant';
    if ((p >= 4 && i === 1) || (p <= 2 && i >= 2)) return 'discordant';
    return 'equivocal';
  }
  const colorOf = function (c) {
    if (c === 'concordant') return '#d4edda';
    if (c === 'discordant') return '#f8d7da';
    return '#fff3cd';
  };

  // Header cells: light grey background (#d0d7de) with DARK text (#1f2328) so
  // axis labels are readable — never pale text on a pale background.
  const HDR_BG = '#d0d7de';
  const HDR_FG = '#1f2328';
  const HDR_BORDER = '#1f2328';

  const table = document.createElement('table');
  table.style.borderCollapse = 'collapse';
  table.style.margin = '6px 0';

  const thead = document.createElement('thead');

  // Axis-title row: "PI-RADS ->" spanning all 5 columns.
  const axisRow = document.createElement('tr');
  const axisCorner = document.createElement('th');
  axisCorner.style.border = '1px solid ' + HDR_BORDER;
  axisCorner.style.padding = '6px 10px';
  axisCorner.style.background = HDR_BG;
  axisCorner.style.color = HDR_FG;
  axisCorner.style.fontSize = '11px';
  axisCorner.style.fontWeight = '700';
  axisCorner.style.textAlign = 'center';
  axisCorner.textContent = 'ISUP \u2193';
  axisRow.appendChild(axisCorner);
  const piradsAxis = document.createElement('th');
  piradsAxis.colSpan = 5;
  piradsAxis.style.border = '1px solid ' + HDR_BORDER;
  piradsAxis.style.padding = '6px 10px';
  piradsAxis.style.background = HDR_BG;
  piradsAxis.style.color = HDR_FG;
  piradsAxis.style.fontSize = '11px';
  piradsAxis.style.fontWeight = '700';
  piradsAxis.style.textAlign = 'center';
  piradsAxis.textContent = 'PI-RADS \u2192';
  axisRow.appendChild(piradsAxis);
  thead.appendChild(axisRow);

  // Column-header row: PI-RADS 1..5
  const hrow = document.createElement('tr');
  const corner = document.createElement('th');
  corner.style.border = '1px solid ' + HDR_BORDER;
  corner.style.padding = '6px 10px';
  corner.style.background = HDR_BG;
  corner.style.fontSize = '11px';
  hrow.appendChild(corner);
  for (let p = 1; p <= 5; p++) {
    const th = document.createElement('th');
    th.textContent = 'PI-RADS ' + p;
    th.style.border = '1px solid ' + HDR_BORDER;
    th.style.padding = '6px 14px';
    th.style.background = HDR_BG;
    th.style.color = HDR_FG;
    th.style.fontSize = '12px';
    th.style.fontWeight = '700';
    th.style.textAlign = 'center';
    hrow.appendChild(th);
  }
  thead.appendChild(hrow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (let i = 1; i <= 5; i++) {
    const tr = document.createElement('tr');
    const rh = document.createElement('th');
    rh.textContent = 'ISUP ' + i;
    rh.style.border = '1px solid ' + HDR_BORDER;
    rh.style.padding = '6px 10px';
    rh.style.background = HDR_BG;
    rh.style.color = HDR_FG;
    rh.style.fontSize = '12px';
    rh.style.fontWeight = '700';
    rh.style.textAlign = 'left';
    tr.appendChild(rh);
    for (let p = 1; p <= 5; p++) {
      const td = document.createElement('td');
      td.style.border = '1px solid #bdc3c7';
      td.style.padding = '8px 14px';
      td.style.textAlign = 'center';
      td.style.fontSize = '14px';
      td.style.background = colorOf(classify(p, i));

      // Native <title> tooltip on every cell.
      const tt = document.createElement('title');
      tt.textContent = 'PI-RADS ' + p + ' x ISUP ' + i;
      td.appendChild(tt);

      const isPatientCell = !missing && (p === pNum && i === iNum);
      if (isPatientCell) {
        // 2px dark stroke plus a dark dot — visible on all cell colors.
        td.style.border = '2px solid #1f2328';
        td.style.fontWeight = '700';
        const dot = document.createElement('span');
        dot.textContent = '\u25CF';
        dot.style.color = '#1f2328';
        dot.style.fontSize = '16px';
        td.appendChild(dot);
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);

  // One-line legend explaining cell color semantics.
  const legend = document.createElement('div');
  legend.style.fontSize = '11px';
  legend.style.marginTop = '6px';
  legend.style.color = '#e6edf3';
  legend.style.display = 'flex';
  legend.style.gap = '16px';
  legend.style.flexWrap = 'wrap';
  const legendItems = [
    { swatch: '#d4edda', label: 'Concordant' },
    { swatch: '#fff3cd', label: 'Mild tension / Equivocal' },
    { swatch: '#f8d7da', label: 'Discordant' }
  ];
  legendItems.forEach(function (item) {
    const span = document.createElement('span');
    span.style.display = 'inline-flex';
    span.style.alignItems = 'center';
    span.style.gap = '5px';
    const sw = document.createElement('span');
    sw.style.display = 'inline-block';
    sw.style.width = '12px';
    sw.style.height = '12px';
    sw.style.background = item.swatch;
    sw.style.border = '1px solid #1f2328';
    sw.style.borderRadius = '2px';
    span.appendChild(sw);
    span.appendChild(document.createTextNode(item.label));
    legend.appendChild(span);
  });
  wrap.appendChild(legend);

  // Status text — light colors for dark-theme AA contrast (>=4.5:1).
  const status = document.createElement('div');
  status.style.fontSize = '12px';
  status.style.marginTop = '6px';

  if (missing) {
    status.style.color = '#d29922';
    status.style.fontWeight = '700';
    status.textContent = 'Insufficient Data for Concordance Analysis';
  } else {
    const cls = classify(pNum, iNum);
    if (cls === 'concordant') {
      status.style.color = '#3fb950';
      status.style.fontWeight = '700';
      status.textContent = 'Concordant';
    } else if (cls === 'discordant') {
      status.style.color = '#f85149';
      status.style.fontWeight = '700';
      status.textContent = 'Discordant';
      const warn = document.createElement('div');
      warn.style.color = '#f85149';
      warn.style.fontSize = '11px';
      warn.style.marginTop = '4px';
      const pLabel = 'PI-RADS ' + pNum;
      const iLabel = 'ISUP ' + iNum;
      let msg;
      if (pNum >= 4 && iNum === 1) {
        msg = pLabel + ' but ' + iLabel +
              ': Possible biopsy targeting miss or indolent index lesion.';
      } else if (pNum <= 2 && iNum >= 2) {
        msg = pLabel + ' but ' + iLabel +
              ': Histologically significant disease despite low-suspicion MRI \u2014 possible occult lesion.';
      } else {
        msg = pLabel + ' vs ' + iLabel +
              ': Imaging-histology discordance warrants multidisciplinary review.';
      }
      warn.textContent = msg;
      wrap.appendChild(status);
      wrap.appendChild(warn);
      container.appendChild(wrap);
      return;
    } else {
      status.style.color = '#d29922';
      status.style.fontWeight = '700';
      status.textContent = 'Equivocal';
    }
  }
  wrap.appendChild(status);
  container.appendChild(wrap);
}

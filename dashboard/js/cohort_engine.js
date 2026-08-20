// CHIMERA-Agent In-Browser Cohort Analytics Engine (cohort_engine.js)
// Zero-Dependency Pure ES6 Module for Real-time Cohort Statistics, PCA, Spearman-Ward & KDE.
// [OFFICIAL: RESEARCHER-APPROVED] Exact mathematical specifications for Phase B artifacts.
// [SUGGESTION: CO-PILOT] High-performance pure JS implementation.

import { safeFloat as importedSafeFloat } from './clinical_engine.js';

export function safeFloat(val) {
  if (typeof importedSafeFloat === 'function') {
    return importedSafeFloat(val);
  }
  if (val === null || val === undefined || typeof val === 'boolean') return null;
  const s = String(val).trim();
  if (s === '' || s.toUpperCase() === 'N/A' || s.toUpperCase() === 'NOT AVAILABLE' || s.toUpperCase() === 'MISSING') {
    return null;
  }
  const n = parseFloat(s);
  return Number.isNaN(n) ? null : n;
}

export const NUMERIC_VARS = [
  'psa', 'psad', 'vol', 'age', 'pirads',
  'bx_isup', 'bx_gl_prim', 'bx_gl_sec',
  'cspca', 'cores_positive', 'cores_total',
];

export const METRIC_UNITS = {
  psa: 'ng/mL',
  psad: 'ng/mL/cc',
  vol: 'cc',
  age: 'years',
  pirads: '',
  bx_isup: '',
  bx_gl_prim: '',
  bx_gl_sec: '',
  cspca: '',
  cores_positive: '',
  cores_total: '',
};

export const MODALITIES = [
  'MRI', 'Biopsy', 'Prostatectomy', 'PSA_Trend', 'Labs', 'FamilyHistory'
];

export const WEYL_PHI = 0.6180339887498949; // (sqrt(5) - 1) / 2

function round6(val) {
  if (val === null || val === undefined || Number.isNaN(val)) return null;
  return Math.round(val * 1e6) / 1e6;
}

function round2(val) {
  if (val === null || val === undefined || Number.isNaN(val)) return null;
  return Math.round(val * 100) / 100;
}

// ---------------------------------------------------------------------------
// 1. Math Utilities (Ranks, Correlation, Silverman KDE, Tukey Box, PCA)
// ---------------------------------------------------------------------------

/**
 * Compute 1-based ranks with average mid-ranks for tied values.
 * @param {number[]} values
 * @returns {number[]}
 */
export function rankData(values) {
  if (!values || !Array.isArray(values) || values.length === 0) return [];
  const n = values.length;
  const indexed = Array.from({ length: n }, (_, i) => i).sort((a, b) => {
    const va = values[a], vb = values[b];
    const naN_a = Number.isNaN(va), naN_b = Number.isNaN(vb);
    if (naN_a && naN_b) return 0;
    if (naN_a) return 1;  // NaN → end of array
    if (naN_b) return -1; // NaN → end of array
    return va - vb;
  });
  const ranks = new Array(n).fill(0);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && values[indexed[j + 1]] === values[indexed[i]]) {
      j++;
    }
    const avgRank = (i + 1 + j + 1) / 2.0;
    for (let k = i; k <= j; k++) {
      ranks[indexed[k]] = avgRank;
    }
    i = j + 1;
  }
  return ranks;
}

/**
 * Pearson correlation coefficient between two numeric vectors.
 * @param {number[]} x
 * @param {number[]} y
 * @returns {number|null}
 */
export function pearsonR(x, y) {
  if (!x || !y || x.length !== y.length) return null;
  const n = x.length;
  if (n < 3) return null;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  const denom = Math.sqrt(sxx * syy);
  return (denom === 0 || Number.isNaN(denom)) ? null : sxy / denom;
}

/**
 * Spearman rank correlation coefficient on pairwise complete observations.
 * Requires at least 5 complete pairs.
 * @param {Array<number|null>} x
 * @param {Array<number|null>} y
 * @returns {number|null}
 */
export function spearmanRho(x, y) {
  if (!x || !y || x.length !== y.length) return null;
  const pairs = [];
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = y[i];
    if (xi !== null && xi !== undefined && !Number.isNaN(xi) &&
        yi !== null && yi !== undefined && !Number.isNaN(yi)) {
      pairs.push([xi, yi]);
    }
  }
  if (pairs.length < 5) return null;
  const xs = pairs.map(p => p[0]);
  const ys = pairs.map(p => p[1]);
  const rx = rankData(xs);
  const ry = rankData(ys);
  return pearsonR(rx, ry);
}

/**
 * Ward's minimum-variance hierarchical agglomerative clustering.
 * Uses the Lance-Williams recurrence for Ward linkage on squared distance D^2 = (1 - |rho|)^2.
 * Returns deterministic left-leaf traversal order of variable indices.
 * @param {Array<Array<number|null>>} rhoMatrix
 * @param {string[]} varNames
 * @returns {number[]}
 */
export function wardCluster(rhoMatrix, varNames) {
  const k = varNames ? varNames.length : (rhoMatrix ? rhoMatrix.length : 0);
  if (k <= 1) return Array.from({ length: k }, (_, i) => i);

  const dist2 = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => {
      if (i === j) return 0.0;
      const r = rhoMatrix[i][j];
      const d = (r === null || r === undefined || Number.isNaN(r)) ? 1.0 : (1.0 - Math.abs(r));
      return d * d;
    })
  );

  const clusterMembers = Array.from({ length: k }, (_, i) => [i]);
  const clusterSizes = new Array(k).fill(1);
  const active = Array.from({ length: k }, (_, i) => i);

  const cd = {};
  for (let a = 0; a < k; a++) {
    for (let b = a + 1; b < k; b++) {
      cd[`${a},${b}`] = dist2[a][b];
    }
  }

  let lastActive = 0;
  while (active.length > 1) {
    let bestPair = null;
    let bestDist = Infinity;

    for (let i = 0; i < active.length; i++) {
      for (let j = i + 1; j < active.length; j++) {
        const a = active[i];
        const b = active[j];
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        const d = cd[key];
        if (d !== undefined && d < bestDist) {
          bestDist = d;
          bestPair = [a, b];
        }
      }
    }

    if (!bestPair) break;
    const [a, b] = bestPair;
    const na = clusterSizes[a];
    const nb = clusterSizes[b];

    clusterMembers[a] = clusterMembers[a].concat(clusterMembers[b]);
    clusterSizes[a] = na + nb;
    const bIdx = active.indexOf(b);
    active.splice(bIdx, 1);
    lastActive = a;

    for (const c of active) {
      if (c === a) continue;
      const nc = clusterSizes[c];
      const keyCA = Math.min(c, a) + ',' + Math.max(c, a);
      const keyCB = Math.min(c, b) + ',' + Math.max(c, b);
      const keyAB = Math.min(a, b) + ',' + Math.max(a, b);

      const dCA = cd[keyCA] !== undefined ? cd[keyCA] : 1.0;
      const dCB = cd[keyCB] !== undefined ? cd[keyCB] : 1.0;
      const dAB = cd[keyAB] !== undefined ? cd[keyAB] : 1.0;

      // Lance-Williams Ward formula on squared distances
      const num = (nc + na) * dCA + (nc + nb) * dCB - nc * dAB;
      const denom = nc + na + nb;
      cd[keyCA] = denom > 0 ? num / denom : 0;
    }

    // Cleanup keys containing b
    for (const key of Object.keys(cd)) {
      const parts = key.split(',');
      if (parts[0] === String(b) || parts[1] === String(b)) {
        delete cd[key];
      }
    }
  }

  const result = clusterMembers[lastActive];
  if (result && result.length === k) return result;
  return Array.from({ length: k }, (_, i) => i);
}

/**
 * Compute Spearman correlation matrix for a dictionary of feature vectors with Ward clustering.
 * @param {Object.<string, Array<number|null>>} featureVectors
 * @param {number} minObs
 * @returns {Object}
 */
export function spearmanCorrelationMatrix(featureVectors, minObs = 5) {
  if (!featureVectors || typeof featureVectors !== 'object') {
    return { matrix: [], features: [], variables: [], dendrogramOrder: [], excluded: [] };
  }
  const featureNames = Object.keys(featureVectors);
  const usableVars = [];
  for (const name of featureNames) {
    const arr = featureVectors[name];
    if (Array.isArray(arr)) {
      const nPresent = arr.filter(x => x !== null && x !== undefined && !Number.isNaN(x)).length;
      if (nPresent >= minObs) usableVars.push(name);
    }
  }

  const k = usableVars.length;
  if (k === 0) {
    return {
      matrix: [],
      features: [],
      variables: [],
      dendrogramOrder: [],
      n_variables: 0,
      method: 'Spearman rho',
      excluded: featureNames,
    };
  }

  const rhoMatrix = Array.from({ length: k }, () => new Array(k).fill(null));
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      if (i === j) {
        rhoMatrix[i][j] = 1.0;
      } else if (j > i) {
        const r = spearmanRho(featureVectors[usableVars[i]], featureVectors[usableVars[j]]);
        rhoMatrix[i][j] = r;
        rhoMatrix[j][i] = r;
      }
    }
  }

  const order = wardCluster(rhoMatrix, usableVars);
  const displayMatrix = rhoMatrix.map(row => row.map(v => v === null ? null : Math.round(v * 10000) / 10000));
  const orderedMatrix = order.map(i => order.map(j => displayMatrix[i][j]));
  const orderedNames = order.map(idx => usableVars[idx]);

  return {
    matrix: orderedMatrix,
    features: orderedNames,
    variables: orderedNames,
    dendrogramOrder: order,
    rawMatrix: rhoMatrix.map(row => [...row]),
    n_variables: orderedNames.length,
    method: 'Spearman rho (Pearson on ranks, pairwise complete). Ward agglomerative clustering on squared distance = (1 - |ρ|)².',
    excluded: featureNames.filter(v => !usableVars.includes(v)),
  };
}

/**
 * Tukey 5-number summary with Type-7 continuous quantile interpolation and 1.5*IQR whiskers.
 * @param {number[]} values
 * @returns {Object|null}
 */
export function tukeyBox(values) {
  if (!values || !Array.isArray(values) || values.length === 0) return null;
  const vs = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v) && typeof v === 'number' && Number.isFinite(v)).sort((a, b) => a - b);
  const n = vs.length;
  if (n === 0) return null;

  if (n === 1) {
    const v0 = vs[0];
    const r0 = round6(v0);
    return {
      q1: r0,
      median: r0,
      q3: r0,
      iqr: 0,
      whisker_lo: r0,
      whisker_hi: r0,
      whiskerMin: r0,
      whiskerMax: r0,
      outliers: [],
      count: 1,
    };
  }

  // Type-7 continuous quantile interpolation (R type 7 / NumPy linear)
  const percentile = p => {
    const k = (n - 1) * p / 100.0;
    const f = Math.floor(k);
    const c = Math.ceil(k);
    if (f === c) return vs[f];
    return vs[f] * (c - k) + vs[c] * (k - f);
  };

  const q1 = percentile(25);
  const med = percentile(50);
  const q3 = percentile(75);
  const iqr = q3 - q1;
  const loWhisker = q1 - 1.5 * iqr;
  const hiWhisker = q3 + 1.5 * iqr;

  // Actual observed datum whiskers
  const loActual = vs.find(v => v >= loWhisker) ?? q1;
  const hiActual = [...vs].reverse().find(v => v <= hiWhisker) ?? q3;
  const outliers = vs.filter(v => v < loWhisker || v > hiWhisker);

  return {
    q1: round6(q1),
    median: round6(med),
    q3: round6(q3),
    iqr: round6(iqr),
    whisker_lo: round6(loActual),
    whisker_hi: round6(hiActual),
    whiskerMin: round6(loActual),
    whiskerMax: round6(hiActual),
    outliers: outliers.map(v => round6(v)),
    count: n,
  };
}

/**
 * Generate deterministic golden-ratio Weyl jitter for sorted values.
 * @param {number[]} values
 * @returns {Array<{ value: number, jitter: number }>}
 */
export function weylJitter(values) {
  if (!values || !Array.isArray(values) || values.length === 0) return [];
  const vs = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v) && Number.isFinite(v)).sort((a, b) => a - b);
  return vs.map((v, i) => ({
    value: round6(v),
    jitter: round6(((i * WEYL_PHI) % 1.0 - 0.5)),
  }));
}

/**
 * Compute the median of a numeric array (matches Python statistics.median).
 * @param {number[]} values
 * @returns {number|null}
 */
export function median(values) {
  if (!values || !Array.isArray(values) || values.length === 0) return null;
  const vs = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v) && typeof v === 'number' && Number.isFinite(v)).sort((a, b) => a - b);
  const n = vs.length;
  if (n === 0) return null;
  if (n % 2 === 1) return vs[(n - 1) / 2];
  return (vs[n / 2 - 1] + vs[n / 2]) / 2.0;
}

/**
 * Type-7 continuous quantile (R type 7 / NumPy linear default).
 * Matches Python CohortStats._percentile_value: pos = (pct/100)*(n-1).
 * @param {number[]} values - sorted or unsorted
 * @param {number} p - percentile in [0, 100]
 * @returns {number|null}
 */
export function quantile(values, p) {
  if (!values || !Array.isArray(values) || values.length === 0) return null;
  const vs = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v) && typeof v === 'number' && Number.isFinite(v)).sort((a, b) => a - b);
  const n = vs.length;
  if (n === 0) return null;
  if (n === 1) return vs[0];
  const k = (n - 1) * p / 100.0;
  const f = Math.floor(k);
  const c = Math.ceil(k);
  if (f === c) return vs[f];
  return vs[f] * (c - k) + vs[c] * (k - f);
}

/**
 * Robust z-score: (x - median) / (IQR / 1.349).
 * Matches the Python CohortStats.robust_z formula exactly.
 * Scale estimator: IQR / 1.349 (consistency constant for normal distribution).
 * Edge cases: empty array or null x -> null; zero IQR -> 0.
 * @param {number[]} values
 * @param {number} x
 * @returns {number|null}
 */
export function robustZ(values, x) {
  if (!values || !Array.isArray(values) || values.length === 0) return null;
  if (x === null || x === undefined || Number.isNaN(x)) return null;
  const med = median(values);
  if (med === null) return null;
  const q1 = quantile(values, 25);
  const q3 = quantile(values, 75);
  const iqr = (q1 !== null && q3 !== null) ? (q3 - q1) : 0;
  if (iqr === 0) return 0;
  return (x - med) / (iqr / 1.349);
}

/**
 * Python 3 compatible round (banker's rounding / round-half-to-even).
 * Python's round() uses "round half to even" whereas JavaScript's
 * Math.round() always rounds 0.5 upward. This helper mirrors Python's
 * int(round(val)) exactly so percentile ranks match the backend.
 * @param {number} val
 * @returns {number}
 */
function pythonRound(val) {
  const floor = Math.floor(val);
  const diff = val - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  // Exact 0.5: round to even (Python 3 banker's rounding)
  return (floor % 2 === 0) ? floor : floor + 1;
}

/**
 * Percentile rank of x within values using the mid-rank formula.
 * Matches Python CohortStats.percentile exactly:
 *   rank_below  = count of values < x
 *   count_equal = count of values == x
 *   pct = ((rank_below + 0.5 * count_equal) / n) * 100.0
 *   return int(round(pct))   # Python banker's rounding
 *
 * Edge cases: empty array or null x -> null.
 * @param {number[]} values
 * @param {number} x
 * @returns {number|null}
 */
export function percentile(values, x) {
  if (!values || !Array.isArray(values) || values.length === 0) return null;
  if (x === null || x === undefined || Number.isNaN(x)) return null;
  const vs = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v) && typeof v === 'number' && Number.isFinite(v)).sort((a, b) => a - b);
  const n = vs.length;
  if (n === 0) return null;
  // Mid-rank percentile (matches Python CohortStats.percentile).
  let rankBelow = 0;
  let countEqual = 0;
  for (let i = 0; i < n; i++) {
    if (vs[i] < x) rankBelow++;
    else if (vs[i] === x) countEqual++;
  }
  const pct = ((rankBelow + 0.5 * countEqual) / n) * 100.0;
  return pythonRound(pct);
}

/**
 * Silverman's rule-of-thumb Gaussian kernel bandwidth selection (AMISE optimal).
 * h = 0.9 * min(sigma, IQR/1.34) * n^(-1/5)
 * @param {number[]} values
 * @returns {number}
 */
export function silvermanBandwidth(values) {
  if (!values || !Array.isArray(values)) return 0.0;
  const vs = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v) && Number.isFinite(v)).sort((a, b) => a - b);
  const n = vs.length;
  if (n < 2) return 0.0;

  const mean = vs.reduce((a, b) => a + b, 0) / n;
  const variance = vs.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (n - 1);
  const sigma = Math.sqrt(variance);

  const pct = p => {
    const k = (n - 1) * p / 100.0;
    const f = Math.floor(k);
    const c = Math.ceil(k);
    if (f === c) return vs[f];
    return vs[f] * (c - k) + vs[c] * (k - f);
  };
  const iqr = pct(75) - pct(25);

  let spread = iqr > 0 ? Math.min(sigma, iqr / 1.34) : sigma;
  if (spread === 0) spread = sigma;
  if (spread === 0) return 0.0;

  return 0.9 * spread * Math.pow(n, -0.2);
}

/**
 * Gaussian Kernel Density Estimation.
 * Supports both:
 *   - Signature A: gaussianKDE(values, bandwidth, evalPoints) -> number[]
 *   - Signature B: gaussianKDE(values, nPoints = 80, padding = 0.15) -> { grid, density, bandwidth, points }
 * @param {number[]} values
 * @param {number|Array} arg2 - bandwidth or nPoints
 * @param {Array|number} arg3 - evalPoints or padding
 * @returns {number[]|Object}
 */
export function gaussianKDE(values, arg2, arg3) {
  if (!values || !Array.isArray(values)) {
    if (Array.isArray(arg3)) return arg3.map(() => 0.0);
    return { grid: [], density: [], bandwidth: 0, points: [] };
  }

  const vs = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v) && Number.isFinite(v));
  const n = vs.length;

  // Signature A: (values, bandwidth, evalPoints)
  if (Array.isArray(arg3)) {
    const bandwidth = typeof arg2 === 'number' ? arg2 : 0;
    const evalPoints = arg3;
    if (n === 0 || bandwidth <= 0) return evalPoints.map(() => 0.0);
    const norm = 1.0 / (n * bandwidth * Math.sqrt(2 * Math.PI));
    return evalPoints.map(x => {
      let s = 0.0;
      for (const v of vs) {
        const u = (x - v) / bandwidth;
        s += Math.exp(-0.5 * u * u);
      }
      return norm * s;
    });
  }

  // Signature B: (values, nPoints = 80, padding = 0.15)
  const nPoints = typeof arg2 === 'number' ? Math.max(2, Math.floor(arg2)) : 80;
  const padding = typeof arg3 === 'number' ? arg3 : 0.15;
  const bw = silvermanBandwidth(vs);

  if (n < 2 || bw <= 0) {
    return { grid: [], density: [], bandwidth: 0, points: [] };
  }

  let lo = Infinity, hi = -Infinity;
  for (const v of vs) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const pad = hi > lo ? (hi - lo) * padding : bw * 3;
  const loEval = lo - pad;
  const hiEval = hi + pad;

  const grid = Array.from({ length: nPoints }, (_, i) => loEval + (hiEval - loEval) * (i / (nPoints - 1)));
  const norm = 1.0 / (n * bw * Math.sqrt(2 * Math.PI));
  const density = grid.map(x => {
    let s = 0.0;
    for (const v of vs) {
      const u = (x - v) / bw;
      s += Math.exp(-0.5 * u * u);
    }
    return norm * s;
  });

  return {
    grid: grid.map(x => round6(x)),
    density: density.map(y => round6(y)),
    bandwidth: round6(bw),
    points: grid.map((x, i) => ({ x: round6(x), y: round6(density[i]) })),
  };
}

/**
 * Fast Dual-Gram Power Iteration for Top-3 PCA in pure JavaScript.
 * Supports both 2D and 3D manifold coordinates, variance explained, and svd_flip sign determinism.
 * Coordinates are scaled by sqrt((n-1) * lambda) matching backend covariance projection Xc @ V.
 * 
 * @param {Array<Array<number>>} vectors - N x D numeric matrix
 * @param {Array<string>|number} caseIdsOrNComponents - Case IDs array or number of components
 * @param {Array<string>} [targetsOrLabels] - Target classes or labels
 * @param {Array<string>} [tasks] - Task names
 * @returns {Object}
 */
export function computePCA(vectors, caseIdsOrNComponents = 3, targetsOrLabels = [], tasks = []) {
  let caseIds = [];
  let targets = [];
  let taskList = [];
  let nComponents = 3;

  if (Array.isArray(caseIdsOrNComponents)) {
    caseIds = caseIdsOrNComponents;
    targets = Array.isArray(targetsOrLabels) ? targetsOrLabels : [];
    taskList = Array.isArray(tasks) ? tasks : [];
    nComponents = 3;
  } else if (typeof caseIdsOrNComponents === 'number') {
    nComponents = Math.min(3, Math.max(1, caseIdsOrNComponents));
    if (Array.isArray(targetsOrLabels)) {
      caseIds = targetsOrLabels;
    }
  }

  if (!vectors || !Array.isArray(vectors) || vectors.length === 0) {
    return {
      points: [],
      variance_explained: [0, 0, 0].slice(0, Math.max(2, nComponents)),
      varianceExplained: [0, 0, 0].slice(0, Math.max(2, nComponents)),
      cumulative_2pc: 0,
      cumulative_3pc: 0,
      totalVariance: 0,
      n: 0,
      method: 'Dual Gram PCA eigendecomposition (Live JavaScript Engine).',
    };
  }

  const n = vectors.length;
  const d = Array.isArray(vectors[0]) ? vectors[0].length : 0;

  if (d === 0) {
    return {
      points: [],
      variance_explained: [0, 0, 0].slice(0, Math.max(2, nComponents)),
      varianceExplained: [0, 0, 0].slice(0, Math.max(2, nComponents)),
      cumulative_2pc: 0,
      cumulative_3pc: 0,
      totalVariance: 0,
      n,
      method: 'Dual Gram PCA eigendecomposition (Live JavaScript Engine).',
    };
  }

  if (n === 1) {
    const pt = {
      case_id: caseIds[0] || '0',
      task: taskList[0] || '',
      target: targets[0] || '',
      label: caseIds[0] || '0',
      pc1: 0,
      pc2: 0,
      pc3: 0,
      x: 0,
      y: 0,
      z: 0,
    };
    return {
      points: [pt],
      variance_explained: [0, 0, 0].slice(0, Math.max(2, nComponents)),
      varianceExplained: [0, 0, 0].slice(0, Math.max(2, nComponents)),
      cumulative_2pc: 0,
      cumulative_3pc: 0,
      totalVariance: 0,
      n: 1,
      method: 'Dual Gram PCA eigendecomposition (Live JavaScript Engine).',
    };
  }

  // Step 1: Center data
  const means = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += vectors[i][j];
    means[j] = sum / n;
  }

  const Xc = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(d);
    for (let j = 0; j < d; j++) row[j] = vectors[i][j] - means[j];
    return row;
  });

  // Step 2: Compute total trace efficiently: trace(K) = sum(||Xc[i]||^2) * scale
  const scale = 1.0 / Math.max(1, n - 1);
  let totalTrace = 0;
  for (let i = 0; i < n; i++) {
    let norm2 = 0;
    const Xi = Xc[i];
    for (let c = 0; c < d; c++) norm2 += Xi[c] * Xi[c];
    totalTrace += norm2;
  }
  totalTrace *= scale;

  if (totalTrace <= 0) {
    const points = Array.from({ length: n }, (_, i) => ({
      case_id: caseIds[i] || String(i),
      task: taskList[i] || '',
      target: targets[i] || '',
      label: caseIds[i] || String(i),
      pc1: 0, pc2: 0, pc3: 0,
      x: 0, y: 0, z: 0,
    }));
    return {
      points,
      variance_explained: [0, 0, 0].slice(0, Math.max(2, nComponents)),
      varianceExplained: [0, 0, 0].slice(0, Math.max(2, nComponents)),
      cumulative_2pc: 0,
      cumulative_3pc: 0,
      totalVariance: 0,
      n,
      method: 'Dual Gram PCA eigendecomposition (Live JavaScript Engine).',
    };
  }

  // -----------------------------------------------------------------------
  // Randomized PCA for large cohorts (N > 200).
  // The exact Gram matrix approach is O(N^2 * D), which becomes prohibitive
  // for large N (e.g., 1000 cases x 1024-d => ~512M multiply-adds => ~1s).
  // Randomized range finding reduces this to O(N * D * k) where k is the
  // oversampled target rank (k = nComponents + 2 = 5), yielding ~10M
  // operations — a ~50x speedup with negligible accuracy loss for top-3 PCs.
  // -----------------------------------------------------------------------
  if (n > 200) {
    const k = Math.min(n, Math.max(2, nComponents) + 2);

    // Deterministic random projection matrix OmegaT (k x D) using Weyl sequence
    const OmegaT = new Float64Array(k * d);
    for (let idx = 0; idx < k * d; idx++) {
      OmegaT[idx] = ((idx * WEYL_PHI) % 1.0) * 2.0 - 1.0;
    }

    // Y = Xc @ OmegaT^T  (N x k)
    const Y = new Float64Array(n * k);
    for (let i = 0; i < n; i++) {
      const Xi = Xc[i];
      const offI = i * k;
      for (let j = 0; j < k; j++) {
        const Oj = OmegaT.subarray(j * d, j * d + d);
        let s = 0;
        for (let c = 0; c < d; c++) s += Xi[c] * Oj[c];
        Y[offI + j] = s;
      }
    }

    // Power iterations (q=2): Y = Xc @ (Xc^T @ Y) to drive sketch toward dominant eigenspace
    // (Halko, Martinsson, Tropp 2011 §1.6). Cost O(q·n·d·k) — negligible vs O(n²·d) exact path.
    for (let q = 0; q < 2; q++) {
      // Z = Xc^T @ Y  (D x k)
      const Z = new Float64Array(d * k);
      for (let c = 0; c < d; c++) {
        for (let j = 0; j < k; j++) {
          let s = 0;
          for (let i = 0; i < n; i++) s += Xc[i][c] * Y[i * k + j];
          Z[c * k + j] = s;
        }
      }
      // Y = Xc @ Z  (N x k)
      for (let i = 0; i < n; i++) {
        const Xi = Xc[i];
        const offI = i * k;
        for (let j = 0; j < k; j++) {
          let s = 0;
          for (let c = 0; c < d; c++) s += Xi[c] * Z[c * k + j];
          Y[offI + j] = s;
        }
      }
    }

    // Modified Gram-Schmidt orthonormalization -> Q (N x k)
    const Q = new Float64Array(n * k);
    Q.set(Y);
    for (let col = 0; col < k; col++) {
      for (let prev = 0; prev < col; prev++) {
        let dot = 0;
        for (let i = 0; i < n; i++) dot += Q[i * k + prev] * Q[i * k + col];
        for (let i = 0; i < n; i++) Q[i * k + col] -= dot * Q[i * k + prev];
      }
      let nrm = 0;
      for (let i = 0; i < n; i++) nrm += Q[i * k + col] * Q[i * k + col];
      nrm = Math.sqrt(nrm);
      if (nrm > 1e-12) {
        for (let i = 0; i < n; i++) Q[i * k + col] /= nrm;
      }
    }

    // B = Q^T @ Xc  (k x D), accumulated row-wise for cache efficiency
    const B = new Float64Array(k * d);
    for (let i = 0; i < n; i++) {
      const Xi = Xc[i];
      for (let j = 0; j < k; j++) {
        const qij = Q[i * k + j];
        if (qij === 0) continue;
        const Bj = B.subarray(j * d, j * d + d);
        for (let c = 0; c < d; c++) Bj[c] += qij * Xi[c];
      }
    }

    // Small Gram matrix: Cb = B @ B^T / (n-1)  (k x k)
    const Cb = Array.from({ length: k }, () => new Float64Array(k));
    for (let i = 0; i < k; i++) {
      const Bi = B.subarray(i * d, i * d + d);
      for (let j = i; j < k; j++) {
        const Bj = B.subarray(j * d, j * d + d);
        let s = 0;
        for (let c = 0; c < d; c++) s += Bi[c] * Bj[c];
        const val = s * scale;
        Cb[i][j] = val;
        Cb[j][i] = val;
      }
    }

    // Power iteration on small Cb (k x k, trivial cost)
    function powerIterSmall(mat, maxIter = 100, tol = 1e-12) {
      let v = new Float64Array(k);
      for (let i = 0; i < k; i++) v[i] = Math.sin(i + 1);
      let nrm = 0;
      for (let i = 0; i < k; i++) nrm += v[i] * v[i];
      nrm = Math.sqrt(nrm);
      if (nrm === 0) nrm = 1.0;
      for (let i = 0; i < k; i++) v[i] /= nrm;
      let lambda = 0;
      for (let iter = 0; iter < maxIter; iter++) {
        const vNext = new Float64Array(k);
        for (let i = 0; i < k; i++) {
          let s = 0;
          for (let j = 0; j < k; j++) s += mat[i][j] * v[j];
          vNext[i] = s;
        }
        nrm = 0;
        for (let i = 0; i < k; i++) nrm += vNext[i] * vNext[i];
        nrm = Math.sqrt(nrm);
        if (nrm === 0) break;
        let diff = 0;
        for (let i = 0; i < k; i++) {
          const next = vNext[i] / nrm;
          diff += (next - v[i]) ** 2;
          v[i] = next;
        }
        lambda = nrm;
        if (Math.sqrt(diff) < tol) break;
      }
      // svd_flip sign determinism
      let maxAbsVal = -1, maxAbsIdx = 0;
      for (let i = 0; i < k; i++) {
        const absVal = Math.abs(v[i]);
        if (absVal > maxAbsVal) { maxAbsVal = absVal; maxAbsIdx = i; }
      }
      if (v[maxAbsIdx] < 0) { for (let i = 0; i < k; i++) v[i] = -v[i]; }
      return { lambda, v };
    }

    const eig1s = powerIterSmall(Cb);
    const Cb2 = Array.from({ length: k }, (_, i) => {
      const row = new Float64Array(k);
      for (let j = 0; j < k; j++) row[j] = Cb[i][j] - eig1s.lambda * eig1s.v[i] * eig1s.v[j];
      return row;
    });
    const eig2s = powerIterSmall(Cb2);
    const Cb3 = Array.from({ length: k }, (_, i) => {
      const row = new Float64Array(k);
      for (let j = 0; j < k; j++) row[j] = Cb2[i][j] - eig2s.lambda * eig2s.v[i] * eig2s.v[j];
      return row;
    });
    const eig3s = powerIterSmall(Cb3);

    // Reconstruct N-dimensional eigenvectors: Veig = Q @ Ub
    const Veig1 = new Float64Array(n);
    const Veig2 = new Float64Array(n);
    const Veig3 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const offI = i * k;
      for (let j = 0; j < k; j++) {
        const qij = Q[offI + j];
        Veig1[i] += qij * eig1s.v[j];
        Veig2[i] += qij * eig2s.v[j];
        Veig3[i] += qij * eig3s.v[j];
      }
    }

    // svd_flip on N-dimensional eigenvectors (consistent with exact path)
    function svdFlipN(v) {
      let maxAbsVal = -1, maxAbsIdx = 0;
      for (let i = 0; i < n; i++) {
        const absVal = Math.abs(v[i]);
        if (absVal > maxAbsVal) { maxAbsVal = absVal; maxAbsIdx = i; }
      }
      if (v[maxAbsIdx] < 0) { for (let i = 0; i < n; i++) v[i] = -v[i]; }
    }
    svdFlipN(Veig1);
    svdFlipN(Veig2);
    svdFlipN(Veig3);

    const var1 = totalTrace > 0 ? (Math.max(0, eig1s.lambda) / totalTrace) * 100 : 0;
    const var2 = totalTrace > 0 ? (Math.max(0, eig2s.lambda) / totalTrace) * 100 : 0;
    const var3 = totalTrace > 0 ? (Math.max(0, eig3s.lambda) / totalTrace) * 100 : 0;
    const cum2pc = var1 + var2;
    const cum3pc = var1 + var2 + var3;

    const nFactor = Math.max(1, n - 1);
    const s1 = Math.sqrt(Math.max(0, eig1s.lambda * nFactor));
    const s2 = Math.sqrt(Math.max(0, eig2s.lambda * nFactor));
    const s3 = Math.sqrt(Math.max(0, eig3s.lambda * nFactor));

    const points = [];
    for (let i = 0; i < n; i++) {
      const pc1 = round6(Veig1[i] * s1);
      const pc2 = round6(Veig2[i] * s2);
      const pc3 = round6(Veig3[i] * s3);
      points.push({
        case_id: caseIds[i] || String(i),
        task: taskList[i] || '',
        target: targets[i] || '',
        label: caseIds[i] || String(i),
        pc1, pc2, pc3,
        x: pc1, y: pc2, z: pc3,
      });
    }

    const varExpl = [round2(var1), round2(var2), round2(var3)];
    return {
      points,
      variance_explained: varExpl.slice(0, Math.max(2, nComponents)),
      varianceExplained: varExpl.slice(0, Math.max(2, nComponents)),
      cumulative_2pc: round2(cum2pc),
      cumulative_3pc: round2(cum3pc),
      totalVariance: round6(totalTrace),
      n,
      method: `Randomized Dual Gram PCA (Live JavaScript Engine). Top-${nComponents} PCs projected via randomized range finding.`,
    };
  }

  // Exact Gram Matrix for small cohorts (N <= 200): K = (1/(n-1)) * Xc @ Xc^T (n x n)
  const K = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let dot = 0;
      for (let c = 0; c < d; c++) dot += Xc[i][c] * Xc[j][c];
      const val = dot * scale;
      K[i][j] = val;
      K[j][i] = val;
    }
  }

  // Step 3: Power Iteration with Rayleigh Quotient & Deflation for Top 3 Eigenvectors
  function powerIter(mat, maxIter = 100, tol = 1e-12) {
    let v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.sin(i + 1); // deterministic non-zero start
    let norm = Math.hypot(...v);
    if (norm === 0) norm = 1.0;
    for (let i = 0; i < n; i++) v[i] /= norm;

    let lambda = 0;
    for (let iter = 0; iter < maxIter; iter++) {
      const vNext = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += mat[i][j] * v[j];
        vNext[i] = s;
      }
      norm = Math.hypot(...vNext);
      if (norm === 0) break;
      let diff = 0;
      for (let i = 0; i < n; i++) {
        const next = vNext[i] / norm;
        diff += (next - v[i]) ** 2;
        v[i] = next;
      }
      lambda = norm;
      if (Math.sqrt(diff) < tol) break;
    }

    // svd_flip sign determinism: ensure element with largest absolute value is positive
    let maxAbsVal = -1;
    let maxAbsIdx = 0;
    for (let i = 0; i < n; i++) {
      const absVal = Math.abs(v[i]);
      if (absVal > maxAbsVal) {
        maxAbsVal = absVal;
        maxAbsIdx = i;
      }
    }
    if (v[maxAbsIdx] < 0) {
      for (let i = 0; i < n; i++) v[i] = -v[i];
    }

    return { lambda, v };
  }

  // PC1
  const eig1 = powerIter(K);

  // Deflate K for 2nd eigenvector: K2 = K - lambda1 * (v1 @ v1^T)
  const K2 = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      row[j] = K[i][j] - eig1.lambda * eig1.v[i] * eig1.v[j];
    }
    return row;
  });
  const eig2 = powerIter(K2);

  // Deflate K2 for 3rd eigenvector: K3 = K2 - lambda2 * (v2 @ v2^T)
  const K3 = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      row[j] = K2[i][j] - eig2.lambda * eig2.v[i] * eig2.v[j];
    }
    return row;
  });
  const eig3 = powerIter(K3);

  const var1 = totalTrace > 0 ? (eig1.lambda / totalTrace) * 100 : 0;
  const var2 = totalTrace > 0 ? (eig2.lambda / totalTrace) * 100 : 0;
  const var3 = totalTrace > 0 ? (eig3.lambda / totalTrace) * 100 : 0;
  const cum2pc = var1 + var2;
  const cum3pc = var1 + var2 + var3;

  // Project coordinates: coords_k = sqrt((n - 1) * lambda_k) * v_k (matches backend Xc @ V scale)
  const nFactor = Math.max(1, n - 1);
  const s1 = Math.sqrt(Math.max(0, eig1.lambda * nFactor));
  const s2 = Math.sqrt(Math.max(0, eig2.lambda * nFactor));
  const s3 = Math.sqrt(Math.max(0, eig3.lambda * nFactor));

  const points = [];
  for (let i = 0; i < n; i++) {
    const pc1 = round6(eig1.v[i] * s1);
    const pc2 = round6(eig2.v[i] * s2);
    const pc3 = round6(eig3.v[i] * s3);
    points.push({
      case_id: caseIds[i] || String(i),
      task: taskList[i] || '',
      target: targets[i] || '',
      label: caseIds[i] || String(i),
      pc1,
      pc2,
      pc3,
      x: pc1,
      y: pc2,
      z: pc3,
    });
  }

  const varExpl = [round2(var1), round2(var2), round2(var3)];

  return {
    points,
    variance_explained: varExpl.slice(0, Math.max(2, nComponents)),
    varianceExplained: varExpl.slice(0, Math.max(2, nComponents)),
    cumulative_2pc: round2(cum2pc),
    cumulative_3pc: round2(cum3pc),
    totalVariance: round6(totalTrace),
    n,
    method: `Dual Gram PCA eigendecomposition (Live JavaScript Engine). Top-${nComponents} PCs projected.`,
  };
}

// ---------------------------------------------------------------------------
// 2. High-Level In-Browser Cohort Analytics Engine
// ---------------------------------------------------------------------------

export class CohortEngine {
  static rankData = rankData;
  static pearsonR = pearsonR;
  static spearmanRho = spearmanRho;
  static wardCluster = wardCluster;
  static spearmanCorrelationMatrix = spearmanCorrelationMatrix;
  static tukeyBox = tukeyBox;
  static weylJitter = weylJitter;
  static median = median;
  static quantile = quantile;
  static robustZ = robustZ;
  static percentile = percentile;
  static silvermanBandwidth = silvermanBandwidth;
  static gaussianKDE = gaussianKDE;
  static computePCA = computePCA;

  static extractTask3Radiology(rr) {
    const out = { vol: null, psad: null, pirads: null };
    if (!rr || typeof rr !== 'string') return out;
    const mVol = rr.match(/Prostate volume:\s*([\d.]+)/i);
    if (mVol) out.vol = safeFloat(mVol[1]);
    const mPsad = rr.match(/PSA density:\s*([\d.]+)/i);
    if (mPsad) out.psad = safeFloat(mPsad[1]);
    const mPirads = rr.match(/PI-RADS:\s*(\d)/i);
    if (mPirads) out.pirads = safeFloat(mPirads[1]);
    return out;
  }

  static extractTask3Pathology(pr) {
    const out = { bx_isup: null, bx_gl_prim: null, bx_gl_sec: null };
    if (!pr || typeof pr !== 'string') return out;
    const mIsup = pr.match(/ISUP grade group (\d)/i);
    if (mIsup) out.bx_isup = safeFloat(mIsup[1]);
    const mGl = pr.match(/Gleason (\d)\+(\d)/i);
    if (mGl) {
      out.bx_gl_prim = safeFloat(mGl[1]);
      out.bx_gl_sec = safeFloat(mGl[2]);
    }
    return out;
  }

  static normalizeCase(trace) {
    if (!trace || typeof trace !== 'object') {
      return {
        case_id: 'UNKNOWN',
        task: 'task1', // defensive default for malformed traces; data.js traces always carry a proper task
        variables: {},
        target: null,
        mri_vec: null,
        bx_vec: null,
        px_vec: null,
        mri_pca_points: null,
        bx_pca_points: null,
        px_pca_points: null,
        missingness: {
          MRI: false,
          Biopsy: false,
          Prostatectomy: false,
          PSA_Trend: false,
          Labs: false,
          FamilyHistory: false,
        },
      };
    }

    const cid = trace.case_id || trace.id || 'UNKNOWN';
    // Defensive default: data.js-sourced traces always carry a proper task field.
    // The 'task1' fallback only triggers for malformed/synthetic traces without one.
    const task = String(trace.task || 'task1').toLowerCase();
    const d = trace.patient_demographics || trace.demographics || {};
    const clin = trace.clinical_records || trace.clinical || {};
    const mod = trace.modality_representations || trace.modality || {};
    const gt = trace.ground_truth || trace.gt || {};
    const sp = trace.structured_prompt || trace.structured || trace.structured_prompt_data || {};

    let target = null;
    if (task === 'task1' || task === 'task2') {
      target = gt.decision ? String(gt.decision).trim().toLowerCase() : (trace.target ? String(trace.target).trim().toLowerCase() : null);
    } else if (task === 'task3') {
      target = gt.event !== undefined && gt.event !== null ? String(gt.event) : (trace.target !== undefined && trace.target !== null ? String(trace.target) : null);
    }

    const variables = {};
    for (const v of NUMERIC_VARS) {
      if (v === 'cspca') {
        variables[v] = safeFloat(d.cspca ?? trace.cspca ?? sp.cspca ?? trace.structured_prompt_data?.cspca ?? clin.cspca);
      } else {
        variables[v] = safeFloat(d[v] ?? trace[v] ?? sp[v] ?? clin[v]);
      }
    }

    // Task 3: supplement from radiology/pathology reports if missing
    if (task === 'task3') {
      const rr = clin.radiology_report || trace.radiology_report || '';
      const pr = clin.pathology_report || trace.pathology_report || '';
      const rad = CohortEngine.extractTask3Radiology(rr);
      const pat = CohortEngine.extractTask3Pathology(pr);
      for (const k of ['vol', 'psad', 'pirads']) {
        if (variables[k] === null || variables[k] === undefined) variables[k] = rad[k];
      }
      for (const k of ['bx_isup', 'bx_gl_prim', 'bx_gl_sec']) {
        if (variables[k] === null || variables[k] === undefined) variables[k] = pat[k];
      }
    }

    const meanPool = raw => {
      if (!raw || !Array.isArray(raw) || raw.length === 0) return null;
      if (typeof raw[0] === 'number') return raw.map(v => round6(v));
      if (!Array.isArray(raw[0]) || raw[0].length === 0) return null;
      const dim = raw[0].length;
      const validRows = raw.filter(r => Array.isArray(r) && r.length === dim);
      if (validRows.length === 0) return null;
      const res = new Float64Array(dim);
      for (const r of validRows) {
        for (let j = 0; j < dim; j++) res[j] += r[j];
      }
      const out = new Array(dim);
      for (let j = 0; j < dim; j++) out[j] = round6(res[j] / validRows.length);
      return out;
    };

    const mriVec = meanPool(mod['MRI image'] || mod.mri_image || mod.mri?.embedding || mod.mri?.vector);
    const bxVec = meanPool(mod['Biopsy slide'] || mod.biopsy_slide || mod.biopsy?.embedding || mod.biopsy?.vector);
    const pxVec = meanPool(mod['Prostatectomy slide'] || mod.prostatectomy_slide || mod.prostatectomy?.embedding || mod.prostatectomy?.vector);

    // Extract precomputed PCA coordinates if embedded in trace
    const mriPcaPoints = mod.mri?.pca_points?.[0]?.pos || mod.mri_pca_points || null;
    const bxPcaPoints = mod.biopsy?.pca_points?.[0]?.pos || mod.biopsy_pca_points || null;
    const pxPcaPoints = mod.prostatectomy?.pca_points?.[0]?.pos || mod.prostatectomy_pca_points || null;

    // Support both raw format and trace representation objects
    const hasMri = mriVec !== null || mriPcaPoints !== null || Boolean(mod.mri && (mod.mri.norm || mod.mri.pca_points)) || Boolean(mod['MRI image']);
    const hasBx = bxVec !== null || bxPcaPoints !== null || Boolean(mod.biopsy && (mod.biopsy.norm || mod.biopsy.pca_points)) || Boolean(mod['Biopsy slide']);
    const hasPx = pxVec !== null || pxPcaPoints !== null || Boolean(mod.prostatectomy && (mod.prostatectomy.norm || mod.prostatectomy.pca_points)) || Boolean(mod['Prostatectomy slide']);
    const hasPsaTrend = (Array.isArray(clin.psa_trend) && clin.psa_trend.length > 0) || Boolean(clin.psa_trend && String(clin.psa_trend).trim());
    const hasLabs = (Array.isArray(clin.laboratory_results) && clin.laboratory_results.length > 0) || Boolean(clin.laboratory_results && String(clin.laboratory_results).trim());
    const hasFh = Boolean(clin.family_history && String(clin.family_history).trim() !== '' && String(clin.family_history).trim().toLowerCase() !== 'none');

    // Clinical Life-Cycle Structural Absence Model (survey_math_spec.md §2.6.2, SCOPE.md R6):
    //   Task 1 (pre-biopsy): Biopsy and Prostatectomy are structurally absent (M=0).
    //   Task 2 (risk stratification): Prostatectomy is structurally absent (M=0).
    //   Task 3 (post-prostatectomy): all modalities may be present.
    // The data-presence flags above are preserved for non-structural modalities; the
    // override below forces structural-absence pairs to false regardless of any stray
    // modality_representations payload, enforcing the clinical staging invariant.
    const structurallyAbsentBiopsy = (task === 'task1');
    const structurallyAbsentProstatectomy = (task === 'task1' || task === 'task2');
    const finalHasBx = structurallyAbsentBiopsy ? false : hasBx;
    const finalHasPx = structurallyAbsentProstatectomy ? false : hasPx;

    return {
      case_id: cid,
      task,
      variables,
      target,
      mri_vec: mriVec,
      bx_vec: bxVec,
      px_vec: pxVec,
      mri_pca_points: mriPcaPoints,
      bx_pca_points: bxPcaPoints,
      px_pca_points: pxPcaPoints,
      missingness: {
        MRI: hasMri,
        Biopsy: finalHasBx,
        Prostatectomy: finalHasPx,
        PSA_Trend: hasPsaTrend,
        Labs: hasLabs,
        FamilyHistory: hasFh,
      },
    };
  }

  static computeComposition(cases) {
    const tasks = {
      task1: { total: 0, classes: { yes: { n: 0, pct: 0 }, no: { n: 0, pct: 0 } } },
      task2: { total: 0, classes: { active_surveillance: { n: 0, pct: 0 }, continued_surveillance: { n: 0, pct: 0 }, watchful_waiting: { n: 0, pct: 0 }, active_treatment: { n: 0, pct: 0 } } },
      task3: { total: 0, classes: { '1': { n: 0, pct: 0 }, '0': { n: 0, pct: 0 } } },
    };

    for (const c of cases) {
      if (!tasks[c.task]) continue;
      tasks[c.task].total++;
      if (c.target && tasks[c.task].classes[c.target]) {
        tasks[c.task].classes[c.target].n++;
      }
    }

    for (const taskKey of Object.keys(tasks)) {
      const tot = tasks[taskKey].total;
      tasks[taskKey].n = tot;
      for (const clsKey of Object.keys(tasks[taskKey].classes)) {
        const n = tasks[taskKey].classes[clsKey].n;
        tasks[taskKey].classes[clsKey].pct = tot > 0 ? Math.round((n / tot) * 1000) / 10 : 0;
      }
    }

    return {
      tasks,
      method: 'Ordered horizontal stacked bars. Exact counts and percentages per task per target class. Zero raster images.',
      target_definitions: {
        task1: 'Biopsy decision: yes/no',
        task2: 'Treatment management: active_surveillance / continued_surveillance / watchful_waiting / active_treatment',
        task3: 'BCR event: 1 (recurred) / 0 (censored)',
      },
    };
  }

  static computeSingleCorrelation(cases, minObs = 5) {
    const usableVars = [];
    const varData = {};
    for (const v of NUMERIC_VARS) {
      varData[v] = cases.map(c => c.variables[v]);
      const nPresent = varData[v].filter(x => x !== null && x !== undefined && !Number.isNaN(x)).length;
      if (nPresent >= minObs) usableVars.push(v);
    }

    const k = usableVars.length;
    if (k === 0) {
      return {
        variables: [],
        matrix: [],
        features: [],
        dendrogramOrder: [],
        n_variables: 0,
        n_cases: cases.length,
        method: 'Spearman rho',
        excluded: NUMERIC_VARS,
      };
    }

    const rhoMatrix = Array.from({ length: k }, () => new Array(k).fill(null));
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        if (i === j) {
          rhoMatrix[i][j] = 1.0;
        } else if (j > i) {
          const r = spearmanRho(varData[usableVars[i]], varData[usableVars[j]]);
          rhoMatrix[i][j] = r;
          rhoMatrix[j][i] = r;
        }
      }
    }

    const order = wardCluster(rhoMatrix, usableVars);
    const displayMatrix = rhoMatrix.map(row => row.map(v => v === null ? null : Math.round(v * 10000) / 10000));
    const orderedMatrix = order.map(i => order.map(j => displayMatrix[i][j]));
    const orderedNames = order.map(idx => usableVars[idx]);

    return {
      variables: orderedNames,
      features: orderedNames,
      matrix: orderedMatrix,
      rawMatrix: rhoMatrix.map(row => [...row]),
      dendrogramOrder: order,
      n_variables: orderedNames.length,
      n_cases: cases.length,
      method: 'Spearman rho (Pearson on ranks, pairwise complete). Ward agglomerative clustering on squared distance = (1 - |ρ|)².',
      excluded: NUMERIC_VARS.filter(v => !usableVars.includes(v)),
    };
  }

  static computeCorrelation(cases) {
    const globalRes = CohortEngine.computeSingleCorrelation(cases, 5);
    const tasksRes = {};
    for (const t of ['task1', 'task2', 'task3']) {
      const taskCases = cases.filter(c => c.task === t);
      tasksRes[t] = CohortEngine.computeSingleCorrelation(taskCases, 5);
    }
    globalRes.tasks = tasksRes;
    return globalRes;
  }

  static computeMissingness(cases) {
    const modalities = MODALITIES;
    const n = cases.length;
    const matrix = cases.map(c => modalities.map(m => (c.missingness[m] ? 1 : 0)));

    const expected_absence = {
      task1: {
        Biopsy: 'Task 1: biopsy expectedly absent (pre-biopsy decision stage)',
        Prostatectomy: 'Task 1: prostatectomy expectedly absent (pre-biopsy decision stage)',
      },
      task2: {
        Prostatectomy: 'Task 2: prostatectomy expectedly absent (post-biopsy, pre-surgery stage)',
      },
      task3: {
        PSA_Trend: 'Task 3: PSA trend expectedly absent (post-prostatectomy survival stage)',
        Labs: 'Task 3: laboratory panel expectedly absent (post-prostatectomy survival stage)',
      },
    };

    const summary = {};
    for (const task of ['task1', 'task2', 'task3']) {
      const taskCases = cases.filter(c => c.task === task);
      summary[task] = {};
      for (const m of modalities) {
        const present = sum1(taskCases, c => c.missingness[m]);
        const total = taskCases.length;
        const missing = total - present;
        const pct = total > 0 ? round2((present / total) * 100) : 0;
        summary[task][m] = { present, missing, total, pct };
      }
    }

    // Availability rate per modality
    const availabilityRate = {};
    for (let j = 0; j < modalities.length; j++) {
      const m = modalities[j];
      const count = matrix.reduce((acc, row) => acc + row[j], 0);
      availabilityRate[m] = n > 0 ? round2((count / n) * 100) : 0;
    }

    // Overall cohort sparsity (fraction of empty cells)
    const totalEntries = n * modalities.length;
    const totalPresent = matrix.reduce((acc, row) => acc + row.reduce((a, b) => a + b, 0), 0);
    const cohortSparsity = totalEntries > 0 ? round6(1.0 - totalPresent / totalEntries) : 0;

    return {
      modalities,
      cases: cases.map(c => ({ case_id: c.case_id, task: c.task })),
      matrix,
      expected_absence,
      summary,
      availabilityRate,
      cohortSparsity,
    };
  }

  static computeRaincloud(cases, metric) {
    const targetOrder = {
      task1: ['yes', 'no'],
      task2: ['active_surveillance', 'continued_surveillance', 'watchful_waiting', 'active_treatment'],
      task3: ['1', '0'],
    };

    const strata = [];
    for (const task of ['task1', 'task2', 'task3']) {
      const taskCases = cases.filter(c => c.task === task);
      for (const target of targetOrder[task]) {
        const values = taskCases
          .filter(c => c.target === target && c.variables[metric] !== null && c.variables[metric] !== undefined && !Number.isNaN(c.variables[metric]))
          .map(c => c.variables[metric]);

        const n = values.length;
        if (n === 0) {
          strata.push({ task, target, n: 0, box: null, kde: null, dots: [], note: 'No data for this stratum.' });
          continue;
        }

        const box = tukeyBox(values);
        const dots = [...values].sort((a, b) => a - b).map((v, i) => ({
          value: round6(v),
          jitter: round6(((i * WEYL_PHI) % 1.0 - 0.5)),
        }));

        const bw = silvermanBandwidth(values);
        let kde = null;
        if (n >= 2 && bw > 0) {
          let lo = Infinity, hi = -Infinity;
          for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
          const padding = hi > lo ? (hi - lo) * 0.15 : bw * 3;
          const loEval = lo - padding;
          const hiEval = hi + padding;
          const nEval = 80;
          const evalPoints = Array.from({ length: nEval }, (_, i) => loEval + (hiEval - loEval) * (i / (nEval - 1)));
          const densities = gaussianKDE(values, bw, evalPoints);
          kde = {
            bandwidth: round6(bw),
            points: evalPoints.map((x, i) => ({ x: round6(x), y: round6(densities[i]) })),
          };
        }

        strata.push({ task, target, n, box, kde, dots });
      }
    }

    return {
      metric,
      unit: METRIC_UNITS[metric] || '',
      strata,
    };
  }

  /**
   * Primary Entry Point: Compute 100% dynamic cohort payload from raw traces.
   * Execution time < 50ms across all 423 cases.
   * 
   * @param {Array<Object>} rawTraces - Array of raw trace objects
   * @param {string} [filter='all'] - Cohort filter ('all', 'task1', 'task2', 'task3')
   * @returns {Object}
   */
  static computeAll(rawTraces = [], filter = 'all') {
    const t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();

    const traces = Array.isArray(rawTraces) ? rawTraces : [];
    const allCases = traces.map(t => CohortEngine.normalizeCase(t));
    const totalCases = allCases.length;

    const filterStr = String(filter || 'all').toLowerCase().trim();
    const cases = (filterStr === 'task1' || filterStr === 'task2' || filterStr === 'task3')
      ? allCases.filter(c => c.task === filterStr)
      : allCases;
    const filteredCases = cases.length;

    // B1: Composition
    const composition = CohortEngine.computeComposition(cases);

    // B2: PCA MRI (Top-3 PCs: 2D & 3D)
    let pca_mri;
    const mriCasesWithVec = cases.filter(c => c.mri_vec !== null);
    if (mriCasesWithVec.length >= 5) {
      pca_mri = computePCA(
        mriCasesWithVec.map(c => c.mri_vec),
        mriCasesWithVec.map(c => c.case_id),
        mriCasesWithVec.map(c => c.target),
        mriCasesWithVec.map(c => c.task)
      );
    } else {
      const mriPoints = [];
      for (const c of cases) {
        const p = c.mri_pca_points || c.mri_vec;
        if (p && p.length >= 2) {
          mriPoints.push({
            case_id: c.case_id,
            task: c.task,
            target: c.target || 'None',
            label: c.case_id,
            pc1: round6(p[0]),
            pc2: round6(p[1]),
            pc3: round6(p[2] || 0),
            x: round6(p[0]),
            y: round6(p[1]),
            z: round6(p[2] || 0),
          });
        }
      }
      pca_mri = {
        points: mriPoints,
        variance_explained: [0, 0, 0],
        varianceExplained: [0, 0, 0],
        cumulative_2pc: 0,
        cumulative_3pc: 0,
        totalVariance: 0,
        n: mriPoints.length,
        method: 'PCA fallback (<5 cases, no live computation)',
      };
    }

    // B2: PCA Biopsy (Top-3 PCs: 2D & 3D)
    let pca_biopsy;
    const bxCasesWithVec = cases.filter(c => c.bx_vec !== null);
    if (bxCasesWithVec.length >= 5) {
      pca_biopsy = computePCA(
        bxCasesWithVec.map(c => c.bx_vec),
        bxCasesWithVec.map(c => c.case_id),
        bxCasesWithVec.map(c => c.target),
        bxCasesWithVec.map(c => c.task)
      );
    } else {
      const bxPoints = [];
      for (const c of cases) {
        const p = c.bx_pca_points || c.bx_vec;
        if (p && p.length >= 2) {
          bxPoints.push({
            case_id: c.case_id,
            task: c.task,
            target: c.target || 'None',
            label: c.case_id,
            pc1: round6(p[0]),
            pc2: round6(p[1]),
            pc3: round6(p[2] || 0),
            x: round6(p[0]),
            y: round6(p[1]),
            z: round6(p[2] || 0),
          });
        }
      }
      pca_biopsy = {
        points: bxPoints,
        variance_explained: [0, 0, 0],
        varianceExplained: [0, 0, 0],
        cumulative_2pc: 0,
        cumulative_3pc: 0,
        totalVariance: 0,
        n: bxPoints.length,
        method: 'PCA fallback (<5 cases, no live computation)',
      };
    }

    // B3: Correlation Matrix
    let correlation;
    if (filterStr === 'all') {
      correlation = CohortEngine.computeCorrelation(cases);
    } else {
      correlation = CohortEngine.computeSingleCorrelation(cases, 5);
      const correlationCopy = { ...correlation };
      delete correlationCopy.tasks;
      correlation.tasks = { [filterStr]: correlationCopy };
    }

    // B4: Missingness Grid
    const missingness = CohortEngine.computeMissingness(cases);

    // B5: Rainclouds
    const raincloud_psa = CohortEngine.computeRaincloud(cases, 'psa');
    const raincloud_psad = CohortEngine.computeRaincloud(cases, 'psad');
    const raincloud_vol = CohortEngine.computeRaincloud(cases, 'vol');
    const raincloud_age = CohortEngine.computeRaincloud(cases, 'age');
    const raincloud_pirads = CohortEngine.computeRaincloud(cases, 'pirads');

    // B6: Robust Z-Scores (engine-only, no UI).
    // Matches Python CohortStats.robust_z: (x - median) / (IQR / 1.349).
    // Per-metric robust z-scores and percentile ranks for key clinical variables.
    // Optimized: compute cohort stats (median, IQR, sorted array) ONCE per metric,
    // then apply formula inline per case — avoids N×re-sort overhead.
    const robustZMetrics = ['psa', 'psad', 'age', 'vol'];
    const robust_z = {};
    for (const metric of robustZMetrics) {
      const cohortValues = cases
        .map(c => c.variables[metric])
        .filter(v => v !== null && v !== undefined && !Number.isNaN(v) && typeof v === 'number' && Number.isFinite(v));
      const sorted = cohortValues.slice().sort((a, b) => a - b);
      const n = sorted.length;
      const med = n > 0 ? median(sorted) : null;
      const q1 = n > 0 ? quantile(sorted, 25) : null;
      const q3 = n > 0 ? quantile(sorted, 75) : null;
      const iqr = (q1 !== null && q3 !== null) ? (q3 - q1) : 0;
      const scale = iqr > 0 ? iqr / 1.349 : 0;
      robust_z[metric] = {
        median: med !== null ? round6(med) : null,
        q1: q1 !== null ? round6(q1) : null,
        q3: q3 !== null ? round6(q3) : null,
        iqr: round6(iqr),
        n: n,
        cases: cases.map(c => {
          const x = c.variables[metric];
          if (x === null || x === undefined || Number.isNaN(x) || typeof x !== 'number' || !Number.isFinite(x) || scale === 0) {
            return { case_id: c.case_id, value: x, robust_z: null, percentile: null };
          }
          const rz = (x - med) / scale;
          // Inline percentile rank using mid-rank formula (matches Python
          // CohortStats.percentile exactly: rank_below + 0.5*count_equal,
          // then int(round(pct)) with Python banker's rounding).
          // Binary search for O(log n) per case on the pre-sorted array.
          let loB = 0, hiB = n;
          while (loB < hiB) { const mid = (loB + hiB) >>> 1; if (sorted[mid] < x) loB = mid + 1; else hiB = mid; }
          const rankBelow = loB;
          let loE = rankBelow, hiE = n;
          while (loE < hiE) { const mid = (loE + hiE) >>> 1; if (sorted[mid] <= x) loE = mid + 1; else hiE = mid; }
          const countEqual = loE - rankBelow;
          const pct = ((rankBelow + 0.5 * countEqual) / n) * 100.0;
          return {
            case_id: c.case_id,
            value: x,
            robust_z: round6(rz),
            percentile: pythonRound(pct),
          };
        }),
      };
    }

    const t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const computationTimeMs = round2(t1 - t0);

    return {
      filter: filterStr,
      totalCases,
      filteredCases,
      n_cases: totalCases,
      n: filteredCases,
      cases,
      composition,
      pca_mri,
      pca_biopsy,
      pca: {
        mri: pca_mri,
        biopsy: pca_biopsy,
        mri3d: pca_mri,
        biopsy3d: pca_biopsy,
      },
      correlation,
      missingness,
      raincloud_psa,
      raincloud_psad,
      raincloud_vol,
      raincloud_age,
      raincloud_pirads,
      rainclouds: {
        psa: raincloud_psa,
        psad: raincloud_psad,
        vol: raincloud_vol,
        age: raincloud_age,
        pirads: raincloud_pirads,
      },
      robust_z,
      metrics: {
        availabilityRate: missingness.availabilityRate,
        cohortSparsity: missingness.cohortSparsity,
      },
      computationTimeMs,
    };
  }
}

function sum1(arr, fn) {
  let c = 0;
  for (let i = 0; i < arr.length; i++) {
    if (fn(arr[i])) c++;
  }
  return c;
}

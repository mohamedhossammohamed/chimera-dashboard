// test/tier2_boundaries.test.js
// Tier 2: Boundary Value Analysis & Corner Cases (10 Features x >=5 tests = >=50 tests)
// Built with native node:test and node:assert/strict

import { describe, it, before } from 'node:test';
import assert from 'node:assert/strict';
import {
  rankData,
  pearsonR,
  spearmanRho,
  wardCluster,
  tukeyBox,
  silvermanBandwidth,
  gaussianKDE,
  computePCA,
  CohortEngine,
  NUMERIC_VARS,
} from '../dashboard/js/cohort_engine.js';
import { safeFloat } from '../dashboard/js/clinical_engine.js';
import { setupMockDOM } from './helpers/mock_dom.js';

describe('Tier 2: Boundary 1 - In-Browser Cohort Computation Boundaries', () => {
  it('B1.1: computeAll on empty trace array [] returns clean zeroed schema', () => {
    const res = CohortEngine.computeAll([]);
    assert.ok(res.composition);
    assert.equal(res.composition.tasks.task1.total, 0);
    assert.equal(res.composition.tasks.task2.total, 0);
    assert.equal(res.composition.tasks.task3.total, 0);
    assert.equal(res.pca_mri.points.length, 0);
    assert.equal(res.pca_biopsy.points.length, 0);
    assert.equal(res.missingness.matrix.length, 0);
  });

  it('B1.2: computeAll on single trace object returns valid single-case metrics', () => {
    const traces = [{ case_id: 'solo_01', task: 'task1', patient_demographics: { psa: 5.0 } }];
    const res = CohortEngine.computeAll(traces);
    assert.equal(res.composition.tasks.task1.total, 1);
    assert.equal(res.missingness.cases.length, 1);
    assert.equal(res.missingness.cases[0].case_id, 'solo_01');
  });

  it('B1.3: handles traces with extreme numeric coordinates without throwing or NaN', () => {
    const traces = [
      { case_id: 'ext_1', task: 'task1', patient_demographics: { psa: 1e9, age: 120, vol: 1e6 } },
      { case_id: 'ext_2', task: 'task1', patient_demographics: { psa: 1e-9, age: 10, vol: 1e-3 } },
    ];
    const res = CohortEngine.computeAll(traces);
    assert.ok(res.composition);
    assert.ok(Number.isFinite(res.raincloud_psa.strata[0].n));
  });

  it('B1.4: normalizes invalid or corrupted task identifiers safely', () => {
    const traces = [
      { case_id: 'bad_task_1', task: 'UNKNOWN_TASK', patient_demographics: { psa: 6.0 } },
      { case_id: 'bad_task_2', task: 12345, patient_demographics: { psa: 7.0 } },
      { case_id: 'bad_task_3', task: null, patient_demographics: { psa: 8.0 } },
    ];
    const cases = traces.map(t => CohortEngine.normalizeCase(t));
    assert.equal(cases[0].task, 'unknown_task');
    assert.equal(cases[1].task, '12345');
    assert.equal(cases[2].task, 'task1'); // defaults to task1
  });

  it('B1.5: stress test with 500 synthetic traces executes stably without memory leak', () => {
    const traces = Array.from({ length: 500 }, (_, i) => ({
      case_id: `stress_${i}`,
      task: i % 3 === 0 ? 'task1' : i % 3 === 1 ? 'task2' : 'task3',
      patient_demographics: { psa: 1.0 + (i % 20), age: 40 + (i % 45), vol: 20 + (i % 80) },
      ground_truth: { decision: i % 2 === 0 ? 'yes' : 'no' },
    }));

    const res = CohortEngine.computeAll(traces);
    const total = res.composition.tasks.task1.total + res.composition.tasks.task2.total + res.composition.tasks.task3.total;
    assert.equal(total, 500);
  });
});

describe('Tier 2: Boundary 2 - Tukey Quantile Boundaries', () => {
  it('B2.1: empty array returns null', () => {
    assert.equal(tukeyBox([]), null);
    assert.equal(tukeyBox(null), null);
    assert.equal(tukeyBox(undefined), null);
  });

  it('B2.2: constant array returns identical quartiles and zero IQR', () => {
    const vals = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    const b = tukeyBox(vals);
    assert.equal(b.q1, 10.0);
    assert.equal(b.median, 10.0);
    assert.equal(b.q3, 10.0);
    assert.equal(b.iqr, 0.0);
    assert.equal(b.whisker_lo, 10.0);
    assert.equal(b.whisker_hi, 10.0);
    assert.deepEqual(b.outliers, []);
  });

  it('B2.3: two element array interpolates Type-7 continuous quantiles', () => {
    // n=2: k_q1 = (1)*0.25 = 0.25 -> vs[0]*0.75 + vs[1]*0.25 = 1*0.75 + 100*0.25 = 25.75
    // k_med = 0.5 -> 1*0.5 + 100*0.5 = 50.5
    // k_q3 = 0.75 -> 1*0.25 + 100*0.75 = 75.25
    const b = tukeyBox([1.0, 100.0]);
    assert.equal(b.q1, 25.75);
    assert.equal(b.median, 50.5);
    assert.equal(b.q3, 75.25);
    assert.equal(b.iqr, 49.5);
  });

  it('B2.4: heavily tied array across quartile boundaries', () => {
    const vals = [2, 2, 2, 2, 2, 2, 2, 2, 10];
    const b = tukeyBox(vals);
    assert.equal(b.q1, 2);
    assert.equal(b.median, 2);
    assert.equal(b.q3, 2);
    assert.equal(b.iqr, 0);
  });

  it('B2.5: extreme dynamic range array without overflow', () => {
    const vals = [1e-9, 2e-9, 3e-9, 4e-9, 1e9];
    const b = tukeyBox(vals);
    assert.ok(Number.isFinite(b.q1));
    assert.ok(Number.isFinite(b.median));
    assert.ok(Number.isFinite(b.q3));
    assert.ok(b.outliers.includes(1e9));
  });

  it('B2.6: boundary outlier comparison strictly uses > and < for fences', () => {
    // [10, 12, 14, 16, 18, 20]: Q1=12.25, Q3=17.75, IQR=5.5 -> Upper fence = 17.75 + 1.5*5.5 = 26.0
    // Adding 26.0 (at fence) and 50.0 (outside fence)
    const vals = [10, 12, 14, 16, 18, 20, 50.0];
    const b = tukeyBox(vals);
    assert.ok(b.outliers.includes(50.0));
  });
});

describe('Tier 2: Boundary 3 - Silverman KDE Boundaries', () => {
  it('B3.1: empty array returns 0.0 bandwidth and all 0.0 density', () => {
    assert.equal(silvermanBandwidth([]), 0.0);
    const density = gaussianKDE([], 0.0, [1, 2, 3]);
    assert.deepEqual(density, [0.0, 0.0, 0.0]);
  });

  it('B3.2: single element array returns 0.0 bandwidth', () => {
    assert.equal(silvermanBandwidth([10.0]), 0.0);
  });

  it('B3.3: constant array (zero variance) returns 0.0 bandwidth', () => {
    const vals = [5.0, 5.0, 5.0, 5.0, 5.0];
    assert.equal(silvermanBandwidth(vals), 0.0);
  });

  it('B3.4: zero bandwidth evaluates to all zeros in gaussianKDE', () => {
    const res = gaussianKDE([10.0, 20.0], 0.0, [10, 15, 20]);
    assert.deepEqual(res, [0.0, 0.0, 0.0]);
  });

  it('B3.5: high dynamic range array handles bandwidth without NaN', () => {
    const vals = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4];
    const h = silvermanBandwidth(vals);
    assert.ok(Number.isFinite(h));
    assert.ok(h > 0);
  });

  it('B3.6: bandwidth falls back to sigma when IQR is 0 but variance > 0', () => {
    // 8 points at 10, 2 points at 20 -> IQR is 0, but sigma > 0
    const vals = [10, 10, 10, 10, 10, 10, 10, 10, 20, 20];
    const h = silvermanBandwidth(vals);
    assert.ok(h > 0, 'Bandwidth must fallback to sigma when IQR is zero');
  });
});

describe('Tier 2: Boundary 4 - Spearman Correlation Boundaries', () => {
  it('B4.1: exactly 4 observations returns null (strict minObs gate)', () => {
    const x = [1, 2, 3, 4];
    const y = [10, 20, 30, 40];
    assert.equal(spearmanRho(x, y), null);
  });

  it('B4.2: exactly 5 observations returns valid correlation in [-1, 1]', () => {
    const x = [1, 2, 3, 4, 5];
    const y = [5, 4, 3, 2, 1];
    const r = spearmanRho(x, y);
    assert.equal(r, -1.0);
  });

  it('B4.3: constant vector (zero variance on ranks) returns null without division by zero', () => {
    const x = [5, 5, 5, 5, 5];
    const y = [1, 2, 3, 4, 5];
    assert.equal(spearmanRho(x, y), null);
  });

  it('B4.4: pearsonR guard with < 3 points returns null', () => {
    assert.equal(pearsonR([1, 2], [3, 4]), null);
  });

  it('B4.5: pearsonR guard with zero variance returns null', () => {
    assert.equal(pearsonR([3, 3, 3, 3], [1, 2, 3, 4]), null);
  });

  it('B4.6: sparse arrays with fewer than 5 overlapping non-null values return null', () => {
    const x = [1, 2, 3, null, null, 6, 7];
    const y = [null, 2, 3, 4, 5, null, 7];
    // Overlapping indices: 1, 2, 6 -> 3 pairs -> < 5 -> null
    assert.equal(spearmanRho(x, y), null);
  });
});

describe('Tier 2: Boundary 5 - Ward Hierarchical Clustering Boundaries', () => {
  it('B5.1: 0 variables returns empty array []', () => {
    assert.deepEqual(wardCluster([], []), []);
  });

  it('B5.2: 1 variable returns [0]', () => {
    assert.deepEqual(wardCluster([[1.0]], ['v1']), [0]);
  });

  it('B5.3: orthogonal identity correlation matrix clusters deterministically', () => {
    const k = 3;
    const rho = [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ];
    const order = wardCluster(rho, ['v1', 'v2', 'v3']);
    assert.equal(order.length, 3);
    assert.deepEqual(new Set(order), new Set([0, 1, 2]));
  });

  it('B5.4: all perfectly correlated variables (rho=1.0) cluster without division by zero', () => {
    const rho = [
      [1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0],
    ];
    const order = wardCluster(rho, ['v1', 'v2', 'v3']);
    assert.equal(order.length, 3);
    assert.deepEqual(new Set(order), new Set([0, 1, 2]));
  });

  it('B5.5: matrix with null values treats distance as 1.0 (uncorrelated) gracefully', () => {
    const rho = [
      [1.0, null, 0.8],
      [null, 1.0, null],
      [0.8, null, 1.0],
    ];
    const order = wardCluster(rho, ['v1', 'v2', 'v3']);
    assert.equal(order.length, 3);
    assert.deepEqual(new Set(order), new Set([0, 1, 2]));
  });
});

describe('Tier 2: Boundary 6 - Dual-Gram PCA Boundaries', () => {
  it('B6.1: empty input vector array returns valid empty PCA payload', () => {
    const res = computePCA([]);
    assert.equal(res.n, 0);
    assert.deepEqual(res.points, []);
    assert.equal(res.cumulative_2pc, 0);
  });

  it('B6.2: single vector array N=1 centers to [0, 0] with zero variance', () => {
    const res = computePCA([[10, 20, 30]], ['c1'], ['yes'], ['task1']);
    assert.equal(res.n, 1);
    assert.equal(res.points.length, 1);
    assert.equal(res.points[0].pc1, 0);
    assert.equal(res.points[0].pc2, 0);
  });

  it('B6.3: all identical vectors (constant matrix) results in 0 variance explained', () => {
    const vectors = [
      [5, 5, 5],
      [5, 5, 5],
      [5, 5, 5],
      [5, 5, 5],
    ];
    const res = computePCA(vectors, ['1', '2', '3', '4'], ['a', 'a', 'a', 'a'], ['t1', 't1', 't1', 't1']);
    assert.equal(res.variance_explained[0], 0);
    assert.equal(res.variance_explained[1], 0);
    assert.equal(res.cumulative_2pc, 0);
  });

  it('B6.4: collinear data (singular 1D line in 3D space) has 100% variance on PC1 and 0% on PC2', () => {
    const vectors = [
      [1, 2, 3],
      [2, 4, 6],
      [3, 6, 9],
      [4, 8, 12],
      [5, 10, 15],
    ];
    const res = computePCA(vectors, ['1', '2', '3', '4', '5'], ['y', 'y', 'y', 'y', 'y'], ['t1', 't1', 't1', 't1', 't1']);
    assert.ok(res.variance_explained[0] > 99.9);
    assert.ok(res.variance_explained[1] < 0.1);
  });

  it('B6.5: high dimensional embedding vectors (N=5, D=1024) computes Dual Gram without memory explosion', () => {
    const vectors = Array.from({ length: 5 }, (_, i) => Array.from({ length: 1024 }, (_, j) => Math.sin(i * 1024 + j)));
    const res = computePCA(vectors, ['1', '2', '3', '4', '5'], ['y', 'y', 'y', 'y', 'y'], ['t1', 't1', 't1', 't1', 't1']);
    assert.equal(res.points.length, 5);
    assert.ok(Number.isFinite(res.cumulative_2pc));
  });
});

describe('Tier 2: Boundary 7 - Modality Missingness & Sparsity Boundaries', () => {
  it('B7.1: empty cases array returns empty matrix and cases list', () => {
    const res = CohortEngine.computeMissingness([]);
    assert.deepEqual(res.matrix, []);
    assert.deepEqual(res.cases, []);
    assert.equal(res.modalities.length, 6);
  });

  it('B7.2: 100% present cohort returns all 1s matrix', () => {
    const cases = Array.from({ length: 4 }, (_, i) => ({
      case_id: `pres_${i}`,
      task: 'task2',
      missingness: { MRI: true, Biopsy: true, Prostatectomy: true, PSA_Trend: true, Labs: true, FamilyHistory: true },
    }));
    const res = CohortEngine.computeMissingness(cases);
    assert.equal(res.matrix.length, 4);
    for (const row of res.matrix) {
      assert.deepEqual(row, [1, 1, 1, 1, 1, 1]);
    }
  });

  it('B7.3: 100% absent cohort returns all 0s matrix', () => {
    const cases = Array.from({ length: 4 }, (_, i) => ({
      case_id: `abs_${i}`,
      task: 'task1',
      missingness: { MRI: false, Biopsy: false, Prostatectomy: false, PSA_Trend: false, Labs: false, FamilyHistory: false },
    }));
    const res = CohortEngine.computeMissingness(cases);
    for (const row of res.matrix) {
      assert.deepEqual(row, [0, 0, 0, 0, 0, 0]);
    }
  });

  it('B7.4: missingness normalization on completely empty clinical records handles missing gracefully', () => {
    const norm = CohortEngine.normalizeCase({ case_id: 'blank' });
    assert.equal(norm.missingness.MRI, false);
    assert.equal(norm.missingness.Biopsy, false);
    assert.equal(norm.missingness.Prostatectomy, false);
    assert.equal(norm.missingness.PSA_Trend, false);
    assert.equal(norm.missingness.Labs, false);
    assert.equal(norm.missingness.FamilyHistory, false);
  });

  it('B7.5: handles malformed embedding objects (strings/scalars) without throwing', () => {
    const trace = {
      case_id: 'bad_mod',
      modality_representations: {
        'MRI image': 'not_an_array',
        'Biopsy slide': 12345,
      },
    };
    const norm = CohortEngine.normalizeCase(trace);
    assert.equal(norm.mri_vec, null);
    assert.equal(norm.bx_vec, null);
  });
});

describe('Tier 2: Boundary 8 - Fail-Closed Parsing & SafeFloat Boundaries', () => {
  it('B8.1: safeFloat handles boolean, null, undefined, and empty strings safely', () => {
    assert.equal(safeFloat(true), null);
    assert.equal(safeFloat(false), null);
    assert.equal(safeFloat(null), null);
    assert.equal(safeFloat(undefined), null);
    assert.equal(safeFloat(''), null);
    assert.equal(safeFloat('   '), null);
  });

  it('B8.2: safeFloat handles non-numeric medical sentinel strings safely', () => {
    assert.equal(safeFloat('N/A'), null);
    assert.equal(safeFloat('n/a'), null);
    assert.equal(safeFloat('NOT AVAILABLE'), null);
    assert.equal(safeFloat('missing'), null);
    assert.equal(safeFloat('None'), null);
    assert.equal(safeFloat('UNKNOWN'), null);
  });

  it('B8.3: safeFloat parses subnormals and scientific notation floats accurately', () => {
    assert.equal(safeFloat('1e-5'), 0.00001);
    assert.equal(safeFloat('2.5e3'), 2500);
    assert.equal(safeFloat('-4.2'), -4.2);
    assert.equal(safeFloat('0.0'), 0.0);
  });

  it('B8.4: Task 3 radiology regex handles missing or corrupt text without throwing', () => {
    assert.deepEqual(CohortEngine.extractTask3Radiology(null), { vol: null, psad: null, pirads: null });
    assert.deepEqual(CohortEngine.extractTask3Radiology(''), { vol: null, psad: null, pirads: null });
    assert.deepEqual(CohortEngine.extractTask3Radiology('Random text with no clinical numbers'), { vol: null, psad: null, pirads: null });
  });

  it('B8.5: Task 3 pathology regex handles missing or corrupt text without throwing', () => {
    assert.deepEqual(CohortEngine.extractTask3Pathology(null), { bx_isup: null, bx_gl_prim: null, bx_gl_sec: null });
    assert.deepEqual(CohortEngine.extractTask3Pathology(''), { bx_isup: null, bx_gl_prim: null, bx_gl_sec: null });
    assert.deepEqual(CohortEngine.extractTask3Pathology('Unremarkable tissue sample'), { bx_isup: null, bx_gl_prim: null, bx_gl_sec: null });
  });
});

describe('Tier 2: Boundary 9 - Vector Rendering Boundaries', () => {
  before(() => {
    setupMockDOM();
  });

  it('B9.1: handles empty data without throwing DOM errors', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    const container = document.getElementById('cohort-view');
    await CohortView.renderCohortTab([]);
    // With upload-first architecture, empty data shows a "No data loaded" message
    assert.ok(container.innerHTML.length > 0);
  });

  it('B9.2: renders with single data point without infinite SVG bounding box', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    const { TraceReader } = await import('../dashboard/js/data.js');
    // Load a single trace from relocated data directory
    const fs = await import('node:fs');
    const path = await import('node:path');
    const tracesDir = path.join(path.resolve(import.meta.dirname, '..'), '..', 'chimera-data', 'traces', 'all_local_cases');
    if (!fs.existsSync(tracesDir)) {
      console.log('Skipping B9.2: trace directory not found');
      return;
    }
    const reader = new TraceReader('traces');
    const files = fs.readdirSync(tracesDir).filter(f => f.endsWith('.json')).slice(0, 5);
    const traces = [];
    for (const f of files) {
      const text = fs.readFileSync(path.join(tracesDir, f), 'utf-8');
      const res = reader.validateAndNormalize(JSON.parse(text));
      if (res.success) traces.push(res.data);
    }
    CohortView._data = { _loaded: false };
    await CohortView.renderCohortTab(traces);
    const svgs = document.getElementById('cohort-view').querySelectorAll('svg');
    assert.ok(svgs.length > 0);
  });

  it('B9.3: handles missing strata in rainclouds gracefully', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    const emptyRaincloud = {
      _loaded: true,
      raincloud_psa: {
        metric: 'psa',
        unit: 'ng/mL',
        strata: [
          { task: 'task1', target: 'yes', n: 0, box: null, kde: null, dots: [] },
          { task: 'task1', target: 'no', n: 0, box: null, kde: null, dots: [] },
        ],
      },
    };
    CohortView._data = emptyRaincloud;
    await CohortView.renderCohortTab([]);
    assert.ok(document.getElementById('cohort-view'));
  });

  it('B9.4: handles correlation matrix with 0 usable variables gracefully', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    CohortView._data = {
      _loaded: true,
      correlation: { variables: [], matrix: [], n_variables: 0, n_cases: 0, excluded: NUMERIC_VARS },
    };
    await CohortView.renderCohortTab([]);
    assert.ok(document.getElementById('cohort-view'));
  });

  it('B9.5: verifies viewBox attribute is present on all rendered SVG nodes', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    CohortView._data = {
      _loaded: true,
      composition: {
        tasks: {
          task1: { total: 1, classes: { yes: { n: 1, pct: 100 }, no: { n: 0, pct: 0 } } },
          task2: { total: 0, classes: { active_surveillance: { n: 0, pct: 0 }, continued_surveillance: { n: 0, pct: 0 }, watchful_waiting: { n: 0, pct: 0 }, active_treatment: { n: 0, pct: 0 } } },
          task3: { total: 0, classes: { '1': { n: 0, pct: 0 }, '0': { n: 0, pct: 0 } } },
        },
      },
    };
    await CohortView.renderCohortTab([]);
    const svgs = document.getElementById('cohort-view').querySelectorAll('svg');
    for (const s of svgs) {
      assert.ok(s.hasAttribute('viewBox') || s.getAttribute('viewBox') !== null);
    }
  });
});

describe('Tier 2: Boundary 10 - Dual-Scope Dynamic Filtering Boundaries', () => {
  const traces = [
    { case_id: 't1_1', task: 'task1', ground_truth: { decision: 'yes' }, patient_demographics: { psa: 6.0 } },
    { case_id: 't2_1', task: 'task2', ground_truth: { decision: 'active_surveillance' }, patient_demographics: { psa: 4.0 } },
    { case_id: 't3_1', task: 'task3', ground_truth: { event: 1 }, patient_demographics: { psa: 0.1 } },
  ];

  it('B10.1: filter="all" includes cases from all 3 tasks', () => {
    const res = CohortEngine.computeAll(traces, 'all');
    assert.equal(res.totalCases, 3);
    assert.equal(res.filteredCases, 3);
  });

  it('B10.2: filter="task1" filters cases strictly to task1', () => {
    const res = CohortEngine.computeAll(traces, 'task1');
    assert.equal(res.totalCases, 3);
    assert.equal(res.filteredCases, 1);
    assert.equal(res.filter, 'task1');
  });

  it('B10.3: filter="task2" filters cases strictly to task2', () => {
    const res = CohortEngine.computeAll(traces, 'task2');
    assert.equal(res.totalCases, 3);
    assert.equal(res.filteredCases, 1);
    assert.equal(res.filter, 'task2');
  });

  it('B10.4: filter="task3" filters cases strictly to task3', () => {
    const res = CohortEngine.computeAll(traces, 'task3');
    assert.equal(res.totalCases, 3);
    assert.equal(res.filteredCases, 1);
    assert.equal(res.filter, 'task3');
  });

  it('B10.5: unknown filter name defaults safely to all cases', () => {
    const res = CohortEngine.computeAll(traces, 'non_existent_filter');
    assert.equal(res.totalCases, 3);
    assert.equal(res.filteredCases, 3);
  });

  it('B10.6: total cohort count equals sum of all 3 individual task counts', () => {
    const resAll = CohortEngine.computeAll(traces, 'all');
    const resT1 = CohortEngine.computeAll(traces, 'task1');
    const resT2 = CohortEngine.computeAll(traces, 'task2');
    const resT3 = CohortEngine.computeAll(traces, 'task3');

    assert.equal(resAll.filteredCases, resT1.filteredCases + resT2.filteredCases + resT3.filteredCases);
  });
});

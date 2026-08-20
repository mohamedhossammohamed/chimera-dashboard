// tests/js/cohort_engine.test.js
import test from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  safeFloat,
  rankData,
  pearsonR,
  spearmanRho,
  spearmanCorrelationMatrix,
  wardCluster,
  silvermanBandwidth,
  gaussianKDE,
  tukeyBox,
  weylJitter,
  computePCA,
  CohortEngine,
  NUMERIC_VARS,
  METRIC_UNITS,
  MODALITIES,
  WEYL_PHI,
} from '../../docs/js/cohort_engine.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '../..');

// ---------------------------------------------------------------------------
// 1. Math Primitives & Rank Transformations
// ---------------------------------------------------------------------------

test('rankData handles unique, tied, and fractional mid-ranks correctly', () => {
  // Unique
  const r1 = rankData([10, 20, 30, 40]);
  assert.deepStrictEqual(r1, [1, 2, 3, 4]);

  // Single tie
  const r2 = rankData([10, 20, 20, 30]);
  assert.deepStrictEqual(r2, [1, 2.5, 2.5, 4]);

  // Multiple ties
  const r3 = rankData([5, 5, 5, 5]);
  assert.deepStrictEqual(r3, [2.5, 2.5, 2.5, 2.5]);

  // Edge cases
  assert.deepStrictEqual(rankData([]), []);
  assert.deepStrictEqual(rankData([42]), [1]);
  assert.deepStrictEqual(rankData([-5, 0, -10, 0]), [2, 3.5, 1, 3.5]);
});

test('pearsonR computes accurate Pearson correlation and respects mathematical guards', () => {
  const x = [1, 2, 3, 4, 5];
  const yPos = [2, 4, 6, 8, 10];
  const yNeg = [10, 8, 6, 4, 2];

  assert.ok(Math.abs(pearsonR(x, yPos) - 1.0) < 1e-6);
  assert.ok(Math.abs(pearsonR(x, yNeg) - (-1.0)) < 1e-6);

  // Guard: n < 3
  assert.strictEqual(pearsonR([1, 2], [3, 4]), null);

  // Guard: zero variance / zero denominator
  assert.strictEqual(pearsonR([5, 5, 5, 5], [1, 2, 3, 4]), null);
  assert.strictEqual(pearsonR([1, 2, 3, 4], [8, 8, 8, 8]), null);

  // Mismatched lengths
  assert.strictEqual(pearsonR([1, 2, 3], [1, 2]), null);
});

test('spearmanRho computes monotonic rank correlation with pairwise completeness', () => {
  const x = [1, 2, 3, 4, 5, 6];
  const y = [1, 8, 27, 64, 125, 216]; // cubic nonlinear monotonic (Pearson < 1.0, Spearman == 1.0)
  const rho = spearmanRho(x, y);
  assert.ok(Math.abs(rho - 1.0) < 1e-6);

  const yInv = [216, 125, 64, 27, 8, 1];
  const rhoInv = spearmanRho(x, yInv);
  assert.ok(Math.abs(rhoInv - (-1.0)) < 1e-6);

  // Guard: < 5 observations
  assert.strictEqual(spearmanRho([1, 2, 3, 4], [2, 4, 6, 8]), null);

  // Pairwise complete observations filtering
  const xMiss = [1, 2, null, 4, 5, 6];
  const yMiss = [2, null, 6, 8, 10, 12];
  // 4 complete pairs (indices 0, 3, 4, 5) -> < 5 complete pairs -> null
  assert.strictEqual(spearmanRho(xMiss, yMiss), null);

  const xMiss5 = [1, 2, 3, 4, null, 6, 7];
  const yMiss5 = [2, 4, 6, 8, 10, 12, 14];
  // 6 complete pairs -> valid rho = 1.0
  const rho5 = spearmanRho(xMiss5, yMiss5);
  assert.ok(Math.abs(rho5 - 1.0) < 1e-6);
});

test('spearmanCorrelationMatrix computes clustered correlation matrix with excluded features', () => {
  const featureVectors = {
    psa: [4.0, 5.2, 6.1, 7.8, 9.5, 12.0, 15.1],
    psad: [0.10, 0.12, 0.15, 0.20, 0.22, 0.28, 0.35],
    age: [60, 62, 65, 68, 70, 72, 75],
    sparse_feature: [null, null, 1.0, null, 2.0, null, null], // < 5 observations
  };

  const res = spearmanCorrelationMatrix(featureVectors, 5);
  assert.strictEqual(res.n_variables, 3);
  assert.strictEqual(res.matrix.length, 3);
  assert.deepStrictEqual(res.excluded, ['sparse_feature']);

  // Diagonal invariant
  for (let i = 0; i < 3; i++) {
    assert.strictEqual(res.matrix[i][i], 1.0);
    for (let j = 0; j < 3; j++) {
      assert.strictEqual(res.matrix[i][j], res.matrix[j][i]);
      assert.ok(res.matrix[i][j] >= -1.0 && res.matrix[i][j] <= 1.0);
    }
  }
});

test('wardCluster Lance-Williams minimum-variance agglomeration and dendrogram ordering', () => {
  // 4 variables: (0, 1) highly correlated, (2, 3) highly correlated
  const rho = [
    [1.0, 0.95, 0.0, 0.0],
    [0.95, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.90],
    [0.0, 0.0, 0.90, 1.0]
  ];
  const varNames = ['a1', 'a2', 'b1', 'b2'];
  const order = wardCluster(rho, varNames);
  assert.strictEqual(order.length, 4);
  const setOrder = new Set(order);
  assert.strictEqual(setOrder.size, 4);

  // Single variable
  assert.deepStrictEqual(wardCluster([[1.0]], ['single']), [0]);
  // Empty
  assert.deepStrictEqual(wardCluster([], []), []);
});

// ---------------------------------------------------------------------------
// 2. Tukey 5-Number Quantiles & Weyl Golden-Ratio Jitter
// ---------------------------------------------------------------------------

test('tukeyBox computes Type-7 continuous linear quantiles, observed datum whiskers, and outliers', () => {
  // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  // n=10: Q1 = 3.25, Med = 5.5, Q3 = 7.75, IQR = 4.5
  // Lower fence = 3.25 - 1.5*4.5 = -3.5 -> observed datum whisker = 1.0
  // Upper fence = 7.75 + 1.5*4.5 = 14.5 -> observed datum whisker = 10.0
  const vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const box = tukeyBox(vals);

  assert.ok(Math.abs(box.q1 - 3.25) < 1e-4);
  assert.ok(Math.abs(box.median - 5.5) < 1e-4);
  assert.ok(Math.abs(box.q3 - 7.75) < 1e-4);
  assert.ok(Math.abs(box.iqr - 4.5) < 1e-4);
  assert.strictEqual(box.whisker_lo, 1);
  assert.strictEqual(box.whisker_hi, 10);
  assert.strictEqual(box.whiskerMin, 1);
  assert.strictEqual(box.whiskerMax, 10);
  assert.strictEqual(box.count, 10);
  assert.deepStrictEqual(box.outliers, []);

  // With outlier
  const valsWithOutlier = [10, 11, 12, 13, 14, 15, 100];
  const boxOut = tukeyBox(valsWithOutlier);
  assert.ok(boxOut.outliers.includes(100));
  assert.strictEqual(boxOut.whisker_hi, 15);
  assert.strictEqual(boxOut.whisker_lo, 10);

  // Single element
  const boxSingle = tukeyBox([42.5]);
  assert.strictEqual(boxSingle.q1, 42.5);
  assert.strictEqual(boxSingle.median, 42.5);
  assert.strictEqual(boxSingle.q3, 42.5);
  assert.strictEqual(boxSingle.iqr, 0);
  assert.strictEqual(boxSingle.whisker_lo, 42.5);
  assert.strictEqual(boxSingle.whisker_hi, 42.5);
  assert.strictEqual(boxSingle.count, 1);
  assert.deepStrictEqual(boxSingle.outliers, []);

  // Constant array
  const boxConst = tukeyBox([5, 5, 5, 5, 5]);
  assert.strictEqual(boxConst.q1, 5);
  assert.strictEqual(boxConst.median, 5);
  assert.strictEqual(boxConst.q3, 5);
  assert.strictEqual(boxConst.iqr, 0);
  assert.strictEqual(boxConst.whisker_lo, 5);
  assert.strictEqual(boxConst.whisker_hi, 5);
  assert.deepStrictEqual(boxConst.outliers, []);

  // Empty / invalid
  assert.strictEqual(tukeyBox([]), null);
  assert.strictEqual(tukeyBox(null), null);
});

test('tukeyBox rejects booleans and non-numeric strings via type-coercion guard', () => {
  // Regression: booleans (true→1, false→0) and strings ('N/A') previously passed
  // through the filter via type coercion, corrupting quartile computation.
  // The filter must match the sentinel set used by safeFloat in clinical_engine.js.
  const dirty = [10, null, NaN, undefined, true, false, 'N/A', 20, 30, 40, 50];
  const box = tukeyBox(dirty);
  assert.ok(box !== null);
  assert.strictEqual(box.count, 5); // only 10, 20, 30, 40, 50
  assert.strictEqual(box.median, 30);
  assert.strictEqual(box.whisker_lo, 10);
  assert.strictEqual(box.whisker_hi, 50);
});

test('weylJitter generates deterministic quasi-random jitter in [-0.5, 0.5]', () => {
  const vals = [10, 20, 30, 40, 50];
  const jittered = weylJitter(vals);
  assert.strictEqual(jittered.length, 5);

  for (let i = 0; i < jittered.length; i++) {
    const pt = jittered[i];
    assert.strictEqual(pt.value, vals[i]);
    assert.ok(pt.jitter >= -0.5 && pt.jitter <= 0.5);
    const expected = (i * WEYL_PHI) % 1.0 - 0.5;
    assert.ok(Math.abs(pt.jitter - expected) < 1e-4);
  }
});

// ---------------------------------------------------------------------------
// 3. Silverman Gaussian KDE
// ---------------------------------------------------------------------------

test('silvermanBandwidth computes AMISE bandwidth and handles edge cases', () => {
  const vals = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];
  const h = silvermanBandwidth(vals);
  assert.ok(h > 0);

  // Zero-variance guard
  assert.strictEqual(silvermanBandwidth([5, 5, 5, 5, 5, 5]), 0);
  assert.strictEqual(silvermanBandwidth([5]), 0);
  assert.strictEqual(silvermanBandwidth([]), 0);

  // Zero-IQR with positive standard deviation fallback
  const zeroIqrVals = [5, 5, 5, 5, 5, 5, 5, 100];
  const hFallback = silvermanBandwidth(zeroIqrVals);
  assert.ok(hFallback > 0);
});

test('gaussianKDE supports both explicit eval grid and automatic 80-point grid', () => {
  const vals = [10, 20, 30, 40, 50];
  const h = silvermanBandwidth(vals);

  // Signature A: (values, bandwidth, evalPoints)
  const grid = [0, 10, 20, 30, 40, 50, 60];
  const densities = gaussianKDE(vals, h, grid);
  assert.strictEqual(densities.length, grid.length);
  for (const d of densities) {
    assert.ok(d >= 0, 'Density must be non-negative');
  }

  // Signature B: (values, nPoints = 80, padding = 0.15)
  const kdeRes = gaussianKDE(vals, 80, 0.15);
  assert.strictEqual(kdeRes.grid.length, 80);
  assert.strictEqual(kdeRes.density.length, 80);
  assert.strictEqual(kdeRes.points.length, 80);
  assert.ok(kdeRes.bandwidth > 0);

  // Approximate Riemann sum integral over finite evaluation grid
  const dx = kdeRes.grid[1] - kdeRes.grid[0];
  const integral = kdeRes.density.reduce((sum, d) => sum + d, 0) * dx;
  assert.ok(integral > 0.80 && integral < 1.05, `Integral ${integral} should be ~1.0 (with finite grid truncation)`);

  // Degenerate input
  const emptyKde = gaussianKDE([], 80, 0.15);
  assert.deepStrictEqual(emptyKde.grid, []);
  assert.strictEqual(emptyKde.bandwidth, 0);
});

// ---------------------------------------------------------------------------
// 4. Dual-Gram Power Iteration PCA Eigendecomposition
// ---------------------------------------------------------------------------

test('computePCA projects high-dimensional embeddings into 2D and 3D with exact variance explained', () => {
  // Synthesize 30 cases with 1024-d embeddings having strong 1st and 2nd components
  const N = 30;
  const D = 1024;
  const vectors = [];
  const caseIds = [];
  const targets = [];
  const tasks = [];

  for (let i = 0; i < N; i++) {
    const row = new Float64Array(D);
    const t = (i - N / 2);
    row[0] = t * 5.0; // dominant PC1
    row[1] = Math.sin(t) * 2.0; // PC2
    row[2] = Math.cos(t) * 1.0; // PC3
    for (let c = 3; c < D; c++) {
      row[c] = (Math.sin(i * 13 + c) * 0.05); // minor noise
    }
    vectors.push(Array.from(row));
    caseIds.push(`CASE_${i}`);
    targets.push(i % 2 === 0 ? 'yes' : 'no');
    tasks.push('task1');
  }

  const pca = computePCA(vectors, caseIds, targets, tasks);
  assert.strictEqual(pca.n, N);
  assert.strictEqual(pca.points.length, N);
  assert.strictEqual(pca.variance_explained.length, 3);
  assert.ok(pca.variance_explained[0] > 70.0, 'PC1 must explain majority variance');
  assert.ok(pca.cumulative_2pc > pca.variance_explained[0]);
  assert.ok(pca.cumulative_3pc >= pca.cumulative_2pc);
  assert.ok(pca.totalVariance > 0);

  // Coordinate integrity
  const pt0 = pca.points[0];
  assert.strictEqual(pt0.case_id, 'CASE_0');
  assert.strictEqual(pt0.task, 'task1');
  assert.strictEqual(pt0.target, 'yes');
  assert.strictEqual(typeof pt0.pc1, 'number');
  assert.strictEqual(typeof pt0.pc2, 'number');
  assert.strictEqual(typeof pt0.pc3, 'number');
  assert.strictEqual(pt0.x, pt0.pc1);
  assert.strictEqual(pt0.y, pt0.pc2);
  assert.strictEqual(pt0.z, pt0.pc3);
});

test('computePCA handles degenerate inputs fail-closed', () => {
  // Empty
  const emptyRes = computePCA([]);
  assert.strictEqual(emptyRes.n, 0);
  assert.deepStrictEqual(emptyRes.points, []);
  assert.deepStrictEqual(emptyRes.variance_explained, [0, 0, 0]);

  // Single sample
  const singleRes = computePCA([[1, 2, 3]], ['CASE_1']);
  assert.strictEqual(singleRes.n, 1);
  assert.strictEqual(singleRes.points.length, 1);
  assert.strictEqual(singleRes.points[0].pc1, 0);
  assert.strictEqual(singleRes.points[0].pc2, 0);
  assert.strictEqual(singleRes.points[0].pc3, 0);

  // Zero variance (all identical vectors)
  const flatRes = computePCA([[1, 2], [1, 2], [1, 2]], ['C1', 'C2', 'C3']);
  assert.strictEqual(flatRes.n, 3);
  assert.strictEqual(flatRes.totalVariance, 0);
  assert.strictEqual(flatRes.points[0].pc1, 0);
});

// ---------------------------------------------------------------------------
// 5. Cohort Normalization, Composition, Missingness & Rainclouds
// ---------------------------------------------------------------------------

test('CohortEngine.normalizeCase robustly extracts variables, targets, and embeddings', () => {
  const rawTrace = {
    case_id: 'PT-TEST-001',
    task: 'task1',
    patient_demographics: {
      age: 65,
      psa: 8.5,
      psad: 0.21,
      vol: 40.5,
      pirads: '4',
      cspca: 1,
    },
    modality_representations: {
      'MRI image': [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], // 2 slides, 3-d
      'Biopsy slide': null,
    },
    clinical_records: {
      psa_trend: [{ date: '2023-01-01', val: 7.0 }, { date: '2023-08-01', val: 8.5 }],
      laboratory_results: [{ name: 'ALP', val: '75 U/L' }],
      family_history: 'Positive (father)',
    },
    ground_truth: {
      decision: 'yes',
    },
  };

  const norm = CohortEngine.normalizeCase(rawTrace);
  assert.strictEqual(norm.case_id, 'PT-TEST-001');
  assert.strictEqual(norm.task, 'task1');
  assert.strictEqual(norm.target, 'yes');
  assert.strictEqual(norm.variables.psa, 8.5);
  assert.strictEqual(norm.variables.psad, 0.21);
  assert.strictEqual(norm.variables.vol, 40.5);
  assert.strictEqual(norm.variables.age, 65);
  assert.strictEqual(norm.variables.pirads, 4);
  assert.strictEqual(norm.variables.cspca, 1);

  // Mean-pooled MRI embedding: [(0.1+0.3)/2, (0.2+0.4)/2, (0.3+0.5)/2] = [0.2, 0.3, 0.4]
  assert.deepStrictEqual(norm.mri_vec, [0.2, 0.3, 0.4]);
  assert.strictEqual(norm.bx_vec, null);

  // Missingness flags
  assert.strictEqual(norm.missingness.MRI, true);
  assert.strictEqual(norm.missingness.Biopsy, false);
  assert.strictEqual(norm.missingness.PSA_Trend, true);
  assert.strictEqual(norm.missingness.Labs, true);
  assert.strictEqual(norm.missingness.FamilyHistory, true);
});

test('CohortEngine.normalizeCase extracts Task 3 pathology and radiology regex fallbacks', () => {
  const rawTraceT3 = {
    case_id: 'PT-TEST-T3',
    task: 'task3',
    patient_demographics: {},
    clinical_records: {
      radiology_report: 'Prostate volume: 45.2 cc. PSA density: 0.18 ng/mL/cc. PI-RADS: 4 assessment.',
      pathology_report: 'ISUP grade group 3. Gleason 4+3 adenocarcinoma. Surgical margins negative.',
    },
    ground_truth: {
      event: 1,
    },
  };

  const norm = CohortEngine.normalizeCase(rawTraceT3);
  assert.strictEqual(norm.task, 'task3');
  assert.strictEqual(norm.target, '1');
  assert.strictEqual(norm.variables.vol, 45.2);
  assert.strictEqual(norm.variables.psad, 0.18);
  assert.strictEqual(norm.variables.pirads, 4);
  assert.strictEqual(norm.variables.bx_isup, 3);
  assert.strictEqual(norm.variables.bx_gl_prim, 4);
  assert.strictEqual(norm.variables.bx_gl_sec, 3);
});

test('CohortEngine.computeMissingness produces 6-channel sparsity metrics', () => {
  const cases = [
    { case_id: 'C1', task: 'task1', missingness: { MRI: true, Biopsy: false, Prostatectomy: false, PSA_Trend: true, Labs: true, FamilyHistory: true } },
    { case_id: 'C2', task: 'task2', missingness: { MRI: true, Biopsy: true, Prostatectomy: false, PSA_Trend: true, Labs: false, FamilyHistory: false } },
    { case_id: 'C3', task: 'task3', missingness: { MRI: true, Biopsy: true, Prostatectomy: true, PSA_Trend: false, Labs: false, FamilyHistory: false } },
  ];

  const res = CohortEngine.computeMissingness(cases);
  assert.deepStrictEqual(res.modalities, MODALITIES);
  assert.strictEqual(res.matrix.length, 3);
  assert.strictEqual(res.matrix[0].length, 6);

  assert.ok(res.availabilityRate.MRI === 100);
  assert.ok(res.cohortSparsity > 0);
  assert.ok(res.expected_absence.task1.Biopsy.includes('Task 1'));
  assert.ok(res.summary.task1.MRI.present === 1);
});

// ---------------------------------------------------------------------------
// 6. Live End-to-End Computation & Performance (<50ms)
// ---------------------------------------------------------------------------

test('CohortEngine.computeAll executes on all 423 local traces with dual-scope filtering under 50ms', () => {
  const tracesDir = path.join(ROOT, '..', 'chimera-data', 'traces', 'all_local_cases');
  if (!fs.existsSync(tracesDir)) {
    console.log('Skipping local trace files test: directory not found');
    return;
  }

  const files = fs.readdirSync(tracesDir).filter(f => f.endsWith('.json'));
  assert.ok(files.length > 0, 'Must have local trace JSON files');

  const rawTraces = files.map(f => JSON.parse(fs.readFileSync(path.join(tracesDir, f), 'utf-8')));
  assert.ok(rawTraces.length >= 400, `Expected ~423 traces, found ${rawTraces.length}`);

  // Test 1: Combined Total Cohort ('all')
  const t0 = performance.now();
  const resAll = CohortEngine.computeAll(rawTraces, 'all');
  const elapsedAll = performance.now() - t0;

  assert.strictEqual(resAll.filter, 'all');
  assert.strictEqual(resAll.totalCases, rawTraces.length);
  assert.strictEqual(resAll.filteredCases, rawTraces.length);

  // Validate composition
  assert.ok(resAll.composition.tasks.task1.total > 0);
  assert.ok(resAll.composition.tasks.task2.total > 0);
  assert.ok(resAll.composition.tasks.task3.total > 0);

  // Validate PCA
  assert.ok(resAll.pca.mri.points.length > 0);
  assert.ok(resAll.pca.biopsy.points.length > 0);
  assert.ok(resAll.pca.mri.variance_explained.length >= 2);
  assert.ok(resAll.pca.biopsy.variance_explained.length >= 2);

  // Validate Correlation
  assert.ok(resAll.correlation.variables.length >= 7);
  assert.ok(resAll.correlation.tasks.task1.variables.length >= 4);

  // Validate Missingness
  assert.strictEqual(resAll.missingness.matrix.length, rawTraces.length);
  assert.strictEqual(resAll.missingness.modalities.length, 6);

  // Validate Rainclouds
  assert.ok(resAll.rainclouds.psa.strata.length === 8); // 2 + 4 + 2 strata
  assert.ok(resAll.rainclouds.psad.strata.length === 8);

  // Performance invariant (< 50ms)
  console.log(`[PERF] CohortEngine.computeAll across ${rawTraces.length} cases: ${elapsedAll.toFixed(2)}ms`);
  assert.ok(elapsedAll < 50.0, `Live computation must finish in <50ms (took ${elapsedAll.toFixed(2)}ms)`);

  // Test 2: Partition Filtering ('task1')
  const resT1 = CohortEngine.computeAll(rawTraces, 'task1');
  assert.strictEqual(resT1.filter, 'task1');
  assert.strictEqual(resT1.totalCases, rawTraces.length);
  assert.ok(resT1.filteredCases < rawTraces.length);
  assert.strictEqual(resT1.composition.tasks.task1.total, resT1.filteredCases);
  assert.strictEqual(resT1.composition.tasks.task2.total, 0);
  assert.strictEqual(resT1.composition.tasks.task3.total, 0);

  // Test 3: Partition Filtering ('task2')
  const resT2 = CohortEngine.computeAll(rawTraces, 'task2');
  assert.strictEqual(resT2.filter, 'task2');
  assert.ok(resT2.filteredCases > 0);

  // Test 4: Partition Filtering ('task3')
  const resT3 = CohortEngine.computeAll(rawTraces, 'task3');
  assert.strictEqual(resT3.filter, 'task3');
  assert.ok(resT3.filteredCases > 0);
});

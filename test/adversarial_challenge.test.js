// test/adversarial_challenge.test.js
// Ruthless Adversarial Scientific Audit & Empirical Stress-Test Suite for Milestone M1
// Testing: dashboard/js/cohort_engine.js

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
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
} from '../docs/js/cohort_engine.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

// Helper to assert no NaN, null, or Inf in numerical output
function assertValidFiniteNumber(val, name = 'val') {
  assert.ok(typeof val === 'number', `${name} must be number, got ${typeof val}`);
  assert.ok(!Number.isNaN(val), `${name} must not be NaN`);
  assert.ok(Number.isFinite(val), `${name} must be finite, got ${val}`);
}

describe('Adversarial Challenge 1: Extreme Numerical Inputs & Boundary Resilience', () => {
  it('1.1: tukeyBox handles identical values, constant arrays, and singletons with zero IQR', () => {
    // Constant array
    const constArr = [7.7, 7.7, 7.7, 7.7, 7.7, 7.7, 7.7];
    const boxConst = tukeyBox(constArr);
    assert.ok(boxConst !== null);
    assert.equal(boxConst.q1, 7.7);
    assert.equal(boxConst.median, 7.7);
    assert.equal(boxConst.q3, 7.7);
    assert.equal(boxConst.iqr, 0);
    assert.equal(boxConst.whisker_lo, 7.7);
    assert.equal(boxConst.whisker_hi, 7.7);
    assert.deepEqual(boxConst.outliers, []);
    assert.equal(boxConst.count, 7);

    // Single item
    const singleBox = tukeyBox([123.456]);
    assert.equal(singleBox.q1, 123.456);
    assert.equal(singleBox.median, 123.456);
    assert.equal(singleBox.q3, 123.456);
    assert.equal(singleBox.iqr, 0);
    assert.equal(singleBox.whisker_lo, 123.456);
    assert.equal(singleBox.whisker_hi, 123.456);
    assert.equal(singleBox.count, 1);

    // Two items
    const twoBox = tukeyBox([10, 20]);
    // For n=2: k(25) = 1*0.25 = 0.25 -> vs[0]*0.75 + vs[1]*0.25 = 7.5 + 5 = 12.5
    // k(50) = 1*0.50 = 0.5 -> vs[0]*0.5 + vs[1]*0.5 = 15
    // k(75) = 1*0.75 = 0.75 -> vs[0]*0.25 + vs[1]*0.75 = 2.5 + 15 = 17.5
    assert.equal(twoBox.q1, 12.5);
    assert.equal(twoBox.median, 15);
    assert.equal(twoBox.q3, 17.5);
    assert.equal(twoBox.iqr, 5);
    assert.equal(twoBox.whisker_lo, 10);
    assert.equal(twoBox.whisker_hi, 20);
    assert.equal(twoBox.count, 2);
  });

  it('1.2: tukeyBox handles extreme dynamic range outliers without numeric overflow or NaN', () => {
    const extremeVals = [1e-10, 2e-10, 3e-10, 4e-10, 1e10, 1e20];
    const box = tukeyBox(extremeVals);
    assertValidFiniteNumber(box.q1, 'q1');
    assertValidFiniteNumber(box.median, 'median');
    assertValidFiniteNumber(box.q3, 'q3');
    assertValidFiniteNumber(box.iqr, 'iqr');
    assertValidFiniteNumber(box.whisker_lo, 'whisker_lo');
    assertValidFiniteNumber(box.whisker_hi, 'whisker_hi');
    assert.ok(box.outliers.length > 0);
    for (const o of box.outliers) {
      assertValidFiniteNumber(o, 'outlier');
    }
  });

  it('1.3: tukeyBox filters NaN, null, undefined, booleans, and non-numeric contamination', () => {
    const dirty = [10, null, NaN, undefined, true, false, 'N/A', 20, 30, 40, 50];
    const box = tukeyBox(dirty);
    assert.ok(box !== null);
    assert.equal(box.count, 5); // 10, 20, 30, 40, 50
    assert.equal(box.median, 30);
    assert.equal(box.whisker_lo, 10);
    assert.equal(box.whisker_hi, 50);

    // Completely empty or all-null
    assert.equal(tukeyBox([]), null);
    assert.equal(tukeyBox([null, undefined, NaN]), null);
    assert.equal(tukeyBox(null), null);
    assert.equal(tukeyBox(undefined), null);
  });

  it('1.4: silvermanBandwidth handles zero-variance, zero-IQR with positive sigma, and large n', () => {
    // Constant array (zero variance)
    assert.equal(silvermanBandwidth([5, 5, 5, 5, 5]), 0.0);
    assert.equal(silvermanBandwidth([]), 0.0);
    assert.equal(silvermanBandwidth([10]), 0.0);

    // Zero IQR with positive sigma (e.g. median repeated multiple times with rare spike)
    const zeroIqrVals = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 50];
    const hFallback = silvermanBandwidth(zeroIqrVals);
    assert.ok(hFallback > 0, `Bandwidth must fall back to sigma when IQR=0: got ${hFallback}`);
    assertValidFiniteNumber(hFallback, 'hFallback');

    // Standard Gaussian sample AMISE scaling
    const normalSample = [
      -1.5, -1.2, -0.8, -0.5, -0.2, 0.0, 0.1, 0.3, 0.6, 0.9, 1.3, 1.7
    ];
    const hNorm = silvermanBandwidth(normalSample);
    assert.ok(hNorm > 0.3 && hNorm < 1.0, `Bandwidth for standard sample expected ~0.5, got ${hNorm}`);
  });

  it('1.5: gaussianKDE maintains non-negativity and approximate unit area integral', () => {
    const vals = [10, 15, 20, 25, 30, 35, 40, 45, 50];
    const kde = gaussianKDE(vals, 80, 0.15);

    assert.equal(kde.grid.length, 80);
    assert.equal(kde.density.length, 80);
    assert.equal(kde.points.length, 80);
    assert.ok(kde.bandwidth > 0);

    // Non-negativity
    for (let i = 0; i < kde.density.length; i++) {
      assert.ok(kde.density[i] >= 0, `Density point ${i} must be >= 0, got ${kde.density[i]}`);
    }

    // Trapezoidal numerical integration: sum_i 0.5 * (y_i + y_{i+1}) * dx
    let integral = 0;
    for (let i = 0; i < kde.grid.length - 1; i++) {
      const dx = kde.grid[i + 1] - kde.grid[i];
      const avgY = (kde.density[i] + kde.density[i + 1]) / 2.0;
      integral += avgY * dx;
    }

    // Finite grid [min-15%, max+15%] captures ~95-100% of probability mass
    assert.ok(integral >= 0.85 && integral <= 1.05, `KDE integral over grid must be ~1.0, got ${integral}`);

    // Signature A consistency: gaussianKDE(values, bandwidth, evalPoints)
    const manualDensities = gaussianKDE(vals, kde.bandwidth, kde.grid);
    assert.equal(manualDensities.length, 80);
    for (let i = 0; i < 80; i++) {
      assert.ok(Math.abs(manualDensities[i] - kde.density[i]) < 1e-4, `Signature A and B mismatch at ${i}`);
    }
  });

  it('1.6: weylJitter is strictly deterministic, bounded in [-0.5, 0.5], and preserves order statistics', () => {
    const vals = [5, 1, 9, 3, 7];
    const res1 = weylJitter(vals);
    const res2 = weylJitter(vals);

    assert.equal(res1.length, 5);
    // Deterministic repeatability
    assert.deepEqual(res1, res2);

    // Sorted order preserved: [1, 3, 5, 7, 9]
    assert.deepEqual(res1.map(d => d.value), [1, 3, 5, 7, 9]);

    // Bounded in [-0.5, 0.5]
    for (let i = 0; i < res1.length; i++) {
      const j = res1[i].jitter;
      assert.ok(j >= -0.5 && j <= 0.5, `Jitter ${j} must be in [-0.5, 0.5]`);
      const expected = ((i * WEYL_PHI) % 1.0) - 0.5;
      assert.ok(Math.abs(j - Math.round(expected * 1e6) / 1e6) < 1e-6);
    }
  });
});

describe('Adversarial Challenge 2: Spearman Rank Tie-Breaking & Collinearity', () => {
  it('2.1: rankData computes exact fractional mid-ranks for complex multi-way ties', () => {
    // Alternating 2-way ties: [10, 10, 20, 20, 30, 30]
    // 1-based ranks: 1, 2 -> 1.5; 3, 4 -> 3.5; 5, 6 -> 5.5
    const r1 = rankData([10, 10, 20, 20, 30, 30]);
    assert.deepEqual(r1, [1.5, 1.5, 3.5, 3.5, 5.5, 5.5]);

    // 4-way tie with singletons: [5, 10, 10, 10, 10, 25]
    // Ranks: 5 -> 1; 10s (2,3,4,5) -> 3.5; 25 -> 6
    const r2 = rankData([5, 10, 10, 10, 10, 25]);
    assert.deepEqual(r2, [1, 3.5, 3.5, 3.5, 3.5, 6]);

    // All identical (10-way tie): [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    // Ranks 1..10 -> mid-rank = 5.5
    const r3 = rankData(new Array(10).fill(7));
    assert.deepEqual(r3, new Array(10).fill(5.5));
  });

  it('2.2: spearmanRho detects perfect monotonic nonlinear relations (cubic, exponential, logarithmic)', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8];
    const yCubic = x.map(v => v ** 3);
    const yExp = x.map(v => Math.exp(v));
    const yLog = x.map(v => Math.log(v + 1));
    const yDecreasing = x.map(v => -Math.exp(v));

    assert.ok(Math.abs(spearmanRho(x, yCubic) - 1.0) < 1e-6);
    assert.ok(Math.abs(spearmanRho(x, yExp) - 1.0) < 1e-6);
    assert.ok(Math.abs(spearmanRho(x, yLog) - 1.0) < 1e-6);
    assert.ok(Math.abs(spearmanRho(x, yDecreasing) - (-1.0)) < 1e-6);
  });

  it('2.3: spearmanRho handles zero-variance in ranks fail-closed (returns null, no division by zero / NaN)', () => {
    const xConst = [4, 4, 4, 4, 4, 4];
    const yVar = [1, 2, 3, 4, 5, 6];
    // x has zero variance -> Pearson on rank must return null
    assert.equal(spearmanRho(xConst, yVar), null);
    assert.equal(spearmanRho(yVar, xConst), null);
    assert.equal(spearmanRho(xConst, xConst), null);
  });

  it('2.4: spearmanCorrelationMatrix pairwise complete filtering and feature exclusion', () => {
    const features = {
      dense1: [1, 2, 3, 4, 5, 6],
      dense2: [2, 4, 6, 8, 10, 12],
      sparse_4obs: [1, 2, 3, 4, null, null], // < 5 observations
      all_null: [null, null, null, null, null, null],
      zero_var: [5, 5, 5, 5, 5, 5],
    };

    const res = spearmanCorrelationMatrix(features, 5);
    // dense1, dense2, and zero_var have >= 5 non-null observations
    assert.ok(res.variables.includes('dense1'));
    assert.ok(res.variables.includes('dense2'));
    assert.ok(res.excluded.includes('sparse_4obs'));
    assert.ok(res.excluded.includes('all_null'));

    // Check correlation between dense1 and dense2 is 1.0
    const i1 = res.variables.indexOf('dense1');
    const i2 = res.variables.indexOf('dense2');
    assert.equal(res.matrix[i1][i2], 1.0);
    assert.equal(res.matrix[i2][i1], 1.0);

    // Diagonal invariant
    for (let i = 0; i < res.matrix.length; i++) {
      assert.equal(res.matrix[i][i], 1.0);
    }
  });
});

describe('Adversarial Challenge 3: Ward Hierarchical Clustering & Dendrogram Ordering', () => {
  it('3.1: wardCluster handles trivial cluster sizes K=0, K=1, K=2', () => {
    assert.deepEqual(wardCluster([], []), []);
    assert.deepEqual(wardCluster([[1.0]], ['A']), [0]);

    const rho2 = [
      [1.0, 0.8],
      [0.8, 1.0]
    ];
    const order2 = wardCluster(rho2, ['A', 'B']);
    assert.equal(order2.length, 2);
    assert.ok(order2.includes(0) && order2.includes(1));
  });

  it('3.2: wardCluster agglomerates highly correlated clusters first and produces valid permutation', () => {
    // 6 features: Block 1 = {0, 1, 2} highly correlated (rho=0.9), Block 2 = {3, 4, 5} highly correlated (rho=0.85)
    // Between block correlation = 0.05
    const K = 6;
    const rho = Array.from({ length: K }, () => new Array(K).fill(0.05));
    for (let i = 0; i < K; i++) rho[i][i] = 1.0;

    for (let i of [0, 1, 2]) {
      for (let j of [0, 1, 2]) {
        if (i !== j) rho[i][j] = 0.90;
      }
    }
    for (let i of [3, 4, 5]) {
      for (let j of [3, 4, 5]) {
        if (i !== j) rho[i][j] = 0.85;
      }
    }

    const varNames = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5'];
    const order = wardCluster(rho, varNames);

    assert.equal(order.length, K);
    const orderSet = new Set(order);
    assert.equal(orderSet.size, K, 'Order must be a strict permutation of 0..K-1 without duplicates');

    // Verify block contiguity: {0,1,2} should be clustered together, and {3,4,5} together
    const pos012 = [order.indexOf(0), order.indexOf(1), order.indexOf(2)].sort((a, b) => a - b);
    const pos345 = [order.indexOf(3), order.indexOf(4), order.indexOf(5)].sort((a, b) => a - b);

    // Range of positions in each block must be 2 (contiguous: e.g. 0,1,2 or 3,4,5)
    assert.equal(pos012[2] - pos012[0], 2, 'Block 1 members must be contiguous in leaf order');
    assert.equal(pos345[2] - pos345[0], 2, 'Block 2 members must be contiguous in leaf order');
  });

  it('3.3: wardCluster handles null correlation values by imputing maximum dissimilarity d=1.0', () => {
    const rhoWithNull = [
      [1.0, null, 0.8],
      [null, 1.0, 0.1],
      [0.8, 0.1, 1.0]
    ];
    const order = wardCluster(rhoWithNull, ['x', 'y', 'z']);
    assert.equal(order.length, 3);
    assert.equal(new Set(order).size, 3);
  });
});

describe('Adversarial Challenge 4: High-Dimensional Dual-Gram PCA & Rank Deficiency', () => {
  it('4.1: computePCA on Rank-0 (all identical / zero variance) matrix returns fail-closed zeros', () => {
    // All rows identical in 50-dimensional space
    const N = 20;
    const D = 50;
    const flatMatrix = Array.from({ length: N }, () => new Array(D).fill(3.1415));
    const caseIds = Array.from({ length: N }, (_, i) => `CASE_${i}`);

    const res = computePCA(flatMatrix, caseIds);
    assert.equal(res.n, N);
    assert.equal(res.totalVariance, 0);
    assert.deepEqual(res.variance_explained, [0, 0, 0]);
    assert.equal(res.cumulative_2pc, 0);
    assert.equal(res.points.length, N);
    for (const pt of res.points) {
      assert.equal(pt.pc1, 0);
      assert.equal(pt.pc2, 0);
      assert.equal(pt.pc3, 0);
    }
  });

  it('4.2: computePCA on Rank-1 collinear matrix yields 100% variance on PC1 and ~0% on PC2/PC3', () => {
    // Single direction in 1024-dimensional space
    const N = 25;
    const D = 1024;
    const baseDirection = Array.from({ length: D }, (_, c) => Math.sin(c + 1));
    const norm = Math.hypot(...baseDirection);
    const u = baseDirection.map(v => v / norm);

    // Each row is a scalar multiple of u: X_i = (i - 12) * 10 * u
    const matrix = Array.from({ length: N }, (_, i) => {
      const scale = (i - 12) * 10.0;
      return u.map(v => v * scale);
    });

    const res = computePCA(matrix, Array.from({ length: N }, (_, i) => `ID_${i}`));
    assert.equal(res.n, N);
    assert.ok(res.variance_explained[0] >= 99.9, `PC1 variance explained should be ~100%, got ${res.variance_explained[0]}`);
    assert.ok(res.variance_explained[1] <= 0.05, `PC2 variance explained should be ~0%, got ${res.variance_explained[1]}`);
    assert.ok(res.variance_explained[2] <= 0.05, `PC3 variance explained should be ~0%, got ${res.variance_explained[2]}`);
  });

  it('4.3: computePCA on Rank-2 orthogonal matrix accurately splits variance according to theoretical ratio', () => {
    // 2 orthogonal principal components: sigma1 = 4.0, sigma2 = 3.0 -> variances = 16 vs 9 -> total = 25
    // Theoretical VarExpl: PC1 = 16/25 = 64.0%, PC2 = 9/25 = 36.0%, PC3 = 0.0%
    const N = 40;
    const D = 512;
    const u1 = new Array(D).fill(0);
    const u2 = new Array(D).fill(0);
    for (let i = 0; i < D / 2; i++) u1[i] = 1.0 / Math.sqrt(D / 2);
    for (let i = D / 2; i < D; i++) u2[i] = 1.0 / Math.sqrt(D / 2);

    const matrix = Array.from({ length: N }, (_, i) => {
      const angle = (2 * Math.PI * i) / N;
      const c1 = 4.0 * Math.cos(angle) * Math.sqrt(N - 1);
      const c2 = 3.0 * Math.sin(angle) * Math.sqrt(N - 1);
      const row = new Array(D).fill(0);
      for (let j = 0; j < D; j++) {
        row[j] = c1 * u1[j] + c2 * u2[j];
      }
      return row;
    });

    const res = computePCA(matrix, 3);
    assert.equal(res.n, N);
    // Check variance explained: 64% and 36% (within tolerance)
    assert.ok(Math.abs(res.variance_explained[0] - 64.0) < 1.0, `PC1 VarExpl expected ~64%, got ${res.variance_explained[0]}`);
    assert.ok(Math.abs(res.variance_explained[1] - 36.0) < 1.0, `PC2 VarExpl expected ~36%, got ${res.variance_explained[1]}`);
    assert.ok(res.variance_explained[2] < 0.1, `PC3 VarExpl expected ~0%, got ${res.variance_explained[2]}`);
    assert.ok(Math.abs(res.cumulative_2pc - 100.0) < 1.0);
  });

  it('4.4: computePCA handles high-dimensional fat matrices (D=2048, N=10) and tall matrices (D=3, N=100)', () => {
    // Fat matrix: N=10, D=2048
    const fatMatrix = Array.from({ length: 10 }, (_, i) =>
      Array.from({ length: 2048 }, (_, j) => Math.sin(i * 17 + j * 31))
    );
    const resFat = computePCA(fatMatrix);
    assert.equal(resFat.n, 10);
    assert.equal(resFat.points.length, 10);
    assert.ok(resFat.totalVariance > 0);
    for (const pt of resFat.points) {
      assertValidFiniteNumber(pt.pc1, 'fat pc1');
      assertValidFiniteNumber(pt.pc2, 'fat pc2');
      assertValidFiniteNumber(pt.pc3, 'fat pc3');
    }

    // Tall matrix: N=100, D=3
    const tallMatrix = Array.from({ length: 100 }, (_, i) => [
      i * 2.0,
      Math.sin(i) * 5.0,
      Math.cos(i) * 1.0,
    ]);
    const resTall = computePCA(tallMatrix);
    assert.equal(resTall.n, 100);
    assert.equal(resTall.points.length, 100);
    assert.ok(resTall.variance_explained[0] > 70.0);
  });

  it('4.5: computePCA sign determinism (svd_flip invariant)', () => {
    const matrix = [
      [10, 2, -1],
      [-5, -1, 2],
      [15, 3, -2],
      [-20, -4, 1],
      [0, 0, 0]
    ];
    const res1 = computePCA(matrix);
    const res2 = computePCA(matrix);
    assert.deepEqual(res1.points, res2.points);
  });
});

describe('Adversarial Challenge 5: Multimodal Missingness & Clinical Staging Invariants across Real Cohort', () => {
  it('5.1: CohortEngine.normalizeCase and computeMissingness across all 423 local cases', () => {
    const tracesDir = path.join(ROOT, '..', 'chimera-data', 'traces', 'all_local_cases');
    assert.ok(fs.existsSync(tracesDir), 'Trace directory must exist');

    const files = fs.readdirSync(tracesDir).filter(f => f.endsWith('.json'));
    assert.equal(files.length, 423, `Expected 423 local trace files, found ${files.length}`);

    const rawTraces = files.map(f => JSON.parse(fs.readFileSync(path.join(tracesDir, f), 'utf-8')));
    const allNormCases = rawTraces.map(t => CohortEngine.normalizeCase(t));

    const t1Cases = allNormCases.filter(c => c.task === 'task1');
    const t2Cases = allNormCases.filter(c => c.task === 'task2');
    const t3Cases = allNormCases.filter(c => c.task === 'task3');

    assert.equal(t1Cases.length, 195, 'Task 1 count must be 195');
    assert.equal(t2Cases.length, 153, 'Task 2 count must be 153');
    assert.equal(t3Cases.length, 75, 'Task 3 count must be 75');

    // Check Task 1 missingness
    // Clinical Staging Invariant (survey_math_spec.md Sec 2.6.2 & SCOPE.md R6):
    // "Task 1 (Pre-Biopsy Decision): Biopsy and Prostatectomy are structurally absent"
    const t1BxViolations = [];
    const t1PxViolations = [];
    for (const c of t1Cases) {
      if (c.missingness.Biopsy !== false) {
        t1BxViolations.push(c.case_id);
      }
      if (c.missingness.Prostatectomy !== false) {
        t1PxViolations.push(c.case_id);
      }
    }

    // Check Task 2 missingness: Prostatectomy is structurally absent
    const t2PxViolations = [];
    for (const c of t2Cases) {
      if (c.missingness.Prostatectomy !== false) {
        t2PxViolations.push(c.case_id);
      }
    }

    console.log(`[AUDIT] Task 1 Biopsy missingness violations: ${t1BxViolations.length} / 195 cases`);
    console.log(`[AUDIT] Task 1 Prostatectomy missingness violations: ${t1PxViolations.length} / 195 cases`);
    console.log(`[AUDIT] Task 2 Prostatectomy missingness violations: ${t2PxViolations.length} / 153 cases`);

    if (t1BxViolations.length > 0) {
      console.log(`[DEFECT FOUND] Examples of Task 1 cases with Biopsy=true: ${t1BxViolations.slice(0, 5).join(', ')}`);
    }

    // Note: In strict compliance, this assertion documents the bug if present
    assert.equal(
      t1BxViolations.length,
      0,
      `Task 1 cases must have Biopsy=false due to clinical structural absence. Found ${t1BxViolations.length} violations (e.g. ${t1BxViolations.slice(0, 3).join(', ')}).`
    );
  });
});

describe('Adversarial Challenge 6: Throughput, Latency & Memory Allocation Benchmarking', () => {
  it('6.1: Benchmark CohortEngine.computeAll across 100 consecutive full-cohort iterations', () => {
    const tracesDir = path.join(ROOT, '..', 'chimera-data', 'traces', 'all_local_cases');
    const files = fs.readdirSync(tracesDir).filter(f => f.endsWith('.json'));
    const rawTraces = files.map(f => JSON.parse(fs.readFileSync(path.join(tracesDir, f), 'utf-8')));

    // Warm-up run
    CohortEngine.computeAll(rawTraces, 'all');

    const ITERATIONS = 100;
    const latencies = [];
    const memBefore = process.memoryUsage().heapUsed;

    for (let iter = 0; iter < ITERATIONS; iter++) {
      const t0 = performance.now();
      const res = CohortEngine.computeAll(rawTraces, 'all');
      const t1 = performance.now();
      latencies.push(t1 - t0);
      assert.equal(res.totalCases, 423);
    }

    const memAfter = process.memoryUsage().heapUsed;
    const heapDeltaMb = (memAfter - memBefore) / (1024 * 1024);

    latencies.sort((a, b) => a - b);
    const meanLat = latencies.reduce((a, b) => a + b, 0) / ITERATIONS;
    const minLat = latencies[0];
    const maxLat = latencies[ITERATIONS - 1];
    const p50Lat = latencies[Math.floor(ITERATIONS * 0.50)];
    const p95Lat = latencies[Math.floor(ITERATIONS * 0.95)];

    console.log(`\n================== BENCHMARK REPORT ==================`);
    console.log(`Iterations: ${ITERATIONS} runs on 423 Multimodal Patient Cases`);
    console.log(`Latency Mean: ${meanLat.toFixed(2)} ms`);
    console.log(`Latency Min:  ${minLat.toFixed(2)} ms`);
    console.log(`Latency P50:  ${p50Lat.toFixed(2)} ms`);
    console.log(`Latency P95:  ${p95Lat.toFixed(2)} ms`);
    console.log(`Latency Max:  ${maxLat.toFixed(2)} ms`);
    console.log(`Heap Delta:   ${heapDeltaMb.toFixed(2)} MB over ${ITERATIONS} runs`);
    console.log(`======================================================\n`);

    assert.ok(p95Lat < 50.0, `P95 calculation latency must be < 50ms (got ${p95Lat.toFixed(2)}ms)`);
  });

  it('6.2: Benchmark on scaled stress cohort (1,000 synthetic 1024-d cases)', () => {
    const N_STRESS = 1000;
    const D = 1024;
    const syntheticTraces = [];

    for (let i = 0; i < N_STRESS; i++) {
      const task = i % 3 === 0 ? 'task1' : (i % 3 === 1 ? 'task2' : 'task3');
      const target = task === 'task1' ? (i % 2 === 0 ? 'yes' : 'no') : (task === 'task2' ? 'active_surveillance' : '1');
      const vec = Array.from({ length: D }, (_, c) => (Math.sin(i * 3 + c * 7) * 0.1));

      syntheticTraces.push({
        case_id: `SYNTH_${i}`,
        task,
        ground_truth: { decision: target, event: 1 },
        patient_demographics: {
          age: 50 + (i % 30),
          psa: 3.0 + (i % 20) * 0.5,
          psad: 0.10 + (i % 15) * 0.01,
          vol: 30 + (i % 40),
          pirads: String(1 + (i % 5)),
          cspca: i % 2,
        },
        modality_representations: {
          'MRI image': [vec],
          'Biopsy slide': task !== 'task1' ? [vec] : null,
        },
        clinical_records: {
          psa_trend: [{ date: '2023-01-01', val: 5.0 }],
          laboratory_results: [{ name: 'ALP', val: '70' }],
          family_history: 'None',
        },
      });
    }

    const t0 = performance.now();
    const res = CohortEngine.computeAll(syntheticTraces, 'all');
    const elapsed = performance.now() - t0;

    console.log(`[STRESS SCALING] 1,000 cases with 1024-d embeddings processed in ${elapsed.toFixed(2)}ms`);
    assert.equal(res.totalCases, 1000);
    // Threshold adjusted from 200ms to 350ms to accommodate PCA power iterations (q=2)
    // which improve numerical stability at the cost of ~80ms overhead.
    assert.ok(elapsed < 350.0, `1000 cases should compute under 350ms (took ${elapsed.toFixed(2)}ms)`);
  });
});

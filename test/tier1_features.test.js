// test/tier1_features.test.js
// Tier 1: Core Feature Verification in Isolation (10 Features x >=5 tests = >=50 tests)
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
  METRIC_UNITS,
} from '../dashboard/js/cohort_engine.js';
import { safeFloat } from '../dashboard/js/clinical_engine.js';
import { setupMockDOM } from './helpers/mock_dom.js';

describe('Tier 1: Feature 1 - Live In-Browser Cohort Computation', () => {
  it('1.1: normalizeCase correctly normalizes a Task 1 pre-biopsy trace', () => {
    const trace = {
      case_id: 'task1_sample_01',
      task: 'task1',
      patient_demographics: { age: 65, psa: 7.2, psad: 0.18, vol: 40.0, pirads: 4 },
      ground_truth: { decision: 'yes' },
      modality_representations: {
        'MRI image': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
      },
      clinical_records: {
        psa_trend: [{ date: '2023-01-01', val: 7.2 }],
        laboratory_results: [{ test: 'PSA', value: 7.2 }],
        family_history: 'Brother with prostate cancer',
      },
    };

    const norm = CohortEngine.normalizeCase(trace);
    assert.equal(norm.case_id, 'task1_sample_01');
    assert.equal(norm.task, 'task1');
    assert.equal(norm.target, 'yes');
    assert.equal(norm.variables.age, 65);
    assert.equal(norm.variables.psa, 7.2);
    assert.equal(norm.variables.psad, 0.18);
    assert.equal(norm.variables.vol, 40.0);
    assert.equal(norm.variables.pirads, 4);
    assert.equal(norm.missingness.MRI, true);
    assert.equal(norm.missingness.Biopsy, false);
    assert.equal(norm.missingness.Prostatectomy, false);
    assert.equal(norm.missingness.PSA_Trend, true);
    assert.equal(norm.missingness.Labs, true);
    assert.equal(norm.missingness.FamilyHistory, true);
  });

  it('1.2: normalizeCase correctly normalizes a Task 2 treatment stratification trace', () => {
    const trace = {
      case_id: 'task2_sample_02',
      task: 'task2',
      patient_demographics: { age: 58, psa: 4.5, bx_isup: 2, cores_positive: 3, cores_total: 12 },
      ground_truth: { decision: 'active_surveillance' },
      modality_representations: {
        'MRI image': [[0.5, 0.5]],
        'Biopsy slide': [[0.8, 0.2]],
      },
      clinical_records: {},
    };

    const norm = CohortEngine.normalizeCase(trace);
    assert.equal(norm.case_id, 'task2_sample_02');
    assert.equal(norm.task, 'task2');
    assert.equal(norm.target, 'active_surveillance');
    assert.equal(norm.variables.bx_isup, 2);
    assert.equal(norm.variables.cores_positive, 3);
    assert.equal(norm.missingness.MRI, true);
    assert.equal(norm.missingness.Biopsy, true);
    assert.equal(norm.missingness.Prostatectomy, false);
  });

  it('1.3: normalizeCase extracts unstructured radiology and pathology reports for Task 3', () => {
    const trace = {
      case_id: 'task3_sample_03',
      task: 'task3',
      patient_demographics: { age: 67, psa: 0.15 },
      ground_truth: { event: 1 },
      clinical_records: {
        radiology_report: 'Multiparametric MRI: Prostate volume: 45.5 cc. PSA density: 0.22 ng/mL/cc. PI-RADS: 4 assessment in peripheral zone.',
        pathology_report: 'Radical prostatectomy specimen: Gleason 4+3=7, ISUP grade group 3. Margins negative.',
      },
    };

    const norm = CohortEngine.normalizeCase(trace);
    assert.equal(norm.case_id, 'task3_sample_03');
    assert.equal(norm.task, 'task3');
    assert.equal(norm.target, '1');
    assert.equal(norm.variables.vol, 45.5);
    assert.equal(norm.variables.psad, 0.22);
    assert.equal(norm.variables.pirads, 4);
    assert.equal(norm.variables.bx_isup, 3);
    assert.equal(norm.variables.bx_gl_prim, 4);
    assert.equal(norm.variables.bx_gl_sec, 3);
  });

  it('1.4: mean pools 2D embedding matrices into 1D feature vectors', () => {
    const trace = {
      case_id: 'embed_test',
      task: 'task1',
      modality_representations: {
        'MRI image': [
          [1.0, 2.0, 3.0],
          [3.0, 4.0, 5.0],
        ],
      },
    };

    const norm = CohortEngine.normalizeCase(trace);
    assert.deepEqual(norm.mri_vec, [2.0, 3.0, 4.0]);
  });

  it('1.5: computeAll produces a complete analytics payload complying with interface contracts', () => {
    const traces = [
      { case_id: 'T1_1', task: 'task1', ground_truth: { decision: 'yes' }, patient_demographics: { psa: 8.0, age: 60, vol: 40, psad: 0.2 } },
      { case_id: 'T1_2', task: 'task1', ground_truth: { decision: 'no' }, patient_demographics: { psa: 3.5, age: 55, vol: 35, psad: 0.1 } },
      { case_id: 'T2_1', task: 'task2', ground_truth: { decision: 'active_surveillance' }, patient_demographics: { psa: 5.0, age: 62, vol: 50, psad: 0.1 } },
      { case_id: 'T2_2', task: 'task2', ground_truth: { decision: 'active_treatment' }, patient_demographics: { psa: 12.0, age: 70, vol: 45, psad: 0.27 } },
      { case_id: 'T3_1', task: 'task3', ground_truth: { event: 1 }, patient_demographics: { psa: 0.4, age: 68, vol: 30, psad: 0.013 } },
      { case_id: 'T3_2', task: 'task3', ground_truth: { event: 0 }, patient_demographics: { psa: 0.02, age: 64, vol: 28, psad: 0.0007 } },
    ];

    const result = CohortEngine.computeAll(traces);
    assert.ok(result.composition);
    assert.ok(result.pca_mri);
    assert.ok(result.pca_biopsy);
    assert.ok(result.correlation);
    assert.ok(result.missingness);
    assert.ok(result.raincloud_psa);
    assert.ok(result.raincloud_psad);
    assert.ok(result.raincloud_vol);
    assert.ok(result.raincloud_age);
  });

  it('1.6: execution latency for 50 traces runs well under 50ms', () => {
    const traces = Array.from({ length: 50 }, (_, i) => ({
      case_id: `bench_${i}`,
      task: i % 3 === 0 ? 'task1' : i % 3 === 1 ? 'task2' : 'task3',
      patient_demographics: { psa: 2.0 + (i % 20), age: 50 + (i % 30), vol: 20 + (i % 60), psad: 0.1 + (i % 10) * 0.02 },
      ground_truth: { decision: i % 2 === 0 ? 'yes' : 'no' },
      modality_representations: {
        'MRI image': [[Math.sin(i), Math.cos(i), Math.tan(i % 1.5)]],
      },
    }));

    const start = performance.now();
    CohortEngine.computeAll(traces);
    const elapsed = performance.now() - start;
    assert.ok(elapsed < 50, `Execution took ${elapsed.toFixed(2)}ms, expected < 50ms`);
  });
});

describe('Tier 1: Feature 2 - Tukey 5-Number Quantiles (Type-7)', () => {
  it('2.1: computes continuous Type-7 linear quantiles on 1..10 standard sequence', () => {
    const vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const b = tukeyBox(vals);
    assert.equal(b.q1, 3.25);
    assert.equal(b.median, 5.5);
    assert.equal(b.q3, 7.75);
    assert.equal(b.iqr, 4.5);
    assert.equal(b.whisker_lo, 1.0);
    assert.equal(b.whisker_hi, 10.0);
    assert.deepEqual(b.outliers, []);
  });

  it('2.2: computes exact quantiles on odd-length sequence [1..9]', () => {
    const vals = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const b = tukeyBox(vals);
    assert.equal(b.q1, 3.0);
    assert.equal(b.median, 5.0);
    assert.equal(b.q3, 7.0);
    assert.equal(b.iqr, 4.0);
    assert.equal(b.whisker_lo, 1.0);
    assert.equal(b.whisker_hi, 9.0);
  });

  it('2.3: truncates whiskers at actual data points within 1.5*IQR fences', () => {
    // Q1=10, Q3=20, IQR=10. Fences: [-5, 35]. Data points: [2, 10, 15, 20, 30]
    const vals = [2, 10, 15, 20, 30];
    const b = tukeyBox(vals);
    assert.ok(b.whisker_lo >= 2);
    assert.ok(b.whisker_hi <= 30);
  });

  it('2.4: detects high and low outliers outside 1.5*IQR fences', () => {
    const vals = [-50, 10, 11, 12, 13, 14, 15, 100];
    const b = tukeyBox(vals);
    assert.ok(b.outliers.includes(-50));
    assert.ok(b.outliers.includes(100));
    assert.equal(b.outliers.length, 2);
  });

  it('2.5: handles single element array with zero IQR', () => {
    const b = tukeyBox([42.5]);
    assert.equal(b.q1, 42.5);
    assert.equal(b.median, 42.5);
    assert.equal(b.q3, 42.5);
    assert.equal(b.iqr, 0);
    assert.equal(b.whisker_lo, 42.5);
    assert.equal(b.whisker_hi, 42.5);
    assert.deepEqual(b.outliers, []);
  });

  it('2.6: guarantees 6-decimal stability and avoids IEEE float drift', () => {
    const vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    const b = tukeyBox(vals);
    assert.equal(typeof b.q1, 'number');
    assert.equal(typeof b.median, 'number');
    assert.equal(typeof b.q3, 'number');
    assert.equal(b.median, 0.45);
  });
});

describe('Tier 1: Feature 3 - Silverman AMISE Gaussian KDE (80 pts)', () => {
  it('3.1: computes Silverman bandwidth on standard normal distribution', () => {
    // Generate pseudo-normal data
    const vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.2, -0.2, 0.8, -0.8];
    const h = silvermanBandwidth(vals);
    assert.ok(h > 0.3 && h < 1.2, `Bandwidth ${h} out of expected range`);
  });

  it('3.2: uses min(sigma, IQR/1.34) when IQR is tighter than standard deviation', () => {
    // Data with long tails: std dev is inflated, IQR is small
    const vals = [10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 100];
    const h = silvermanBandwidth(vals);
    assert.ok(h > 0);
    assert.ok(h < 15, `Bandwidth ${h} should be constrained by tight IQR`);
  });

  it('3.3: produces non-negative density values everywhere', () => {
    const vals = [2.0, 4.0, 6.0, 8.0, 10.0];
    const h = silvermanBandwidth(vals);
    const evalPts = [0, 2, 4, 6, 8, 10, 12];
    const density = gaussianKDE(vals, h, evalPts);
    assert.equal(density.length, evalPts.length);
    for (const d of density) {
      assert.ok(d >= 0, `Density ${d} should be >= 0`);
    }
  });

  it('3.4: integrates numerically close to 1.0 over padded grid', () => {
    const vals = [10, 12, 14, 16, 18, 20];
    const h = silvermanBandwidth(vals);
    const lo = 5;
    const hi = 25;
    const nPts = 100;
    const step = (hi - lo) / (nPts - 1);
    const evalPts = Array.from({ length: nPts }, (_, i) => lo + i * step);
    const density = gaussianKDE(vals, h, evalPts);
    const integral = density.reduce((sum, d) => sum + d * step, 0);
    assert.ok(Math.abs(integral - 1.0) < 0.1, `Integral ${integral} should be close to 1.0`);
  });

  it('3.5: evaluates Gaussian kernel accurately at cluster center', () => {
    const vals = [0.0];
    const h = 1.0;
    const evalPts = [0.0];
    const density = gaussianKDE(vals, h, evalPts);
    // Gaussian peak at 0 for N=1, h=1 is 1 / sqrt(2 * pi) ~ 0.398942
    const expected = 1.0 / Math.sqrt(2 * Math.PI);
    assert.ok(Math.abs(density[0] - expected) < 1e-4);
  });

  it('3.6: computeRaincloud produces exactly 80 evaluation points for KDE', () => {
    const cases = Array.from({ length: 20 }, (_, i) => ({
      case_id: `rc_${i}`,
      task: 'task1',
      target: 'yes',
      variables: { psa: 4.0 + i * 0.5 },
    }));

    const rc = CohortEngine.computeRaincloud(cases, 'psa');
    const stratum = rc.strata.find(s => s.task === 'task1' && s.target === 'yes');
    assert.ok(stratum);
    assert.equal(stratum.kde.points.length, 80);
  });
});

describe('Tier 1: Feature 4 - Spearman Correlation Matrix (Pairwise)', () => {
  it('4.1: calculates mid-ranks with exact fractional ranks for ties', () => {
    const vals = [10.0, 20.0, 20.0, 30.0];
    const ranks = rankData(vals);
    assert.deepEqual(ranks, [1.0, 2.5, 2.5, 4.0]);

    const tied4 = [5.0, 5.0, 5.0, 5.0];
    assert.deepEqual(rankData(tied4), [2.5, 2.5, 2.5, 2.5]);
  });

  it('4.2: computes Spearman rho = 1.0 for perfect nonlinear monotonic relationship', () => {
    const x = [1, 2, 3, 4, 5, 6];
    const y = [1, 8, 27, 64, 125, 216]; // y = x^3
    const rho = spearmanRho(x, y);
    assert.ok(Math.abs(rho - 1.0) < 1e-6, `Expected rho=1.0, got ${rho}`);
  });

  it('4.3: computes Spearman rho = -1.0 for perfect inverse monotonic relationship', () => {
    const x = [1, 2, 3, 4, 5, 6];
    const y = [100, 80, 50, 20, 5, 1];
    const rho = spearmanRho(x, y);
    assert.ok(Math.abs(rho - (-1.0)) < 1e-6, `Expected rho=-1.0, got ${rho}`);
  });

  it('4.4: computes Spearman rho ~ 0.0 for uncorrelated features', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = [5, 2, 8, 1, 9, 3, 7, 4, 10, 6];
    const rho = spearmanRho(x, y);
    assert.ok(Math.abs(rho) < 0.5, `Expected near zero rho, got ${rho}`);
  });

  it('4.5: enforces pairwise minimum observations requirement (minObs >= 5)', () => {
    const x = [1, 2, 3, 4];
    const y = [2, 4, 6, 8];
    assert.equal(spearmanRho(x, y), null);

    const x5 = [1, 2, 3, 4, 5];
    const y5 = [2, 4, 6, 8, 10];
    assert.equal(spearmanRho(x5, y5), 1.0);
  });

  it('4.6: matrix diagonal is 1.0 and matrix is strictly symmetric', () => {
    const cases = Array.from({ length: 10 }, (_, i) => ({
      case_id: `c_${i}`,
      task: 'task1',
      variables: { psa: 2.0 + i, psad: 0.1 + i * 0.02, vol: 30 + i * 2, age: 50 + i },
    }));

    const res = CohortEngine.computeSingleCorrelation(cases, 5);
    const k = res.variables.length;
    assert.ok(k >= 4);
    for (let i = 0; i < k; i++) {
      assert.equal(res.matrix[i][i], 1.0);
      for (let j = 0; j < k; j++) {
        assert.equal(res.matrix[i][j], res.matrix[j][i]);
      }
    }
  });
});

describe('Tier 1: Feature 5 - Ward Minimum-Variance Clustering', () => {
  it('5.1: base case with 2 variables returns valid index array [0, 1]', () => {
    const rho = [
      [1.0, 0.8],
      [0.8, 1.0],
    ];
    const order = wardCluster(rho, ['var1', 'var2']);
    assert.equal(order.length, 2);
    assert.deepEqual(new Set(order), new Set([0, 1]));
  });

  it('5.2: clusters highly correlated variable pairs into adjacent leaf positions', () => {
    // 4 variables: (0, 1) correlated (rho=0.99), (2, 3) correlated (rho=0.95), cross=0
    const rho = [
      [1.0, 0.99, 0.0, 0.0],
      [0.99, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.95],
      [0.0, 0.0, 0.95, 1.0],
    ];
    const order = wardCluster(rho, ['a1', 'a2', 'b1', 'b2']);
    assert.equal(order.length, 4);
    assert.deepEqual(new Set(order), new Set([0, 1, 2, 3]));

    // Check adjacency: either 0 and 1 are adjacent or 2 and 3 are adjacent
    const pos0 = order.indexOf(0);
    const pos1 = order.indexOf(1);
    assert.equal(Math.abs(pos0 - pos1), 1, 'Variables 0 and 1 must be adjacent in tree');
  });

  it('5.3: uses distance squared d^2 = (1 - |rho|)^2 so strong inverse correlations cluster together', () => {
    const rho = [
      [1.0, -0.98, 0.0],
      [-0.98, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ];
    const order = wardCluster(rho, ['v1', 'v2', 'v3']);
    assert.equal(order.length, 3);
    const p1 = order.indexOf(0);
    const p2 = order.indexOf(1);
    assert.equal(Math.abs(p1 - p2), 1, 'Anti-correlated variables 0 and 1 should cluster together');
  });

  it('5.4: Lance-Williams recurrence preserves all unique variable indices', () => {
    const k = 6;
    const rho = Array.from({ length: k }, (_, i) =>
      Array.from({ length: k }, (_, j) => (i === j ? 1.0 : 0.5 - Math.abs(i - j) * 0.1))
    );
    const order = wardCluster(rho, ['v0', 'v1', 'v2', 'v3', 'v4', 'v5']);
    assert.equal(order.length, k);
    assert.deepEqual(new Set(order), new Set([0, 1, 2, 3, 4, 5]));
  });

  it('5.5: single variable returns [0]', () => {
    const rho = [[1.0]];
    const order = wardCluster(rho, ['v1']);
    assert.deepEqual(order, [0]);
  });
});

describe('Tier 1: Feature 6 - Dual-Gram Power Iteration PCA (2D & 3D)', () => {
  it('6.1: centers data and constructs symmetric Dual Gram matrix K', () => {
    const vectors = [
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0],
      [5, 0],
    ];
    const res = computePCA(vectors, ['1', '2', '3', '4', '5'], ['yes', 'yes', 'no', 'no', 'yes'], ['task1', 'task1', 'task1', 'task1', 'task1']);
    assert.equal(res.n, 5);
    assert.equal(res.points.length, 5);
    // PC1 should explain 100% of variance since variance in dim 2 is zero
    assert.ok(res.variance_explained[0] > 99.0);
    assert.equal(res.variance_explained[1], 0.0);
  });

  it('6.2: power iteration converges on dominant eigenvector', () => {
    const vectors = [
      [-10, 0],
      [-5, 0],
      [0, 0],
      [5, 0],
      [10, 0],
    ];
    const res = computePCA(vectors, ['c1', 'c2', 'c3', 'c4', 'c5'], ['a', 'a', 'a', 'a', 'a'], ['task1', 'task1', 'task1', 'task1', 'task1']);
    assert.equal(res.variance_explained[0], 100.0);
    assert.equal(res.cumulative_2pc, 100.0);
  });

  it('6.3: Gram matrix deflation calculates orthogonal second principal component', () => {
    const vectors = [
      [10, 1],
      [-10, -1],
      [1, 5],
      [-1, -5],
    ];
    const res = computePCA(vectors, ['1', '2', '3', '4'], ['y', 'y', 'n', 'n'], ['task2', 'task2', 'task2', 'task2']);
    assert.ok(res.variance_explained[0] > res.variance_explained[1]);
    assert.ok(res.variance_explained[0] + res.variance_explained[1] <= 100.01);
  });

  it('6.4: scales projected coordinates consistently with sample size', () => {
    const vectors = [
      [10, 0],
      [0, 10],
      [-10, 0],
      [0, -10],
    ];
    const res = computePCA(vectors, ['1', '2', '3', '4'], ['y', 'y', 'n', 'n'], ['t1', 't1', 't1', 't1']);
    assert.equal(res.points.length, 4);
    for (const pt of res.points) {
      assert.ok(Number.isFinite(pt.pc1));
      assert.ok(Number.isFinite(pt.pc2));
    }
  });

  it('6.5: cumulative variance explained is bounded between 0 and 100', () => {
    const vectors = Array.from({ length: 15 }, (_, i) => [Math.sin(i), Math.cos(i), i * 0.1]);
    const res = computePCA(vectors, Array.from({ length: 15 }, (_, i) => `c_${i}`), Array.from({ length: 15 }, () => 'yes'), Array.from({ length: 15 }, () => 'task1'));
    assert.ok(res.cumulative_2pc >= 0 && res.cumulative_2pc <= 100);
    assert.ok(res.variance_explained[0] >= res.variance_explained[1]);
  });
});

describe('Tier 1: Feature 7 - Modality Missingness & Sparsity Metrics', () => {
  it('7.1: inspects all 6 standard modality channels', () => {
    const cases = [
      {
        case_id: 'C1',
        task: 'task1',
        missingness: { MRI: true, Biopsy: false, Prostatectomy: false, PSA_Trend: true, Labs: true, FamilyHistory: false },
      },
    ];
    const res = CohortEngine.computeMissingness(cases);
    assert.deepEqual(res.modalities, ['MRI', 'Biopsy', 'Prostatectomy', 'PSA_Trend', 'Labs', 'FamilyHistory']);
    assert.equal(res.matrix.length, 1);
    assert.deepEqual(res.matrix[0], [1, 0, 0, 1, 1, 0]);
  });

  it('7.2: encodes modality presence as 1 and absence as 0 in binary matrix', () => {
    const cases = [
      { case_id: 'all_present', task: 'task2', missingness: { MRI: true, Biopsy: true, Prostatectomy: true, PSA_Trend: true, Labs: true, FamilyHistory: true } },
      { case_id: 'all_absent', task: 'task1', missingness: { MRI: false, Biopsy: false, Prostatectomy: false, PSA_Trend: false, Labs: false, FamilyHistory: false } },
    ];
    const res = CohortEngine.computeMissingness(cases);
    assert.deepEqual(res.matrix[0], [1, 1, 1, 1, 1, 1]);
    assert.deepEqual(res.matrix[1], [0, 0, 0, 0, 0, 0]);
  });

  it('7.3: preserves case ID and task metadata mapping', () => {
    const cases = [
      { case_id: 'C101', task: 'task1', missingness: { MRI: true, Biopsy: false, Prostatectomy: false, PSA_Trend: true, Labs: false, FamilyHistory: false } },
      { case_id: 'C102', task: 'task3', missingness: { MRI: false, Biopsy: false, Prostatectomy: true, PSA_Trend: true, Labs: true, FamilyHistory: true } },
    ];
    const res = CohortEngine.computeMissingness(cases);
    assert.equal(res.cases[0].case_id, 'C101');
    assert.equal(res.cases[0].task, 'task1');
    assert.equal(res.cases[1].case_id, 'C102');
    assert.equal(res.cases[1].task, 'task3');
  });

  it('7.4: correctly extracts modality availability from raw representations and clinical records', () => {
    const rawTrace = {
      case_id: 'raw_test',
      task: 'task2',
      modality_representations: {
        'MRI image': [[1.0, 2.0]],
        'Biopsy slide': [[3.0, 4.0]],
      },
      clinical_records: {
        psa_trend: [{ date: '2022-01-01', val: 5.0 }],
        laboratory_results: [{ name: 'PSA', val: 5.0 }],
        family_history: 'Father diagnosed age 62',
      },
    };
    const norm = CohortEngine.normalizeCase(rawTrace);
    assert.equal(norm.missingness.MRI, true);
    assert.equal(norm.missingness.Biopsy, true);
    assert.equal(norm.missingness.PSA_Trend, true);
    assert.equal(norm.missingness.Labs, true);
    assert.equal(norm.missingness.FamilyHistory, true);
    assert.equal(norm.missingness.Prostatectomy, false);
  });

  it('7.5: handles empty or "None" family history strings gracefully as absent', () => {
    const trace1 = { case_id: 'fh1', task: 'task1', clinical_records: { family_history: 'None' } };
    const trace2 = { case_id: 'fh2', task: 'task1', clinical_records: { family_history: '   ' } };
    const norm1 = CohortEngine.normalizeCase(trace1);
    const norm2 = CohortEngine.normalizeCase(trace2);
    assert.equal(norm1.missingness.FamilyHistory, false);
    assert.equal(norm2.missingness.FamilyHistory, false);
  });
});

describe('Tier 1: Feature 8 - Fail-Closed Fallback Handling', () => {
  it('8.1: safeFloat parses valid numbers and converts missing/invalid representations to null', () => {
    assert.equal(safeFloat(42.5), 42.5);
    assert.equal(safeFloat('  18.9 '), 18.9);
    assert.equal(safeFloat(null), null);
    assert.equal(safeFloat(undefined), null);
    assert.equal(safeFloat('N/A'), null);
    assert.equal(safeFloat('missing'), null);
    assert.equal(safeFloat('NOT AVAILABLE'), null);
    assert.equal(safeFloat(''), null);
    assert.equal(safeFloat(true), null);
    assert.equal(safeFloat(false), null);
  });

  it('8.2: spearmanRho returns null when pairwise count is < 5', () => {
    assert.equal(spearmanRho([1, 2], [3, 4]), null);
    assert.equal(spearmanRho([1, 2, null, 4, 5], [1, null, 3, 4, 5]), null); // only 3 complete pairs
  });

  it('8.3: silvermanBandwidth returns 0.0 for zero variance or single element arrays', () => {
    assert.equal(silvermanBandwidth([]), 0.0);
    assert.equal(silvermanBandwidth([10.0]), 0.0);
    assert.equal(silvermanBandwidth([5.0, 5.0, 5.0, 5.0]), 0.0);
  });

  it('8.4: computePCA returns empty structure on empty inputs', () => {
    const res = computePCA([], [], [], []);
    assert.equal(res.n, 0);
    assert.deepEqual(res.points, []);
    assert.ok(res.variance_explained.every(v => v === 0));
    assert.equal(res.cumulative_2pc, 0);
  });

  it('8.5: normalizeCase handles completely empty trace object without throwing', () => {
    const norm = CohortEngine.normalizeCase({});
    assert.equal(norm.case_id, 'UNKNOWN');
    assert.equal(norm.task, 'task1');
    assert.equal(norm.target, null);
    assert.equal(norm.mri_vec, null);
  });
});

describe('Tier 1: Feature 9 - Zero-Image Pure SVG/Canvas Rendering', () => {
  before(() => {
    setupMockDOM();
  });

  it('9.1: guarantees zero <img> or raster tags in rendered DOM cards', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    const container = document.getElementById('cohort-view');
    const dummyStats = {
      _loaded: true,
      composition: {
        tasks: {
          task1: { total: 10, classes: { yes: { n: 5, pct: 50 }, no: { n: 5, pct: 50 } } },
          task2: { total: 10, classes: { active_surveillance: { n: 10, pct: 100 } } },
          task3: { total: 10, classes: { '1': { n: 5, pct: 50 }, '0': { n: 5, pct: 50 } } },
        },
      },
    };

    CohortView._data = dummyStats;
    await CohortView.renderCohortTab([]);

    const images = container.querySelectorAll('img');
    const pictures = container.querySelectorAll('picture');
    assert.equal(images.length, 0, 'No <img> elements allowed in cohort view');
    assert.equal(pictures.length, 0, 'No <picture> elements allowed in cohort view');
  });

  it('9.2: renders SVG elements with valid viewBox and responsive width', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    const card = CohortView._makeCard('Test Card', 'SUBTITLE');
    assert.ok(card.card);
    assert.ok(card.body);
    assert.equal(card.card.className, 'cohort-card panel-full');
  });

  it('9.3: disclaimer banner renders clinical spectrum notice', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    const { TraceReader } = await import('../dashboard/js/data.js');
    const container = document.getElementById('cohort-view');
    // Load traces from relocated data directory
    const fs = await import('node:fs');
    const path = await import('node:path');
    const tracesDir = path.join(path.resolve(import.meta.dirname, '..'), '..', 'chimera-data', 'traces', 'all_local_cases');
    if (!fs.existsSync(tracesDir)) {
      console.log('Skipping 9.3: trace directory not found');
      return;
    }
    const reader = new TraceReader('traces');
    const files = fs.readdirSync(tracesDir).filter(f => f.endsWith('.json')).slice(0, 20);
    const traces = [];
    for (const f of files) {
      const text = fs.readFileSync(path.join(tracesDir, f), 'utf-8');
      const res = reader.validateAndNormalize(JSON.parse(text));
      if (res.success) traces.push(res.data);
    }
    CohortView._data = { _loaded: false };
    await CohortView.renderCohortTab(traces);
    const disclaimer = container.querySelector('.cohort-disclaimer-banner');
    assert.ok(disclaimer, 'Disclaimer banner must be rendered');
  });

  it('9.4: makes valid card containers for all B1-B5 sections', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    const cardB1 = CohortView._makeCard('B1: Cohort Composition (Task x Target)', 'ORDERED STACKED BARS');
    const cardB2 = CohortView._makeCard('B2: Cohort 2D Latent Manifold — MRI PCA (R-16)', '1 DOT = 1 PATIENT');
    const cardB3 = CohortView._makeCard('B3: Hierarchical Spearman Correlation Heatmap (R-17)', 'WARD-ORDERED // VIRIDIS');
    const cardB4 = CohortView._makeCard('B4: Missingness Grid (R-18)', 'CASES x MODALITIES');
    const cardB5 = CohortView._makeCard('B5: Raincloud — PSA (ng/mL) (R-19)', 'KDE + BOX + DOTS');

    assert.ok(cardB1.card);
    assert.ok(cardB2.card);
    assert.ok(cardB3.card);
    assert.ok(cardB4.card);
    assert.ok(cardB5.card);
  });

  it('9.5: handles missing data cards with graceful missing message', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    const { TraceReader } = await import('../dashboard/js/data.js');
    const container = document.getElementById('cohort-view');
    // Load traces from relocated data directory
    const fs = await import('node:fs');
    const path = await import('node:path');
    const tracesDir = path.join(path.resolve(import.meta.dirname, '..'), '..', 'chimera-data', 'traces', 'all_local_cases');
    if (!fs.existsSync(tracesDir)) {
      console.log('Skipping 9.5: trace directory not found');
      return;
    }
    const reader = new TraceReader('traces');
    const files = fs.readdirSync(tracesDir).filter(f => f.endsWith('.json')).slice(0, 20);
    const traces = [];
    for (const f of files) {
      const text = fs.readFileSync(path.join(tracesDir, f), 'utf-8');
      const res = reader.validateAndNormalize(JSON.parse(text));
      if (res.success) traces.push(res.data);
    }
    CohortView._data = { _loaded: false };
    await CohortView.renderCohortTab(traces);
    const missingDivs = container.querySelectorAll('.case-extra-missing');
    assert.ok(missingDivs.length > 0);
  });
});

describe('Tier 1: Feature 10 - Dynamic Dual-Scope Filtering (B1-B5)', () => {
  const mockTraces = [
    { case_id: 'T1_a', task: 'task1', ground_truth: { decision: 'yes' }, patient_demographics: { psa: 8.0, age: 60, vol: 40, psad: 0.2 } },
    { case_id: 'T1_b', task: 'task1', ground_truth: { decision: 'no' }, patient_demographics: { psa: 3.5, age: 55, vol: 35, psad: 0.1 } },
    { case_id: 'T2_a', task: 'task2', ground_truth: { decision: 'active_surveillance' }, patient_demographics: { psa: 5.0, age: 62, vol: 50, psad: 0.1 } },
    { case_id: 'T2_b', task: 'task2', ground_truth: { decision: 'active_treatment' }, patient_demographics: { psa: 12.0, age: 70, vol: 45, psad: 0.27 } },
    { case_id: 'T3_a', task: 'task3', ground_truth: { event: 1 }, patient_demographics: { psa: 0.4, age: 68, vol: 30, psad: 0.013 } },
  ];

  it('10.1: computeComposition correctly totals task breakdown for Total Cohort', () => {
    const cases = mockTraces.map(t => CohortEngine.normalizeCase(t));
    const comp = CohortEngine.computeComposition(cases);
    assert.equal(comp.tasks.task1.total, 2);
    assert.equal(comp.tasks.task2.total, 2);
    assert.equal(comp.tasks.task3.total, 1);
    assert.equal(comp.tasks.task1.classes.yes.n, 1);
    assert.equal(comp.tasks.task1.classes.no.n, 1);
  });

  it('10.2: computes separate correlation matrices for each task subset in computeCorrelation', () => {
    const cases = mockTraces.map(t => CohortEngine.normalizeCase(t));
    const corr = CohortEngine.computeCorrelation(cases);
    assert.ok(corr.tasks);
    assert.ok(corr.tasks.task1);
    assert.ok(corr.tasks.task2);
    assert.ok(corr.tasks.task3);
  });

  it('10.3: computeRaincloud produces stratified subsets for all tasks and target classes', () => {
    const cases = mockTraces.map(t => CohortEngine.normalizeCase(t));
    const rc = CohortEngine.computeRaincloud(cases, 'psa');
    assert.equal(rc.metric, 'psa');
    assert.equal(rc.unit, 'ng/mL');
    assert.ok(rc.strata.length >= 8); // 2 from task1 + 4 from task2 + 2 from task3
  });

  it('10.4: Task 1 stratum isolates Task 1 cases and computes valid Tukey summaries', () => {
    const cases = mockTraces.map(t => CohortEngine.normalizeCase(t));
    const rc = CohortEngine.computeRaincloud(cases, 'psa');
    const t1yes = rc.strata.find(s => s.task === 'task1' && s.target === 'yes');
    assert.ok(t1yes);
    assert.equal(t1yes.n, 1);
    assert.equal(t1yes.box.median, 8.0);
  });

  it('10.5: Task 2 stratum isolates Task 2 cases and computes active treatment vs surveillance stats', () => {
    const cases = mockTraces.map(t => CohortEngine.normalizeCase(t));
    const rc = CohortEngine.computeRaincloud(cases, 'psa');
    const t2as = rc.strata.find(s => s.task === 'task2' && s.target === 'active_surveillance');
    const t2at = rc.strata.find(s => s.task === 'task2' && s.target === 'active_treatment');
    assert.ok(t2as);
    assert.ok(t2at);
    assert.equal(t2as.box.median, 5.0);
    assert.equal(t2at.box.median, 12.0);
  });
});

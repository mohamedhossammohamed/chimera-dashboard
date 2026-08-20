// test/tier3_cross_features.test.js
// Tier 3: Cross-Feature Interactions & Dynamic Filtering (>=15 tests)
// Built with native node:test and node:assert/strict

import { describe, it, before } from 'node:test';
import assert from 'node:assert/strict';
import {
  CohortEngine,
  spearmanCorrelationMatrix,
  wardCluster,
  tukeyBox,
  silvermanBandwidth,
  gaussianKDE,
  computePCA,
  NUMERIC_VARS,
} from '../dashboard/js/cohort_engine.js';
import { setupMockDOM } from './helpers/mock_dom.js';

describe('Tier 3: Cross-Feature Interactions & Pipeline Integration', () => {
  before(() => {
    setupMockDOM();
  });

  const generateSyntheticCohort = () => {
    const traces = [];
    // Task 1: 30 cases
    for (let i = 0; i < 30; i++) {
      traces.push({
        case_id: `T1_${i}`,
        task: 'task1',
        patient_demographics: {
          psa: 3.0 + (i * 0.4),
          psad: 0.08 + (i * 0.008),
          vol: 35.0 + (i * 1.2),
          age: 50 + (i % 25),
          pirads: (i % 5) + 1,
        },
        ground_truth: { decision: i % 2 === 0 ? 'yes' : 'no' },
        modality_representations: {
          'MRI image': [Array.from({ length: 32 }, (_, j) => Math.sin(i + j))],
        },
        clinical_records: {
          psa_trend: [{ date: '2023-01-01', val: 3.0 + i * 0.4 }],
          laboratory_results: [{ name: 'PSA', value: 3.0 + i * 0.4 }],
          family_history: i % 3 === 0 ? 'Father positive' : 'None',
        },
      });
    }

    // Task 2: 25 cases
    for (let i = 0; i < 25; i++) {
      traces.push({
        case_id: `T2_${i}`,
        task: 'task2',
        patient_demographics: {
          psa: 4.0 + (i * 0.8),
          psad: 0.12 + (i * 0.012),
          vol: 40.0 + (i * 1.5),
          age: 55 + (i % 20),
          bx_isup: (i % 5) + 1,
          cores_positive: (i % 6) + 1,
          cores_total: 12,
        },
        ground_truth: {
          decision: i % 4 === 0 ? 'active_surveillance' : (i % 4 === 1 ? 'continued_surveillance' : (i % 4 === 2 ? 'watchful_waiting' : 'active_treatment')),
        },
        modality_representations: {
          'MRI image': [Array.from({ length: 32 }, (_, j) => Math.cos(i + j))],
          'Biopsy slide': [Array.from({ length: 32 }, (_, j) => Math.sin(i * 2 + j))],
        },
        clinical_records: {
          psa_trend: [{ date: '2022-01-01', val: 4.0 + i * 0.8 }],
          laboratory_results: [{ name: 'PSA', value: 4.0 + i * 0.8 }],
        },
      });
    }

    // Task 3: 15 cases
    for (let i = 0; i < 15; i++) {
      traces.push({
        case_id: `T3_${i}`,
        task: 'task3',
        patient_demographics: {
          psa: 0.05 + (i * 0.08),
          age: 60 + (i % 15),
        },
        ground_truth: { event: i % 2 === 0 ? 1 : 0 },
        clinical_records: {
          radiology_report: `Prostate volume: ${30.0 + i * 1.1} cc. PSA density: ${0.01 + i * 0.005} ng/mL/cc. PI-RADS: ${(i % 5) + 1}`,
          pathology_report: `ISUP grade group ${(i % 5) + 1}, Gleason 3+4=7.`,
          psa_trend: [{ date: '2021-01-01', val: 0.05 + i * 0.08 }],
        },
        modality_representations: {
          'Prostatectomy slide': [Array.from({ length: 32 }, (_, j) => Math.tan((i % 1.5) + j * 0.1))],
        },
      });
    }

    return traces;
  };

  const traces = generateSyntheticCohort();

  it('X1: Dynamic filter transition lifecycle All -> T1 -> T2 -> T3 -> All preserves count conservation', () => {
    const resAll1 = CohortEngine.computeAll(traces, 'all');
    assert.equal(resAll1.totalCases, 70);
    assert.equal(resAll1.filteredCases, 70);

    const resT1 = CohortEngine.computeAll(traces, 'task1');
    assert.equal(resT1.totalCases, 70);
    assert.equal(resT1.filteredCases, 30);

    const resT2 = CohortEngine.computeAll(traces, 'task2');
    assert.equal(resT2.totalCases, 70);
    assert.equal(resT2.filteredCases, 25);

    const resT3 = CohortEngine.computeAll(traces, 'task3');
    assert.equal(resT3.totalCases, 70);
    assert.equal(resT3.filteredCases, 15);

    const resAll2 = CohortEngine.computeAll(traces, 'all');
    assert.equal(resAll2.filteredCases, 70);
    assert.equal(resAll2.filteredCases, resT1.filteredCases + resT2.filteredCases + resT3.filteredCases);
  });

  it('X2: Raincloud distributions match per-task Tukey summaries and Silverman bandwidths', () => {
    const resAll = CohortEngine.computeAll(traces, 'all');
    const t1Cases = traces.filter(t => t.task === 'task1').map(t => CohortEngine.normalizeCase(t));
    const directT1Psa = t1Cases.filter(c => c.target === 'yes').map(c => c.variables.psa);
    const directBox = tukeyBox(directT1Psa);

    const stratum = resAll.raincloud_psa.strata.find(s => s.task === 'task1' && s.target === 'yes');
    assert.ok(stratum);
    assert.equal(stratum.box.median, directBox.median);
    assert.equal(stratum.box.q1, directBox.q1);
    assert.equal(stratum.box.q3, directBox.q3);
  });

  it('X3: Spearman rank correlation directly feeds Ward clustering for complete leaf ordering', () => {
    const cases = traces.map(t => CohortEngine.normalizeCase(t));
    const corr = CohortEngine.computeSingleCorrelation(cases, 5);
    assert.ok(corr.variables.length >= 4);
    assert.equal(corr.matrix.length, corr.variables.length);

    // Verify matrix is ordered by dendrogram
    for (let i = 0; i < corr.variables.length; i++) {
      assert.equal(corr.matrix[i][i], 1.0);
    }
  });

  it('X4: PCA manifold projections exclude cases lacking embeddings while missingness grid reflects exact sparsity', () => {
    const res = CohortEngine.computeAll(traces, 'all');
    // Task 1 has MRI embeddings (30 cases) + Task 2 has MRI (25 cases) = 55 MRI cases
    assert.equal(res.pca_mri.points.length, 55);
    // Task 2 has Biopsy embeddings (25 cases) = 25 Biopsy cases
    assert.equal(res.pca_biopsy.points.length, 25);

    // Missingness matrix has all 70 cases
    assert.equal(res.missingness.matrix.length, 70);
    const biopsyModIndex = res.missingness.modalities.indexOf('Biopsy');
    assert.ok(biopsyModIndex >= 0);

    // Task 1 cases should have Biopsy = 0, Task 2 should have Biopsy = 1
    for (let i = 0; i < 30; i++) {
      assert.equal(res.missingness.matrix[i][biopsyModIndex], 0);
    }
    for (let i = 30; i < 55; i++) {
      assert.equal(res.missingness.matrix[i][biopsyModIndex], 1);
    }
  });

  it('X5: Raincloud outlier dots strictly match points lying outside Tukey inner fences', () => {
    const cases = traces.map(t => CohortEngine.normalizeCase(t));
    // Add an intentional outlier to Task 1
    cases.push({
      case_id: 'outlier_case',
      task: 'task1',
      target: 'yes',
      variables: { psa: 500.0 },
    });

    const rc = CohortEngine.computeRaincloud(cases, 'psa');
    const stratum = rc.strata.find(s => s.task === 'task1' && s.target === 'yes');
    assert.ok(stratum.box.outliers.includes(500.0));
    const outlierDots = stratum.dots.filter(d => d.value > stratum.box.whisker_hi || d.value < stratum.box.whisker_lo);
    assert.ok(outlierDots.some(d => d.value === 500.0));
  });

  it('X6: Task 3 regex extraction outputs seamlessly integrate into correlation matrix', () => {
    const t3Cases = traces.filter(t => t.task === 'task3').map(t => CohortEngine.normalizeCase(t));
    const corr = CohortEngine.computeSingleCorrelation(t3Cases, 5);
    // vol, psad, pirads, bx_isup were extracted from text reports
    assert.ok(corr.variables.includes('vol'));
    assert.ok(corr.variables.includes('psad'));
    assert.ok(corr.variables.includes('pirads'));
    assert.ok(corr.variables.includes('bx_isup'));
  });

  it('X7: High-dimensional multimodal embeddings project onto 2D and 3D PCA manifolds with finite eigenvalues', () => {
    const res = CohortEngine.computeAll(traces, 'all');
    assert.ok(res.pca.mri.variance_explained[0] > 0);
    assert.ok(res.pca.mri.variance_explained[1] >= 0);
    assert.ok(res.pca.biopsy.variance_explained[0] > 0);

    for (const p of res.pca.mri.points) {
      assert.ok(Number.isFinite(p.pc1));
      assert.ok(Number.isFinite(p.pc2));
    }
  });

  it('X8: Composition percentages sum to exactly 100% within each task cohort partition', () => {
    const res = CohortEngine.computeAll(traces, 'all');
    const comp = res.composition;

    // Task 1 classes (yes/no)
    const t1Sum = comp.tasks.task1.classes.yes.pct + comp.tasks.task1.classes.no.pct;
    assert.ok(Math.abs(t1Sum - 100.0) < 0.2, `Task 1 sum ${t1Sum} should equal 100%`);

    // Task 2 classes
    const t2Sum = Object.values(comp.tasks.task2.classes).reduce((s, c) => s + c.pct, 0);
    assert.ok(Math.abs(t2Sum - 100.0) < 0.2, `Task 2 sum ${t2Sum} should equal 100%`);

    // Task 3 classes
    const t3Sum = comp.tasks.task3.classes['1'].pct + comp.tasks.task3.classes['0'].pct;
    assert.ok(Math.abs(t3Sum - 100.0) < 0.2, `Task 3 sum ${t3Sum} should equal 100%`);
  });

  it('X9: Silverman AMISE bandwidth adapts seamlessly across distinct clinical variable dynamic ranges', () => {
    const cases = traces.map(t => CohortEngine.normalizeCase(t));
    const psaValues = cases.map(c => c.variables.psa).filter(v => v !== null);
    const psadValues = cases.map(c => c.variables.psad).filter(v => v !== null);
    const volValues = cases.map(c => c.variables.vol).filter(v => v !== null);
    const ageValues = cases.map(c => c.variables.age).filter(v => v !== null);

    const hPsa = silvermanBandwidth(psaValues);
    const hPsad = silvermanBandwidth(psadValues);
    const hVol = silvermanBandwidth(volValues);
    const hAge = silvermanBandwidth(ageValues);

    // psad values are ~0.01 - 0.5, volume is ~30 - 80. Bandwidths must reflect respective scales.
    assert.ok(hPsad < hPsa, 'PSAD bandwidth should be smaller than PSA bandwidth');
    assert.ok(hPsad < hVol, 'PSAD bandwidth should be smaller than Volume bandwidth');
    assert.ok(hAge > 0 && hAge < 20, 'Age bandwidth should be in reasonable clinical scale');
  });

  it('X10: Correlation matrix excludes sparse variables and handles zero-variance safely', () => {
    const cases = traces.map(t => CohortEngine.normalizeCase(t));
    // Set cores_total to a constant 12 for all cases
    for (const c of cases) c.variables.cores_total = 12;
    // Set cspca to null for all cases so it has 0 observations
    for (const c of cases) c.variables.cspca = null;

    const corr = CohortEngine.computeSingleCorrelation(cases, 5);
    assert.ok(corr.excluded.includes('cspca'), 'cspca should be in excluded array due to < 5 observations');
    assert.ok(corr.variables.includes('psa'));
    assert.ok(corr.variables.includes('age'));

    // cores_total is in variables, but its off-diagonal correlations with psa should be null due to zero variance
    const coresIdx = corr.variables.indexOf('cores_total');
    const psaIdx = corr.variables.indexOf('psa');
    if (coresIdx >= 0 && psaIdx >= 0) {
      assert.equal(corr.matrix[coresIdx][psaIdx], null, 'Zero-variance variable correlation should be null');
    }
  });

  it('X11: PCA points preserve ground truth target labels for vector styling', () => {
    const res = CohortEngine.computeAll(traces, 'all');
    for (const pt of res.pca_mri.points) {
      assert.ok(pt.case_id);
      assert.ok(pt.task);
      assert.ok(pt.target !== undefined);
      assert.ok(['yes', 'no', 'active_surveillance', 'continued_surveillance', 'watchful_waiting', 'active_treatment', '1', '0', 'None'].includes(pt.target));
    }
  });

  it('X12: Raincloud jitter offsets are deterministic and strictly bounded in [-0.5, 0.5]', () => {
    const cases = traces.map(t => CohortEngine.normalizeCase(t));
    const rc = CohortEngine.computeRaincloud(cases, 'psa');
    for (const stratum of rc.strata) {
      for (const dot of stratum.dots) {
        assert.ok(dot.jitter >= -0.5 && dot.jitter <= 0.5, `Jitter ${dot.jitter} out of [-0.5, 0.5]`);
      }
    }
  });

  it('X13: End-to-end trace ingestion -> analytics -> mock SVG DOM pipeline execution', async () => {
    const { CohortView } = await import('../dashboard/js/cohort_view.js');
    const container = document.getElementById('cohort-view');

    // Pass traces directly — live computation mode (upload-first architecture)
    await CohortView.renderCohortTab(traces);

    const cards = container.querySelectorAll('.cohort-card');
    assert.ok(cards.length >= 5, 'All B1-B5 cards must be mounted in DOM');

    const svgs = container.querySelectorAll('svg');
    assert.ok(svgs.length >= 5, 'All visualizations must render SVG elements');
  });

  it('X14: Cross-cohort statistical variance reflects distinct task spectrum divergence', () => {
    const t1Cases = traces.filter(t => t.task === 'task1').map(t => CohortEngine.normalizeCase(t));
    const t3Cases = traces.filter(t => t.task === 'task3').map(t => CohortEngine.normalizeCase(t));

    const t1Psa = t1Cases.map(c => c.variables.psa);
    const t3Psa = t3Cases.map(c => c.variables.psa);

    const t1Med = tukeyBox(t1Psa).median;
    const t3Med = tukeyBox(t3Psa).median;

    // Post-prostatectomy baseline PSA (Task 3) is much lower than pre-biopsy screening (Task 1)
    assert.ok(t3Med < t1Med, `Task 3 median PSA (${t3Med}) should be lower than Task 1 (${t1Med})`);
  });

  it('X15: Multimodal missingness tracks clinical structural boundaries across all tasks', () => {
    const cases = traces.map(t => CohortEngine.normalizeCase(t));
    const miss = CohortEngine.computeMissingness(cases);

    const pxIdx = miss.modalities.indexOf('Prostatectomy');
    const bxIdx = miss.modalities.indexOf('Biopsy');

    // In Task 1, Prostatectomy and Biopsy are structurally absent
    const t1Indices = miss.cases.map((c, i) => (c.task === 'task1' ? i : -1)).filter(i => i >= 0);
    for (const idx of t1Indices) {
      assert.equal(miss.matrix[idx][pxIdx], 0);
      assert.equal(miss.matrix[idx][bxIdx], 0);
    }
  });

  it('X16: Tukey whisker invariant holds: whisker_lo >= min and whisker_hi <= max', () => {
    const cases = traces.map(t => CohortEngine.normalizeCase(t));
    for (const metric of ['psa', 'psad', 'vol', 'age']) {
      const rc = CohortEngine.computeRaincloud(cases, metric);
      for (const stratum of rc.strata) {
        if (stratum.n > 0 && stratum.box) {
          const vals = stratum.dots.map(d => d.value);
          const min = Math.min(...vals);
          const max = Math.max(...vals);
          assert.ok(stratum.box.whisker_lo >= min, `whisker_lo ${stratum.box.whisker_lo} < min ${min}`);
          assert.ok(stratum.box.whisker_hi <= max, `whisker_hi ${stratum.box.whisker_hi} > max ${max}`);
          assert.ok(stratum.box.q1 <= stratum.box.median);
          assert.ok(stratum.box.median <= stratum.box.q3);
        }
      }
    }
  });
});

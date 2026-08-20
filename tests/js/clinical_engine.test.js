// tests/js/clinical_engine.test.js
import test from 'node:test';
import assert from 'node:assert';
import {
  safeFloat,
  DateParser,
  PSAKinetics,
  EAURiskClassifier,
  CAPRAScorer,
  CAPRASScorer,
  ClinicalBundleGenerator
} from '../../docs/js/clinical_engine.js';

test('safeFloat helper parses valid numbers and guards against invalid inputs', () => {
  assert.strictEqual(safeFloat(12.34), 12.34);
  assert.strictEqual(safeFloat('12.34'), 12.34);
  assert.strictEqual(safeFloat(0), 0);
  assert.strictEqual(safeFloat('0'), 0);
  assert.strictEqual(safeFloat(null), null);
  assert.strictEqual(safeFloat(undefined), null);
  assert.strictEqual(safeFloat(false), null);
  assert.strictEqual(safeFloat(true), null);
  assert.strictEqual(safeFloat('N/A'), null);
  assert.strictEqual(safeFloat('NOT AVAILABLE'), null);
  assert.strictEqual(safeFloat('MISSING'), null);
  assert.strictEqual(safeFloat(''), null);
});

test('DateParser parses various date formats and handles invalid strings', () => {
  // YYYY-MM-DD
  const d1 = DateParser.parse('2022-01-15');
  assert.strictEqual(d1.getFullYear(), 2022);
  assert.strictEqual(d1.getMonth(), 0);
  assert.strictEqual(d1.getDate(), 15);

  // Mon YYYY / Month YYYY
  const d2 = DateParser.parse('Jan 2022');
  assert.strictEqual(d2.getFullYear(), 2022);
  assert.strictEqual(d2.getMonth(), 0);

  const d3 = DateParser.parse('15 Jan 2022');
  assert.strictEqual(d3.getFullYear(), 2022);
  assert.strictEqual(d3.getMonth(), 0);
  assert.strictEqual(d3.getDate(), 15);

  // Invalid
  assert.strictEqual(DateParser.parse(null), null);
  assert.strictEqual(DateParser.parse(''), null);
  assert.strictEqual(DateParser.parse('invalid-date'), null);
});

test('PSAKinetics Carter gating rules (N >= 3 and span >= 6 months)', () => {
  // 0 points
  const k0 = new PSAKinetics([]);
  assert.strictEqual(k0.isInsufficient, true);
  assert.strictEqual(k0.calculatePSAV(), null);
  assert.strictEqual(k0.calculatePSADT(), null);

  // 2 points
  const k2 = new PSAKinetics([
    { date: '1 Jan 2020', val: 4.0 },
    { date: '1 Jan 2021', val: 5.0 }
  ]);
  assert.strictEqual(k2.isInsufficient, true);
  assert.strictEqual(k2.calculatePSAV(), null);
  assert.strictEqual(k2.calculatePSADT(), null);

  // 3 points spanning < 6 months
  const kShort = new PSAKinetics([
    { date: '1 Jan 2020', val: 4.0 },
    { date: '1 Feb 2020', val: 4.5 },
    { date: '1 Mar 2020', val: 5.0 }
  ]);
  assert.strictEqual(kShort.isInsufficient, true);
  assert.strictEqual(kShort.calculatePSAV(), null);
  assert.strictEqual(kShort.calculatePSADT(), null);
});

test('PSAKinetics OLS PSAV regression slope', () => {
  const trend = [
    { date: '1 Jan 2020', val: 4.0 },
    { date: '1 Jan 2021', val: 5.0 },
    { date: '1 Jan 2022', val: 6.0 }
  ];
  const k = new PSAKinetics(trend);
  assert.strictEqual(k.isInsufficient, false);
  const psav = k.calculatePSAV();
  assert.ok(Math.abs(psav - 1.0) < 0.05);

  // Flat trend
  const flatTrend = [
    { date: '1 Jan 2020', val: 5.0 },
    { date: '1 Jul 2020', val: 5.0 },
    { date: '1 Jan 2021', val: 5.0 }
  ];
  const kFlat = new PSAKinetics(flatTrend);
  assert.strictEqual(kFlat.calculatePSAV(), 0.0);
});

test('PSAKinetics log-linear PSADT doubling time and non-rising guard', () => {
  // Doubling every 12 months
  const trend = [
    { date: '1 Jan 2020', val: 2.0 },
    { date: '1 Jan 2021', val: 4.0 },
    { date: '1 Jan 2022', val: 8.0 }
  ];
  const k = new PSAKinetics(trend);
  const psadt = k.calculatePSADT();
  assert.ok(psadt !== null);
  assert.ok(Math.abs(psadt - 12.0) < 0.5);

  // Declining PSA (k <= 0)
  const decTrend = [
    { date: '1 Jan 2020', val: 6.0 },
    { date: '1 Jan 2021', val: 5.0 },
    { date: '1 Jan 2022', val: 4.0 }
  ];
  const kDec = new PSAKinetics(decTrend);
  // Fix: declining PSA returns Infinity (stable/not doubling), not null
  assert.strictEqual(kDec.calculatePSADT(), Infinity);
});

test('EAURiskClassifier 5-tier classification logic and precedence', () => {
  // High: PSA > 20 (EAU 2025 — no "Very High" tier)
  const r1 = EAURiskClassifier.classify(25.0, null, null);
  assert.strictEqual(r1.tier, 'High');

  // Locally Advanced: cT3b (cT3-4 takes precedence)
  const r2 = EAURiskClassifier.classify(6.0, 5, 'cT3b');
  assert.strictEqual(r2.tier, 'Locally Advanced');

  // Locally Advanced: cT4
  const r3 = EAURiskClassifier.classify(5.0, 1, 'cT4');
  assert.strictEqual(r3.tier, 'Locally Advanced');

  // Unfavorable Intermediate: PSA 10-20 (not High — EAU 2025)
  const r4 = EAURiskClassifier.classify(12.0, null, null);
  assert.strictEqual(r4.tier, 'Unfavorable Intermediate');

  // High: ISUP 4/5
  const r5 = EAURiskClassifier.classify(6.0, 4, 'cT1c');
  assert.strictEqual(r5.tier, 'High');

  // Locally Advanced: cT3a (not High — EAU 2025)
  const r6 = EAURiskClassifier.classify(6.0, 1, 'cT3a');
  assert.strictEqual(r6.tier, 'Locally Advanced');

  // High: cT2c (EAU 2025 — cT2c is High, not Unfavorable Intermediate)
  const r6b = EAURiskClassifier.classify(6.0, 2, 'cT2c');
  assert.strictEqual(r6b.tier, 'High');

  // Unfavorable Intermediate: ISUP 3
  const r7 = EAURiskClassifier.classify(6.0, 3, 'cT1c');
  assert.strictEqual(r7.tier, 'Unfavorable Intermediate');

  // Unfavorable Intermediate: cT2b
  const r8 = EAURiskClassifier.classify(6.0, 1, 'cT2b');
  assert.strictEqual(r8.tier, 'Unfavorable Intermediate');

  // Favorable Intermediate: PSA < 10, ISUP 2, cT in {cT1, cT2a}
  const r9 = EAURiskClassifier.classify(6.0, 2, 'cT1c');
  assert.strictEqual(r9.tier, 'Favorable Intermediate');

  // Low: PSA < 10, ISUP 1, cT in {cT1, cT2a}
  const r10 = EAURiskClassifier.classify(4.0, 1, 'cT1c');
  assert.strictEqual(r10.tier, 'Low');

  // Missing — now uses canonical sentinel '[DATA NOT RECORDED]'
  const r11 = EAURiskClassifier.classify(4.0, null, null);
  assert.strictEqual(r11.tier, '[DATA NOT RECORDED]');
});

test('CAPRAScorer pre-treatment nomogram scoring (0-10)', () => {
  // Score 0: Age < 50, PSA <= 6, Gleason 3+3, cT1-2, Cores < 34%
  const res0 = CAPRAScorer.calculate(45, 4.0, 3, 3, 'cT1c', 2, 10);
  assert.strictEqual(res0.score, 0);

  // Score 10: Age >= 50 (1), PSA > 30 (4), Gleason 4+3 (3), cT3a (1), Cores >= 34% (1)
  const res10 = CAPRAScorer.calculate(65, 40.0, 4, 3, 'cT3a', 6, 10);
  assert.strictEqual(res10.score, 10);

  // Intermediate score: Age 60 (1), PSA 8 (1), Gleason 3+4 (1), cT1c (0), Cores 20% (0) -> 3
  const res3 = CAPRAScorer.calculate(60, 8.0, 3, 4, 'cT1c', 2, 10);
  assert.strictEqual(res3.score, 3);
});

test('CAPRASScorer post-surgical nomogram scoring (0-12) & report extraction', () => {
  const allPosReport =
    'Radical prostatectomy: Gleason 4+5; positive margins (R1); extraprostatic extension was present; seminal vesicle invasion present; lymph node metastasis present.';
  const res12 = CAPRASScorer.calculate(25.0, allPosReport);
  assert.strictEqual(res12.score, 12);
  assert.strictEqual(res12.imputed.length, 0);

  const allNegReport =
    'Radical prostatectomy: Gleason 3+3; negative margins (R0); extraprostatic extension absent; seminal vesicles clear; lymph nodes negative.';
  const res0 = CAPRASScorer.calculate(4.0, allNegReport);
  assert.strictEqual(res0.score, 0);
  assert.strictEqual(res0.imputed.length, 0);

  // Missing report imputation
  const resImp = CAPRASScorer.calculate(null, null);
  assert.strictEqual(resImp.score, 0);
  assert.ok(resImp.imputed.includes('PSA'));
  assert.ok(resImp.imputed.includes('pGleason'));
  assert.ok(resImp.imputed.includes('Margin'));
});

test('ClinicalBundleGenerator generates markdown bundle without errors', () => {
  const sampleTrace = {
    case_id: 'PT-TEST-001',
    task: 'task1',
    patient_demographics: { age: 65, psa: 7.2, psad: 0.18, vol: 40.0, pirads: 4, ct: 'cT1c', bx_isup: 2 },
    clinical_records: { psa_trend: [{ date: '1 Jan 2020', val: 5.0 }, { date: '1 Jul 2020', val: 6.0 }, { date: '1 Jan 2021', val: 7.2 }] },
    ground_truth: { decision: 'yes' },
    model_prediction: { decision: 'yes', confidence: 'high', free_text: 'Recommend biopsy due to PIRADS 4 and elevated PSAD.' },
    evaluation: { is_correct: true }
  };

  const md = ClinicalBundleGenerator.generateMarkdown(sampleTrace);
  assert.ok(md.includes('# Clinical Case Summary: PT-TEST-001'));
  assert.ok(md.includes('EAU 2026 Risk Category'));
  assert.ok(md.includes('UCSF CAPRA Score'));
  assert.ok(md.includes('PSA Velocity'));
});

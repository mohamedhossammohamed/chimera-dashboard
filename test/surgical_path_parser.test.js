// test/surgical_path_parser.test.js
// Surgical Pathology Parser — negation-aware field extraction verification.
// Built with native node:test and node:assert/strict.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { parseSurgicalPathology } from '../docs/js/surgical_path_parser.js';

describe('Surgical Path Parser - Gleason parsing', () => {
  it('direct format "Gleason 3+4" yields prim=3, sec=4', () => {
    const r = parseSurgicalPathology('Gleason 3+4');
    assert.equal(r.gleason_prim, 3);
    assert.equal(r.gleason_sec, 4);
  });

  it('with "score" keyword "Gleason score 3+4" yields prim=3, sec=4', () => {
    const r = parseSurgicalPathology('Gleason score 3+4');
    assert.equal(r.gleason_prim, 3);
    assert.equal(r.gleason_sec, 4);
  });

  it('with "pattern" keyword "Gleason pattern 3+4" yields prim=3, sec=4', () => {
    const r = parseSurgicalPathology('Gleason pattern 3+4');
    assert.equal(r.gleason_prim, 3);
    assert.equal(r.gleason_sec, 4);
  });

  it('sum-first parenthesized "Gleason score: 7 (3+4)" yields prim=3, sec=4', () => {
    const r = parseSurgicalPathology('Gleason score: 7 (3+4)');
    assert.equal(r.gleason_prim, 3);
    assert.equal(r.gleason_sec, 4);
  });

  it('sum-first with equals "Gleason: 7 = 3+4" yields prim=3, sec=4', () => {
    const r = parseSurgicalPathology('Gleason: 7 = 3+4');
    assert.equal(r.gleason_prim, 3);
    assert.equal(r.gleason_sec, 4);
  });

  it('separate patterns "Gleason pattern 3, pattern 4" yields prim=3, sec=4', () => {
    const r = parseSurgicalPathology('Gleason pattern 3, pattern 4');
    assert.equal(r.gleason_prim, 3);
    assert.equal(r.gleason_sec, 4);
  });

  it('highest-grade selection "Gleason 3+4. Gleason 4+3" yields prim=4, sec=3 (tiebreak by higher primary)', () => {
    const r = parseSurgicalPathology('Gleason 3+4. Gleason 4+3');
    assert.equal(r.gleason_prim, 4);
    assert.equal(r.gleason_sec, 3);
  });

  it('no match "No Gleason reported" yields prim=null, sec=null', () => {
    const r = parseSurgicalPathology('No Gleason reported');
    assert.equal(r.gleason_prim, null);
    assert.equal(r.gleason_sec, null);
  });
});

describe('Surgical Path Parser - Margin classification', () => {
  it('R0 explicit "R0 resection" yields negative', () => {
    const r = parseSurgicalPathology('R0 resection');
    assert.equal(r.margin, 'negative');
  });

  it('R1 "R1 margin" yields positive', () => {
    const r = parseSurgicalPathology('R1 margin');
    assert.equal(r.margin, 'positive');
  });

  it('positive margin "margins positive for tumor" yields positive', () => {
    const r = parseSurgicalPathology('margins positive for tumor');
    assert.equal(r.margin, 'positive');
  });

  it('negative margin "margins negative" yields negative', () => {
    const r = parseSurgicalPathology('margins negative');
    assert.equal(r.margin, 'negative');
  });

  it('mixed margins "Margins negative for tumor, positive at apex" yields positive', () => {
    const r = parseSurgicalPathology('Margins negative for tumor, positive at apex');
    assert.equal(r.margin, 'positive');
  });

  it('negated positive "no positive margins" yields negative', () => {
    const r = parseSurgicalPathology('no positive margins');
    assert.equal(r.margin, 'negative');
  });

  it('no mention "Gleason 3+4, seminal vesicles unremarkable" yields null margin', () => {
    const r = parseSurgicalPathology('Gleason 3+4, seminal vesicles unremarkable');
    assert.equal(r.margin, null);
  });
});

describe('Surgical Path Parser - ECE (extraprostatic extension)', () => {
  it('present "extraprostatic extension present" yields present', () => {
    const r = parseSurgicalPathology('extraprostatic extension present');
    assert.equal(r.ece, 'present');
  });

  it('absent "no extraprostatic extension" yields absent', () => {
    const r = parseSurgicalPathology('no extraprostatic extension');
    assert.equal(r.ece, 'absent');
  });

  it('negated list "no evidence of ECE, SVI" yields ece absent', () => {
    const r = parseSurgicalPathology('no evidence of ECE, SVI');
    assert.equal(r.ece, 'absent');
  });

  it('no mention "nothing about ECE" yields null', () => {
    const r = parseSurgicalPathology('nothing about ECE');
    assert.equal(r.ece, null);
  });
});

describe('Surgical Path Parser - SVI (seminal vesicle invasion)', () => {
  it('present "seminal vesicle invasion" yields present', () => {
    const r = parseSurgicalPathology('seminal vesicle invasion');
    assert.equal(r.svi, 'present');
  });

  it('absent "no SVI" yields absent', () => {
    const r = parseSurgicalPathology('no SVI');
    assert.equal(r.svi, 'absent');
  });

  it('no mention yields null', () => {
    const r = parseSurgicalPathology('prostate adenocarcinoma, Gleason 3+4');
    assert.equal(r.svi, null);
  });
});

describe('Surgical Path Parser - LNI (lymph node involvement)', () => {
  it('present "lymph node positive" yields present', () => {
    const r = parseSurgicalPathology('lymph node positive');
    assert.equal(r.lni, 'present');
  });

  it('absent numeric "0/15 lymph nodes positive" yields absent', () => {
    const r = parseSurgicalPathology('0/15 lymph nodes positive');
    assert.equal(r.lni, 'absent');
  });

  it('pN0 yields absent', () => {
    const r = parseSurgicalPathology('pN0');
    assert.equal(r.lni, 'absent');
  });

  it('pN1 yields present', () => {
    const r = parseSurgicalPathology('pN1');
    assert.equal(r.lni, 'present');
  });

  it('pNx yields null (not assessed)', () => {
    const r = parseSurgicalPathology('pNx');
    assert.equal(r.lni, null);
  });

  it('distant mets "bone metastasis" yields null (excluded from LNI)', () => {
    const r = parseSurgicalPathology('bone metastasis');
    assert.equal(r.lni, null);
  });

  it('not assessable "no lymph nodes removed" yields null', () => {
    const r = parseSurgicalPathology('no lymph nodes removed');
    assert.equal(r.lni, null);
  });

  it('no mention yields null', () => {
    const r = parseSurgicalPathology('prostate adenocarcinoma, Gleason 3+4');
    assert.equal(r.lni, null);
  });
});

describe('Surgical Path Parser - LVI (lymphovascular invasion)', () => {
  it('present "lymphovascular invasion present" yields present', () => {
    const r = parseSurgicalPathology('lymphovascular invasion present');
    assert.equal(r.lvi, 'present');
  });

  it('absent "no lymphovascular invasion" yields absent', () => {
    const r = parseSurgicalPathology('no lymphovascular invasion');
    assert.equal(r.lvi, 'absent');
  });

  it('not noted "LVI: not noted" yields null', () => {
    const r = parseSurgicalPathology('LVI: not noted');
    assert.equal(r.lvi, null);
  });

  it('NOS across comma clause "lymphovascular invasion, not otherwise specified" yields null (bug fix)', () => {
    const r = parseSurgicalPathology('lymphovascular invasion, not otherwise specified');
    assert.equal(r.lvi, null);
  });
});

describe('Surgical Path Parser - NOT_ASSESSED across comma clause', () => {
  it('"ECE present, not noted" yields ece null (bug fix: not-noted suppresses positive)', () => {
    const r = parseSurgicalPathology('ECE present, not noted');
    assert.equal(r.ece, null);
  });
});

describe('Surgical Path Parser - List positive propagation', () => {
  it('"ECE, SVI, LNI present" propagates present to all three fields', () => {
    const r = parseSurgicalPathology('ECE, SVI, LNI present');
    assert.equal(r.ece, 'present');
    assert.equal(r.svi, 'present');
    assert.equal(r.lni, 'present');
  });
});

describe('Surgical Path Parser - List negation with positive term guard', () => {
  it('"no evidence of ECE, SVI, margins positive" negates ECE/SVI but not margin', () => {
    const r = parseSurgicalPathology('no evidence of ECE, SVI, margins positive');
    assert.equal(r.ece, 'absent');
    assert.equal(r.svi, 'absent');
    assert.notEqual(r.margin, 'negative');
    assert.equal(r.margin, 'positive');
  });
});

describe('Surgical Path Parser - Null/empty/non-string input', () => {
  const expectAllNull = (r, label) => {
    assert.equal(r.gleason_prim, null, `${label}: gleason_prim`);
    assert.equal(r.gleason_sec, null, `${label}: gleason_sec`);
    assert.equal(r.margin, null, `${label}: margin`);
    assert.equal(r.ece, null, `${label}: ece`);
    assert.equal(r.svi, null, `${label}: svi`);
    assert.equal(r.lni, null, `${label}: lni`);
    assert.equal(r.lvi, null, `${label}: lvi`);
  };

  it('null input returns all null fields', () => {
    expectAllNull(parseSurgicalPathology(null), 'null');
  });

  it('undefined input returns all null fields', () => {
    expectAllNull(parseSurgicalPathology(undefined), 'undefined');
  });

  it('empty string input returns all null fields', () => {
    expectAllNull(parseSurgicalPathology(''), 'empty string');
  });

  it('non-string (123) input returns all null fields', () => {
    expectAllNull(parseSurgicalPathology(123), 'number');
  });
});

// PARSER TEST SUITE COMPLETE

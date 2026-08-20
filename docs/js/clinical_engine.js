// CHIMERA-Agent In-Browser Clinical Computation Engine (clinical_engine.js)
// Zero-Dependency Pure ES6 Module for Clinical Scoring, Nomograms, Kinetics & Live Bundle Rendering.
// [OFFICIAL: RESEARCHER-APPROVED] EAU 2026, CAPRA, CAPRA-S, PSA Kinetics standard formulas.
// [SUGGESTION: CO-PILOT] Pure client-side JS implementation.

import { parseSurgicalPathology } from './surgical_path_parser.js';
import { EAU_SENTINEL } from './constants.js';

// Re-export so existing consumers that imported EAU_SENTINEL from
// clinical_engine.js continue to resolve without a second import site.
export { EAU_SENTINEL };

export function safeFloat(val) {
  if (val === null || val === undefined || typeof val === 'boolean') return null;
  const s = String(val).trim();
  if (s === '' || s.toUpperCase() === 'N/A' || s.toUpperCase() === 'NOT AVAILABLE' || s.toUpperCase() === 'MISSING') {
    return null;
  }
  const n = parseFloat(s);
  return (Number.isNaN(n) || !Number.isFinite(n)) ? null : n;
}

export class DateParser {
  static parse(dateStr) {
    if (!dateStr || typeof dateStr !== 'string') return null;
    const s = dateStr.trim();

    // 1. YYYY-MM-DD
    let m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
    if (m) {
      return new Date(parseInt(m[1], 10), parseInt(m[2], 10) - 1, parseInt(m[3], 10));
    }

    // 2. YYYY-MM
    m = s.match(/^(\d{4})-(\d{1,2})$/);
    if (m) {
      return new Date(parseInt(m[1], 10), parseInt(m[2], 10) - 1, 1);
    }

    // 3. YYYY
    m = s.match(/^(\d{4})$/);
    if (m) {
      return new Date(parseInt(m[1], 10), 0, 1);
    }

    // 4. DD-MM-YYYY or DD/MM/YYYY
    m = s.match(/^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$/);
    if (m) {
      return new Date(parseInt(m[3], 10), parseInt(m[2], 10) - 1, parseInt(m[1], 10));
    }

    // 5. Month name YYYY (e.g. Jan 2022, January 2022)
    const months = {
      jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5,
      jul: 6, aug: 7, sep: 8, oct: 9, nov: 10, dec: 11
    };
    m = s.match(/^([a-zA-Z]{3,9})\s+(\d{4})$/);
    if (m) {
      const mon = months[m[1].toLowerCase().slice(0, 3)];
      if (mon !== undefined) {
        return new Date(parseInt(m[2], 10), mon, 1);
      }
    }

    const d = new Date(s);
    return Number.isNaN(d.getTime()) ? null : d;
  }
}

export class PSAKinetics {
  static DAYS_PER_YEAR = 365.25;
  static DAYS_PER_MONTH = 30.4375;

  constructor(psaTrend = []) {
    this.points = [];
    if (Array.isArray(psaTrend)) {
      for (const item of psaTrend) {
        if (!item || typeof item !== 'object') continue;
        const d = DateParser.parse(item.date);
        const v = safeFloat(item.val);
        if (d && v !== null && v > 0) {
          this.points.push({ date: d, val: v });
        }
      }
    }
    this.points.sort((a, b) => a.date.getTime() - b.date.getTime());
  }

  get nPoints() {
    return this.points.length;
  }

  get spanMonths() {
    if (this.points.length < 2) return 0.0;
    const ms = this.points[this.points.length - 1].date.getTime() - this.points[0].date.getTime();
    const days = ms / (1000 * 60 * 60 * 24);
    return days / PSAKinetics.DAYS_PER_MONTH;
  }

  get isInsufficient() {
    return this.points.length < 3 || this.spanMonths < 6.0;
  }

  calculatePSAV() {
    if (this.isInsufficient) return null;
    const baseTime = this.points[0].date.getTime();
    const ts = this.points.map(p => (p.date.getTime() - baseTime) / (1000 * 60 * 60 * 24 * PSAKinetics.DAYS_PER_YEAR));
    const vs = this.points.map(p => p.val);
    const n = ts.length;

    const tMean = ts.reduce((a, b) => a + b, 0) / n;
    const vMean = vs.reduce((a, b) => a + b, 0) / n;

    const ssT = ts.reduce((acc, t) => acc + (t - tMean) ** 2, 0);
    if (ssT === 0) return null;

    const ssTV = ts.reduce((acc, t, i) => acc + (t - tMean) * (vs[i] - vMean), 0);
    return Math.round((ssTV / ssT) * 1000) / 1000;
  }

  /**
   * PSA Doubling Time via log-linear regression (ln(PSA) vs time in years).
   *
   * Return contract:
   *   - null      → insufficient data (fewer than 3 PSA values, or span < 6 months,
   *                 or zero time variance). Caller MUST treat as "not computable".
   *   - Infinity  → PSA stable or declining (regression slope k <= 0; no doubling
   *                 occurring). Clinically distinct from "insufficient data".
   *   - number    → valid doubling time in months (rounded to 1 decimal).
   *
   * Consumers MUST distinguish null from Infinity: null means "we don't know",
   * Infinity means "PSA is not rising". Bundling both under null produced false
   * "Stable / Not Doubling" reassurance on cases with too few measurements.
   *
   * @returns {number|null|Infinity}
   */
  calculatePSADT() {
    if (this.isInsufficient) return null;
    const baseTime = this.points[0].date.getTime();
    const ts = this.points.map(p => (p.date.getTime() - baseTime) / (1000 * 60 * 60 * 24 * PSAKinetics.DAYS_PER_YEAR));
    const logVs = this.points.map(p => Math.log(p.val));
    const n = ts.length;

    const tMean = ts.reduce((a, b) => a + b, 0) / n;
    const logMean = logVs.reduce((a, b) => a + b, 0) / n;

    const ssT = ts.reduce((acc, t) => acc + (t - tMean) ** 2, 0);
    if (ssT === 0) return null;

    const ssTLog = ts.reduce((acc, t, i) => acc + (t - tMean) * (logVs[i] - logMean), 0);
    const k = ssTLog / ssT;

    if (k <= 0) return Infinity; // Stable or declining — clinically distinct from insufficient data
    const dtYears = Math.LN2 / k;
    const dtMonths = dtYears * 12.0;
    return Math.round(dtMonths * 10) / 10;
  }

  /**
   * Classify PSADT into clinically actionable trajectory categories.
   * Matches Python generate_bundles.py:276-287 exactly.
   *
   * Thresholds:
   *   < 6 months  → "Aggressive"
   *   < 12 months → "Rapid"
   *   < 24 months → "Moderate"
   *   >= 24       → "Indolent"
   *
   * Null handling (preserves the calculatePSADT null/Infinity contract):
   *   - null + insufficient data (N<3 or span<6mo) → EAU_SENTINEL ("[DATA NOT RECORDED]")
   *   - null + sufficient data (slope k <= 0, PSA not rising) → "Stable/Declining"
   *   - Infinity is never passed by callers; calculatePSADT returns Infinity only
   *     when k <= 0, which the caller maps to null-equivalent "Stable/Declining"
   *     semantics. Pass the raw PSADT value through.
   *
   * @param {number|null|Infinity} psadtVal - PSADT in months from calculatePSADT()
   * @returns {string} 'Aggressive' | 'Rapid' | 'Moderate' | 'Indolent' | 'Stable/Declining' | '[DATA NOT RECORDED]'
   */
  trajectory(psadtVal) {
    if (psadtVal === null || Number.isNaN(psadtVal)) {
      return this.isInsufficient ? EAU_SENTINEL : 'Stable/Declining';
    }
    if (psadtVal === Infinity) {
      return 'Stable/Declining';
    }
    if (psadtVal < 6) return 'Aggressive';
    if (psadtVal < 12) return 'Rapid';
    if (psadtVal < 24) return 'Moderate';
    return 'Indolent';
  }
}

export class EAURiskClassifier {
  static parseCT(ct) {
    if (!ct || typeof ct !== 'string') return null;
    const m = ct.trim().match(/^[cCpP]?T([1-4])([a-cA-C]?)$/i);
    if (!m) return null;
    const major = parseInt(m[1], 10);
    let minor = 0;
    if (m[2]) {
      minor = m[2].toLowerCase().charCodeAt(0) - 'a'.charCodeAt(0) + 1;
    }
    return { major, minor };
  }

  static classify(psa, bxIsup, ct, highRiskPatterns) {
    const psaVal = safeFloat(psa);
    let isup = bxIsup !== null && bxIsup !== undefined ? safeFloat(bxIsup) : null;
    if (isup !== null) isup = Math.floor(isup);
    const ctRank = EAURiskClassifier.parseCT(ct);
    const hasHighRisk = Boolean(highRiskPatterns) &&
      !['', 'none', 'null', 'n/a', 'false'].includes(String(highRiskPatterns).trim().toLowerCase());

    const criteria = [];

    // EAU 2026 Risk Classification (5 tiers — no "Very High" tier exists)
    // Order: Locally advanced → High → Unfavorable Intermediate → Favorable Intermediate → Low

    // 1. Locally Advanced: cT3-4 (takes precedence over all other criteria)
    if (ctRank && ctRank.major >= 3) {
      criteria.push(`Clinical stage ${ct} (cT3-4)`);
      return { tier: 'Locally Advanced', reason: criteria.join('; ') };
    }

    // 2. High: ISUP 4/5 OR PSA > 20 OR cT2c
    if (isup !== null && (isup === 4 || isup === 5)) {
      criteria.push(`ISUP ${isup}`);
      return { tier: 'High', reason: criteria.join('; ') };
    }
    if (psaVal !== null && psaVal > 20) {
      criteria.push(`PSA ${psaVal} > 20`);
      return { tier: 'High', reason: criteria.join('; ') };
    }
    if (ctRank && ctRank.major === 2 && ctRank.minor === 3) {
      criteria.push(`Clinical stage ${ct} (cT2c)`);
      return { tier: 'High', reason: criteria.join('; ') };
    }

    if (isup === null && ctRank === null && psaVal === null) {
      return { tier: EAU_SENTINEL, reason: 'Insufficient data: ISUP grade, clinical T stage, and PSA all missing' };
    }

    // 3. Unfavorable Intermediate: ISUP 3 OR (ISUP 2 AND PSA 10-20) OR cT2b
    if (isup !== null && isup === 3) {
      criteria.push(`ISUP 3`);
      return { tier: 'Unfavorable Intermediate', reason: criteria.join('; ') };
    }
    // 3b. Favourable Intermediate: ISUP 1 AND PSA 10-20 AND cT1-2a AND no high-risk
    if (psaVal !== null && psaVal >= 10 && psaVal <= 20 && isup === 1 &&
        ctRank && (ctRank.major === 1 || (ctRank.major === 2 && ctRank.minor <= 1)) && !hasHighRisk) {
      criteria.push(`PSA ${psaVal} 10-20; ISUP 1; ${ct}; no high-risk patterns`);
      return { tier: 'Favorable Intermediate', reason: criteria.join('; ') };
    }
    if (psaVal !== null && psaVal >= 10 && psaVal <= 20) {
      if (isup !== null && isup === 2) {
        criteria.push(`PSA ${psaVal} 10-20; ISUP 2`);
        return { tier: 'Unfavorable Intermediate', reason: criteria.join('; ') };
      }
      // PSA 10-20 alone with no ISUP info — still intermediate
      if (isup === null) {
        criteria.push(`PSA ${psaVal} 10-20`);
        return { tier: 'Unfavorable Intermediate', reason: criteria.join('; ') };
      }
    }
    if (ctRank && ctRank.major === 2 && ctRank.minor === 2) {
      criteria.push(`Clinical stage ${ct} (cT2b)`);
      return { tier: 'Unfavorable Intermediate', reason: criteria.join('; ') };
    }

    // 4. Favorable Intermediate: ISUP 2 AND PSA < 10 AND cT1-2a AND no high-risk patterns
    if (psaVal !== null && psaVal < 10 && isup === 2 && ctRank && (ctRank.major === 1 || (ctRank.major === 2 && ctRank.minor <= 1)) && !hasHighRisk) {
      criteria.push(`PSA ${psaVal} < 10; ISUP 2; ${ct}; no high-risk patterns`);
      return { tier: 'Favorable Intermediate', reason: criteria.join('; ') };
    }

    // 5. Low: ISUP 1 AND PSA < 10 AND cT1-2a
    if (psaVal !== null && psaVal < 10 && isup === 1 && ctRank && (ctRank.major === 1 || (ctRank.major === 2 && ctRank.minor <= 1))) {
      criteria.push(`PSA ${psaVal} < 10; ISUP 1; ${ct}`);
      return { tier: 'Low', reason: criteria.join('; ') };
    }

    // Partial classification fallback: ISUP 2 with high-risk patterns upgrade
    if (psaVal !== null && psaVal < 10 && isup === 2 && hasHighRisk) {
      criteria.push(`PSA ${psaVal} < 10; ISUP 2; high-risk patterns present — upgraded`);
      return { tier: 'Unfavorable Intermediate', reason: criteria.join('; ') };
    }

    return { tier: EAU_SENTINEL, reason: 'Non-standard risk parameter constellation' };
  }
}

export class CAPRAScorer {
  static calculate(age, psa, bxGlPrim, bxGlSec, ct, coresPos, coresTot) {
    let total = 0;
    const breakdown = [];

    // Age (max 1)
    const ageVal = safeFloat(age);
    if (ageVal !== null && ageVal >= 50) {
      total += 1;
      breakdown.push('Age [+1] (>=50)');
    } else {
      breakdown.push('Age [+0] (<50)');
    }

    // PSA (max 4) — boundaries inclusive per UCSF CAPRA table
    const psaVal = safeFloat(psa);
    let psaPts = 0;
    if (psaVal === null || psaVal <= 6) psaPts = 0;
    else if (psaVal <= 10) psaPts = 1;
    else if (psaVal <= 20) psaPts = 2;
    else if (psaVal <= 30) psaPts = 3;
    else psaPts = 4;
    total += psaPts;
    breakdown.push(`PSA [+${psaPts}]`);

    // Gleason (max 3)
    const gp = safeFloat(bxGlPrim);
    const gs = safeFloat(bxGlSec);
    let glPts = 0;
    if (gp !== null && gs !== null) {
      if (gp >= 4) glPts = 3;
      else if (gs >= 4) glPts = 1;
      else glPts = 0;
      breakdown.push(`Gleason [+${glPts}] (${gp}+${gs})`);
    } else {
      breakdown.push('Gleason [+0] (data missing)');
    }
    total += glPts;

    // T-Stage (max 1)
    const ctRank = EAURiskClassifier.parseCT(ct);
    let ctPts = 0;
    if (ctRank && ctRank.major > 2) {
      ctPts = 1;
      breakdown.push(`T-Stage [+1] (${ct})`);
    } else {
      breakdown.push(`T-Stage [+0] (${ct || 'cT1/cT2'})`);
    }
    total += ctPts;

    // Positive Cores (max 1)
    const cp = safeFloat(coresPos);
    const ctot = safeFloat(coresTot);
    let corePts = 0;
    if (cp !== null && ctot !== null && ctot > 0) {
      const pct = (cp / ctot) * 100;
      if (pct >= 34) corePts = 1;
      breakdown.push(`Cores [+${corePts}] (${pct.toFixed(1)}%)`);
    } else {
      breakdown.push('Cores [+0] (not recorded)');
    }
    total += corePts;

    const score = Math.min(10, total);
    return { score, breakdown: breakdown.join(', ') };
  }
}

// UCSF CAPRA-S (post-surgical) nomogram — Cooperberg et al., Cancer 2011;
// 117:5039-5046, PMCID: PMC3170662, doi:10.1002/cncr.26169.
// Weights (verified against published Table 1):
//   PSA:  0-6 -> 0, 6.01-10 -> 1, 10.01-20 -> 2, >20 -> 3       (max 3)
//   pGS:  2-6 -> 0, 3+4 -> 1, 4+3 -> 2, 8-10 -> 3               (max 3)
//   SM:   negative -> 0, positive -> 2                           (max 2)
//   ECE:  absent -> 0, present -> 1                              (max 1)
//   SVI:  absent -> 0, present -> 2                              (max 2)
//   LNI:  negative -> 0, positive -> 1                           (max 1)
//   Total max = 3+3+2+1+2+1 = 12
export class CAPRASScorer {
  // Surgical pathology extraction is delegated to the canonical
  // parseSurgicalPathology() in surgical_path_parser.js (negation-aware,
  // single source of truth). CAPRASScorer consumes its structured output
  // and applies the verified CAPRA-S scoring weights below.
  static calculate(psa, surgicalReportText) {
    const parsed = parseSurgicalPathology(surgicalReportText);
    const comps = {
      gleasonPrim: parsed.gleason_prim,
      gleasonSec: parsed.gleason_sec,
      margin: parsed.margin,
      ece: parsed.ece,
      svi: parsed.svi,
      lni: parsed.lni,
    };
    const breakdown = [];
    const imputed = [];
    let total = 0;

    // PSA (max 3)
    const psaVal = safeFloat(psa);
    let psaPts = 0;
    if (psaVal === null) {
      psaPts = 0;
      imputed.push('PSA');
      breakdown.push('PSA [+0](imputed)');
    } else if (psaVal <= 6) {
      psaPts = 0;
      breakdown.push('PSA [+0]');
    } else if (psaVal <= 10) {
      psaPts = 1;
      breakdown.push('PSA [+1]');
    } else if (psaVal <= 20) {
      psaPts = 2;
      breakdown.push('PSA [+2]');
    } else {
      psaPts = 3;
      breakdown.push('PSA [+3]');
    }
    total += psaPts;

    // Gleason (max 3)
    const gp = comps.gleasonPrim;
    const gs = comps.gleasonSec;
    let glPts = 0;
    if (gp === null || gs === null) {
      glPts = 0;
      imputed.push('pGleason');
      breakdown.push('pGleason [+0](imputed)');
    } else {
      const gsum = gp + gs;
      if (gsum <= 6) glPts = 0;
      else if (gp === 3 && gs === 4) glPts = 1;
      else if (gp === 4 && gs === 3) glPts = 2;
      else glPts = 3;
      breakdown.push(`pGleason [+${glPts}] (${gp}+${gs})`);
    }
    total += glPts;

    // Margin (max 2)
    let marginPts = 0;
    if (comps.margin === null) {
      marginPts = 0;
      imputed.push('Margin');
      breakdown.push('Margin [+0](imputed)');
    } else if (comps.margin === 'positive') {
      marginPts = 2;
      breakdown.push('Margin [+2]');
    } else {
      marginPts = 0;
      breakdown.push('Margin [+0]');
    }
    total += marginPts;

    // ECE (max 1)
    let ecePts = 0;
    if (comps.ece === null) {
      ecePts = 0;
      imputed.push('ECE');
      breakdown.push('ECE [+0](imputed)');
    } else if (comps.ece === 'present') {
      ecePts = 1;
      breakdown.push('ECE [+1]');
    } else {
      ecePts = 0;
      breakdown.push('ECE [+0]');
    }
    total += ecePts;

    // SVI (max 2)
    let sviPts = 0;
    if (comps.svi === null) {
      sviPts = 0;
      imputed.push('SVI');
      breakdown.push('SVI [+0](imputed)');
    } else if (comps.svi === 'present') {
      sviPts = 2;
      breakdown.push('SVI [+2]');
    } else {
      sviPts = 0;
      breakdown.push('SVI [+0]');
    }
    total += sviPts;

    // LNI (max 1)
    let lniPts = 0;
    if (comps.lni === null) {
      lniPts = 0;
      imputed.push('LNI');
      breakdown.push('LNI [+0](imputed)');
    } else if (comps.lni === 'present') {
      lniPts = 1;
      breakdown.push('LNI [+1]');
    } else {
      lniPts = 0;
      breakdown.push('LNI [+0]');
    }
    total += lniPts;

    const score = Math.min(12, total);
    return { score, breakdown: breakdown.join(', '), imputed };
  }
}

export class ClinicalBundleGenerator {
  static generateMarkdown(trace, cohortStats = null) {
    if (!trace) return '# No Trace Loaded';
    const cid = trace.case_id || 'UNKNOWN';
    const task = (trace.task || 'task1').toLowerCase();
    const d = trace.patient_demographics || {};
    const clin = trace.clinical_records || {};
    const pred = trace.model_prediction || {};
    const gt = trace.ground_truth || {};
    const evalData = trace.evaluation || {};

    const lines = [];

    // Header
    lines.push(`# Clinical Case Summary: ${cid}`);
    lines.push(`**Task Context:** ${task.toUpperCase()} | **Generated:** ${new Date().toISOString().slice(0, 10)} (Live In-Browser Engine)\n`);

    // 1. Demographics & Biomarkers Table
    lines.push(`## 1. Patient Demographics & Key Biomarkers`);
    lines.push(`| Metric | Value | Reference / Normal | Status |`);
    lines.push(`| :--- | :--- | :--- | :--- |`);
    lines.push(`| Age | ${d.age !== undefined ? d.age + ' yrs' : 'missing'} | — | — |`);
    lines.push(`| Serum PSA | ${d.psa !== undefined ? d.psa + ' ng/mL' : 'missing'} | < 4.0 ng/mL | ${d.psa > 10 ? '🔴 Elevated' : (d.psa >= 4 ? '🟡 Borderline' : '🟢 Normal')} |`);
    lines.push(`| PSA Density (PSAD) | ${d.psad !== undefined ? d.psad + ' ng/mL/cc' : 'missing'} | < 0.15 ng/mL/cc | ${d.psad >= 0.15 ? '🔴 High Risk' : '🟢 Low Risk'} |`);
    lines.push(`| Prostate Volume | ${d.vol !== undefined ? d.vol + ' cc' : 'missing'} | 20 – 30 cc | ${d.vol > 30 ? '🟡 Enlarged' : '🟢 Normal'} |`);
    lines.push(`| PI-RADS v2.1 | ${d.pirads !== undefined ? d.pirads : 'missing'} | 1 – 5 scale | ${d.pirads >= 4 ? '🔴 Malignant Suspicion' : (d.pirads === 3 ? '🟡 Equivocal' : '🟢 Benign')} |`);
    lines.push(`| Biopsy ISUP Grade | ${d.bx_isup !== undefined ? d.bx_isup : 'missing'} | Grade Group 1 – 5 | ${d.bx_isup >= 3 ? '🔴 High Grade' : (d.bx_isup ? '🟡 Low/Intermediate' : '—')} |`);
    lines.push(`| Clinical T Stage | ${d.ct || 'missing'} | cT1 – cT4 | — |\n`);

    // 2. Clinical Guideline Risk Scores
    lines.push(`## 2. Standardized Clinical Guideline Scores`);
    const highRisk = d.high_risk_patterns || d.highRiskPatterns || clin.high_risk_patterns || trace.high_risk_patterns;
    const eau = EAURiskClassifier.classify(d.psa, d.bx_isup, d.ct, highRisk);
    lines.push(`* **EAU 2026 Risk Category:** **${eau.tier}** (${eau.reason})`);

    if (task !== 'task3') {
      const capra = CAPRAScorer.calculate(d.age, d.psa, d.bx_gl_prim, d.bx_gl_sec, d.ct, d.cores_positive, d.cores_total);
      lines.push(`* **UCSF CAPRA Score (Pre-Treatment):** **${capra.score} / 10** [Breakdown: ${capra.breakdown}]`);
    } else {
      const surgReport = clin.surgical_pathology_report || clin.pathology_report;
      const capras = CAPRASScorer.calculate(d.psa, surgReport);
      lines.push(`* **UCSF CAPRA-S Score (Post-Surgical):** **${capras.score} / 12** [Breakdown: ${capras.breakdown}]`);
      if (capras.imputed && capras.imputed.length > 0) {
        lines.push(`  * ⚠️ **Imputed components (scored as 0 due to missing data):** ${capras.imputed.join(', ')}. Score may underestimate risk.`);
      }
    }

    // 3. PSA Kinetics
    if (clin.psa_trend && Array.isArray(clin.psa_trend) && clin.psa_trend.length > 0) {
      const kinetics = new PSAKinetics(clin.psa_trend);
      const psav = kinetics.calculatePSAV();
      const psadt = kinetics.calculatePSADT();
      lines.push(`\n## 3. Longitudinal PSA Kinetics`);
      lines.push(`* Total Recorded Readings: ${kinetics.nPoints} over ${kinetics.spanMonths.toFixed(1)} months`);
      lines.push(`* **PSA Velocity (PSAV):** ${psav !== null ? psav + ' ng/mL/year' : 'Insufficient time span or points'}`);
      let psadtLabel;
      if (psadt === null) psadtLabel = 'Insufficient data (fewer than 3 readings or < 6-month span)';
      else if (psadt === Infinity) psadtLabel = 'Stable / Not Doubling (PSA not rising)';
      else psadtLabel = psadt + ' months';
      lines.push(`* **PSA Doubling Time (PSADT):** ${psadtLabel}`);
      const trajectory = kinetics.trajectory(psadt);
      lines.push(`* **Trajectory Classification:** ${trajectory}`);
    }

    // 4. Decision Comparison & Evaluation
    lines.push(`\n## 4. Model Decision vs Ground Truth`);
    lines.push(`* **Ground Truth:** \`${JSON.stringify(gt.decision || gt)}\``);
    lines.push(`* **Model Prediction:** \`${JSON.stringify(pred.decision || 'N/A')}\` (Confidence: ${pred.confidence || 'uncertain'})`);
    lines.push(`* **Agreement Status:** ${evalData.is_correct ? '✅ CONCORDANT' : '❌ DISCORDANT'}`);
    if (pred.free_text) {
      lines.push(`\n### Clinical Reasoning Narrative\n> ${pred.free_text}`);
    }

    return lines.join('\n');
  }
}

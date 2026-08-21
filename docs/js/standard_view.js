// CHIMERA-Agent 1D/2D Standard View Engine (standard_view.js)
// Zero-Dependency, Pure DOM & SVG Scientific Workbench
import { renderClevelandBulletStrip, renderEAUScorecard, renderConcordanceMatrix, isMissingClinicalValue } from './standard_components.js';
import { parseSurgicalPathology } from './surgical_path_parser.js';
import { showFeedback } from './feedback.js';
import { PSAKinetics } from './clinical_engine.js';
import { CohortEngine } from './cohort_engine.js';
import { EAU_SENTINEL } from './constants.js';

// Shared clinical threshold constants — single source of truth for both the
// demographics status badges and the Cleveland-McGill bullet strips. Aligning
// these guarantees the badge tier and the bullet-strip band never disagree.
const PSA_THRESHOLDS = {
  domain: [0, 20], ticks: [0, 4, 10, 20],
  normal_max: 4.0, borderline_max: 10.0,
  pathological_label: 'High', unit: 'ng/mL'
};
const PSAD_THRESHOLDS = {
  domain: [0, 0.6], ticks: [0, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60],
  normal_max: 0.15, borderline_max: 0.20,
  pathological_label: 'High', unit: 'ng/mL/cc'
};
const VOL_THRESHOLDS = {
  domain: [0, 90], ticks: [0, 30, 60, 90],
  normal_max: 30, borderline_max: 60,
  normal_min: 20, // Lower bound of the healthy range; <20 cc is abnormally small (M-7)
  pathological_label: 'Enlarged', unit: 'cc'
};

// Trajectory categorization color map — single source of truth for the
// demographics trajectory badge. Thresholds mirror the Python trajectory()
// contract exactly (Aggressive<6, Rapid<12, Moderate<24, Indolent>=24,
// Stable/Declining for non-rising PSA, [DATA NOT RECORDED] for insufficient
// data). See UIWIRING_DECISION.md (HYBRID) and PROTOCOL.md.
const TRAJECTORY_COLORS = {
  'Aggressive': '#d9534f',        // red — PSADT < 6 months
  'Rapid': '#e67e22',             // orange — PSADT 6–12 months
  'Moderate': '#f0ad4e',          // yellow — PSADT 12–24 months
  'Indolent': '#28a745',          // green — PSADT >= 24 months
  'Stable/Declining': '#6c757d',  // gray — PSA not rising
  [EAU_SENTINEL]: '#6c757d' // gray — insufficient data
};

export class StandardView {
  static render(trace) {
    if (!trace) return;

    this.activeTrace = trace;
    const task = (trace.task || 'task1').toLowerCase();

    // 1. Demographics & Biometrics Panel
    this.renderDemographics('panel-demographics', trace.patient_demographics, trace.case_id, task, trace.clinical_records);

    // 2. Decision Comparator Panel
    this.renderDecision('panel-decision', {
      task: task,
      groundTruth: trace.ground_truth,
      modelPrediction: trace.model_prediction,
      evaluation: trace.evaluation,
      demographics: trace.patient_demographics
    });

    // 3. Variable Importance Weights Panel
    this.renderWeights('panel-weights', trace.model_prediction?.variable_weights);

    // 4. Embedding Statistical Summary Panel
    this.renderEmbeddings('panel-embeddings', trace.modality_representations);

    // 5. Clinical EHR Text Reader Panel
    this.renderClinicalText('panel-clinical-text', trace.clinical_records);

    // 6. Interactive Raw JSON Tree Inspector Panel
    this.renderJSONTree('panel-json-tree', trace);
  }

  // --- 1. Demographics & Biometrics Table ---
  static renderDemographics(containerId, demographics = {}, caseId, task, clinicalRecords = {}) {
    const el = document.getElementById(containerId);
    if (!el) return;

    // Trajectory categorization via PSAKinetics engine. Computes PSADT then
    // classifies the longitudinal PSA trajectory. Falls back to a local
    // classifier matching the Python contract if the engine has not yet
    // exposed the trajectory() method on PSAKinetics (defensive).
    const psaTrend = (clinicalRecords && Array.isArray(clinicalRecords.psa_trend)) ? clinicalRecords.psa_trend : [];
    const trajectoryResult = this.computeTrajectory(psaTrend);

    const rows = [
      { key: 'age', label: 'Patient Age', unit: 'years', ref: '—', val: demographics.age, status: '—', provenance: 'uploaded' },
      { key: 'psa', label: 'Serum PSA', unit: 'ng/mL', ref: '< 4.0 (normal)', val: demographics.psa, status: this.getPSAStatus(demographics.psa), provenance: 'uploaded' },
      { key: 'psad', label: 'PSA Density (PSAD)', unit: 'ng/mL/cc', ref: '< 0.15 (low risk)', val: demographics.psad, status: this.getPSADStatus(demographics.psad), provenance: 'uploaded' },
      { key: 'vol', label: 'Prostate Volume', unit: 'cc', ref: '20 – 30 (normal)', val: demographics.vol, status: this.getVolStatus(demographics.vol), provenance: 'uploaded' },
      { key: 'pirads', label: 'PI-RADS v2.1 Score', unit: '—', ref: '1 – 5 scale', val: demographics.pirads, status: this.getPIRADSStatus(demographics.pirads), provenance: 'uploaded' },
      { key: 'dre', label: 'Digital Rectal Exam (DRE)', unit: '—', ref: 'Normal / Smooth', val: demographics.dre, status: this.getDREStatus(demographics.dre), provenance: 'uploaded' },
      { key: 'ct', label: 'Clinical T Stage', unit: '—', ref: 'cT1 – cT4', val: demographics.ct, status: '—', provenance: 'uploaded' },
      { key: 'bx_isup', label: 'Biopsy ISUP Grade Group', unit: '—', ref: '1 – 5 scale', val: demographics.bx_isup, status: this.getISUPStatus(demographics.bx_isup), provenance: 'uploaded' },
      { key: 'bx_gl_prim', label: 'Gleason Primary Pattern', unit: '—', ref: '3 – 5 pattern', val: demographics.bx_gl_prim, status: '—', provenance: 'uploaded' },
      { key: 'bx_gl_sec', label: 'Gleason Secondary Pattern', unit: '—', ref: '3 – 5 pattern', val: demographics.bx_gl_sec, status: '—', provenance: 'uploaded' },
      { key: 'trajectory', label: 'PSA Trajectory', unit: '—', ref: 'PSADT-based', val: trajectoryResult.label, status: this.getTrajectoryBadge(trajectoryResult.category), provenance: 'calculated', method: 'psa-kinetics-psadt' }
    ];

    const tableRows = rows.map(r => {
      const isMissing = r.val === undefined || r.val === null || r.val === '';
      const isLiteralNA = String(r.val).trim().toUpperCase() === 'N/A';
      
      let valHtml = '';
      if (isMissing) {
        valHtml = '<span class="badge-status badge-missing">MISSING</span>';
      } else if (isLiteralNA) {
        valHtml = '<span style="font-family: var(--font-mono); color: var(--text-muted);">N/A</span>';
      } else {
        const rawJsonVal = typeof r.val === 'string' ? `"${r.val}"` : String(r.val);
        valHtml = `<span class="copyable-cell" data-copy='${this.escapeAttr(rawJsonVal)}'>${this.escapeHTML(String(r.val))}</span>`;
      }

      const provBadge = r.provenance === 'calculated'
        ? `<a href="computations.html#${r.method || ''}" class="provenance-badge calculated" title="Calculated in-browser — click for documentation" target="_blank">CALCULATED</a>`
        : '<span class="provenance-badge uploaded" title="Parsed from uploaded JSON trace">UPLOADED</span>';

      return `
        <tr>
          <td style="font-weight: 600; color: var(--text-main);">${r.label}${provBadge}</td>
          <td class="num-cell">${valHtml}</td>
          <td style="font-family: var(--font-mono); color: var(--text-muted);">${r.unit}</td>
          <td style="color: var(--text-muted); font-size: 11px;">${r.ref}</td>
          <td>${r.status}</td>
        </tr>
      `;
    }).join('');

    el.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th style="width: 28%;">Clinical Indicator</th>
            <th style="width: 22%; text-align: right;">Value</th>
            <th style="width: 14%;">Unit</th>
            <th style="width: 20%;">Reference Range</th>
            <th style="width: 16%;">Diagnostic Status</th>
          </tr>
        </thead>
        <tbody>
          ${tableRows}
        </tbody>
      </table>
    `;

    this.bindCopyableCells(el);

    // --- Additive: Cleveland-McGill bullet strips for key biometrics ---
    const bulletWrap = document.createElement('div');
    bulletWrap.className = 'bullet-strip-group';
    bulletWrap.style.marginTop = '14px';
    bulletWrap.style.display = 'flex';
    bulletWrap.style.flexDirection = 'column';
    bulletWrap.style.gap = '10px';

    const stripDefs = [
      { label: 'Serum PSA (ng/mL)', value: demographics.psa, unit: 'ng/mL',
        thresholds: PSA_THRESHOLDS },
      { label: 'PSA Density (ng/mL/cc)', value: demographics.psad, unit: 'ng/mL/cc',
        thresholds: PSAD_THRESHOLDS },
      { label: 'Prostate Volume (cc)', value: demographics.vol, unit: 'cc',
        thresholds: VOL_THRESHOLDS }
    ];

    stripDefs.forEach(def => {
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.alignItems = 'center';
      row.style.gap = '12px';

      const lbl = document.createElement('div');
      lbl.style.fontFamily = 'var(--font-mono)';
      lbl.style.fontSize = '11px';
      lbl.style.fontWeight = '600';
      lbl.style.color = 'var(--text-main)';
      lbl.style.minWidth = '180px';
      lbl.style.flexShrink = '0';
      lbl.textContent = def.label;

      const stripContainer = document.createElement('div');
      stripContainer.style.flex = '1';
      stripContainer.style.minWidth = '0';

      row.appendChild(lbl);
      row.appendChild(stripContainer);
      bulletWrap.appendChild(row);

      renderClevelandBulletStrip(stripContainer, def.value, def.thresholds);
    });

    el.appendChild(bulletWrap);
  }

  static getPSAStatus(val) {
    if (isMissingClinicalValue(val) || isNaN(Number(val))) return '—';
    const num = Number(val);
    const nm = PSA_THRESHOLDS.normal_max;
    const bm = PSA_THRESHOLDS.borderline_max;
    if (num > bm) return '<span class="badge-status badge-high">HIGH (>' + bm + ')</span>';
    if (num > nm) return '<span class="badge-status badge-elevated">ELEVATED (>' + nm + ')</span>';
    return '<span class="badge-status badge-normal">NORMAL</span>';
  }

  static getPSADStatus(val) {
    if (isMissingClinicalValue(val) || isNaN(Number(val))) return '—';
    const num = Number(val);
    const nm = PSAD_THRESHOLDS.normal_max;
    const bm = PSAD_THRESHOLDS.borderline_max;
    if (num > bm) return '<span class="badge-status badge-high">HIGH (>' + bm + ')</span>';
    if (num > nm) return '<span class="badge-status badge-elevated">ELEVATED (>' + nm + ')</span>';
    return '<span class="badge-status badge-normal">LOW RISK (≤' + nm + ')</span>';
  }

  static getVolStatus(val) {
    if (isMissingClinicalValue(val) || isNaN(Number(val))) return '—';
    const num = Number(val);
    const nm = VOL_THRESHOLDS.normal_max;
    const bm = VOL_THRESHOLDS.borderline_max;
    const nmin = VOL_THRESHOLDS.normal_min;
    if (num > bm) return '<span class="badge-status badge-high">ENLARGED (>' + bm + ')</span>';
    if (num > nm) return '<span class="badge-status badge-elevated">ELEVATED (>' + nm + ')</span>';
    if (typeof nmin === 'number' && num < nmin) return '<span class="badge-status badge-intermediate">ATROPHIC (<' + nmin + ' cc)</span>';
    return '<span class="badge-status badge-normal">NORMAL</span>';
  }

  static getPIRADSStatus(val) {
    if (!val || val === 'N/A') return '—';
    const s = String(val).trim();
    if (s === '4' || s === '5') return '<span class="badge-status badge-high">SUSPICIOUS (v' + s + ')</span>';
    if (s === '3') return '<span class="badge-status badge-intermediate">EQUIVOCAL (v3)</span>';
    if (s === '1' || s === '2') return '<span class="badge-status badge-normal">LOW RISK (v' + s + ')</span>';
    return '—';
  }

  static getDREStatus(val) {
    if (!val) return '—';
    const s = String(val).toLowerCase();
    if (s.includes('abnormal')) return '<span class="badge-status badge-alert">ABNORMAL</span>';
    if (s.includes('normal')) return '<span class="badge-status badge-normal">NORMAL</span>';
    return '—';
  }

  static getISUPStatus(val) {
    if (isMissingClinicalValue(val) || isNaN(Number(val))) return '—';
    const num = Number(val);
    if (num >= 4) return '<span class="badge-status badge-high">HIGH RISK (ISUP ' + num + ')</span>';
    if (num === 3) return '<span class="badge-status badge-elevated">UNFAVORABLE (ISUP 3)</span>';
    if (num === 2) return '<span class="badge-status badge-intermediate">FAVORABLE (ISUP 2)</span>';
    if (num === 1) return '<span class="badge-status badge-normal">LOW RISK (ISUP 1)</span>';
    return '—';
  }

  // --- PSA Trajectory Categorization ---
  // Computes PSADT via PSAKinetics and classifies the longitudinal trajectory.
  // Prefers the engine's trajectory() method when available; falls back to
  // a local classifier that mirrors the Python contract exactly so the UI is
  // functional even if the engine method is absent.
  static computeTrajectory(psaTrend) {
    const EMPTY = { category: EAU_SENTINEL, label: null, psadt: null };
    if (!Array.isArray(psaTrend) || psaTrend.length === 0) return EMPTY;
    let kinetics, psadt;
    try {
      kinetics = new PSAKinetics(psaTrend);
      psadt = kinetics.calculatePSADT();
    } catch (e) {
      return EMPTY;
    }
    // Prefer engine method when present.
    if (kinetics && typeof kinetics.trajectory === 'function') {
      let category;
      try {
        category = kinetics.trajectory(psadt);
      } catch (e) {
        // Defensive fallback if the engine method throws. Pass isInsufficient
        // so the local classifier can distinguish null+insufficient (missing
        // data) from null+sufficient (PSA not rising → Stable/Declining),
        // matching the engine's trajectory() null-handling contract exactly.
        category = this.classifyTrajectoryLocal(psadt, kinetics.isInsufficient);
      }
      return { category, label: this.trajectoryLabel(psadt), psadt };
    }
    // Local fallback — matches Python trajectory() thresholds exactly. This
    // branch is now dead code once PSAKinetics.trajectory() lands (it always
    // takes precedence above), but is retained as a defensive contract mirror.
    // Kept reachable only for engines that lack the trajectory() method.
    const category = this.classifyTrajectoryLocal(psadt, kinetics ? kinetics.isInsufficient : true);
    return { category, label: this.trajectoryLabel(psadt), psadt };
  }

  // Local classifier mirroring the PSAKinetics.trajectory() engine contract:
  //   null + insufficient (N<3 or span<6mo) -> '[DATA NOT RECORDED]'
  //   null + sufficient (slope k <= 0)       -> 'Stable/Declining'
  //   Infinity                               -> 'Stable/Declining'
  //   < 6 months                             -> 'Aggressive'
  //   6 – < 12 months                        -> 'Rapid'
  //   12 – < 24 months                       -> 'Moderate'
  //   >= 24 months                           -> 'Indolent'
  // The isInsufficient flag (defaulting to true) lets the null branch
  // distinguish missing-data from not-rising, matching the engine exactly.
  static classifyTrajectoryLocal(psadt, isInsufficient = true) {
    if (psadt === null || psadt === undefined || (typeof psadt === 'number' && isNaN(psadt))) {
      return isInsufficient ? EAU_SENTINEL : 'Stable/Declining';
    }
    if (psadt === Infinity) return 'Stable/Declining';
    if (psadt < 6) return 'Aggressive';
    if (psadt < 12) return 'Rapid';
    if (psadt < 24) return 'Moderate';
    return 'Indolent';
  }

  // Human-readable label for the trajectory row value cell (shows PSADT when
  // computable, otherwise a sentinel). Distinct from the colored status badge.
  static trajectoryLabel(psadt) {
    if (psadt === null || psadt === undefined || (typeof psadt === 'number' && isNaN(psadt))) {
      return EAU_SENTINEL;
    }
    if (psadt === Infinity) return 'Stable / Declining';
    return `${Number(psadt).toFixed(1)} mo`;
  }

  // Colored badge for the trajectory status cell. Uses inline styling with
  // TRAJECTORY_COLORS to guarantee exact color parity regardless of CSS theme.
  static getTrajectoryBadge(category) {
    const color = TRAJECTORY_COLORS[category] || TRAJECTORY_COLORS[EAU_SENTINEL];
    return `<span class="badge-status" style="background: ${color}; color: #fff; border: 1px solid ${color};">${this.escapeHTML(category)}</span>`;
  }

  // --- 2. Diagnostic Decision Comparator ---
  static renderDecision(containerId, { task, groundTruth, modelPrediction, evaluation, demographics }) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const gt = groundTruth || {};
    const pred = modelPrediction || {};
    const demo = demographics || {};

    let comparisonHtml = '';

    if (task === 'task3') {
      const gtMonths = gt.months_to_recurrence !== undefined ? `${gt.months_to_recurrence} m` : '<span class="badge-status badge-missing">MISSING</span>';
      const predMonths = pred.months_to_recurrence !== undefined ? `${pred.months_to_recurrence} m` : '<span class="badge-status badge-missing">MISSING</span>';
      
      let deltaHtml = '—';
      if (gt.months_to_recurrence !== undefined && pred.months_to_recurrence !== undefined) {
        const delta = Math.abs(gt.months_to_recurrence - pred.months_to_recurrence).toFixed(1);
        const deltaClass = delta < 6 ? 'badge-normal' : (delta < 12 ? 'badge-intermediate' : 'badge-high');
        deltaHtml = `<span class="badge-status ${deltaClass}">Δ = ${delta} m</span>`;
      }

      const gtEvent = gt.event === 1 ? 'Observed BCR (+)' : (gt.event === 0 ? 'Right-Censored (0)' : 'MISSING');
      const predRisk = pred.event_risk !== undefined ? `${(pred.event_risk * 100).toFixed(1)}% Recurrence Risk` : 'MISSING';

      comparisonHtml = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Evaluation Milestone</th>
              <th>Ground Truth Cohort Target</th>
              <th>Model Predictive Inference</th>
              <th>Concordance / Deviation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="font-weight: 600;">Time to Recurrence</td>
              <td class="num-cell" style="text-align: left;">${gtMonths}</td>
              <td class="num-cell" style="text-align: left; color: var(--color-blue-fg);">${predMonths}</td>
              <td>${deltaHtml}</td>
            </tr>
            <tr>
              <td style="font-weight: 600;">Event Censoring Status</td>
              <td><span class="badge-status ${gt.event === 1 ? 'badge-high' : 'badge-normal'}">${gtEvent}</span></td>
              <td><span style="font-family: var(--font-mono); font-weight: 600;">${predRisk}</span></td>
              <td><span class="badge-status badge-normal">${(typeof pred.confidence === 'number' && pred.confidence > 0) ? pred.confidence : (typeof pred.confidence === 'string' && pred.confidence.trim().length > 0 ? this.escapeHTML(pred.confidence) : 'UNVERIFIED')}</span></td>
            </tr>
          </tbody>
        </table>
      `;
    } else {
      const gtDec = gt.decision !== undefined && gt.decision !== null ? String(gt.decision) : null;
      const predDec = pred.decision !== undefined && pred.decision !== null ? String(pred.decision) : null;

      const isCorrect = evaluation?.is_correct !== null && evaluation?.is_correct !== undefined
        ? evaluation.is_correct
        : (gtDec && predDec ? gtDec.toLowerCase().trim() === predDec.toLowerCase().trim() : null);

      const matchBadge = isCorrect === true
        ? '<span class="badge-status badge-concordant">✓ CONCORDANT</span>'
        : (isCorrect === false
          ? '<span class="badge-status badge-discordant">✗ DISCORDANT</span>'
          : '<span class="badge-status badge-missing">UNVERIFIED</span>');

      comparisonHtml = `
        <table class="data-table">
          <thead>
            <tr>
              <th style="width: 30%;">Diagnostic Task</th>
              <th style="width: 25%;">Ground Truth Decision</th>
              <th style="width: 25%;">Model Decision</th>
              <th style="width: 20%;">Concordance Match</th>
            </tr>
          </thead>
          <tbody>
            <tr class="${isCorrect === false ? 'discordant-row' : ''}">
              <td style="font-weight: 600;">${task === 'task1' ? 'Biopsy Indication Decision' : 'Clinical Management Protocol'}</td>
              <td style="font-family: var(--font-mono); font-weight: 700;">${gtDec ? this.escapeHTML(gtDec) : '<span class="badge-status badge-missing">MISSING</span>'}</td>
              <td style="font-family: var(--font-mono); font-weight: 700; color: ${isCorrect === false ? 'var(--color-red-fg)' : 'var(--color-green-fg)'};">${predDec ? this.escapeHTML(predDec) : '<span class="badge-status badge-missing">MISSING</span>'}</td>
              <td>${matchBadge}</td>
            </tr>
          </tbody>
        </table>
      `;
    }

    const freeText = pred.free_text && String(pred.free_text).trim().length > 0 ? pred.free_text : null;
    const rationaleHtml = freeText
      ? `<div class="rationale-quote"><div class="rationale-title">Model Decision Rationale</div>${this.escapeHTML(freeText)}</div>`
      : `<div class="rationale-quote"><div class="rationale-title">Model Decision Rationale</div><div class="rationale-empty">NO RATIONALE PROVIDED</div></div>`;

    el.innerHTML = comparisonHtml + rationaleHtml;

    // --- Additive: EAU Risk Stratification Scorecard ---
    const eauSection = document.createElement('div');
    eauSection.style.marginTop = '16px';
    const eauTitle = document.createElement('div');
    eauTitle.style.fontFamily = 'var(--font-mono)';
    eauTitle.style.fontSize = '11px';
    eauTitle.style.fontWeight = '700';
    eauTitle.style.color = 'var(--text-main)';
    eauTitle.style.marginBottom = '8px';
    eauTitle.style.borderBottom = '1px solid var(--border-subtle)';
    eauTitle.style.paddingBottom = '4px';
    eauTitle.textContent = 'EAU 2026 Risk Stratification Scorecard';
    eauSection.appendChild(eauTitle);

    const eauProvBadge = document.createElement('a');
    eauProvBadge.href = 'computations.html#eau-risk-classification';
    eauProvBadge.className = 'provenance-badge calculated';
    eauProvBadge.target = '_blank';
    eauProvBadge.textContent = 'CALCULATED';
    eauProvBadge.style.marginBottom = '8px';
    eauProvBadge.style.display = 'inline-flex';
    eauSection.appendChild(eauProvBadge);

    const eauContainer = document.createElement('div');
    eauSection.appendChild(eauContainer);
    el.appendChild(eauSection);

    const highRisk = demo.high_risk_patterns || demo.highRiskPatterns || this.activeTrace?.clinical_records?.high_risk_patterns || this.activeTrace?.high_risk_patterns;
    const eauResult = this.computeEAUTier(demo.psa, demo.bx_isup, demo.ct, highRisk);
    renderEAUScorecard(eauContainer, eauResult.tier, eauResult.criteria);

    // --- Additive: CAPRA-S (post-surgical) Scorecard (Task 3 only) ---
    // UCSF CAPRA-S (Cooperberg et al., Cancer 2011)
    if (task === 'task3') {
      const surgPath = this.activeTrace?.clinical_records?.surgical_pathology_report;
      const caprasResult = this.computeCAPRAS(demo.psa, surgPath);

      const caprasSection = document.createElement('div');
      caprasSection.style.marginTop = '12px';

      const caprasTitle = document.createElement('div');
      caprasTitle.style.fontFamily = 'var(--font-mono)';
      caprasTitle.style.fontSize = '11px';
      caprasTitle.style.fontWeight = '700';
      caprasTitle.style.color = 'var(--text-main)';
      caprasTitle.style.marginBottom = '6px';
      caprasTitle.style.borderBottom = '1px solid var(--border-subtle)';
      caprasTitle.style.paddingBottom = '4px';
      caprasTitle.textContent = 'CAPRA-S (Post-Surgical) Scorecard';
      caprasSection.appendChild(caprasTitle);

      const caprasProvBadge = document.createElement('a');
      caprasProvBadge.href = 'computations.html#capra-s';
      caprasProvBadge.className = 'provenance-badge calculated';
      caprasProvBadge.target = '_blank';
      caprasProvBadge.textContent = 'CALCULATED';
      caprasProvBadge.style.marginBottom = '6px';
      caprasProvBadge.style.display = 'inline-flex';
      caprasSection.appendChild(caprasProvBadge);

      const caprasLine = document.createElement('div');
      caprasLine.style.fontFamily = 'var(--font-mono)';
      caprasLine.style.fontSize = '11px';
      caprasLine.style.color = 'var(--text-main)';
      caprasLine.style.lineHeight = '1.6';
      const imputedSuffix = caprasResult.imputed ? ` (imputed: ${caprasResult.imputed})` : '';
      caprasLine.textContent = `CAPRA-S (post-surgical): ${caprasResult.score}/12 (components: ${caprasResult.breakdown})${imputedSuffix}`;
      caprasSection.appendChild(caprasLine);

      el.appendChild(caprasSection);
    }

    // --- Additive: MRI vs Histology Concordance Matrix ---
    const concSection = document.createElement('div');
    concSection.style.marginTop = '16px';
    const concTitle = document.createElement('div');
    concTitle.style.fontFamily = 'var(--font-mono)';
    concTitle.style.fontSize = '11px';
    concTitle.style.fontWeight = '700';
    concTitle.style.color = 'var(--text-main)';
    concTitle.style.marginBottom = '8px';
    concTitle.style.borderBottom = '1px solid var(--border-subtle)';
    concTitle.style.paddingBottom = '4px';
    concTitle.textContent = 'MRI (PI-RADS) vs Histology (ISUP) Concordance Matrix';
    concSection.appendChild(concTitle);

    const concProvBadge = document.createElement('a');
    concProvBadge.href = 'computations.html#concordance-matrix';
    concProvBadge.className = 'provenance-badge calculated';
    concProvBadge.target = '_blank';
    concProvBadge.textContent = 'CALCULATED';
    concProvBadge.style.marginBottom = '8px';
    concProvBadge.style.display = 'inline-flex';
    concSection.appendChild(concProvBadge);

    const concContainer = document.createElement('div');
    concSection.appendChild(concContainer);
    el.appendChild(concSection);

    renderConcordanceMatrix(concContainer, demo.pirads, demo.bx_isup);
  }

  // --- EAU 2025 Risk Tier Computation (JS mirror of Python generator rules) ---
  static computeEAUTier(psa, bxIsup, ct, highRiskPatterns = null) {
    const psaNum = (psa === null || psa === undefined || psa === '' || String(psa).trim().toUpperCase() === 'N/A' || String(psa).trim().toUpperCase() === 'MISSING' || String(psa).trim().toUpperCase() === 'NOT AVAILABLE')
      ? null : Number(psa);
    const isupNum = (bxIsup === null || bxIsup === undefined || bxIsup === '' || String(bxIsup).trim().toUpperCase() === 'N/A' || String(bxIsup).trim().toUpperCase() === 'MISSING' || String(bxIsup).trim().toUpperCase() === 'NOT AVAILABLE')
      ? null : Math.floor(Number(bxIsup));
    const ctStr = (ct === null || ct === undefined || ct === '' || String(ct).trim().toUpperCase() === 'N/A' || String(ct).trim().toUpperCase() === 'MISSING' || String(ct).trim().toUpperCase() === 'NOT AVAILABLE')
      ? null : String(ct).trim();

    const hasHighRisk = Boolean(highRiskPatterns) &&
      !['', 'none', 'null', 'n/a', 'false'].includes(String(highRiskPatterns).trim().toLowerCase());

    const ctRank = ctStr ? this.ctStageRank(ctStr) : 0;
    const hasCT = ctStr !== null && ctRank > 0;
    const hasISUP = isupNum !== null && !isNaN(isupNum);
    const hasPSA = psaNum !== null && !isNaN(psaNum);

    // EAU 2025 Risk Classification (5 tiers — no "Very High" tier exists)
    // Order: Locally advanced → High → Unfavorable Intermediate → Favorable Intermediate → Low

    // 1. Locally Advanced: cT3-4 (takes precedence over all other criteria)
    if (hasCT && ctRank >= 3.0) {
      return { tier: 'Locally Advanced', criteria: `Clinical stage ${ctStr} (cT3-4)` };
    }

    // 2. High: ISUP 4/5 OR PSA > 20 OR cT2c
    if (hasISUP && (isupNum === 4 || isupNum === 5)) {
      return { tier: 'High', criteria: `ISUP ${isupNum}` };
    }
    if (hasPSA && psaNum > 20) {
      return { tier: 'High', criteria: `PSA ${psaNum} > 20` };
    }
    if (ctRank === 2.3) {
      return { tier: 'High', criteria: `Clinical stage ${ctStr} (cT2c)` };
    }

    if (!hasISUP && !hasCT && !hasPSA) {
      return { tier: EAU_SENTINEL, criteria: 'Insufficient data: ISUP grade, clinical T stage, and PSA all missing' };
    }

    // 3. Unfavorable Intermediate: ISUP 3 OR (ISUP 2 AND PSA 10-20) OR cT2b
    if (hasISUP && isupNum === 3) {
      return { tier: 'Unfavorable Intermediate', criteria: `ISUP 3` };
    }
    // 3b. Unfavorable Intermediate: ISUP 1 AND PSA 10-20 AND high-risk patterns
    if (hasPSA && psaNum >= 10 && psaNum <= 20 && hasISUP && isupNum === 1 && hasHighRisk) {
      return { tier: 'Unfavorable Intermediate', criteria: `PSA ${psaNum} 10-20; ISUP 1; high-risk patterns present` };
    }
    // 3c. Favorable Intermediate: ISUP 2 AND PSA 10-20 AND cT1-2a AND no high-risk
    if (hasPSA && psaNum >= 10 && psaNum <= 20 && hasISUP && isupNum === 2 &&
        hasCT && ctRank <= 2.1 && !hasHighRisk) {
      return { tier: 'Favorable Intermediate', criteria: `PSA ${psaNum} 10-20; ISUP 2; ${ctStr}; no high-risk patterns` };
    }
    if (hasPSA && psaNum >= 10 && psaNum <= 20) {
      if (hasISUP && isupNum === 2) {
        return { tier: 'Unfavorable Intermediate', criteria: `PSA ${psaNum} 10-20; ISUP 2` };
      }
      if (!hasISUP) {
        return { tier: 'Intermediate (unclassified)', criteria: `PSA ${psaNum} 10-20; ISUP unclassified` };
      }
    }
    if (ctRank === 2.2) {
      return { tier: 'Unfavorable Intermediate', criteria: `Clinical stage ${ctStr} (cT2b)` };
    }

    // 4. Favorable Intermediate: ISUP 2 AND PSA < 10 AND cT1-2a AND no high-risk
    if (hasPSA && psaNum < 10 && hasISUP && isupNum === 2) {
      if (hasCT && ctRank <= 2.1 && !hasHighRisk) {
        return { tier: 'Favorable Intermediate', criteria: `PSA ${psaNum} < 10; ISUP 2; ${ctStr}; no high-risk patterns` };
      }
    }

    // 5. Low: ISUP 1 AND PSA < 10 AND cT1-2a
    if (hasPSA && psaNum < 10 && hasISUP && isupNum === 1) {
      if (hasCT && ctRank <= 2.1) {
        return { tier: 'Low', criteria: `PSA ${psaNum} < 10; ISUP 1; ${ctStr}` };
      }
    }

    // Partial classification fallback: ISUP 2 with high-risk patterns upgrade
    if (hasPSA && psaNum < 10 && hasISUP && isupNum === 2 && hasHighRisk) {
      return { tier: 'Unfavorable Intermediate', criteria: `PSA ${psaNum} < 10; ISUP 2; high-risk patterns present — upgraded` };
    }

    // 6. Catch-all: PSA 10-20 with valid ISUP + CT but no high-risk — not enough
    // criteria for Favorable/Unfavorable, but NOT missing data either.
    if (hasPSA && psaNum >= 10 && psaNum <= 20 && hasISUP && hasCT && !hasHighRisk) {
      return { tier: 'Intermediate (unclassified)', criteria: `PSA ${psaNum} 10-20; ISUP ${isupNum}; ${ctStr}; insufficient criteria for Favorable/Unfavorable` };
    }

    return { tier: EAU_SENTINEL, criteria: 'Insufficient staging data for tier classification' };
  }

  // --- CAPRA-S (post-surgical) Computation (JS mirror of Python CAPRAS_Score) ---
  // Task 3 only. Published weights: Cooperberg et al., Cancer 2011 (PMCID: PMC3170662).
  // Weights verified against Cooperberg et al. (2011) Cancer 117:5039-5046, Table 1.
  // Extraction delegated to the canonical surgical_path_parser.js to eliminate the
  // three-way divergence that previously caused negation-scope false positives.
  static computeCAPRAS(psa, surgicalPathReport) {
    const parsed = parseSurgicalPathology(surgicalPathReport);
    const comps = {
      gleasonPrim: parsed.gleason_prim,
      gleasonSec: parsed.gleason_sec,
      margin: parsed.margin,
      ece: parsed.ece,
      svi: parsed.svi,
      lni: parsed.lni
    };

    const breakdown = [];
    const imputed = [];
    let total = 0;

    // PSA (max 3): <=6 -> 0, 6.01-10 -> 1, 10.01-20 -> 2, >20 -> 3
    const psaNum = (psa === null || psa === undefined || psa === '' || String(psa).trim().toUpperCase() === 'N/A')
      ? null : Number(psa);
    if (psaNum === null || isNaN(psaNum)) {
      total += 0; imputed.push('PSA'); breakdown.push('PSA [+0](imputed)');
    } else if (psaNum <= 6) {
      total += 0; breakdown.push('PSA [+0]');
    } else if (psaNum <= 10) {
      total += 1; breakdown.push('PSA [+1]');
    } else if (psaNum <= 20) {
      total += 2; breakdown.push('PSA [+2]');
    } else {
      total += 3; breakdown.push('PSA [+3]');
    }

    // Pathologic Gleason (max 3): 2-6 -> 0, 3+4 -> 1, 4+3 -> 2, 8-10 -> 3
    const gp = comps.gleasonPrim, gs = comps.gleasonSec;
    if (gp === null || gs === null) {
      total += 0; imputed.push('pGleason'); breakdown.push('pGleason [+0](imputed)');
    } else {
      const gsum = gp + gs;
      let pts;
      if (gsum <= 6) pts = 0;
      else if (gp === 3 && gs === 4) pts = 1;
      else if (gp === 4 && gs === 3) pts = 2;
      else pts = 3; // gsum >= 8
      total += pts; breakdown.push(`pGleason [+${pts}] (${gp}+${gs})`);
    }

    // Surgical Margin (max 2): negative -> 0, positive -> 2
    if (comps.margin === null) {
      total += 0; imputed.push('Margin'); breakdown.push('Margin [+0](imputed)');
    } else if (comps.margin === 'positive') {
      total += 2; breakdown.push('Margin [+2]');
    } else {
      total += 0; breakdown.push('Margin [+0]');
    }

    // ECE (max 1): absent -> 0, present -> 1
    if (comps.ece === null) {
      total += 0; imputed.push('ECE'); breakdown.push('ECE [+0](imputed)');
    } else if (comps.ece === 'present') {
      total += 1; breakdown.push('ECE [+1]');
    } else {
      total += 0; breakdown.push('ECE [+0]');
    }

    // SVI (max 2): absent -> 0, present -> 2
    if (comps.svi === null) {
      total += 0; imputed.push('SVI'); breakdown.push('SVI [+0](imputed)');
    } else if (comps.svi === 'present') {
      total += 2; breakdown.push('SVI [+2]');
    } else {
      total += 0; breakdown.push('SVI [+0]');
    }

    // LNI (max 1): negative -> 0, positive -> 1
    if (comps.lni === null) {
      total += 0; imputed.push('LNI'); breakdown.push('LNI [+0](imputed)');
    } else if (comps.lni === 'present') {
      total += 1; breakdown.push('LNI [+1]');
    } else {
      total += 0; breakdown.push('LNI [+0]');
    }

    const score = Math.min(total, 12);
    return { score, breakdown: breakdown.join(', '), imputed: imputed.join(', ') };
  }

  static ctStageRank(ctStr) {
    if (!ctStr || typeof ctStr !== 'string') return 0;
    // Aligned with EAURiskClassifier.parseCT (clinical_engine.js): optional
    // c/C/p/P prefix, mandatory T, digit 1-4, optional substage a-c. Anchored
    // to reject embedded garbage. Bare "T3a", "cT3a", and "pT3a" all rank 3.1.
    const m = ctStr.trim().match(/^[cCpP]?T([1-4])([a-cA-C]?)$/i);
    if (!m) return 0;
    const major = parseInt(m[1], 10);
    const minor = m[2] ? m[2].toLowerCase().charCodeAt(0) - 96 : 0; // a->1, b->2, c->3
    if (major >= 4) return 4.0;
    return major + minor / 10;
  }

  // --- 3. Variable Importance Ranking ---
  static renderWeights(containerId, weights = {}) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const entries = Object.entries(weights || {});
    if (entries.length === 0) {
      el.innerHTML = '<div style="color: var(--text-muted); font-style: italic; font-size: 12px;">No variable importance weights recorded in trace payload.</div>';
      return;
    }

    const rankOrder = { decisive: 4, important: 3, noted: 2, not_used: 1 };
    entries.sort((a, b) => (rankOrder[b[1]] || 0) - (rankOrder[a[1]] || 0));

    const rows = entries.map(([variable, weight]) => {
      const safeWeight = this.escapeAttr(String(weight));
      const wClass = `weight-${safeWeight}`;
      return `
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 600; color: var(--text-main);">${this.escapeHTML(variable)}</td>
          <td><span class="weight-badge ${wClass}">${this.escapeHTML(weight)}</span></td>
        </tr>
      `;
    }).join('');

    el.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Clinical Predictor Variable</th>
            <th>Attribution Importance Class</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
    `;
  }

  // --- 4. Embedding Statistical Summary & Raw Vector Sample ---
  static renderEmbeddings(containerId, modalityRepresentations = {}) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const modalities = [
      { key: 'mri', baseTitle: 'MRI Representation', fallbackDim: 1024 },
      { key: 'biopsy', baseTitle: 'Biopsy WSI Representation', fallbackDim: 960 },
      { key: 'prostatectomy', baseTitle: 'Prostatectomy Representation', fallbackDim: 960 }
    ];

    const cardsHtml = modalities.map(m => {
      const rep = modalityRepresentations[m.key];
      const isPresent = rep && rep.shape && rep.shape.length > 0 && rep.shape[0] > 0;

      // Derive dimensionality from rep.shape when available; fall back to the
      // hardcoded model-specific default only when shape is absent (L-3). This
      // keeps the title honest if the embedding architecture changes while the
      // shape column already reports the true dimensionality.
      const dim = (rep && Array.isArray(rep.shape) && rep.shape.length > 0 &&
                   typeof rep.shape[0] === 'number' && rep.shape[0] > 0)
        ? rep.shape[0] : m.fallbackDim;
      const title = `${m.baseTitle} (${dim}-d)`;

      if (!isPresent) {
        return `
          <div style="margin-bottom: 14px; padding: 10px 14px; background: var(--bg-table-zebra); border: 1px solid var(--border-subtle); border-radius: 4px;">
            <div style="font-weight: 600; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center;">
              <span>${this.escapeHTML(title)}</span>
              <span class="badge-status badge-missing">SPARSE / ABSENT</span>
            </div>
          </div>
        `;
      }

      const shapeStr = this.escapeHTML(JSON.stringify(rep.shape));
      const normStr = rep.norm !== undefined ? Number(rep.norm).toFixed(4) : '<span class="badge-status badge-missing">MISSING</span>';
      const meanStr = rep.mean !== undefined ? Number(rep.mean).toFixed(4) : '<span class="badge-status badge-missing">MISSING</span>';
      const stdStr = rep.std !== undefined ? Number(rep.std).toFixed(4) : '<span class="badge-status badge-missing">MISSING</span>';
      const minStr = rep.min !== undefined ? Number(rep.min).toFixed(4) : '<span class="badge-status badge-missing">MISSING</span>';
      const maxStr = rep.max !== undefined ? Number(rep.max).toFixed(4) : '<span class="badge-status badge-missing">MISSING</span>';

      // Raw vector sample inspection
      let sampleValues = [];
      if (Array.isArray(rep.vector_sample)) {
        sampleValues = rep.vector_sample;
      } else if (Array.isArray(rep.vector_full)) {
        sampleValues = rep.vector_full.slice(0, 20);
      }

      const samplePre = sampleValues.length > 0
        ? `[ ${sampleValues.map(v => Number(v).toFixed(6)).join(', ')} ... ]`
        : 'No raw vector sample array present in trace JSON.';

      return `
        <div style="margin-bottom: 16px; border: 1px solid var(--border-subtle); border-radius: 4px; overflow: hidden;">
          <div style="padding: 8px 12px; background: var(--bg-card-header); font-weight: 700; font-family: var(--font-mono); font-size: 11px; color: var(--text-link);">
            ${this.escapeHTML(title)}
          </div>
          <table class="data-table">
            <thead>
              <tr>
                <th>Tensor Shape</th>
                <th style="text-align: right;">L2 Norm</th>
                <th style="text-align: right;">Mean</th>
                <th style="text-align: right;">Std Dev</th>
                <th style="text-align: right;">Min</th>
                <th style="text-align: right;">Max</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="num-cell" style="text-align: left;">${shapeStr}</td>
                <td class="num-cell">${normStr}</td>
                <td class="num-cell">${meanStr}</td>
                <td class="num-cell">${stdStr}</td>
                <td class="num-cell">${minStr}</td>
                <td class="num-cell">${maxStr}</td>
              </tr>
            </tbody>
          </table>
          <details class="vector-details">
            <summary>Raw Vector Sample (first ${sampleValues.length} values)</summary>
            <pre class="vector-pre">${this.escapeHTML(samplePre)}</pre>
          </details>
        </div>
      `;
    }).join('');

    el.innerHTML = cardsHtml;
  }

  // --- 6. Clinical EHR Text Reader (Sanitized + Entity Highlights) ---
  static renderClinicalText(containerId, records = {}) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const sections = [
      { key: 'radiology_report', title: 'Radiology Report (mpMRI)' },
      { key: 'pathology_report', title: 'Biopsy Pathology Report' },
      { key: 'surgical_pathology_report', title: 'Surgical Resection Pathology' },
      { key: 'psa_trend', title: 'Longitudinal PSA Kinetics' },
      { key: 'laboratory_results', title: 'Serum Laboratory Panel' },
      { key: 'previous_notes', title: 'Prior Clinical Notes' },
      { key: 'family_history', title: 'Oncological Family History' }
    ];

    const NULLISH_STRINGS = new Set(['unknown', 'none', 'n/a', 'not available', 'not recorded', 'no data', 'null', 'nil']);
    const accordionsHtml = sections.map(s => {
      const val = records[s.key];
      const hasContent = val !== null && val !== undefined && (typeof val === 'object' ? Object.keys(val).length > 0 : String(val).trim().length > 0);
      const isUnknownString = typeof val === 'string' && NULLISH_STRINGS.has(val.trim().toLowerCase());
      const badgeState = !hasContent ? 'sparse' : (isUnknownString ? 'unknown' : 'recorded');

      let bodyHtml = '';
      if (!hasContent) {
        bodyHtml = '<div style="color: var(--text-muted); font-style: italic;">No records on file for this clinical section.</div>';
      } else if (isUnknownString) {
        bodyHtml = '<div style="color: var(--text-muted); font-style: italic;">Field present but value is &ldquo;' + this.escapeHTML(val) + '&rdquo;&mdash;no meaningful clinical data recorded.</div>';
      } else if (s.key === 'psa_trend' && Array.isArray(val)) {
        bodyHtml = `
          <table class="data-table">
            <thead>
              <tr><th>Measurement Date</th><th style="text-align: right;">Serum PSA Value (ng/mL)</th></tr>
            </thead>
            <tbody>
              ${val.map(p => `<tr><td>${this.escapeHTML(p.date || '—')}</td><td class="num-cell">${this.escapeHTML(String(p.val))}</td></tr>`).join('')}
            </tbody>
          </table>
        `;
      } else if (s.key === 'laboratory_results' && Array.isArray(val)) {
        bodyHtml = `
          <table class="data-table">
            <thead>
              <tr><th>Assay Parameter</th><th style="text-align: right;">Result</th><th>Date</th><th>Flag</th></tr>
            </thead>
            <tbody>
              ${val.map(l => `
                <tr>
                  <td style="font-weight: 600;">${this.escapeHTML(l.name || '—')}</td>
                  <td class="num-cell">${this.escapeHTML(String(l.val || '—'))}</td>
                  <td>${this.escapeHTML(l.date || '—')}</td>
                  <td>${l.flag ? `<span class="badge-status badge-alert">${this.escapeHTML(l.flag)}</span>` : '—'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `;
      } else if (Array.isArray(val)) {
        bodyHtml = val.map(item => `
          <div style="margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--border-subtle);">
            ${item.date ? `<div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-link); font-weight: 700; margin-bottom: 2px;">${this.escapeHTML(item.date)} ${item.author ? `— ${this.escapeHTML(item.author)}` : ''}</div>` : ''}
            <div>${this.highlightEntities(item.text || JSON.stringify(item))}</div>
          </div>
        `).join('');
      } else if (typeof val === 'object') {
        bodyHtml = `<pre class="vector-pre">${this.escapeHTML(JSON.stringify(val, null, 2))}</pre>`;
      } else {
        bodyHtml = `<div>${this.highlightEntities(String(val))}</div>`;
      }

      const badgeClass = badgeState === 'recorded' ? 'badge-normal' : (badgeState === 'unknown' ? 'badge-unknown' : 'badge-missing');
      const badgeText = badgeState === 'recorded' ? 'RECORDED' : (badgeState === 'unknown' ? 'UNKNOWN' : 'SPARSE');

      return `
        <details class="clinical-accordion" ${badgeState === 'recorded' ? 'open' : ''}>
          <summary class="clinical-summary">
            <span>${s.title}</span>
            <span class="badge-status ${badgeClass}">${badgeText}</span>
          </summary>
          <div class="clinical-body">${bodyHtml}</div>
        </details>
      `;
    }).join('');

    el.innerHTML = accordionsHtml;
  }

  // Sanitized Entity Highlighter — M-104: escapes untrusted text FIRST,
  // then applies entity highlighting so no raw HTML can ever reach the DOM.
  static highlightEntities(rawText) {
    if (!rawText) return '';
    const escaped = StandardView.escapeHTML(rawText);
    return escaped
      .replace(/\b(PI-RADS\s*[1-5]?)\b/gi, '<mark class="entity-pirads">$1</mark>')
      .replace(/\b(Gleason\s*\d\s*\+\s*\d|\bGG\s*[1-5]\b)/gi, '<mark class="entity-gleason">$1</mark>')
      .replace(/\b(ISUP\s*(?:grade\s*group\s*)?[1-5]?)\b/gi, '<mark class="entity-isup">$1</mark>')
      .replace(/\b(cribriform\w*|intraductal\w*)\b/gi, '<mark class="entity-cribriform">$1</mark>')
      .replace(/\b(margin\s*positive|positive\s*margins?|\bR1\b|\bR0\b)\b/gi, '<mark class="entity-margin">$1</mark>')
      .replace(/\b(extraprostatic\s*extension|\bEPE\b|seminal\s*vesicle\s*invasion|\bSVI\b)\b/gi, '<mark class="entity-epe">$1</mark>');
  }

  // --- 7. Interactive Raw JSON Tree Inspector ---
  static renderJSONTree(containerId, trace) {
    const el = document.getElementById(containerId);
    if (!el) return;

    el.innerHTML = `
      <div class="json-toolbar">
        <button type="button" class="json-btn" id="btn-expand-json">Expand All</button>
        <button type="button" class="json-btn" id="btn-collapse-json">Collapse All</button>
        <input type="text" class="json-search-input" id="input-json-search" placeholder="Search JSON keys & values..." />
        <button type="button" class="json-btn" id="btn-copy-json">Copy JSON</button>
        <button type="button" class="json-btn" id="btn-download-json">Download JSON</button>
      </div>
      <div class="json-tree-container" id="json-tree-mount">
        ${this.buildDOMTree(trace, 'trace_root')}
      </div>
    `;

    // Event Handlers
    const mount = document.getElementById('json-tree-mount');
    
    document.getElementById('btn-expand-json')?.addEventListener('click', () => {
      mount?.querySelectorAll('details').forEach(d => d.open = true);
    });

    document.getElementById('btn-collapse-json')?.addEventListener('click', () => {
      mount?.querySelectorAll('details').forEach((d, i) => {
        if (i > 0) d.open = false;
      });
    });

    document.getElementById('btn-copy-json')?.addEventListener('click', () => {
      navigator.clipboard.writeText(JSON.stringify(trace, null, 2))
        .then(() => showFeedback('Full trace JSON copied to clipboard.', 'success'))
        .catch(() => showFeedback('Failed to copy trace JSON to clipboard.', 'error'));
    });

    document.getElementById('btn-download-json')?.addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(trace, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${StandardView.sanitizeFilename(trace.case_id || 'trace')}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });

    document.getElementById('input-json-search')?.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      const nodes = mount?.querySelectorAll('.json-node-row');
      if (!nodes) return;

      if (!q) {
        nodes.forEach(n => n.style.display = '');
        return;
      }

      nodes.forEach(n => {
        const text = n.textContent.toLowerCase();
        if (text.includes(q)) {
          n.style.display = '';
          // Ensure parent details are open
          let p = n.parentElement;
          while (p && p !== mount) {
            if (p.tagName === 'DETAILS') p.open = true;
            p = p.parentElement;
          }
        } else {
          n.style.display = 'none';
        }
      });
    });
  }

  static buildDOMTree(obj, keyName = 'root') {
    if (obj === null) return `<div class="json-node-row"><span class="json-key">${this.escapeHTML(keyName)}</span>: <span class="json-null">null</span></div>`;
    if (typeof obj === 'boolean') return `<div class="json-node-row"><span class="json-key">${this.escapeHTML(keyName)}</span>: <span class="json-bool">${obj}</span></div>`;
    if (typeof obj === 'number') return `<div class="json-node-row"><span class="json-key">${this.escapeHTML(keyName)}</span>: <span class="json-number">${obj}</span></div>`;
    if (typeof obj === 'string') {
      const displayStr = obj.length > 200 ? `${obj.substring(0, 200)}...` : obj;
      return `<div class="json-node-row"><span class="json-key">${this.escapeHTML(keyName)}</span>: <span class="json-string">"${this.escapeHTML(displayStr)}"</span></div>`;
    }

    if (Array.isArray(obj)) {
      if (obj.length === 0) return `<div class="json-node-row"><span class="json-key">${this.escapeHTML(keyName)}</span>: [ ]</div>`;
      const children = obj.map((item, idx) => `<div class="json-node">${this.buildDOMTree(item, `[${idx}]`)}</div>`).join('');
      return `
        <details class="json-node-row" open>
          <summary><span class="json-key">${this.escapeHTML(keyName)}</span> <span style="color: var(--text-muted);">[ ${obj.length} items ]</span></summary>
          ${children}
        </details>
      `;
    }

    if (typeof obj === 'object') {
      const keys = Object.keys(obj);
      if (keys.length === 0) return `<div class="json-node-row"><span class="json-key">${this.escapeHTML(keyName)}</span>: { }</div>`;
      const children = keys.map(k => `<div class="json-node">${this.buildDOMTree(obj[k], k)}</div>`).join('');
      return `
        <details class="json-node-row" open>
          <summary><span class="json-key">${this.escapeHTML(keyName)}</span> <span style="color: var(--text-muted);">{ ${keys.length} keys }</span></summary>
          ${children}
        </details>
      `;
    }

    return `<div class="json-node-row"><span class="json-key">${this.escapeHTML(keyName)}</span>: ${this.escapeHTML(String(obj))}</div>`;
  }

  // --- Utilities ---
  static escapeHTML(str) {
    if (str === undefined || str === null) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  static escapeAttr(str) {
    if (str === undefined || str === null) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // M-106 — sanitize download filenames: replace any character outside
  // [a-zA-Z0-9._-] with underscore to prevent path/injection in filenames.
  static sanitizeFilename(name) {
    return String(name || '').replace(/[^a-zA-Z0-9._-]/g, '_');
  }

  static bindCopyableCells(rootEl) {
    rootEl.querySelectorAll('.copyable-cell').forEach(cell => {
      cell.addEventListener('click', () => {
        const val = cell.getAttribute('data-copy');
        if (val) {
          navigator.clipboard.writeText(val);
          const existingToast = cell.querySelector('.copied-toast');
          if (!existingToast) {
            const toast = document.createElement('span');
            toast.className = 'copied-toast';
            toast.textContent = 'COPIED';
            cell.appendChild(toast);
            setTimeout(() => toast.remove(), 1500);
          }
        }
      });
    });
  }
}

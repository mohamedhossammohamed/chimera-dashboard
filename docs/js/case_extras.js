// [OFFICIAL: RESEARCHER-APPROVED] CHIMERA-Agent Phase B Case-Level Extras
// [SUGGESTION: CO-PILOT] CCI keyword mapping and surgical pathology extraction
//
// CaseExtras: ES module that appends case-level interpretability components
// into existing Standard View panels without replacing their content.
//
// Components:
//   R-15: Surgical Pathology Flag Strip (Task 3 only, #panel-survival)
//   R-14: Charlson/CCI Actuarial Line (#panel-demographics)
//   R-11: AI-vs-Human ISUP Discordance Gauge (#panel-decision)
//   R-12: Embedding Signature Panel (#panel-embeddings)
//
// Anti-misleading invariants:
//   - Missing data → "NOT RECORDED" / "MISSING" / "NOT AVAILABLE" (never assumed negative)
//   - No fabricated AI predictions
//   - Every aggregate shows n
//   - Every axis labelled
//   - Zero 3D, zero external libraries, pure vanilla JS + SVG

import { parseSurgicalPathology } from './surgical_path_parser.js';

// Escapes HTML-sensitive characters to prevent stored XSS when interpolating
// trace-controlled data into innerHTML.
function escapeHTML(str) {
  if (str === undefined || str === null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ===========================================================================
// R-14: Charlson Comorbidity Index (CCI) Keyword → Weight Mapping Table
// ===========================================================================
// CONFIDENCE: MEDIUM — keyword matching is approximate; free-text pmhx entries
// may use variant spellings, abbreviations, or non-standard phrasing.
//
// Mapping table (Charlson et al., J Chronic Dis 1987; Quan et al., Med Care 2005):
// ┌──────────────────────────────────────────┬───────┐
// │ Keyword pattern (case-insensitive)       │ Weight│
// ├──────────────────────────────────────────┼───────┤
// │ "chronic kidney disease" / "CKD"         │   2   │
// │ "diabetes" (uncomplicated)               │   1   │
// │ "diabetes" + "end-organ" / "nephropathy" │   2   │
// │ "myocardial infarction" / " MI "         │   1   │
// │ "congestive heart failure" / "CHF"       │   1   │
// │ "COPD" / "chronic obstructive pulmonary" │   1   │
// │ "cerebrovascular disease" / "stroke"     │   1   │
// │ "peripheral vascular disease" / "PVD"    │   1   │
// │ "dementia"                                │   1   │
// │ "liver disease" (mild / uncomplicated)   │   1   │
// │ "liver disease" (moderate-severe / cirrh)│   3   │
// │ "hemiplegia" / "hemiparesis"             │   2   │
// │ "paraplegia" / "paralysis"               │   2   │
// │ "leukemia" / "lymphoma"                  │   2   │
// │ "solid tumor" / "cancer" (metastatic)    │   6   │
// │ "AIDS" / "HIV"                            │   6   │
// │ "ulcer" / "peptic ulcer disease"         │   1   │
// │ "connective tissue disease" / "rheumatoid│   1   │
// └──────────────────────────────────────────┴───────┘
//
// Deduplication: each rule carries a `category` field grouping rules that represent
// the same Charlson comorbidity family. computeCCI keeps only the highest-weight
// match per category across all pmhx entries, preventing cross-entry double-counting
// (e.g. "diabetes" + "diabetic nephropathy" → category 'diabetes' → score 2, not 3).
//
// Age-adjusted CCI (Charlson et al., 1994):
//   age_adjusted = raw_CCI + (age >= 50 ? floor((age - 40) / 10) : 0)
//   i.e. age 50-59 → +1, 60-69 → +2, 70-79 → +3, 80-89 → +4, 90+ → +5

const CCI_KEYWORD_MAP = [
  // Order matters: more specific (higher-weight) patterns checked first.
  // The `category` field groups rules that represent the same Charlson comorbidity
  // family so computeCCI can deduplicate across pmhx entries (keep highest weight
  // per category). This prevents cross-entry double-counting, e.g. "diabetes" +
  // "diabetic nephropathy" both map to category 'diabetes' → only the weight-2
  // end-organ match is retained.
  { patterns: [/metastat/i, /metast/i], weight: 6, label: 'Solid tumor (metastatic)', category: 'metastatic_solid_tumor' },
  { patterns: [/\bAIDS\b/, /\bHIV\b/i], weight: 6, label: 'AIDS/HIV', category: 'aids' },
  { patterns: [/moderate.*severe.*liver|severe.*liver|cirrhos|liver.*transplant/i], weight: 3, label: 'Liver disease (moderate-severe)', category: 'liver' },
  { patterns: [/leukemia|lymphoma/i], weight: 2, label: 'Leukemia/Lymphoma', category: 'malignancy' },
  { patterns: [/cancer|carcinoma|tumor|neoplasm|malignancy/i], weight: 2, label: 'Any prior malignancy (non-metastatic)', category: 'malignancy' },
  { patterns: [/hemipleg|hemipares/i], weight: 2, label: 'Hemiplegia', category: 'paraplegia' },
  { patterns: [/\bparapleg(?:ia|ic)\b/i, /\bparalysis\b/i, /\bhemipleg(?:ia|ic)\b/i], weight: 2, label: 'Paraplegia', category: 'paraplegia' },
  { patterns: [/chronic kidney|CKD/i], weight: 2, label: 'Chronic kidney disease', category: 'renal' },
  { patterns: [/diabetes.*end.?organ|diabetes.*nephropath|diabetes.*retinopath|diabetic.*nephropath/i], weight: 2, label: 'Diabetes (end-organ damage)', category: 'diabetes' },
  { patterns: [/diabetes|diabetic/i], weight: 1, label: 'Diabetes (uncomplicated)', category: 'diabetes' },
  { patterns: [/myocardial infarction|\bMI\b/i], weight: 1, label: 'Myocardial infarction', category: 'mi' },
  { patterns: [/congestive heart failure|\bCHF\b|heart failure/i], weight: 1, label: 'Congestive heart failure', category: 'chf' },
  { patterns: [/COPD|chronic obstructive pulmonary/i], weight: 1, label: 'COPD', category: 'copd' },
  { patterns: [/cerebrovascular|stroke|\bTIA\b|transient ischemic/i], weight: 1, label: 'Cerebrovascular disease', category: 'cerebrovascular' },
  { patterns: [/peripheral vascular|\bPVD\b/i], weight: 1, label: 'Peripheral vascular disease', category: 'pvd' },
  { patterns: [/dementia/i], weight: 1, label: 'Dementia', category: 'dementia' },
  { patterns: [/liver disease|mild.*liver/i], weight: 1, label: 'Liver disease (mild)', category: 'liver' },
  { patterns: [/peptic ulcer|ulcer disease/i], weight: 1, label: 'Peptic ulcer disease', category: 'ulcer' },
  { patterns: [/connective tissue|rheumatoid|lupus|SLE/i], weight: 1, label: 'Connective tissue disease', category: 'connective' },
];

function computeCCI(pmhx, age) {
  if (!Array.isArray(pmhx) || pmhx.length === 0) {
    return { raw: null, ageAdjusted: null, components: [], missing: true };
  }

  const rawComponents = [];

  for (let i = 0; i < pmhx.length; i++) {
    const entry = String(pmhx[i] || '').trim();
    if (!entry) continue;
    for (const rule of CCI_KEYWORD_MAP) {
      if (rule.patterns.some(p => p.test(entry))) {
        // First match wins (CCI_KEYWORD_MAP is ordered by weight descending, so the
        // first matching rule is the highest-weight rule for this entry).
        rawComponents.push({ label: rule.label, weight: rule.weight, category: rule.category, source: entry });
        break;
      }
    }
    // If no CCI condition matched, the entry is a non-CCI comorbidity (e.g. hypercholesterolaemia)
    // — it contributes 0 to CCI but is not an error.
  }

  // Category-based deduplication: keep only the highest-weight match per clinical
  // category. This prevents cross-entry double-counting when the same disease family
  // appears as separate pmhx entries with varying specificity — e.g. ["diabetes",
  // "diabetic nephropathy"] both map to category 'diabetes'; only the weight-2
  // end-organ match is retained (correct Charlson = 2, not 1+2=3).
  const bestByCategory = new Map();
  for (const comp of rawComponents) {
    const existing = bestByCategory.get(comp.category);
    if (!existing || comp.weight > existing.weight) {
      bestByCategory.set(comp.category, comp);
    }
  }
  const components = Array.from(bestByCategory.values());
  const raw = components.reduce((sum, c) => sum + c.weight, 0);

  // Age adjustment: age >= 50 → +floor((age - 40) / 10)
  let ageAdj = raw;
  const ageNum = (age === null || age === undefined || age === '' || String(age).trim().toUpperCase() === 'N/A')
    ? null : Number(age);
  if (ageNum !== null && !isNaN(ageNum) && ageNum >= 50) {
    ageAdj = raw + Math.floor((ageNum - 40) / 10);
  }

  return { raw, ageAdjusted: ageAdj, components, missing: false };
}

// ===========================================================================
// R-15: Surgical Pathology Regex Extraction
// ===========================================================================
// Same extraction patterns as CAPRA-S (standard_view.js computeCAPRAS):
//   Margin: positive|r1|positive margins|margins were positive → PRESENT
//           negative|r0|negative margins|margins were negative → ABSENT
//   EPE:    extraprostatic extension|extracapsular extension|ECE
//           present|positive|identified|noted → PRESENT
//           absent|negative|not identified|no extraprostatic → ABSENT
//   SVI:    seminal vesicle
//           invasion|invaded|positive|present → PRESENT
//           absent|negative|no seminal vesicle|clear|not invaded → ABSENT
//   LVI:    lymphovascular invasion
//           present|positive|identified → PRESENT
//           absent|negative|not identified → ABSENT
//   N1:     lymph node|nodal (NOT lymphovascular)
//           metastasis.*present|lymph node.*positive|lymph node.*involved → PRESENT
//           no lymph node|lymph node.*negative|lymph node.*absent → ABSENT
//           not removed|no lymph nodes were removed|no lymph node dissection → NOT RECORDED

function extractSurgicalPathFlags(report) {
  if (!report || typeof report !== 'string') {
    return { margin: null, epe: null, svi: null, lvi: null, n1: null };
  }

  // Delegate regex extraction to the shared parser module.
  // The shared module returns string enums ('positive'/'negative', 'present'/'absent');
  // this wrapper maps them back to the boolean tri-state (true/false/null) expected
  // by the flag-strip badge renderer (makeFlagBadge).
  const parsed = parseSurgicalPathology(report);

  const toBool = (val, presentStr, absentStr) => {
    if (val === presentStr) return true;
    if (val === absentStr) return false;
    return null;
  };

  // LVI (Lymphovascular Invasion) is now extracted by the shared parser, which
  // applies negation-aware classification and the "not noted" → null invariant
  // (unknown, not absent). Map the string enum to the boolean tri-state.
  return {
    margin: toBool(parsed.margin, 'positive', 'negative'),
    epe: toBool(parsed.ece, 'present', 'absent'),
    svi: toBool(parsed.svi, 'present', 'absent'),
    lvi: toBool(parsed.lvi, 'present', 'absent'),
    n1: toBool(parsed.lni, 'present', 'absent'),
  };
}

// ===========================================================================
// DOM Helpers
// ===========================================================================

function makeSection(titleText) {
  const section = document.createElement('div');
  section.className = 'case-extra-section';

  const title = document.createElement('div');
  title.className = 'case-extra-title';
  title.textContent = titleText;
  section.appendChild(title);

  return section;
}

function makeFlagBadge(value, presentLabel, absentLabel) {
  const badge = document.createElement('span');
  if (value === true) {
    badge.className = 'forest-strip-flag forest-flag-positive';
    badge.textContent = presentLabel;
  } else if (value === false) {
    badge.className = 'forest-strip-flag forest-flag-negative';
    badge.textContent = absentLabel;
  } else {
    badge.className = 'forest-strip-flag forest-flag-unknown';
    badge.textContent = 'NOT RECORDED';
  }
  return badge;
}

function makeMissingBox(text) {
  const box = document.createElement('div');
  box.className = 'case-extra-missing';
  box.textContent = text;
  return box;
}

// ===========================================================================
// R-15: Surgical Pathology Flag Strip (Task 3 only)
// ===========================================================================

function renderSurgicalPathologyFlagStrip(panel, trace) {
  const task = (trace.task || '').toLowerCase();
  if (task !== 'task3') return;

  const report = trace.clinical_records?.surgical_pathology_report;
  if (!report || typeof report !== 'string') {
    const section = makeSection('R-15: Surgical Pathology Flag Strip');
    section.appendChild(makeMissingBox('No surgical pathology report available for this case.'));
    panel.appendChild(section);
    return;
  }

  const flags = extractSurgicalPathFlags(report);

  // Weight-ordered by CAPRA-S weight (descending): SVI(2), Margin(2), N1(1), EPE(1), LVI(N/A)
  const orderedFlags = [
    { key: 'svi', label: 'SVI', presentLabel: 'PRESENT', absentLabel: 'ABSENT', value: flags.svi,
      desc: 'Seminal Vesicle Invasion' },
    { key: 'margin', label: 'Margin', presentLabel: 'R1', absentLabel: 'R0', value: flags.margin,
      desc: 'Surgical Margin Status' },
    { key: 'n1', label: 'N1', presentLabel: 'PRESENT', absentLabel: 'ABSENT', value: flags.n1,
      desc: 'Lymph Node Metastasis' },
    { key: 'epe', label: 'EPE', presentLabel: 'PRESENT', absentLabel: 'ABSENT', value: flags.epe,
      desc: 'Extraprostatic Extension' },
    { key: 'lvi', label: 'LVI', presentLabel: 'PRESENT', absentLabel: 'ABSENT', value: flags.lvi,
      desc: 'Lymphovascular Invasion' },
  ];

  const section = makeSection('R-15: Surgical Pathology Flag Strip (Weight-Ordered)');

  // Provenance badge — computed via NLP regex extraction
  const surgProv = document.createElement('a');
  surgProv.href = 'computations.html#surgical-pathology-parser';
  surgProv.className = 'provenance-badge computed';
  surgProv.target = '_blank';
  surgProv.textContent = 'COMPUTED';
  surgProv.style.marginBottom = '6px';
  surgProv.style.display = 'inline-flex';
  section.appendChild(surgProv);

  for (const f of orderedFlags) {
    const row = document.createElement('div');
    row.className = 'forest-strip-row';

    const label = document.createElement('span');
    label.className = 'forest-strip-label';
    label.textContent = f.label;

    const badge = makeFlagBadge(f.value, f.presentLabel, f.absentLabel);

    const desc = document.createElement('span');
    desc.style.color = 'var(--text-muted)';
    desc.style.fontSize = '10px';
    desc.textContent = f.desc;

    row.appendChild(label);
    row.appendChild(badge);
    row.appendChild(desc);
    section.appendChild(row);
  }

  // Extraction provenance note
  const note = document.createElement('div');
  note.style.fontFamily = 'var(--font-mono)';
  note.style.fontSize = '10px';
  note.style.color = 'var(--text-dim)';
  note.style.marginTop = '6px';
  note.textContent = 'Extracted via documented regex from surgical_pathology_report. Unextractable → NOT RECORDED (never assumed negative).';
  section.appendChild(note);

  panel.appendChild(section);
}

// ===========================================================================
// R-14: Charlson/CCI Actuarial Line
// ===========================================================================

function renderCCILine(panel, trace) {
  const demo = trace.patient_demographics || {};
  const pmhx = demo.pmhx;
  const age = demo.age;

  const section = makeSection('R-14: Charlson Comorbidity Index (CCI) — Actuarial Line');

  // Provenance badge — computed via keyword matching + age adjustment
  const cciProv = document.createElement('a');
  cciProv.href = 'computations.html#charlson-comorbidity-index';
  cciProv.className = 'provenance-badge computed';
  cciProv.target = '_blank';
  cciProv.textContent = 'COMPUTED';
  cciProv.style.marginBottom = '6px';
  cciProv.style.display = 'inline-flex';
  section.appendChild(cciProv);

  if (!Array.isArray(pmhx) || pmhx.length === 0) {
    section.appendChild(makeMissingBox('pmhx absent or empty — CCI cannot be computed (MISSING).'));
    panel.appendChild(section);
    return;
  }

  const cci = computeCCI(pmhx, age);

  if (cci.missing || cci.components.length === 0) {
    // pmhx exists but no CCI-mappable conditions found
    const line = document.createElement('div');
    line.className = 'cci-line';
    const ageNum = (age === null || age === undefined) ? null : Number(age);
    const ageAdj = (ageNum !== null && !isNaN(ageNum) && ageNum >= 50)
      ? Math.floor((ageNum - 40) / 10) : 0;
    line.innerHTML = `CCI: <span class="cci-score">0</span> (raw) / <span class="cci-score">${ageAdj}</span> (age-adjusted) — Components: [none mapped from ${pmhx.length} pmhx entr${pmhx.length === 1 ? 'y' : 'ies'}]`;
    section.appendChild(line);

    const note = document.createElement('div');
    note.style.fontFamily = 'var(--font-mono)';
    note.style.fontSize = '10px';
    note.style.color = 'var(--text-dim)';
    note.style.marginTop = '4px';
    note.textContent = 'CONFIDENCE: MEDIUM — keyword matching is approximate. pmhx entries may not map to CCI conditions.';
    section.appendChild(note);

    panel.appendChild(section);
    return;
  }

  const compLabels = cci.components.map(c => `${escapeHTML(c.label)} (+${escapeHTML(c.weight)})`);
  const line = document.createElement('div');
  line.className = 'cci-line';
  line.innerHTML = `CCI: <span class="cci-score">${escapeHTML(cci.raw)}</span> (raw) / <span class="cci-score">${escapeHTML(cci.ageAdjusted)}</span> (age-adjusted) — Components: [${compLabels.join(', ')}]`;
  section.appendChild(line);

  const note = document.createElement('div');
  note.style.fontFamily = 'var(--font-mono)';
  note.style.fontSize = '10px';
  note.style.color = 'var(--text-dim)';
  note.style.marginTop = '4px';
  note.textContent = 'CONFIDENCE: MEDIUM — keyword matching is approximate. Age adjustment: +floor((age-40)/10) for age >= 50.';
  section.appendChild(note);

  panel.appendChild(section);
}

// ===========================================================================
// R-11: AI-vs-Human ISUP Discordance Gauge
// ===========================================================================
// Cohort-wide availability check: bx_isup_pred is present in 0/423 trace JSONs.
// In structured-prompt.json source files: 195/195 task1 have the field but ALL null,
// 0/153 task2 and 0/75 task3 have the field at all.
// Conclusion: bx_isup_pred is NOT AVAILABLE in this cohort → R-11 renders
// "NOT AVAILABLE IN THIS COHORT" for all cases.

function renderDiscordanceGauge(panel, trace) {
  const section = makeSection('R-11: AI-vs-Human ISUP Discordance Gauge');

  const demo = trace.patient_demographics || {};
  const bxIsupPred = demo.bx_isup_pred;

  if (bxIsupPred === undefined || bxIsupPred === null) {
    // Cohort-wide check confirms bx_isup_pred is not available
    const box = document.createElement('div');
    box.className = 'discordance-gauge-missing';
    box.textContent = 'NOT AVAILABLE IN THIS COHORT — bx_isup_pred is null/absent in all 423 traces (0% non-null).';
    section.appendChild(box);

    const note = document.createElement('div');
    note.style.fontFamily = 'var(--font-mono)';
    note.style.fontSize = '10px';
    note.style.color = 'var(--text-dim)';
    note.style.marginTop = '4px';
    note.textContent = 'AI ISUP predictions are not recorded in the evaluation traces. No fabricated predictions.';
    section.appendChild(note);

    panel.appendChild(section);
    return;
  }

  // If bx_isup_pred becomes available in future cohorts:
  const humanIsup = demo.bx_isup;
  const aiIsup = Number(bxIsupPred);

  if (humanIsup === undefined || humanIsup === null || isNaN(Number(humanIsup))) {
    section.appendChild(makeMissingBox('Human bx_isup is MISSING — cannot compute discordance.'));
    panel.appendChild(section);
    return;
  }

  const humanNum = Number(humanIsup);
  const delta = aiIsup - humanNum;
  const maxDelta = 4; // ISUP ranges 1-5, max difference = 4
  const deltaPct = Math.abs(delta) / maxDelta;

  // Diverging bar from center
  const wrap = document.createElement('div');
  wrap.style.fontFamily = 'var(--font-mono)';
  wrap.style.fontSize = '11px';
  wrap.style.color = 'var(--text-main)';
  wrap.style.padding = '8px 12px';
  wrap.style.background = 'var(--bg-table-zebra)';
  wrap.style.border = '1px solid var(--border-subtle)';
  wrap.style.borderRadius = '4px';

  const barRow = document.createElement('div');
  barRow.style.display = 'flex';
  barRow.style.alignItems = 'center';
  barRow.style.gap = '8px';
  barRow.style.marginBottom = '6px';

  const humanLabel = document.createElement('span');
  humanLabel.textContent = `Human ISUP: ${humanNum}`;
  humanLabel.style.fontWeight = '600';

  const aiLabel = document.createElement('span');
  aiLabel.textContent = `AI ISUP: ${aiIsup}`;
  aiLabel.style.fontWeight = '600';
  aiLabel.style.color = 'var(--color-blue-fg)';

  const deltaLabel = document.createElement('span');
  deltaLabel.textContent = `Δ = ${delta >= 0 ? '+' : ''}${delta}`;
  deltaLabel.style.fontWeight = '700';
  deltaLabel.style.color = Math.abs(delta) >= 2 ? 'var(--color-red-fg)' : (Math.abs(delta) >= 1 ? 'var(--color-amber-fg)' : 'var(--color-green-fg)');

  barRow.appendChild(humanLabel);
  barRow.appendChild(aiLabel);
  barRow.appendChild(deltaLabel);
  wrap.appendChild(barRow);

  // Diverging bar visualization
  const barContainer = document.createElement('div');
  barContainer.style.position = 'relative';
  barContainer.style.height = '12px';
  barContainer.style.background = 'var(--bg-input)';
  barContainer.style.borderRadius = '2px';
  barContainer.style.overflow = 'hidden';

  const centerLine = document.createElement('div');
  centerLine.style.position = 'absolute';
  centerLine.style.left = '50%';
  centerLine.style.top = '0';
  centerLine.style.bottom = '0';
  centerLine.style.width = '1px';
  centerLine.style.background = 'var(--text-muted)';
  barContainer.appendChild(centerLine);

  const fillBar = document.createElement('div');
  fillBar.style.position = 'absolute';
  fillBar.style.top = '0';
  fillBar.style.bottom = '0';
  if (delta >= 0) {
    fillBar.style.left = '50%';
    fillBar.style.width = `${deltaPct * 50}%`;
    fillBar.style.background = 'var(--color-blue-fg)';
  } else {
    fillBar.style.right = '50%';
    fillBar.style.width = `${deltaPct * 50}%`;
    fillBar.style.background = 'var(--color-red-fg)';
  }
  barContainer.appendChild(fillBar);
  wrap.appendChild(barContainer);

  section.appendChild(wrap);
  panel.appendChild(section);
}

// ===========================================================================
// R-12: Embedding Signature Panel
// ===========================================================================

function computeVectorStats(vec) {
  if (!Array.isArray(vec) || vec.length === 0) return null;
  let sum = 0, min = Infinity, max = -Infinity;
  for (const v of vec) {
    const n = Number(v);
    if (isNaN(n)) continue;
    sum += n;
    if (n < min) min = n;
    if (n > max) max = n;
  }
  const n = vec.length;
  const mean = sum / n;
  let sqSum = 0;
  for (const v of vec) {
    const nn = Number(v);
    if (isNaN(nn)) continue;
    sqSum += (nn - mean) ** 2;
  }
  const std = Math.sqrt(sqSum / n);
  return { mean, std, min, max, n };
}

function renderEmbeddingSignature(panel, trace) {
  const mods = trace.modality_representations || {};
  const modalityDefs = [
    { key: 'mri', label: 'MRI (1024-d)' },
    { key: 'biopsy', label: 'Biopsy WSI (960-d)' },
    { key: 'prostatectomy', label: 'Prostatectomy (960-d)' },
  ];

  const section = makeSection('R-12: Embedding Signature Panel (Backend-Computed)');

  // Provenance badge — vector stats computed in-browser from uploaded embedding vectors
  const embProv = document.createElement('a');
  embProv.href = 'computations.html#vector-statistics';
  embProv.className = 'provenance-badge computed';
  embProv.target = '_blank';
  embProv.textContent = 'COMPUTED';
  embProv.style.marginBottom = '6px';
  embProv.style.display = 'inline-flex';
  section.appendChild(embProv);

  // Check if any modality has data
  const hasAny = modalityDefs.some(m => {
    const rep = mods[m.key];
    return rep && rep.shape && Array.isArray(rep.shape) && rep.shape[0] > 0;
  });

  if (!hasAny) {
    section.appendChild(makeMissingBox('No modality representations available for this case.'));
    panel.appendChild(section);
    return;
  }

  const table = document.createElement('table');
  table.className = 'embedding-sig-table';

  // Header
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  ['Modality', 'Shape', 'L2 Norm', 'Mean', 'Std', 'Min', 'Max'].forEach(h => {
    const th = document.createElement('th');
    th.textContent = h;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');

  // Track whether any modality's stats were recomputed from vector_sample
  // (vs. read directly from backend-computed trace JSON fields).
  let anyRecomputed = false;

  for (const m of modalityDefs) {
    const rep = mods[m.key];
    const row = document.createElement('tr');

    const isPresent = rep && rep.shape && Array.isArray(rep.shape) && rep.shape[0] > 0;

    // Modality label
    const labelCell = document.createElement('td');
    labelCell.style.fontWeight = '600';
    labelCell.style.color = 'var(--text-main)';
    labelCell.textContent = m.label;
    row.appendChild(labelCell);

    if (!isPresent) {
      const absentCell = document.createElement('td');
      absentCell.colSpan = 6;
      absentCell.style.color = 'var(--text-muted)';
      absentCell.style.fontStyle = 'italic';
      absentCell.textContent = 'SPARSE / ABSENT';
      row.appendChild(absentCell);
      tbody.appendChild(row);
      continue;
    }

    // Shape
    const shapeStr = rep.shape ? `[${rep.shape.join(', ')}]` : '—';
    const shapeCell = document.createElement('td');
    shapeCell.className = 'num-cell';
    shapeCell.textContent = shapeStr;
    row.appendChild(shapeCell);

    // Compute stats: prefer vector_sample if available, else use backend-computed values
    let stats = null;
    if (Array.isArray(rep.vector_sample) && rep.vector_sample.length > 0) {
      stats = computeVectorStats(rep.vector_sample);
      anyRecomputed = true;
    }

    // L2 Norm
    const normCell = document.createElement('td');
    normCell.className = 'num-cell';
    normCell.textContent = (rep.norm !== undefined && rep.norm !== null) ? Number(rep.norm).toFixed(4) : '—';
    row.appendChild(normCell);

    // Mean
    const meanCell = document.createElement('td');
    meanCell.className = 'num-cell';
    if (stats) {
      meanCell.textContent = stats.mean.toFixed(4);
    } else if (rep.mean !== undefined && rep.mean !== null) {
      meanCell.textContent = Number(rep.mean).toFixed(4);
    } else {
      meanCell.textContent = '—';
    }
    row.appendChild(meanCell);

    // Std
    const stdCell = document.createElement('td');
    stdCell.className = 'num-cell';
    if (stats) {
      stdCell.textContent = stats.std.toFixed(4);
    } else if (rep.std !== undefined && rep.std !== null) {
      stdCell.textContent = Number(rep.std).toFixed(4);
    } else {
      stdCell.textContent = '—';
    }
    row.appendChild(stdCell);

    // Min
    const minCell = document.createElement('td');
    minCell.className = 'num-cell';
    if (stats) {
      minCell.textContent = stats.min.toFixed(4);
    } else if (rep.min !== undefined && rep.min !== null) {
      minCell.textContent = Number(rep.min).toFixed(4);
    } else {
      minCell.textContent = '—';
    }
    row.appendChild(minCell);

    // Max
    const maxCell = document.createElement('td');
    maxCell.className = 'num-cell';
    if (stats) {
      maxCell.textContent = stats.max.toFixed(4);
    } else if (rep.max !== undefined && rep.max !== null) {
      maxCell.textContent = Number(rep.max).toFixed(4);
    } else {
      maxCell.textContent = '—';
    }
    row.appendChild(maxCell);

    tbody.appendChild(row);
  }

  table.appendChild(tbody);
  section.appendChild(table);

  // Provenance label: clarify whether stats are backend-computed or client-recomputed
  // from a (possibly truncated) vector_sample. When vector_sample is present, the
  // recomputed mean/std/min/max may diverge from the backend full-vector statistics
  // rendered in the Standard View embeddings panel above (H3 divergence guard).
  if (anyRecomputed) {
    const sampledLabel = document.createElement('div');
    sampledLabel.className = 'backend-label sampled-stats-label';
    sampledLabel.style.marginTop = '6px';
    sampledLabel.style.fontWeight = '600';
    sampledLabel.style.color = 'var(--color-amber-fg)';
    sampledLabel.textContent = 'Sampled statistics (client-recomputed from vector_sample)';
    section.appendChild(sampledLabel);

    const sampledNote = document.createElement('div');
    sampledNote.className = 'backend-label sampled-stats-note';
    sampledNote.style.marginTop = '4px';
    sampledNote.style.fontFamily = 'var(--font-mono)';
    sampledNote.style.fontSize = '10px';
    sampledNote.style.color = 'var(--text-dim)';
    sampledNote.textContent = 'Note: Statistics computed from available vector sample. May differ from backend full-vector statistics shown above.';
    section.appendChild(sampledNote);
  } else {
    const label = document.createElement('div');
    label.className = 'backend-label';
    label.style.marginTop = '6px';
    label.textContent = 'Backend-computed — statistics from trace JSON, not recomputed in browser';
    section.appendChild(label);
  }

  panel.appendChild(section);
}

// ===========================================================================
// Main CaseExtras Export
// ===========================================================================

export const CaseExtras = {
  /**
   * Mounts case-level extras into existing Standard View panels.
   * Appends sub-sections; does NOT replace existing panel content.
   * @param {Object} trace - Normalized trace object from TraceReader
   */
  mount(trace) {
    if (!trace) return;

    // R-14: CCI Actuarial Line → #panel-demographics
    const panelDemo = document.getElementById('panel-demographics');
    if (panelDemo) {
      renderCCILine(panelDemo, trace);
    }

    // R-11: AI-vs-Human ISUP Discordance Gauge → #panel-decision
    const panelDecision = document.getElementById('panel-decision');
    if (panelDecision) {
      renderDiscordanceGauge(panelDecision, trace);
    }

    // R-12: Embedding Signature Panel → #panel-embeddings
    const panelEmb = document.getElementById('panel-embeddings');
    if (panelEmb) {
      renderEmbeddingSignature(panelEmb, trace);
    }

    // R-15: Surgical Pathology Flag Strip → #panel-survival (Task 3 only)
    const panelSurvival = document.getElementById('panel-survival');
    if (panelSurvival) {
      renderSurgicalPathologyFlagStrip(panelSurvival, trace);
    }
  },
};

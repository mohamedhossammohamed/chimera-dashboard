// Handshake Protocol & Schema Ingestion Engine (data.js)

// M-105 — Strip prototype-pollution keys (__proto__, constructor, prototype)
// from any parsed JSON before the data is trusted by downstream logic.
export function sanitizeJson(obj) {
  if (Array.isArray(obj)) {
    return obj.map(sanitizeJson);
  }
  if (obj !== null && typeof obj === 'object') {
    const cleaned = {};
    for (const k of Object.keys(obj)) {
      if (k === '__proto__' || k === 'constructor' || k === 'prototype') continue;
      cleaned[k] = sanitizeJson(obj[k]);
    }
    return cleaned;
  }
  return obj;
}

/**
 * Detect whether a set of uploaded files is in CHIMERA train_release split format
 * (5 separate JSON files per case directory) rather than the merged trace format
 * the app expects.
 *
 * train_release directory structure:
 *   task1/PT-pseudo_xxx/
 *     structured-prompt.json                              → demographics + metadata
 *     prostate-biopsy-decision-clinical-data.json         → clinical records
 *     prostate-modality-level-neural-representations.json → embeddings
 *     prostate-biopsy-decision-reasoning.json             → model prediction
 *     prostate-biopsy-decision.json                       → ground truth (bare string)
 *
 *   task2/T2-xxx/
 *     structured-prompt.json
 *     prostate-treatment-decision-clinical-data.json
 *     prostate-modality-level-neural-representations.json
 *     prostate-treatment-decision-reasoning.json
 *     prostate-treatment-decision.json                    → ground truth (bare string)
 *
 *   task3/T3-xxx/
 *     structured-prompt.json
 *     prostate-time-to-recurrence-or-last-follow-up-clinical-data.json
 *     prostate-modality-level-neural-representations.json
 *     prostate-time-to-recurrence-or-last-follow-up.json  → ground truth (object with months_to_recurrence, event)
 *
 * @param {Array<{name: string, content: any, relativePath?: string}>} files
 * @returns {boolean} true if files appear to be in train_release split format
 */
export function isTrainReleaseSplitFormat(files) {
  let structuredPromptCount = 0;
  let neuralRepCount = 0;
  let clinicalDataCount = 0;
  for (const f of files) {
    const name = f.name;
    if (name === 'structured-prompt.json') structuredPromptCount++;
    if (name === 'prostate-modality-level-neural-representations.json') neuralRepCount++;
    if (/^prostate-.*-clinical-data\.json$/.test(name)) clinicalDataCount++;
  }
  // Need at least one case worth of files (structured-prompt + neural-rep + clinical-data)
  return structuredPromptCount > 0 && neuralRepCount > 0 && clinicalDataCount > 0;
}

/**
 * Merge a group of files belonging to the same case directory into a single
 * trace object matching the app's expected schema.
 *
 * @param {Array<{name: string, content: any}>} caseFiles - files from one case dir
 * @returns {Object|null} merged trace, or null if insufficient data
 */
export function mergeTrainReleaseCase(caseFiles) {
  let structuredPrompt = null;
  let clinicalData = null;
  let neuralReps = null;
  let reasoning = null;
  let decision = null;

  for (const f of caseFiles) {
    const name = f.name;
    const content = f.content;
    if (name === 'structured-prompt.json') structuredPrompt = content;
    else if (name === 'prostate-modality-level-neural-representations.json') neuralReps = content;
    else if (/^prostate-.*-clinical-data\.json$/.test(name)) clinicalData = content;
    else if (/^prostate-.*-reasoning\.json$/.test(name)) reasoning = content;
    else if (/^prostate-(?:biopsy-decision|treatment-decision|time-to-recurrence-or-last-follow-up)\.json$/.test(name)) decision = content;
  }

  if (!structuredPrompt) return null;

  // Extract case_id and task from structured-prompt
  const caseId = structuredPrompt.case_id || structuredPrompt.id || structuredPrompt.pid;
  const taskNum = structuredPrompt.task; // integer 1, 2, or 3
  const task = `task${taskNum}`;

  // Build patient_demographics from structured-prompt fields
  const patient_demographics = {
    age: structuredPrompt.age ?? null,
    psa: structuredPrompt.psa ?? null,
    psad: structuredPrompt.psad ?? null,
    vol: structuredPrompt.vol ?? null,
    pirads: structuredPrompt.pirads ?? null,
    dre: structuredPrompt.dre ?? null,
    ct: structuredPrompt.ct ?? null,
    bx_isup: structuredPrompt.bx_isup ?? null,
    bx_gl_prim: structuredPrompt.bx_gl_prim ?? null,
    bx_gl_sec: structuredPrompt.bx_gl_sec ?? null,
    bx_gl_tert: structuredPrompt.bx_gl_tert ?? null,
    bx_isup_pred: structuredPrompt.bx_isup_pred ?? null,
    pmhx: structuredPrompt.pmhx ?? [],
  };

  // Build modality_representations from neural representations
  // Neural rep file has keys: "MRI image", "Biopsy slide", "Prostatectomy slide"
  // Each value is an array of arrays (embedding vectors) or empty array
  const modality_representations = {};
  if (neuralReps) {
    // MRI
    const mriRaw = neuralReps['MRI image'];
    if (Array.isArray(mriRaw) && mriRaw.length > 0) {
      const vec = Array.isArray(mriRaw[0]) ? mriRaw[0] : mriRaw;
      modality_representations.mri = {
        shape: [1, vec.length],
        norm: Math.sqrt(vec.reduce((s, v) => s + v * v, 0)),
        vector_sample: vec.slice(0, 6),
        embedding: mriRaw,
      };
    } else {
      modality_representations.mri = { shape: [0], norm: 0, vector_sample: [], embedding: [] };
    }
    // Biopsy
    const bxRaw = neuralReps['Biopsy slide'];
    if (Array.isArray(bxRaw) && bxRaw.length > 0) {
      const vec = Array.isArray(bxRaw[0]) ? bxRaw[0] : bxRaw;
      modality_representations.biopsy = {
        shape: [1, vec.length],
        norm: Math.sqrt(vec.reduce((s, v) => s + v * v, 0)),
        vector_sample: vec.slice(0, 6),
        embedding: bxRaw,
      };
    } else {
      modality_representations.biopsy = { shape: [0], norm: 0, vector_sample: [], embedding: [] };
    }
    // Prostatectomy
    const pxRaw = neuralReps['Prostatectomy slide'];
    if (Array.isArray(pxRaw) && pxRaw.length > 0) {
      const vec = Array.isArray(pxRaw[0]) ? pxRaw[0] : pxRaw;
      modality_representations.prostatectomy = {
        shape: [1, vec.length],
        norm: Math.sqrt(vec.reduce((s, v) => s + v * v, 0)),
        vector_sample: vec.slice(0, 6),
        embedding: pxRaw,
      };
    } else {
      modality_representations.prostatectomy = { shape: [0], norm: 0, vector_sample: [], embedding: [] };
    }
  }

  // clinical_records from clinical-data.json
  const clinical_records = clinicalData || {};

  // ground_truth from decision file
  // Task 1/2: decision.json is a bare string like "no" or "active_surveillance"
  // Task 3: decision.json is an object { months_to_recurrence, event }
  let ground_truth = {};
  if (decision !== null) {
    if (typeof decision === 'string') {
      ground_truth = { decision };
    } else if (typeof decision === 'object' && !Array.isArray(decision)) {
      ground_truth = { ...decision };
    }
  }

  // model_prediction from reasoning file
  let model_prediction = null;
  if (reasoning && typeof reasoning === 'object' && !Array.isArray(reasoning)) {
    model_prediction = reasoning;
  }

  const merged = {
    case_id: String(caseId),
    task,
    patient_demographics,
    modality_representations,
    clinical_records,
    ground_truth,
  };
  if (model_prediction) merged.model_prediction = model_prediction;

  return merged;
}

/**
 * Group uploaded files by case directory and merge each group into a single trace.
 *
 * @param {Array<{name: string, content: any, relativePath?: string}>} files
 * @returns {Array<Object>} merged traces ready for validateAndNormalize
 */
export function mergeTrainReleaseFiles(files) {
  // Group files by their directory path (from relativePath or webkitRelativePath)
  // If no path info, group by parent directory inferred from filename patterns
  const groups = new Map(); // dirKey → Array<{name, content}>

  for (const f of files) {
    // Try to get directory path from relativePath or webkitRelativePath
    const relPath = f.relativePath || f.webkitRelativePath || '';
    let dirKey;
    if (relPath && relPath.includes('/')) {
      // Use the directory containing the file as the group key
      const parts = relPath.split('/');
      dirKey = parts.slice(0, -1).join('/');
    } else {
      // No path info — group by case_id from structured-prompt, or skip
      // For files without path info, we can't reliably group them
      // Fall back to a single group per unique filename pattern
      dirKey = '__no_path__';
    }

    if (!groups.has(dirKey)) groups.set(dirKey, []);
    groups.get(dirKey).push({ name: f.name, content: f.content });
  }

  // If all files are in __no_path__, try to group by case_id from structured-prompt
  if (groups.has('__no_path__')) {
    const noPathFiles = groups.get('__no_path__');
    groups.delete('__no_path__');

    // Try to extract case_id from each structured-prompt.json
    // and group clinical-data/neural-rep/decision with it
    const caseGroups = new Map(); // caseId → files
    const ungrouped = [];

    for (const f of noPathFiles) {
      if (f.name === 'structured-prompt.json' && f.content?.case_id) {
        const cid = f.content.case_id;
        if (!caseGroups.has(cid)) caseGroups.set(cid, []);
        caseGroups.get(cid).push(f);
      } else {
        ungrouped.push(f);
      }
    }

    // If we found structured-prompt files with case_ids, try to match other files
    if (caseGroups.size > 0) {
      // For ungrouped files, we can't determine which case they belong to
      // without path info. Add them to a special group that will be skipped.
      // In practice, folder upload always provides relativePath.
      for (const [cid, caseFiles] of caseGroups) {
        groups.set(cid, caseFiles);
      }
    } else {
      // No structured-prompt found — put everything back in one group
      groups.set('__no_path__', noPathFiles);
    }
  }

  const merged = [];
  for (const [dirKey, caseFiles] of groups) {
    const trace = mergeTrainReleaseCase(caseFiles);
    if (trace) merged.push(trace);
  }

  return merged;
}

export class TraceReader {
  constructor(basePath = 'traces') {
    this.basePath = basePath;
  }

  async _fetchWithTimeout(url, timeoutMs = 30000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, { signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  async loadTrace(traceRelativePath) {
    try {
      const resp = await this._fetchWithTimeout(traceRelativePath);
      if (!resp.ok) throw new Error(`HTTP ${resp.status} loading trace`);
      const trace = await resp.json();
      return this.validateAndNormalize(trace);
    } catch (err) {
      return {
        success: false,
        error: `Failed to load trace (${traceRelativePath}): ${err.message}`
      };
    }
  }

  validateAndNormalize(trace) {
    // M-105 — sanitize prototype-pollution keys before any processing.
    trace = sanitizeJson(trace);
    const errors = [];
    if (!trace || typeof trace !== 'object' || Array.isArray(trace)) {
      return {
        success: false,
        isSchemaViolation: true,
        error: 'SCHEMA VIOLATION: Root trace payload must be a valid JSON Object.'
      };
    }

    if (!trace.case_id || typeof trace.case_id !== 'string') {
      errors.push('Missing or invalid "case_id" (string required).');
    }
    if (!trace.task || !['task1', 'task2', 'task3'].includes(String(trace.task).toLowerCase())) {
      errors.push('Missing or invalid "task" (must be "task1", "task2", or "task3").');
    }

    const validTasks = ['task1', 'task2', 'task3'];
    const taskRaw = trace.task ? String(trace.task).toLowerCase() : '';
    const task = validTasks.includes(taskRaw) ? taskRaw : 'unknown';
    if (trace.task && !validTasks.includes(taskRaw)) {
      console.warn(`[data.js] Unrecognized task "${trace.task}" for case ${trace.case_id || '?'} — defaulting to "unknown".`);
    }

    // Deep-clone helper: structuredClone with JSON fallback for older browsers.
    const clone = (obj) => {
      try { return structuredClone(obj); }
      catch (e) { return JSON.parse(JSON.stringify(obj)); }
    };

    // Object-field guard: missing (null/undefined) → silent default (fail-closed
    // normalization); present-but-wrong-type (array or non-object) → default + error.
    const objField = (value, defaultVal, name) => {
      if (value === undefined || value === null) return defaultVal;
      if (typeof value === 'object' && !Array.isArray(value)) return value;
      errors.push(`${name} must be a non-array object.`);
      return defaultVal;
    };

    // Task-specific default schemas.
    // Task 1 & 2 produce clinical decisions (active_surveillance, etc.).
    // Task 3 produces survival predictions (months_to_recurrence, event_risk, survival_curve).
    // Using a one-size-fits-all default would fabricate a non-existent "decision"
    // field for Task 3 and drop survival fields. Branch by task.
    const isTask3 = task === 'task3';

    const DEFAULT_MODEL_PREDICTION = isTask3
      ? {
          months_to_recurrence: null,
          event_risk: null,
          survival_curve: { time_points: [], survival_probabilities: [] },
          free_text: 'No model prediction available.'
        }
      : {
          decision: 'N/A',
          confidence: 'uncertain',
          variable_weights: {},
          reveal_sequence: [],
          free_text: 'No model prediction available.'
        };

    const DEFAULT_GROUND_TRUTH = isTask3
      ? { months_to_recurrence: null, event: null }
      : { decision: 'N/A' };

    // Fail-closed normalization with explicit validation tracking.
    const normalized = {
      case_id: (typeof trace.case_id === 'string' && trace.case_id.length > 0) ? trace.case_id : 'UNKNOWN_CASE',
      task: task,
      schema_errors: errors,
      patient_demographics: objField(trace.patient_demographics, {}, 'patient_demographics'),
      modality_representations: objField(trace.modality_representations, {}, 'modality_representations'),
      clinical_records: objField(trace.clinical_records, {}, 'clinical_records'),
      ground_truth: { ...DEFAULT_GROUND_TRUTH, ...objField(trace.ground_truth, DEFAULT_GROUND_TRUTH, 'ground_truth') },
      // Merge partial model_prediction with task-specific defaults so missing
      // fields are populated rather than left undefined.
      model_prediction: { ...DEFAULT_MODEL_PREDICTION, ...objField(trace.model_prediction, DEFAULT_MODEL_PREDICTION, 'model_prediction') },
      evaluation: objField(trace.evaluation, { is_correct: null }, 'evaluation')
    };

    // Check for duplicate case_id
    const existingKey = `${normalized.task}_${normalized.case_id}`;
    if (this._seenCaseIds && this._seenCaseIds.has(existingKey)) {
      console.warn(`[CHIMERA] Duplicate case_id detected: ${normalized.case_id} (task: ${normalized.task}). Previous trace will be overwritten.`);
    }
    if (!this._seenCaseIds) this._seenCaseIds = new Set();
    this._seenCaseIds.add(existingKey);

    // Set after nested-field guards so nested type violations are reflected.
    normalized.schema_valid = errors.length === 0;

    // Deep-clone so downstream mutations cannot corrupt the original parsed trace.
    return {
      success: true,
      data: clone(normalized)
    };
  }
}

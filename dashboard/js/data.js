// Handshake Protocol & Schema Ingestion Engine (data.js)
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

  async loadManifest() {
    try {
      const resp = await this._fetchWithTimeout(`${this.basePath}/index.json`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching trace index`);
      const manifest = await resp.json();
      return { success: true, data: manifest };
    } catch (err) {
      return { success: false, error: err.message };
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
      // fields are populated rather than left undefined (XFI-2).
      model_prediction: { ...DEFAULT_MODEL_PREDICTION, ...objField(trace.model_prediction, DEFAULT_MODEL_PREDICTION, 'model_prediction') },
      evaluation: objField(trace.evaluation, { is_correct: null }, 'evaluation')
    };

    // Set after nested-field guards so nested type violations are reflected.
    normalized.schema_valid = errors.length === 0;

    // Deep-clone so downstream mutations cannot corrupt the original parsed trace.
    // CRITICAL: coupled with app.js preloadAllTraces dedup fix (FIX-AP) — both must ship together.
    return {
      success: true,
      data: clone(normalized)
    };
  }
}

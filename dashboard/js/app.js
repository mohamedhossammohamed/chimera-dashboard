// Master Standard View Application Coordinator (app.js)
// [OFFICIAL: RESEARCHER-APPROVED] CHIMERA-Agent Phase A Integration
// [SUGGESTION: CO-PILOT] In-browser live execution and bundle generation
import { TraceReader } from './data.js';
import { StandardView } from './standard_view.js';
import { renderClevelandBulletStrip, renderKaplanMeierSVG, renderEAUScorecard, renderConcordanceMatrix } from './standard_components.js';
import { CohortView } from './cohort_view.js';
import { CaseExtras } from './case_extras.js';
import { ClinicalBundleGenerator } from './clinical_engine.js';
import { CohortEngine } from './cohort_engine.js';

class StandardWorkbenchApp {
  constructor() {
    this.reader = new TraceReader('traces');
    this.manifest = { traces: [] };
    this.activeTrace = null;
    this.activeBundle = null;
    this.loadedTraces = [];
  }

  // Non-blocking transient feedback for clinical workflow messages.
  // Generalizes the bundle-feedback pattern: lazily creates a shared
  // feedback host element, applies a CSS class per severity, and
  // auto-dismisses after ~3 seconds.
  showFeedback(message, type = 'info') {
    let host = document.getElementById('app-feedback');
    if (!host) {
      host = document.createElement('div');
      host.id = 'app-feedback';
      host.style.position = 'fixed';
      host.style.bottom = '12px';
      host.style.right = '12px';
      host.style.zIndex = '9999';
      host.style.fontFamily = 'var(--font-mono)';
      host.style.fontSize = '12px';
      host.style.padding = '8px 12px';
      host.style.borderRadius = '4px';
      host.style.maxWidth = '360px';
      host.style.opacity = '0';
      host.style.transition = 'opacity 0.3s ease';
      host.style.pointerEvents = 'none';
      document.body.appendChild(host);
    }

    host.textContent = message;
    host.className = `feedback-${type}`;
    host.style.opacity = '1';

    if (this._feedbackTimer) clearTimeout(this._feedbackTimer);
    this._feedbackTimer = setTimeout(() => {
      host.style.opacity = '0';
    }, 3000);
  }

  async init() {
    // Upload-first architecture: no manifest fetch, start with empty state.
    // User uploads a folder of trace JSON files via the upload button or drag-drop.
    this.manifest = { traces: [] };
    this.populateCaseSelector();

    this.injectBundleControls();
    this.injectCaseSearch();
    this.bindEvents();
    this.bindDropZone();
    this.bindFolderUpload();
    this.bindTabToggle();
  }

  populateCaseSelector() {
    const sel = document.getElementById('case-select');
    if (!sel) return;

    sel.innerHTML = '';

    if (!this.manifest || !this.manifest.traces || this.manifest.traces.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.disabled = true;
      opt.selected = true;
      opt.textContent = 'No data loaded — upload a folder';
      sel.appendChild(opt);
      const countLabel = document.getElementById('case-count-label');
      if (countLabel) countLabel.textContent = 'EVALUATION TRACES (0):';
      return;
    }

    // Defensive deduplication by (task, case_id)
    const seen = new Set();
    const uniqueTraces = (this.manifest.traces || []).filter(t => {
      const key = `${(t.task || '').toLowerCase()}::${t.case_id}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    for (const t of uniqueTraces) {
      const opt = document.createElement('option');
      opt.value = t.file || t.case_id;
      opt.textContent = `[${(t.task || '').toUpperCase()}] ${t.case_id}`;
      sel.appendChild(opt);
    }

    const countLabel = document.getElementById('case-count-label');
    if (countLabel) {
      countLabel.textContent = `EVALUATION TRACES (${uniqueTraces.length} CASES):`;
    }
  }

  // Rebuilds the manifest from loaded traces after a folder upload or drag-drop.
  rebuildManifestFromLoadedTraces() {
    this.manifest = {
      traces: this.loadedTraces.map(t => ({
        file: null,
        task: t.task,
        case_id: t.case_id
      }))
    };
  }

  injectBundleControls() {
    const controls = document.querySelector('.controls-group');
    if (!controls) return;
    if (document.getElementById('btn-copy-bundle')) return;

    const wrap = document.createElement('div');
    wrap.className = 'bundle-controls';
    wrap.style.display = 'inline-flex';
    wrap.style.gap = '8px';
    wrap.style.marginLeft = '10px';
    wrap.style.alignItems = 'center';

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.id = 'btn-copy-bundle';
    copyBtn.className = 'json-btn';
    copyBtn.textContent = 'Copy Bundle';
    copyBtn.style.fontFamily = 'var(--font-mono)';
    copyBtn.style.fontSize = '11px';
    copyBtn.style.padding = '4px 10px';
    copyBtn.style.cursor = 'pointer';

    const dlBtn = document.createElement('button');
    dlBtn.type = 'button';
    dlBtn.id = 'btn-download-bundle';
    dlBtn.className = 'json-btn';
    dlBtn.textContent = 'Download Bundle';
    dlBtn.style.fontFamily = 'var(--font-mono)';
    dlBtn.style.fontSize = '11px';
    dlBtn.style.padding = '4px 10px';
    dlBtn.style.cursor = 'pointer';

    wrap.appendChild(copyBtn);
    wrap.appendChild(dlBtn);
    controls.appendChild(wrap);

    copyBtn.addEventListener('click', () => this.copyBundle());
    dlBtn.addEventListener('click', () => this.downloadBundle());
  }

  copyBundle() {
    if (!this.activeBundle) {
      this.showFeedback('Bundle not loaded for this case.', 'error');
      return;
    }
    if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
      this.showFeedback('Clipboard not available', 'error');
      return;
    }
    navigator.clipboard.writeText(this.activeBundle).then(() => {
      this.showFeedback('Copied!', 'success');
    }).catch(() => {
      this.showFeedback('Copy failed', 'error');
    });
  }

  downloadBundle() {
    if (!this.activeBundle) {
      this.showFeedback('Bundle not loaded for this case.', 'error');
      return;
    }
    const t = this.activeTrace;
    const filename = `${(t && t.case_id) || 'case'}_bundle.md`;
    const blob = new Blob([this.activeBundle], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  injectCaseSearch() {
    const controls = document.querySelector('.controls-group');
    if (!controls) return;
    if (document.getElementById('case-search')) return;

    const wrap = document.createElement('div');
    wrap.style.display = 'inline-flex';
    wrap.style.alignItems = 'center';
    wrap.style.gap = '4px';
    wrap.style.marginLeft = '6px';

    const input = document.createElement('input');
    input.type = 'text';
    input.id = 'case-search';
    input.placeholder = 'Filter cases...';
    input.style.fontFamily = 'var(--font-mono)';
    input.style.fontSize = '11px';
    input.style.padding = '4px 8px';
    input.style.background = 'var(--bg-input)';
    input.style.border = '1px solid var(--border-strong)';
    input.style.borderRadius = '4px';
    input.style.color = 'var(--text-main)';
    input.style.width = '140px';

    const count = document.createElement('span');
    count.id = 'case-search-count';
    count.style.fontFamily = 'var(--font-mono)';
    count.style.fontSize = '10px';
    count.style.color = 'var(--text-muted)';
    count.style.minWidth = '50px';

    wrap.appendChild(input);
    wrap.appendChild(count);
    controls.appendChild(wrap);

    input.addEventListener('input', () => this.filterCases(input.value));
  }

  filterCases(query) {
    const sel = document.getElementById('case-select');
    if (!sel || !this.manifest) return;
    const q = query.trim().toLowerCase();

    // Defensive deduplication by (task, case_id) — mirrors populateCaseSelector
    const seen = new Set();
    const uniqueTraces = (this.manifest.traces || []).filter(t => {
      const key = `${(t.task || '').toLowerCase()}::${t.case_id}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    const filtered = q === '' ? uniqueTraces : uniqueTraces.filter(t =>
      (t.case_id || '').toLowerCase().includes(q) ||
      (t.task || '').toLowerCase().includes(q)
    );
    sel.innerHTML = '';
    for (const t of filtered) {
      const opt = document.createElement('option');
      opt.value = t.file || t.case_id;
      opt.textContent = `[${(t.task || '').toUpperCase()}] ${t.case_id}`;
      sel.appendChild(opt);
    }
    const countEl = document.getElementById('case-search-count');
    if (countEl) {
      countEl.textContent = q === '' ? '' : `${filtered.length}/${uniqueTraces.length}`;
    }
  }

  bindEvents() {
    const sel = document.getElementById('case-select');
    if (sel) {
      sel.addEventListener('change', e => this.selectCase(e.target.value));
    }
  }

  // Shared file processing pipeline for both folder upload and drag-drop.
  async processFiles(files) {
    if (files.length > 1) {
      this.showFeedback(`Processing ${files.length} files...`, 'info');
    }

    let loadedCount = 0;
    let lastValid = null;

    for (const file of files) {
      if (!file.name.endsWith('.json')) continue;
      if (file.size > 10 * 1024 * 1024) {
        this.showFeedback(`Skipped "${file.name}" — exceeds 10MB limit.`, 'error');
        continue;
      }
      try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        const res = this.reader.validateAndNormalize(parsed);
        if (res.success) {
          const newKey = `${(res.data.task || '').toLowerCase()}::${res.data.case_id}`;
          const exists = this.loadedTraces.some(t =>
            `${(t.task || '').toLowerCase()}::${t.case_id}` === newKey
          );
          if (!exists) {
            this.loadedTraces.push(res.data);
            loadedCount++;
            lastValid = res.data;
          }
        } else {
          this.showFeedback(`Schema Validation Warning for "${file.name}": ${res.error}`, 'error');
        }
      } catch (err) {
        this.showFeedback(`Invalid JSON in "${file.name}": ${err.message}`, 'error');
      }
    }

    if (loadedCount > 0) {
      this.showFeedback(`Loaded ${loadedCount} traces.`, 'success');
      this.rebuildManifestFromLoadedTraces();
      this.populateCaseSelector();
      if (lastValid) {
        this.activeTrace = lastValid;
        this.loadBundle(this.activeTrace);
        this.render();
      }
    } else if (files.length > 0) {
      this.showFeedback('No valid trace files found.', 'error');
    }
  }

  // Recursively reads a dropped directory entry, collecting all File objects.
  async readDirectoryEntries(reader) {
    const files = [];
    const entries = await new Promise(resolve => reader.readEntries(resolve));
    for (const entry of entries) {
      if (entry.isFile) {
        const file = await new Promise(resolve => entry.file(resolve));
        files.push(file);
      } else if (entry.isDirectory) {
        const subFiles = await this.readDirectoryEntries(entry.createReader());
        files.push(...subFiles);
      }
    }
    return files;
  }

  bindDropZone() {
    window.addEventListener('dragover', e => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });

    window.addEventListener('drop', async e => {
      e.preventDefault();

      // Try webkitGetAsEntry first for folder drag-drop support
      const items = e.dataTransfer && e.dataTransfer.items;
      if (items && items.length > 0) {
        const entries = [];
        for (const item of items) {
          const entry = item.webkitGetAsEntry?.();
          if (entry) entries.push(entry);
        }
        if (entries.length > 0) {
          const files = [];
          for (const entry of entries) {
            if (entry.isFile) {
              const file = await new Promise(resolve => entry.file(resolve));
              files.push(file);
            } else if (entry.isDirectory) {
              const dirFiles = await this.readDirectoryEntries(entry.createReader());
              files.push(...dirFiles);
            }
          }
          await this.processFiles(files);
          return;
        }
      }

      // Fallback to flat file list
      const files = e.dataTransfer && e.dataTransfer.files;
      if (!files || files.length === 0) return;
      await this.processFiles(Array.from(files));
    });
  }

  bindFolderUpload() {
    const input = document.getElementById('folder-upload');
    const btn = document.getElementById('btn-upload-folder');
    if (!input || !btn) return;

    btn.addEventListener('click', () => input.click());

    input.addEventListener('change', async (e) => {
      const files = Array.from(e.target.files);
      if (files.length === 0) return;
      await this.processFiles(files);
      input.value = '';
    });
  }

  async selectCase(tracePath) {
    // Upload-first: traces are already in memory. Find by case_id.
    const loaded = this.loadedTraces.find(t => t.case_id === tracePath);
    if (loaded) {
      this.activeTrace = loaded;
      this.loadBundle(this.activeTrace);
      this.render();
      return;
    }

    // Fallback: try HTTP fetch (for local dev with serve.py + pre-bundled data)
    this._selectCaseSeq = (this._selectCaseSeq || 0) + 1;
    const mySeq = this._selectCaseSeq;
    const res = await this.reader.loadTrace(tracePath);
    if (mySeq !== this._selectCaseSeq) return;
    if (res.success) {
      this.activeTrace = res.data;
      this.loadBundle(this.activeTrace);
      this.render();
    } else {
      console.error(res.error);
      this.showFeedback(`Failed to load case: ${res.error}`, 'error');
    }
  }

  loadBundle(trace) {
    if (!trace || !trace.task || !trace.case_id) {
      this.activeBundle = null;
      return;
    }
    // 100% In-Browser Live Markdown Bundle Generation (Option 1)
    this.activeBundle = ClinicalBundleGenerator.generateMarkdown(trace);
  }

  render() {
    if (!this.activeTrace) return;
    const t = this.activeTrace;

    const caseIdEl = document.getElementById('current-case-id');
    const taskBadgeEl = document.getElementById('current-task-badge');

    if (caseIdEl) caseIdEl.textContent = t.case_id || 'UNKNOWN_CASE';
    if (taskBadgeEl) taskBadgeEl.textContent = (t.task || '').toUpperCase();

    // Render all Standard View Panels
    StandardView.render(t);

    // Phase B: Mount case-level extras into existing panels
    CaseExtras.mount(t);
  }

  // --- Phase B: Tab Toggle between Standard View and Cohort Analytics ---
  bindTabToggle() {
    const tabStandard = document.getElementById('tab-standard');
    const tabCohort = document.getElementById('tab-cohort');
    const standardView = document.getElementById('standard-view');
    const cohortView = document.getElementById('cohort-view');
    if (!tabStandard || !tabCohort) return;

    tabStandard.addEventListener('click', () => {
      standardView.style.display = '';
      cohortView.style.display = 'none';
      tabStandard.classList.add('active');
      tabCohort.classList.remove('active');
    });

    tabCohort.addEventListener('click', async () => {
      standardView.style.display = 'none';
      cohortView.style.display = '';
      tabStandard.classList.remove('active');
      tabCohort.classList.add('active');
      await this.loadCohortData();
    });
  }

  // --- Phase B: In-Browser Live Cohort Analytics Render via CohortView ---
  async loadCohortData() {
    try {
      if (!this.loadedTraces || this.loadedTraces.length === 0) {
        this.showFeedback('No data loaded — upload a folder first.', 'error');
        return;
      }
      await CohortView.renderCohortTab(this.loadedTraces);
    } catch (err) {
      console.error('[CohortView] Failed to compute cohort tab:', err);
      const root = document.getElementById('cohort-view');
      if (root) {
        root.innerHTML = '<div style="padding: 20px; color: var(--color-red-fg); font-family: var(--font-mono); font-size: 12px;">Failed to compute cohort analytics. Check console for details.</div>';
      }
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new StandardWorkbenchApp();
  app.init();
});

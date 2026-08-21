# CHIMERA-Agent — Clinical Interpretability Workbench

A browser-based clinical interpretability workbench for prostate cancer evaluation traces. Upload a folder of trace JSON files (drag-drop or file picker), and it renders two views: a per-case Standard View and a cohort-level bird's eye view.

**Live:** https://mohamedhossammohamed.github.io/chimera-dashboard/

---

## Standard View — one patient at a time

- **Demographics panel** with Cleveland bullet strips (PSA, age, Gleason, ISUP, PI-RADS — each plotted against clinical reference ranges)
- **PSA kinetics**: PSADT (doubling time), PSAV (velocity), trajectory classification — calculated in-browser from the PSA series
- **EAU 2026 risk tier** scorecard with the exact criteria that fired
- **CAPRA-S** post-surgical nomogram score (0–10) with component breakdown
- **PI-RADS × ISUP concordance matrix** — flags AI-vs-human grade discordance
- **Negation-aware surgical pathology parser** — extracts margin status, extraprostatic extension, seminal vesicle invasion, lymph node involvement from free-text reports
- **Charlson Comorbidity Index** — keyword-matched from past medical history, age-adjusted
- **Embedding signature panel** — vector statistics (mean/std/min/max/shape) for MRI, biopsy, and prostatectomy slide embeddings
- Variable weights, clinical text, raw JSON tree for full reproducibility

## Cohort View — bird's eye across all uploaded cases

- **Composition**: case counts by task, risk tier distribution
- **PCA manifolds**: dual-Gram power iteration on MRI (1024-d) and biopsy (960-d) embeddings, projected to 2D
- **Spearman correlation matrix** across all clinical variables
- **Ward hierarchical clustering** dendrogram
- **Missingness grid**: 6-channel multimodal coverage heatmap
- **Raincloud plots**: Tukey five-number summary + KDE (Silverman bandwidth) + jittered scatter for every continuous variable
- **Kaplan-Meier** survival estimator (cohort-level)

## Provenance

Every calculated value carries a provenance badge — **CALCULATED** (clickable, links to the formula and code in `computations.html`) or **UPLOADED** (parsed from the raw trace). No black boxes. The math is documented, the code is visible, the formulas are rendered in LaTeX.

## Parsing

The parsing layer handles schema validation, `train_release` split-file merging, deduplication by `(task, case_id)`, and input sanitization. Drag a folder, it works.

## Privacy

Uploaded data persists in the browser's IndexedDB across sessions — no re-upload on refresh. A "Clear Memory" button wipes it. Nothing leaves the browser. No server, no backend, no telemetry.

## Repository

https://github.com/mohamedhossammohamed/chimera-dashboard

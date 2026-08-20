#!/usr/bin/env python3
"""CHIMERA-Agent Phase B: Cohort Analytics Precompute Engine.

[OFFICIAL: RESEARCHER-APPROVED] CHIMERA-Agent Phase B Cohort Analytics
[SUGGESTION: CO-PILOT] Implementation architecture and SVG rendering

This script loads all 423 CHIMERA prostate-cohort cases from train_release/,
computes cohort-level statistical artifacts (PCA, Spearman correlation with
Ward clustering, missingness grid, raincloud plot data), and emits finished
JSON to dashboard/cohort/ for the browser-side SVG renderer.

numpy is used ONLY for PCA eigendecomposition and KDE evaluation.
All other logic uses Python stdlib (json, os, re, math, statistics, collections).

Math documentation:
  PCA: Center data → covariance matrix → eigendecomposition → top-2 projection.
  Spearman: Rank-transform each variable → Pearson r on ranks (pairwise complete).
  Ward: Agglomerative clustering minimising within-cluster variance increase.
  KDE: Silverman bandwidth h = 0.9 * min(sigma, IQR/1.34) * n^(-1/5); Gaussian kernel.
  Box: Tukey five-number (Q1/median/Q3) with 1.5*IQR whiskers.
"""

import json
import os
import re
import math
import sys
from collections import Counter, OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "train_release")
OUT_DIR = os.path.join(ROOT, "dashboard", "cohort")

TASK_DIRS = {"task1": "task1", "task2": "task2", "task3": "task3"}
TASK_CLINICAL = {
    "task1": "prostate-biopsy-decision-clinical-data.json",
    "task2": "prostate-treatment-decision-clinical-data.json",
    "task3": "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",
}
TASK_DECISION = {
    "task1": "prostate-biopsy-decision.json",
    "task2": "prostate-treatment-decision.json",
    "task3": "prostate-time-to-recurrence-or-last-follow-up.json",
}

NUMERIC_VARS = [
    "psa", "psad", "vol", "age", "pirads",
    "bx_isup", "bx_gl_prim", "bx_gl_sec",
    "cspca", "cores_positive", "cores_total",
]

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_float(val):
    """Convert a value to float, returning None if missing/non-numeric."""
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    s = str(val).strip()
    if s == "" or s.upper() == "N/A" or s.upper() == "NOT AVAILABLE":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def extract_task3_radiology(rr):
    """Extract vol, psad, pirads from task3 radiology_report text."""
    out = {"vol": None, "psad": None, "pirads": None}
    if not rr:
        return out
    m = re.search(r"Prostate volume:\s*([\d.]+)", rr, re.I)
    if m:
        out["vol"] = safe_float(m.group(1))
    m = re.search(r"PSA density:\s*([\d.]+)", rr, re.I)
    if m:
        out["psad"] = safe_float(m.group(1))
    m = re.search(r"PI-RADS:\s*(\d)", rr, re.I)
    if m:
        out["pirads"] = safe_float(m.group(1))
    return out


def extract_task3_pathology(pr):
    """Extract bx_isup, bx_gl_prim, bx_gl_sec from task3 pathology_report text."""
    out = {"bx_isup": None, "bx_gl_prim": None, "bx_gl_sec": None}
    if not pr:
        return out
    m = re.search(r"ISUP grade group (\d)", pr, re.I)
    if m:
        out["bx_isup"] = safe_float(m.group(1))
    m = re.search(r"Gleason (\d)\+(\d)", pr, re.I)
    if m:
        out["bx_gl_prim"] = safe_float(m.group(1))
        out["bx_gl_sec"] = safe_float(m.group(2))
    return out


def get_target(task, decision_data):
    """Extract the verified target class per task."""
    if task == "task1":
        return str(decision_data).strip().lower() if decision_data else None
    elif task == "task2":
        return str(decision_data).strip().lower() if decision_data else None
    elif task == "task3":
        if isinstance(decision_data, dict):
            ev = decision_data.get("event")
            if ev is not None:
                return str(ev)
        return None
    return None


def load_trace_ground_truth():
    """Load ground-truth targets from pre-generated trace files.
    These cover all 423 cases and serve as the canonical target source.
    The train_release decision JSONs are missing for ~44% of cases (no file emitted),
    so traces are the authoritative fallback."""
    traces_dir = os.path.join(ROOT, "dashboard", "traces", "all_local_cases")
    gt_map = {}
    if not os.path.isdir(traces_dir):
        return gt_map
    for fname in os.listdir(traces_dir):
        if not fname.endswith(".json"):
            continue
        try:
            d = load_json(os.path.join(traces_dir, fname))
        except (json.JSONDecodeError, OSError):
            continue
        cid = d.get("case_id")
        task = d.get("task")
        gt = d.get("ground_truth", {})
        if cid and task:
            gt_map[(task, cid)] = gt
    return gt_map


def load_cohort():
    """Load all 423 cases into a list of case dicts."""
    trace_gt = load_trace_ground_truth()
    cases = []
    for task_name in ["task1", "task2", "task3"]:
        tdir = os.path.join(DATA_DIR, task_name)
        if not os.path.isdir(tdir):
            continue
        for case_dir in sorted(os.listdir(tdir)):
            cpath = os.path.join(tdir, case_dir)
            if not os.path.isdir(cpath):
                continue
            sp_path = os.path.join(cpath, "structured-prompt.json")
            if not os.path.exists(sp_path):
                continue
            sp = load_json(sp_path)
            task = sp.get("task", int(task_name[-1]))
            task_str = "task" + str(task) if isinstance(task, int) else str(task)

            # Modality representations
            mod_path = os.path.join(cpath, "prostate-modality-level-neural-representations.json")
            modality = load_json(mod_path) if os.path.exists(mod_path) else {}

            def mean_pool(raw):
                """Average-pool a [k, d] list-of-lists into a single [d] vector.
                This handles variable k (0-3 biopsy slides, 0-3 prostatectomy slides)
                by averaging across the k dimension, producing a fixed-d representation."""
                if not raw or len(raw) == 0 or len(raw[0]) == 0:
                    return None
                d = len(raw[0])
                rows = [[float(x) for x in row] for row in raw if len(row) == d]
                if not rows:
                    return None
                return [sum(row[j] for row in rows) / len(rows) for j in range(d)]

            mri_vec = mean_pool(modality.get("MRI image", []))
            bx_vec = mean_pool(modality.get("Biopsy slide", []))
            px_vec = mean_pool(modality.get("Prostatectomy slide", []))

            # Clinical data
            clin_path = os.path.join(cpath, TASK_CLINICAL.get(task_str, ""))
            clin = load_json(clin_path) if os.path.exists(clin_path) else {}

            # Decision / target — prefer trace ground truth (covers all 423 cases),
            # fall back to train_release decision JSON where available.
            dec_path = os.path.join(cpath, TASK_DECISION.get(task_str, ""))
            dec = load_json(dec_path) if os.path.exists(dec_path) else None
            target = get_target(task_str, dec)
            if target is None:
                gt = trace_gt.get((task_str, sp.get("case_id", case_dir)))
                if gt:
                    if task_str == "task3":
                        target = str(gt.get("event")) if gt.get("event") is not None else None
                    else:
                        target = str(gt.get("decision", "")).strip().lower() or None

            # Build numeric variable dict
            variables = {}
            for v in NUMERIC_VARS:
                variables[v] = safe_float(sp.get(v))

            # Task3: supplement from radiology/pathology reports
            if task_str == "task3":
                rr = clin.get("radiology_report", "")
                pr = clin.get("pathology_report", "")
                rad = extract_task3_radiology(rr)
                pat = extract_task3_pathology(pr)
                for k in ("vol", "psad", "pirads"):
                    if variables[k] is None:
                        variables[k] = rad[k]
                for k in ("bx_isup", "bx_gl_prim", "bx_gl_sec"):
                    if variables[k] is None:
                        variables[k] = pat[k]

            # Missingness modalities
            has_mri = mri_vec is not None
            has_bx = bx_vec is not None
            has_px = px_vec is not None
            has_psa_trend = clin.get("psa_trend") is not None and (
                (isinstance(clin.get("psa_trend"), list) and len(clin.get("psa_trend")) > 0)
                or (isinstance(clin.get("psa_trend"), str) and clin.get("psa_trend").strip())
            )
            has_labs = clin.get("laboratory_results") is not None and (
                (isinstance(clin.get("laboratory_results"), list) and len(clin.get("laboratory_results")) > 0)
            )
            has_fh = clin.get("family_history") is not None and str(clin.get("family_history")).strip() != ""

            case = {
                "case_id": sp.get("case_id", case_dir),
                "task": task_str,
                "dir": case_dir,
                "variables": variables,
                "target": target,
                "mri_vec": mri_vec,
                "bx_vec": bx_vec,
                "px_vec": px_vec,
                "clinical": clin,
                "structured": sp,
                "missingness": {
                    "MRI": has_mri,
                    "Biopsy": has_bx,
                    "Prostatectomy": has_px,
                    "PSA_Trend": has_psa_trend,
                    "Labs": has_labs,
                    "FamilyHistory": has_fh,
                },
            }
            cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# B1: Cohort Composition
# ---------------------------------------------------------------------------

def compute_composition(cases):
    """Compute task/target composition counts and percentages."""
    composition = OrderedDict()
    task_order = ["task1", "task2", "task3"]

    # Define target class order per task
    target_order = {
        "task1": ["yes", "no"],
        "task2": ["active_surveillance", "continued_surveillance", "watchful_waiting", "active_treatment"],
        "task3": ["1", "0"],
    }

    for task in task_order:
        task_cases = [c for c in cases if c["task"] == task]
        total = len(task_cases)
        classes = target_order[task]
        counts = {}
        for cls in classes:
            counts[cls] = sum(1 for c in task_cases if c["target"] == cls)
        # Include any unexpected targets
        for c in task_cases:
            if c["target"] not in counts:
                counts[c["target"]] = counts.get(c["target"], 0) + 1
        composition[task] = {
            "total": total,
            "classes": OrderedDict(),
        }
        for cls in classes:
            n = counts.get(cls, 0)
            pct = (n / total * 100) if total > 0 else 0.0
            composition[task]["classes"][cls] = {"n": n, "pct": round(pct, 1)}
        # Any remaining
        for cls, n in counts.items():
            if cls not in composition[task]["classes"]:
                pct = (n / total * 100) if total > 0 else 0.0
                composition[task]["classes"][cls] = {"n": n, "pct": round(pct, 1)}

    return composition


# ---------------------------------------------------------------------------
# B2: PCA
# ---------------------------------------------------------------------------

def compute_pca(vectors, case_ids, targets, tasks):
    """
    PCA via numpy:
      1. Center data (subtract column means)
      2. Compute covariance matrix (1/(n-1) * X^T X)
      3. Eigendecomposition (eigh for symmetric)
      4. Sort eigenvalues descending
      5. Project onto top-2 PCs
      6. Variance explained = eigenvalue_i / sum(all eigenvalues)

    Returns dict with coords, variance_explained, cumulative_2pc.
    """
    if len(vectors) == 0:
        return {"points": [], "variance_explained": [0, 0], "cumulative_2pc": 0, "n": 0}

    X = np.array(vectors, dtype=np.float64)  # shape (n, d)
    n, d = X.shape

    # Step 1: Center
    mean = X.mean(axis=0)
    Xc = X - mean  # centered

    # Step 2: Covariance matrix (d x d)
    # cov = (1/(n-1)) * Xc^T @ Xc
    cov = np.cov(Xc, rowvar=False)

    # Step 3: Eigendecomposition (eigh returns ascending for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 4: Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Enforce svd_flip sign determinism (ensure largest absolute coordinate is positive)
    max_abs_rows = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[max_abs_rows, np.arange(eigenvectors.shape[1])])
    signs[signs == 0] = 1.0
    eigenvectors = eigenvectors * signs

    # Step 5: Project onto top-2 PCs
    # coords = Xc @ eigenvectors[:, :2]  → shape (n, 2)
    top2 = eigenvectors[:, :2]
    coords = Xc @ top2

    # Step 6: Variance explained
    total_var = eigenvalues.sum()
    if total_var > 0:
        var_explained = (eigenvalues / total_var * 100).tolist()
    else:
        var_explained = [0.0] * len(eigenvalues)

    cum_2pc = var_explained[0] + var_explained[1] if len(var_explained) >= 2 else 0.0

    points = []
    for i in range(n):
        points.append({
            "case_id": case_ids[i],
            "task": tasks[i],
            "target": targets[i],
            "pc1": round(float(coords[i, 0]), 6),
            "pc2": round(float(coords[i, 1]), 6),
        })

    return {
        "points": points,
        "variance_explained": [round(v, 2) for v in var_explained[:2]],
        "cumulative_2pc": round(cum_2pc, 2),
        "n": n,
        "method": "Centered covariance eigendecomposition (numpy.linalg.eigh), top-2 PCs projected.",
    }


# ---------------------------------------------------------------------------
# B3: Spearman Correlation + Ward Clustering
# ---------------------------------------------------------------------------

def rank_data(values):
    """Rank a list of values (1-based, average ties). Returns list of ranks."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_r(x, y):
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(x)
    if n < 3:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    denom = math.sqrt(sxx * syy)
    if denom == 0:
        return None
    return sxy / denom


def spearman_rho(x, y):
    """Spearman rho = Pearson r on ranks. Uses pairwise complete observations."""
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if len(pairs) < 5:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rx = rank_data(xs)
    ry = rank_data(ys)
    return pearson_r(rx, ry)


def compute_single_correlation(cases, min_obs=5):
    """Compute Spearman correlation matrix and Ward clustering over a subset of cases."""
    var_data = {}
    for v in NUMERIC_VARS:
        var_data[v] = [c["variables"].get(v) for c in cases]

    usable_vars = []
    for v in NUMERIC_VARS:
        n_present = sum(1 for x in var_data[v] if x is not None)
        if n_present >= min_obs:
            usable_vars.append(v)

    k = len(usable_vars)
    if k == 0:
        return {
            "variables": [],
            "matrix": [],
            "n_variables": 0,
            "n_cases": len(cases),
            "method": "Spearman rho (Pearson on ranks, pairwise complete).",
            "excluded": list(NUMERIC_VARS),
        }

    rho_matrix = [[None] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                rho_matrix[i][j] = 1.0
            elif j > i:
                r = spearman_rho(var_data[usable_vars[i]], var_data[usable_vars[j]])
                rho_matrix[i][j] = round(r, 4) if r is not None else None
                rho_matrix[j][i] = rho_matrix[i][j]

    order = ward_cluster(rho_matrix, usable_vars)
    ordered_matrix = []
    for i in order:
        row = []
        for j in order:
            row.append(rho_matrix[i][j])
        ordered_matrix.append(row)

    ordered_names = [usable_vars[idx] for idx in order]
    return {
        "variables": ordered_names,
        "matrix": ordered_matrix,
        "n_variables": len(ordered_names),
        "n_cases": len(cases),
        "method": "Spearman rho (Pearson on ranks, pairwise complete). Ward agglomerative clustering on distance = (1 - |rho|)².",
        "excluded": [v for v in NUMERIC_VARS if v not in usable_vars],
    }


def compute_correlation(cases):
    """Compute global and per-task Spearman correlation matrices."""
    global_res = compute_single_correlation(cases, min_obs=5)

    tasks_res = {}
    for task_name in ["task1", "task2", "task3"]:
        task_cases = [c for c in cases if c["task"] == task_name]
        tasks_res[task_name] = compute_single_correlation(task_cases, min_obs=5)
        print(f"  [CORRELATION] {task_name}: {tasks_res[task_name]['n_variables']} variables usable (n={len(task_cases)})")

    global_res["tasks"] = tasks_res
    return global_res


def ward_cluster(rho_matrix, var_names):
    """
    Ward agglomerative clustering on variables.
    Distance metric: d(i,j) = 1 - |rho(i,j)|
    Ward merge: minimise increase in total within-cluster variance.

    Returns list of original indices in dendrogram order (left-leaf traversal).
    """
    k = len(var_names)
    if k <= 1:
        return list(range(k))

    # Distance matrix
    dist = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            r = rho_matrix[i][j]
            if r is None:
                d = 1.0
            else:
                d = 1.0 - abs(r)
            dist[i][j] = d

    # Each cluster is a list of original indices
    clusters = [[i] for i in range(k)]
    # Cluster sizes and centroids (mean of distances within cluster)
    # For Ward, we track the "error sum of squares" for each cluster
    # ESS(cluster) = sum over members of squared distance to centroid
    # But for variable clustering with distance matrix, we use the Lance-Williams Ward formula:
    # d(C_k, C_ij) = sqrt( ((n_k+n_i)*d(C_k,C_i) + (n_k+n_j)*d(C_k,C_j) - n_k*d(C_i,C_j)) / (n_k+n_i+n_j) )
    # We work with squared distances for Ward.

    # Use squared distances
    dist2 = [[dist[i][j] ** 2 for j in range(k)] for i in range(k)]

    # Map cluster index -> list of member indices
    # Active cluster distances
    n_clusters = k
    cluster_members = [[i] for i in range(k)]
    cluster_sizes = [1] * k
    active = list(range(k))

    # Distance between active clusters (squared)
    # We maintain a dict of (ci, cj) -> squared distance
    cd = {}
    for a in range(k):
        for b in range(a + 1, k):
            cd[(a, b)] = dist2[a][b]

    merge_order = []

    while len(active) > 1:
        # Find pair with minimum Ward merge distance
        best_pair = None
        best_dist = float('inf')
        for (a, b), d in cd.items():
            if a in active and b in active and d < best_dist:
                best_dist = d
                best_pair = (a, b)

        if best_pair is None:
            break

        a, b = best_pair
        na = cluster_sizes[a]
        nb = cluster_sizes[b]
        new_cluster = active.index(a)  # reuse a's slot
        cluster_members[a] = cluster_members[a] + cluster_members[b]
        cluster_sizes[a] = na + nb
        active.remove(b)

        # Update distances using Lance-Williams Ward formula
        new_active = [c for c in active if c != a]
        for c in new_active:
            nc = cluster_sizes[c]
            d_ca = cd.get((min(a, c), max(a, c)), cd.get((min(c, a), max(c, a)), 0))
            d_cb = cd.get((min(b, c), max(b, c)), cd.get((min(c, b), max(c, b)), 0))
            d_ab = cd.get((min(a, b), max(a, b)), 0)
            # Lance-Williams Ward (squared distances)
            numerator = (nc + na) * d_ca + (nc + nb) * d_cb - nc * d_ab
            denom = nc + na + nb
            new_d = numerator / denom if denom > 0 else 0
            cd[(min(a, c), max(a, c))] = new_d

        # Remove b entries
        keys_to_remove = [key for key in cd if b in key]
        for key in keys_to_remove:
            del cd[key]

        merge_order.append((a, b))

    # Dendrogram order: left-leaf traversal of the final merged tree
    # Since we merged into cluster a, the order is just cluster_members[a]
    # but we want a meaningful dendrogram order. We'll use the merge sequence
    # to build a tree and do in-order traversal.
    # Simpler: the final cluster_members[a] gives a reasonable order based on merge sequence.
    order = cluster_members[a]
    return order


# ---------------------------------------------------------------------------
# B4: Missingness Grid
# ---------------------------------------------------------------------------

def compute_missingness(cases):
    """Compute cases x modalities binary missingness matrix."""
    modalities = ["MRI", "Biopsy", "Prostatectomy", "PSA_Trend", "Labs", "FamilyHistory"]
    matrix = []
    for c in cases:
        row = []
        for m in modalities:
            row.append(1 if c["missingness"].get(m, False) else 0)
        matrix.append(row)

    # Task-expected absence annotations
    expected_absence = {
        "task1": {
            "Biopsy": "Task 1: biopsy expectedly absent (pre-biopsy decision stage)",
            "Prostatectomy": "Task 1: prostatectomy expectedly absent (pre-biopsy decision stage)",
        },
        "task2": {
            "Prostatectomy": "Task 2: prostatectomy expectedly absent (post-biopsy, pre-surgery stage)",
        },
        "task3": {
            "PSA_Trend": "Task 3: PSA trend expectedly absent (post-prostatectomy survival stage)",
            "Labs": "Task 3: laboratory panel expectedly absent (post-prostatectomy survival stage)",
        },
    }

    # Per-task per-modality missingness summary
    summary = {}
    for task in ["task1", "task2", "task3"]:
        task_cases = [c for c in cases if c["task"] == task]
        summary[task] = {}
        for m in modalities:
            present = sum(1 for c in task_cases if c["missingness"].get(m, False))
            total = len(task_cases)
            summary[task][m] = {"present": present, "missing": total - present, "total": total}

    return {
        "modalities": modalities,
        "cases": [{"case_id": c["case_id"], "task": c["task"]} for c in cases],
        "matrix": matrix,
        "expected_absence": expected_absence,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# B5: Raincloud Plots
# ---------------------------------------------------------------------------

def tukey_box(values):
    """Compute Tukey five-number summary with 1.5*IQR whiskers."""
    if len(values) == 0:
        return None
    vs = sorted(values)
    n = len(vs)

    def percentile(p):
        if n == 1:
            return vs[0]
        k = (n - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vs[int(k)]
        return vs[f] * (c - k) + vs[c] * (k - f)

    q1 = percentile(25)
    med = percentile(50)
    q3 = percentile(75)
    iqr = q3 - q1
    lo_whisker = q1 - 1.5 * iqr
    hi_whisker = q3 + 1.5 * iqr

    # Actual whisker ends: most extreme data point within whisker range
    lo_actual = min(v for v in vs if v >= lo_whisker) if any(v >= lo_whisker for v in vs) else q1
    hi_actual = max(v for v in vs if v <= hi_whisker) if any(v <= hi_whisker for v in vs) else q3

    outliers = [v for v in vs if v < lo_whisker or v > hi_whisker]

    return {
        "q1": round(q1, 6),
        "median": round(med, 6),
        "q3": round(q3, 6),
        "iqr": round(iqr, 6),
        "whisker_lo": round(lo_actual, 6),
        "whisker_hi": round(hi_actual, 6),
        "outliers": [round(v, 6) for v in outliers],
    }


def silverman_bandwidth(values):
    """
    Silverman's rule of thumb:
      h = 0.9 * min(sigma, IQR/1.34) * n^(-1/5)
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    sigma = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0

    vs = sorted(values)
    def pct(p):
        k = (n - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vs[int(k)]
        return vs[f] * (c - k) + vs[c] * (k - f)
    iqr = pct(75) - pct(25)

    spread = min(sigma, iqr / 1.34) if iqr > 0 else sigma
    if spread == 0:
        spread = sigma
    if spread == 0:
        return 0.0

    h = 0.9 * spread * (n ** (-1.0 / 5.0))
    return h


def gaussian_kde(values, bandwidth, eval_points):
    """
    Gaussian KDE:
      f(x) = (1 / (n * h)) * sum_i K((x - x_i) / h)
      K(u) = (1 / sqrt(2*pi)) * exp(-u^2 / 2)
    """
    n = len(values)
    if n == 0 or bandwidth == 0:
        return [0.0] * len(eval_points)
    norm = 1.0 / (n * bandwidth * math.sqrt(2 * math.pi))
    densities = []
    for x in eval_points:
        s = 0.0
        for xi in values:
            u = (x - xi) / bandwidth
            s += math.exp(-0.5 * u * u)
        densities.append(norm * s)
    return densities


def compute_raincloud(cases, metric):
    """Compute raincloud (box + jittered dots + KDE) per task per target stratum."""
    task_order = ["task1", "task2", "task3"]
    target_order = {
        "task1": ["yes", "no"],
        "task2": ["active_surveillance", "continued_surveillance", "watchful_waiting", "active_treatment"],
        "task3": ["1", "0"],
    }

    result = {
        "metric": metric,
        "unit": METRIC_UNITS.get(metric, ""),
        "strata": [],
    }

    for task in task_order:
        task_cases = [c for c in cases if c["task"] == task]
        classes = target_order[task]
        for cls in classes:
            stratum_values = []
            for c in task_cases:
                if c["target"] == cls:
                    v = c["variables"].get(metric)
                    if v is not None:
                        stratum_values.append(v)
            n = len(stratum_values)
            if n == 0:
                result["strata"].append({
                    "task": task,
                    "target": cls,
                    "n": 0,
                    "box": None,
                    "kde": None,
                    "dots": [],
                    "note": "No data for this stratum.",
                })
                continue

            box = tukey_box(stratum_values)

            # Jittered dots (deterministic jitter based on index)
            dots = []
            for i, v in enumerate(sorted(stratum_values)):
                jitter = ((i * 0.6180339887498949) % 1.0 - 0.5)  # golden-ratio jitter in [-0.5, 0.5]
                dots.append({"value": round(v, 6), "jitter": round(jitter, 4)})

            # KDE
            bw = silverman_bandwidth(stratum_values)
            if n >= 2 and bw > 0:
                lo = min(stratum_values)
                hi = max(stratum_values)
                padding = (hi - lo) * 0.15 if hi > lo else bw * 3
                lo_eval = lo - padding
                hi_eval = hi + padding
                n_eval = 80
                eval_points = [lo_eval + (hi_eval - lo_eval) * i / (n_eval - 1) for i in range(n_eval)]
                densities = gaussian_kde(stratum_values, bw, eval_points)
                kde = {
                    "bandwidth": round(bw, 6),
                    "points": [{"x": round(x, 6), "y": round(y, 6)} for x, y in zip(eval_points, densities)],
                }
            else:
                kde = None

            result["strata"].append({
                "task": task,
                "target": cls,
                "n": n,
                "box": box,
                "kde": kde,
                "dots": dots,
            })

    return result


METRIC_UNITS = {
    "psa": "ng/mL",
    "psad": "ng/mL/cc",
    "vol": "cc",
    "age": "years",
}


# ---------------------------------------------------------------------------
# Composition JSON (B1 data)
# ---------------------------------------------------------------------------

def emit_composition(composition):
    """Emit composition data for B1."""
    out = {
        "tasks": composition,
        "method": "Ordered horizontal stacked bars. Exact counts and percentages per task per target class. No pie charts.",
        "target_definitions": {
            "task1": "Biopsy decision: yes/no",
            "task2": "Treatment management: active_surveillance / continued_surveillance / watchful_waiting / active_treatment",
            "task3": "BCR event: 1 (recurred) / 0 (censored)",
        },
    }
    with open(os.path.join(OUT_DIR, "composition.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Emitted composition.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("CHIMERA-Agent Phase B: Cohort Analytics Precompute Engine")
    print("=" * 70)

    # 1. Load all cases
    print("\n[1] Loading cohort...")
    cases = load_cohort()
    print(f"  Loaded {len(cases)} cases.")
    for task in ["task1", "task2", "task3"]:
        n = sum(1 for c in cases if c["task"] == task)
        print(f"    {task}: {n} cases")

    # 2. B1: Composition
    print("\n[2] Computing cohort composition (B1)...")
    composition = compute_composition(cases)
    for task, info in composition.items():
        print(f"  {task} (n={info['total']}):")
        for cls, data in info["classes"].items():
            print(f"    {cls}: n={data['n']} ({data['pct']}%)")
    emit_composition(composition)

    # 3. B2: PCA
    print("\n[3] Computing PCA (B2)...")
    # MRI PCA: all cases with MRI vectors
    mri_cases = [c for c in cases if c["mri_vec"] is not None]
    mri_vectors = [c["mri_vec"] for c in mri_cases]
    mri_ids = [c["case_id"] for c in mri_cases]
    mri_targets = [c["target"] for c in mri_cases]
    mri_tasks = [c["task"] for c in mri_cases]
    print(f"  MRI PCA: {len(mri_cases)} cases with MRI vectors (1024-d)")
    pca_mri = compute_pca(mri_vectors, mri_ids, mri_targets, mri_tasks)
    print(f"  MRI variance explained: PC1={pca_mri['variance_explained'][0]}%, PC2={pca_mri['variance_explained'][1]}%")
    print(f"  MRI cumulative 2-PC: {pca_mri['cumulative_2pc']}%")
    with open(os.path.join(OUT_DIR, "pca_mri.json"), "w", encoding="utf-8") as fh:
        json.dump(pca_mri, fh, indent=2)
    print(f"  Emitted pca_mri.json")

    # Biopsy PCA: all cases with biopsy vectors
    bx_cases = [c for c in cases if c["bx_vec"] is not None]
    bx_vectors = [c["bx_vec"] for c in bx_cases]
    bx_ids = [c["case_id"] for c in bx_cases]
    bx_targets = [c["target"] for c in bx_cases]
    bx_tasks = [c["task"] for c in bx_cases]
    print(f"  Biopsy PCA: {len(bx_cases)} cases with biopsy vectors (960-d)")
    pca_bx = compute_pca(bx_vectors, bx_ids, bx_targets, bx_tasks)
    print(f"  Biopsy variance explained: PC1={pca_bx['variance_explained'][0]}%, PC2={pca_bx['variance_explained'][1]}%")
    print(f"  Biopsy cumulative 2-PC: {pca_bx['cumulative_2pc']}%")
    with open(os.path.join(OUT_DIR, "pca_biopsy.json"), "w", encoding="utf-8") as fh:
        json.dump(pca_bx, fh, indent=2)
    print(f"  Emitted pca_biopsy.json")

    # 4. B3: Spearman Correlation + Ward
    print("\n[4] Computing Spearman correlation + Ward clustering (B3)...")
    correlation = compute_correlation(cases)
    with open(os.path.join(OUT_DIR, "correlation.json"), "w", encoding="utf-8") as fh:
        json.dump(correlation, fh, indent=2)
    print(f"  Emitted correlation.json ({correlation['n_variables']} variables)")
    print(f"  Variable order: {correlation['variables']}")
    if correlation["excluded"]:
        print(f"  Excluded (insufficient data): {correlation['excluded']}")

    # 5. B4: Missingness
    print("\n[5] Computing missingness grid (B4)...")
    missingness = compute_missingness(cases)
    with open(os.path.join(OUT_DIR, "missingness.json"), "w", encoding="utf-8") as fh:
        json.dump(missingness, fh, indent=2)
    print(f"  Emitted missingness.json ({len(cases)} cases x {len(missingness['modalities'])} modalities)")

    # 6. B5: Raincloud plots
    print("\n[6] Computing raincloud plots (B5)...")
    for metric in ["psa", "psad", "vol", "age"]:
        rc = compute_raincloud(cases, metric)
        out_path = os.path.join(OUT_DIR, f"raincloud_{metric}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(rc, fh, indent=2)
        print(f"  Emitted raincloud_{metric}.json ({len(rc['strata'])} strata)")

    print("\n" + "=" * 70)
    print("Phase B precompute complete. Artifacts in dashboard/cohort/")
    print("=" * 70)


if __name__ == "__main__":
    main()

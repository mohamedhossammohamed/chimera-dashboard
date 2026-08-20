#!/usr/bin/env python3
# [OFFICIAL: RESEARCHER-APPROVED] CHIMERA-Agent Phase A Bundle Generator
# Mathematical formulas: EAU 2026 Guidelines, UCSF CAPRA Nomogram, Cleveland-McGill perceptual principles
# [SUGGESTION: CO-PILOT] Implementation architecture and robust z-score formulation
"""CHIMERA-Agent Phase A: 9-View Clinical Bundle Generator.

Generates Markdown "Agentic Bundles" for LLM consumption from the CHIMERA-Agent
MICCAI 2026 challenge dataset. Pure Python standard library only.

Output: dashboard/bundles/task{1,2,3}/<case_id>.md
"""

import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "train_release"
OUTPUT_DIR = BASE_DIR / "dashboard" / "bundles"

MISSING = "[DATA NOT RECORDED]"

MONTHS_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.4375
LN2 = math.log(2.0)

# Clinical-data JSON filename per task
CLINICAL_DATA_FILE = {
    1: "prostate-biopsy-decision-clinical-data.json",
    2: "prostate-treatment-decision-clinical-data.json",
    3: "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value):
    """Return float if value is numeric or numeric string, else None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s or s.upper() in ('N/A', 'NOT AVAILABLE', 'MISSING'):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _fmt_num(value, decimals=4):
    """Format a float to a fixed number of decimals, stripping trailing zeros."""
    if value is None:
        return MISSING
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return MISSING
    formatted = f"{value:.{decimals}f}"
    return formatted


def _fmt_display(value):
    """Format a value for display: ints stay ints, floats keep natural representation."""
    if value is None:
        return MISSING
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return str(value)
    return str(value)


def _or_missing(value):
    """Return str(value) if value is not None/empty, else MISSING."""
    if value is None:
        return MISSING
    if isinstance(value, str) and value.strip() == "":
        return MISSING
    return str(value)


# ---------------------------------------------------------------------------
# DateParser
# ---------------------------------------------------------------------------

class DateParser:
    """Parse date strings in 'Mon YYYY' and 'DD Mon YYYY' formats."""

    @staticmethod
    def parse(date_str):
        if not date_str or not isinstance(date_str, str):
            return None
        s = date_str.strip()
        # Try "DD Mon YYYY"
        m = re.match(r"^(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})$", s)
        if m:
            day = int(m.group(1))
            mon = MONTHS_MAP.get(m.group(2))
            if mon is None:
                return None
            year = int(m.group(3))
            try:
                return datetime(year, mon, day)
            except ValueError:
                return None
        # Try "Mon YYYY"
        m = re.match(r"^([A-Za-z]{3})\s+(\d{4})$", s)
        if m:
            mon = MONTHS_MAP.get(m.group(1))
            if mon is None:
                return None
            year = int(m.group(2))
            try:
                return datetime(year, mon, 1)
            except ValueError:
                return None
        return None


# ---------------------------------------------------------------------------
# CohortStats
# ---------------------------------------------------------------------------

class CohortStats:
    """Collect cohort-wide metrics and compute robust z-scores and percentiles."""

    def __init__(self):
        self._values = {"psa": [], "psad": [], "vol": []}
        self._sorted = {}
        self._median = {}
        self._q1 = {}
        self._q3 = {}
        self._iqr = {}

    def add(self, metric, value):
        if value is not None and not (isinstance(value, float) and
                                      (math.isnan(value) or math.isinf(value))):
            self._values[metric].append(value)

    def finalize(self):
        for metric, vals in self._values.items():
            s = sorted(vals)
            self._sorted[metric] = s
            n = len(s)
            if n == 0:
                self._median[metric] = None
                self._q1[metric] = None
                self._q3[metric] = None
                self._iqr[metric] = 0.0
                continue
            self._median[metric] = statistics.median(s)
            self._q1[metric] = self._percentile_value(s, 25)
            self._q3[metric] = self._percentile_value(s, 75)
            self._iqr[metric] = self._q3[metric] - self._q1[metric]

    @staticmethod
    def _percentile_value(sorted_vals, pct):
        """Compute the p-th percentile (0-100) via linear interpolation."""
        n = len(sorted_vals)
        if n == 0:
            return None
        if n == 1:
            return sorted_vals[0]
        pos = (pct / 100.0) * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return sorted_vals[lo]
        frac = pos - lo
        return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    def percentile(self, metric, value):
        """Percentile of value within the cohort (linear interpolation for ties)."""
        s = self._sorted.get(metric, [])
        n = len(s)
        if n == 0 or value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        rank_below = sum(1 for v in s if v < value)
        count_equal = sum(1 for v in s if v == value)
        pct = ((rank_below + 0.5 * count_equal) / n) * 100.0
        return int(round(pct))

    def robust_z(self, metric, value):
        """Robust z-score: (x - median) / (IQR / 1.349)."""
        if value is None:
            return None
        med = self._median.get(metric)
        iqr = self._iqr.get(metric, 0.0)
        if med is None:
            return None
        if iqr == 0:
            return 0.0
        return (value - med) / (iqr / 1.349)

    def n(self, metric):
        return len(self._values.get(metric, []))


# ---------------------------------------------------------------------------
# PSAKinetics
# ---------------------------------------------------------------------------

class PSAKinetics:
    """Compute PSA Velocity (PSAV) and PSA Doubling Time (PSADT)."""

    def __init__(self, psa_trend):
        self.points = []
        if isinstance(psa_trend, list):
            for entry in psa_trend:
                if not isinstance(entry, dict):
                    continue
                date = DateParser.parse(entry.get("date"))
                val = _safe_float(entry.get("val"))
                if date is not None and val is not None and val > 0:
                    self.points.append((date, val))
        self.points.sort(key=lambda p: p[0])

    @property
    def n_points(self):
        return len(self.points)

    @property
    def span_months(self):
        if len(self.points) < 2:
            return 0.0
        delta = self.points[-1][0] - self.points[0][0]
        return delta.days / DAYS_PER_MONTH

    @property
    def insufficient(self):
        return len(self.points) < 3 or self.span_months < 6.0

    def psav(self):
        """PSA Velocity: linear regression slope of PSA vs time (years)."""
        if self.insufficient:
            return None
        ts = [(p[0] - self.points[0][0]).days / DAYS_PER_YEAR for p in self.points]
        vs = [p[1] for p in self.points]
        n = len(ts)
        t_mean = sum(ts) / n
        v_mean = sum(vs) / n
        num = sum((ts[i] - t_mean) * (vs[i] - v_mean) for i in range(n))
        den = sum((ts[i] - t_mean) ** 2 for i in range(n))
        if den == 0:
            return None
        return num / den

    def psadt(self):
        """PSA Doubling Time: ln(2) / slope of ln(PSA) vs time (months)."""
        if self.insufficient:
            return None
        ts = [(p[0] - self.points[0][0]).days / DAYS_PER_MONTH for p in self.points]
        lvs = [math.log(p[1]) for p in self.points]
        n = len(ts)
        t_mean = sum(ts) / n
        lv_mean = sum(lvs) / n
        num = sum((ts[i] - t_mean) * (lvs[i] - lv_mean) for i in range(n))
        den = sum((ts[i] - t_mean) ** 2 for i in range(n))
        if den == 0:
            return None
        slope = num / den
        if slope <= 0:
            return None  # signal "non-rising" — JSON collapses JS Infinity to null anyway
        return round(LN2 / slope, 1)  # round to 1 decimal to match JS

    def trajectory(self, psadt_val):
        if psadt_val is None or (isinstance(psadt_val, float) and math.isnan(psadt_val)):
            if self.insufficient:
                return MISSING
            return "Stable/Declining"
        if psadt_val is None:
            return "Stable/Declining"
        if psadt_val < 6:
            return "Aggressive"
        if psadt_val < 12:
            return "Rapid"
        if psadt_val < 24:
            return "Moderate"
        return "Indolent"


# ---------------------------------------------------------------------------
# EAURiskTier
# ---------------------------------------------------------------------------

def _parse_ct(ct):
    """Parse clinical T stage string into (major, minor) tuple, or None."""
    if not ct or not isinstance(ct, str):
        return None
    m = re.match(r"^[cCpP]?T(\d)([a-cA-C]?)$", ct.strip())
    if not m:
        return None
    major = int(m.group(1))
    minor = 0
    if m.group(2):
        minor = ord(m.group(2).lower()) - ord("a") + 1
    return (major, minor)


class EAURiskTier:
    """Deterministic EAU 2026 risk tier classification."""

    LOW_CT = {(1, 1), (1, 2), (1, 3), (2, 1)}

    @classmethod
    def classify(cls, psa, bx_isup, ct, high_risk_patterns):
        psa = _safe_float(psa)
        bx_isup = bx_isup if isinstance(bx_isup, int) and not isinstance(bx_isup, bool) else None
        if bx_isup is None:
            bx_isup = _safe_float(bx_isup)
            bx_isup = int(bx_isup) if bx_isup is not None else None
        ct_rank = _parse_ct(ct)
        has_high_risk = bool(high_risk_patterns) and str(high_risk_patterns).strip() not in ("", "None", "null")

        criteria = []

        # EAU 2026 Risk Classification (5 tiers — no "Very High" tier exists)
        # Order: Locally advanced → High → Unfavorable Intermediate → Favorable Intermediate → Low

        # 1. Locally Advanced: cT3-4 (takes precedence over all other criteria)
        if ct_rank is not None and ct_rank[0] >= 3:
            criteria.append(f"Clinical stage {ct} (cT3-4)")
            return "Locally Advanced", "; ".join(criteria)

        # 2. High: ISUP 4/5 OR PSA > 20 OR cT2c
        if bx_isup is not None and bx_isup in (4, 5):
            criteria.append(f"ISUP {bx_isup}")
            return "High", "; ".join(criteria)
        if psa is not None and psa > 20:
            criteria.append(f"PSA {psa} > 20")
            return "High", "; ".join(criteria)
        if ct_rank is not None and ct_rank == (2, 3):
            criteria.append(f"Clinical stage {ct} (cT2c)")
            return "High", "; ".join(criteria)

        if bx_isup is None and ct_rank is None and psa is None:
            return MISSING, "Insufficient data: ISUP grade, clinical T stage, and PSA all missing"

        # 3. Unfavorable Intermediate: ISUP 3 OR (ISUP 2 AND PSA 10-20) OR cT2b
        if bx_isup is not None and bx_isup == 3:
            criteria.append(f"ISUP 3")
            return "Unfavorable Intermediate", "; ".join(criteria)
        # 3b. Favourable Intermediate: ISUP 1 AND PSA 10-20 AND cT1-2a AND no high-risk
        if psa is not None and 10 <= psa <= 20 and bx_isup == 1:
            if ct_rank in cls.LOW_CT and not has_high_risk:
                criteria.append(f"PSA {psa} 10-20; ISUP 1; {ct}; no high-risk patterns")
                return "Favorable Intermediate", "; ".join(criteria)
        if psa is not None and 10 <= psa <= 20:
            if bx_isup is not None and bx_isup == 2:
                criteria.append(f"PSA {psa} 10-20; ISUP 2")
                return "Unfavorable Intermediate", "; ".join(criteria)
            if bx_isup is None:
                criteria.append(f"PSA {psa} 10-20")
                return "Unfavorable Intermediate", "; ".join(criteria)
        if ct_rank is not None and ct_rank == (2, 2):
            criteria.append(f"Clinical stage {ct} (cT2b)")
            return "Unfavorable Intermediate", "; ".join(criteria)

        # 4. Favorable Intermediate: ISUP 2 AND PSA < 10 AND cT1-2a AND no high-risk
        if psa is not None and psa < 10 and bx_isup == 2:
            if ct_rank in cls.LOW_CT and not has_high_risk:
                criteria.append(f"PSA {psa} < 10; ISUP 2; {ct}; no high-risk patterns")
                return "Favorable Intermediate", "; ".join(criteria)

        # 5. Low: ISUP 1 AND PSA < 10 AND cT1-2a
        if psa is not None and psa < 10 and bx_isup == 1:
            if ct_rank in cls.LOW_CT:
                criteria.append(f"PSA {psa} < 10; ISUP 1; {ct}")
                return "Low", "; ".join(criteria)

        # Partial classification fallback
        if psa is not None and psa < 10 and bx_isup == 2 and has_high_risk:
            criteria.append(f"PSA {psa} < 10; ISUP 2; high-risk patterns present — upgraded")
            return "Unfavorable Intermediate", "; ".join(criteria)

        return MISSING, "Insufficient staging data for tier classification"


# ---------------------------------------------------------------------------
# CAPRAScore
# ---------------------------------------------------------------------------

class CAPRAScore:
    """UCSF CAPRA nomogram additive score (capped at 10).

    Published components (Cooperberg et al., UCSF):
      Age at diagnosis:      <50 -> 0, >=50 -> 1                         (max 1)
      PSA at diagnosis:      <=6 -> 0, 6.01-10 -> 1, 10.01-20 -> 2,
                             20.01-30 -> 3, >30 -> 4                     (max 4)
      Gleason (prim/sec):    no pattern 4/5 -> 0, secondary 4/5 -> 1,
                             primary 4/5 -> 3                             (max 3)
      Clinical T stage:      T1-T2 -> 0, T3a -> 1                        (max 1)
      % positive cores:      <34% -> 0, >=34% -> 1                       (max 1)
      Total max = 1+4+3+1+1 = 10
    """

    @staticmethod
    def compute(age, psa, bx_gl_prim, bx_gl_sec, bx_gl_tert, ct, cores_positive, cores_total):
        breakdown = []
        total = 0

        # Age (max 1) — published CAPRA: binary <50 / >=50
        age_val = _safe_float(age)
        if age_val is None:
            pts = 0
            breakdown.append(f"Age [+{pts}](imputed: data not recorded)")
        elif age_val < 50:
            pts = 0
            breakdown.append(f"Age [+{pts}]")
        else:
            pts = 1
            breakdown.append(f"Age [+{pts}]")
        total += pts

        # PSA (max 4)
        psa_val = _safe_float(psa)
        if psa_val is None:
            pts = 0
            breakdown.append(f"PSA [+{pts}](imputed: data not recorded)")
        elif psa_val <= 6:
            pts = 0
            breakdown.append(f"PSA [+{pts}]")
        elif psa_val <= 10:
            pts = 1
            breakdown.append(f"PSA [+{pts}]")
        elif psa_val <= 20:
            pts = 2
            breakdown.append(f"PSA [+{pts}]")
        elif psa_val <= 30:
            pts = 3
            breakdown.append(f"PSA [+{pts}]")
        else:
            pts = 4
            breakdown.append(f"PSA [+{pts}]")
        total += pts

        # Gleason (max 3) — published CAPRA: primary/secondary pattern 4/5
        gl_prim = _safe_float(bx_gl_prim)
        gl_sec = _safe_float(bx_gl_sec)
        if gl_prim is None or gl_sec is None:
            pts = 0
            breakdown.append(f"Gleason [+{pts}](imputed: data not recorded)")
        else:
            gp = int(gl_prim)
            gs = int(gl_sec)
            if gp >= 4:
                pts = 3  # primary pattern 4 or 5
            elif gs >= 4:
                pts = 1  # secondary pattern 4 or 5 (primary is 3)
            else:
                pts = 0  # no pattern 4 or 5
            breakdown.append(f"Gleason [+{pts}] ({gp}+{gs})")
        total += pts

        # Clinical T stage (max 1) — published CAPRA: T1-T2 -> 0, T3a -> 1
        ct_rank = _parse_ct(ct)
        if ct_rank is None:
            pts = 0
            breakdown.append(f"T-Stage [+{pts}](imputed: data not recorded)")
        elif ct_rank[0] <= 2:
            pts = 0
            breakdown.append(f"T-Stage [+{pts}] ({ct})")
        else:
            pts = 1  # T3a or higher
            breakdown.append(f"T-Stage [+{pts}] ({ct})")
        total += pts

        # % positive cores (max 1) — published CAPRA: <34% -> 0, >=34% -> 1
        cp = _safe_float(cores_positive)
        ctot = _safe_float(cores_total)
        if cp is None or ctot is None or ctot == 0:
            pts = 0
            breakdown.append(f"Cores [+{pts}](imputed: data not recorded)")
        else:
            pct = (cp / ctot) * 100.0
            if pct < 34:
                pts = 0
            else:
                pts = 1
            breakdown.append(f"Cores [+{pts}] ({pct:.1f}%)")
        total += pts

        score = min(total, 10)
        return score, ", ".join(breakdown)


# ---------------------------------------------------------------------------
# CAPRAS_Score (post-surgical, Task 3 only)
# ---------------------------------------------------------------------------

class CAPRAS_Score:
    """UCSF CAPRA-S (post-surgical) nomogram additive score (capped at 12).

    # [OFFICIAL: RESEARCHER-APPROVED] UCSF CAPRA-S (Cooperberg et al., JAMA 2006)
    # [SUGGESTION: CO-PILOT] Regex extraction implementation from surgical pathology reports

    Published components (Cooperberg et al., Cancer 2011; 117:5039-5046,
    PMCID: PMC3170662, doi:10.1002/cncr.26169):
      PSA (pre-treatment):    0-6 -> 0, 6.01-10 -> 1, 10.01-20 -> 2, >20 -> 3       (max 3)
      Pathologic Gleason:     2-6 -> 0, 3+4 -> 1, 4+3 -> 2, 8-10 -> 3               (max 3)
      Surgical Margin:        negative -> 0, positive -> 2                           (max 2)
      ECE:                    absent -> 0, present -> 1                              (max 1)
      SVI:                    absent -> 0, present -> 2                              (max 2)
      LNI:                    negative -> 0, positive -> 1                           (max 1)
      Total max = 3+3+2+1+2+1 = 12

    Weights verified against the published Table 1 in Cooperberg et al. 2011 and
    the multi-institutional validation (Hinotsu et al., J Urol 2012). Both the
    Python and JavaScript implementations use these authoritative values.
    """

    # Clause delimiters — mirrors JS clauseBounds(): . ; ! ? , \n
    _CLAUSE_DELIM_RE = re.compile(r'[.;!?,\n]')

    @staticmethod
    def _clause_bounds(lower, pos):
        """Find the clause boundaries around `pos` in lowercased text.

        Clauses are delimited by [.;!?,\\n] (period, semicolon, bang, question,
        comma, newline). Returns (start, end) half-open interval. Mirrors the JS
        `clauseBounds()` helper exactly so negation words from adjacent
        clauses cannot bleed into the current field's context window.
        """
        delims = CAPRAS_Score._CLAUSE_DELIM_RE
        start = 0
        end = len(lower)
        for i in range(pos - 1, -1, -1):
            if delims.match(lower[i]):
                start = i + 1
                break
        for i in range(pos, len(lower)):
            if delims.match(lower[i]):
                end = i
                break
        return start, end

    @staticmethod
    def _is_negated(lower, start, end, negation_re, evidence_neg_re):
        """Check whether a mention at [start, end) is negated.

        Scans within the current clause (delimited by . ; ! ? , \\n) plus a
        ±20 char window, but never crosses clause boundaries. Special case:
        "no evidence of" can look back past one comma boundary (mirrors JS
        `isNegated()` + EVIDENCE_NEG_RE).
        """
        clause_start, clause_end = CAPRAS_Score._clause_bounds(lower, start)
        win_start = max(clause_start, start - 20)
        win_end = min(clause_end, end + 20)
        window = lower[win_start:win_end]
        if negation_re.search(window):
            return True
        # "no evidence of" may span a single comma boundary (not periods).
        # CRITICAL: Only span if "no evidence of" is at the END of the previous
        # clause (nothing meaningful follows it). If a field trigger follows
        # "no evidence of" in the previous clause, that trigger already consumed
        # the negation — spanning would falsely negate the current field.
        # E.g. "No evidence of ECE, SVI present" must NOT negate SVI.
        if clause_start > 0 and lower[clause_start - 1] == ',':
            prev_start, prev_end = CAPRAS_Score._clause_bounds(lower, clause_start - 1)
            prev_text = lower[prev_start:prev_end]
            m = evidence_neg_re.search(prev_text)
            if m:
                after_evidence = prev_text[m.end():].strip()
                if after_evidence == '' or re.match(r'^[,;.\s]*$', after_evidence):
                    return True
        return False

    @staticmethod
    def _find_list_negation_ranges(lower):
        """Find ranges where list negation propagates.

        "no evidence of ECE, SVI, LVI" → all three are negated.
        Scans for negation phrases followed by comma-separated terms until
        a sentence boundary (period, semicolon, or end of text).
        Mirrors JS `findListNegationRanges()`.
        """
        ranges = []
        list_neg_re = re.compile(
            r'\b(?:no\s+evidence\s+of|no\s+sign\s+of|free\s+of)\b',
            re.IGNORECASE,
        )
        for m in list_neg_re.finditer(lower):
            phrase_start = m.start()
            phrase_end = m.end()
            pos = phrase_end
            while pos < len(lower) and lower[pos].isspace():
                pos += 1
            if pos >= len(lower):
                continue
            list_end = pos
            while list_end < len(lower):
                ch = lower[list_end]
                if ch in '.;\n':
                    break
                list_end += 1
            ranges.append((phrase_start, list_end))
        return ranges

    @staticmethod
    def _classify_field(lower, trigger_re, positive_re, check_numeric,
                        not_assessable_re, not_assessed_re, numeric_neg_re,
                        negation_re, evidence_neg_re, absent_label, present_label,
                        exclude_re=None, list_neg_ranges=None):
        """Negation-aware classification of a binary pathology field.

        Two-phase scan mirroring the JS `classifyField()`:
          Phase 1: Find all mentions of the field trigger terms.
          Phase 2: For each mention, check the clause-bounded ±20 char window
                   for not-assessable / not-assessed / numeric negation /
                   negation scope / positive keywords.

        Aggregation (any-positive-wins, matching JS):
          - any un-negated positive mention → present
          - else any negated mention → absent
          - else None (not assessable / unextractable)

        `not_assessable_re` (e.g. no lymph node dissection) and
        `not_assessed_re` (e.g. not noted / not evaluated) cause the mention
        to be skipped entirely — neither positive nor negated.
        """
        has_positive = False
        has_negated = False
        for m in trigger_re.finditer(lower):
            start = m.start()
            end = m.end()
            clause_start, clause_end = CAPRAS_Score._clause_bounds(lower, start)
            win_start = max(clause_start, start - 20)
            win_end = min(clause_end, end + 20)
            window = lower[win_start:win_end]

            # Not assessable (e.g. no dissection) → skip this mention entirely
            if not_assessable_re is not None and not_assessable_re.search(window):
                continue
            # Distant metastasis exclusion (e.g. "bone metastasis" → not LNI)
            if exclude_re is not None and exclude_re.search(window):
                continue
            # List negation: "no evidence of ECE, SVI, LVI" → all negated
            if list_neg_ranges is not None and any(
                r[0] <= start and end <= r[1] for r in list_neg_ranges
            ):
                has_negated = True
                continue
            # Not assessed (e.g. not noted / not evaluated) → skip this mention
            if not_assessed_re is not None and not_assessed_re.search(window):
                continue
            # Numeric negation: "0/15 lymph nodes positive" → absent
            if check_numeric and numeric_neg_re.search(window):
                has_negated = True
                continue

            if CAPRAS_Score._is_negated(lower, start, end, negation_re, evidence_neg_re):
                has_negated = True
            elif positive_re.search(window):
                has_positive = True

        if has_positive:
            return present_label
        if has_negated:
            return absent_label
        return None

    @staticmethod
    def _extract_from_surgical_report(text):
        """Extract CAPRA-S components from surgical pathology report text.

        Returns dict with keys: gleason_prim, gleason_sec, margin, ece, svi, lni.
        Values are 'present'/'absent' for binary features, ints for Gleason,
        or None if unextractable ([DATA NOT RECORDED]).

        Unextractable component = None, never assumed negative.

        Negation-aware parsing: for each binary field, clauses matching the field
        trigger are checked for negation scope (no/not/none/without/free of/no
        evidence of) before positive keywords. Negation words override positive
        keywords to handle bidirectional clinical negation:
          - Pre-negation:  "no positive lymph nodes", "no evidence of SVI"
          - Post-negation: "lymph nodes not involved", "metastasis not present"
          - Numeric:       "0/15 lymph nodes positive"
        """
        result = {
            'gleason_prim': None,
            'gleason_sec': None,
            'margin': None,
            'ece': None,
            'svi': None,
            'lni': None,
            'lvi': None,
        }

        if not text or not isinstance(text, str):
            return result

        # Full lowercased report — mirrors JS `parseSurgicalPathology` which
        # scans the entire lowercased text with clause-bounded ±20 windows
        # (no upfront clause split). Clause isolation is enforced inside
        # `_classify_field` via `_clause_bounds` using delimiters [.;!?,\n].
        lower = text.lower()
        list_neg_ranges = CAPRAS_Score._find_list_negation_ranges(lower)

        # --- Pathologic Gleason: /Gleason (?:score|pattern)? (\d)\+(\d)/i ---
        # Highest-grade selection (ISUP 2019 / Epstein 2005 / Kunz 2009):
        # highest sum (prim+sec) correlates best with biochemical recurrence;
        # tiebreak by higher primary (4+3 > 3+4).
        # Optional intervening word "score" or "pattern" handles all clinical
        # phrasings: "Gleason 3+4", "Gleason score 3+4", "Gleason pattern 3+4".
        gleason_matches = re.findall(r'Gleason\s+(?:score\s+|pattern\s+)?[:=]?\s*(\d{1,2})\s*\+\s*(\d{1,2})', text, re.IGNORECASE)
        if gleason_matches:
            best = max(gleason_matches, key=lambda mt: (int(mt[0]) + int(mt[1]), int(mt[0])))
            result['gleason_prim'] = int(best[0])
            result['gleason_sec'] = int(best[1])

        # Negation scope detector — mirrors JS NEGATION_RE exactly.
        # \bclear(?!ly) prevents matching "clearly" (e.g. "clearly present"
        # is affirmative, not negated). `no evidence of` / `negative for` are
        # covered by the bare `no` / `negative` word boundaries.
        negation_re = re.compile(
            r'\b(?:no|not|without|negative|absent|none|free|clear(?!ly)|unremarkable|uninvolved)\b|\(-\)',
            re.IGNORECASE,
        )

        # Extended negation: "no evidence of" can span a comma before the field
        # name (mirrors JS EVIDENCE_NEG_RE).
        evidence_neg_re = re.compile(r'\bno\s+evidence\s+of\b', re.IGNORECASE)

        # Numeric negation: "0/15", "0 / 15" — zero positive out of N sampled.
        numeric_neg_re = re.compile(r'\b0\s*/\s*\d+\b')

        # Not-assessable patterns (LNI: no dissection performed → skip mention).
        # Mirrors JS NOT_ASSESSABLE_RE (includes "not dissected").
        not_assessable_re = re.compile(
            r'\b(?:not removed|no lymph nodes?\s+(?:were\s+)?removed|no lymph node dissection|not dissected)\b',
            re.IGNORECASE,
        )

        # Not-assessed patterns (e.g. "not noted", "not evaluated") → skip
        # mention entirely (neither positive nor negated). Mirrors the JS
        # NOT_ASSESSED_RE added by FIX-LVI.
        not_assessed_re = re.compile(
            r'\b(?:not noted|not assessed|not evaluated|not reported|not applicable)\b',
            re.IGNORECASE,
        )

        # Distant metastasis patterns — exclude from LNI classification
        distant_mets_re = re.compile(
            r'\b(?:distant|remote|bone|visceral|hepatic|pulmonary|brain|liver|lung)\s+metastas(?:is|es)\b'
            r'|\bmetastas(?:is|es)\s+(?:to\s+)?(?:distant|remote|bone|visceral|hepatic|pulmonary|brain|liver|lung)\b',
            re.IGNORECASE,
        )

        # --- Surgical Margin — maps present→'positive', absent→'negative' ---
        # R0 is a negative margin (no residual tumor). R1 is positive.
        # Trigger/positive regexes mirror JS exactly.
        margin_raw = CAPRAS_Score._classify_field(
            lower,
            re.compile(r'\bmargins?\b|\br0\b|\br1\b', re.IGNORECASE),
            re.compile(r'\b(?:positive|r1|involved)\b', re.IGNORECASE),
            False,
            None,
            not_assessed_re,
            numeric_neg_re,
            negation_re,
            evidence_neg_re,
            'absent',
            'present',
            list_neg_ranges=list_neg_ranges,
        )
        # Handle R0 explicitly: if "R0" appears and no R1/positive, it's negative
        if margin_raw is None and re.search(r'\br0\b', text, re.IGNORECASE) \
                and not re.search(r'\br1\b', text, re.IGNORECASE):
            result['margin'] = 'negative'
        else:
            result['margin'] = (
                'positive' if margin_raw == 'present'
                else 'negative' if margin_raw == 'absent'
                else None
            )

        # --- ECE (Extraprostatic / Extracapsular Extension) ---
        result['ece'] = CAPRAS_Score._classify_field(
            lower,
            re.compile(
                r'\b(?:extraprostatic extensions?|extracapsular extensions?|ece)\b',
                re.IGNORECASE,
            ),
            re.compile(r'\b(?:present|positive|identified|noted|invasion|invasive|seen|involvement|yes)\b|\(\+\)', re.IGNORECASE),
            False,
            None,
            not_assessed_re,
            numeric_neg_re,
            negation_re,
            evidence_neg_re,
            'absent',
            'present',
            list_neg_ranges=list_neg_ranges,
        )

        # --- SVI (Seminal Vesicle Invasion) ---
        result['svi'] = CAPRAS_Score._classify_field(
            lower,
            re.compile(r'\b(?:seminal vesicles?|svi)\b', re.IGNORECASE),
            re.compile(r'\b(?:invasion|invaded|positive|present|seen|involvement|yes)\b|\(\+\)', re.IGNORECASE),
            False,
            None,
            not_assessed_re,
            numeric_neg_re,
            negation_re,
            evidence_neg_re,
            'absent',
            'present',
            list_neg_ranges=list_neg_ranges,
        )

        # --- LNI (Lymph Node Involvement) ---
        # Trigger: lymph node|nodal|LNI|metastasis (NOT lymphovascular — LVI is
        # distinct). Not-assessable (no dissection) → skip. Numeric negation
        # (0/N) → absent.
        result['lni'] = CAPRAS_Score._classify_field(
            lower,
            re.compile(r'\b(?:lymph nodes?|nodal|lni|metastas(?:is|es))\b', re.IGNORECASE),
            re.compile(
                r'\b(?:positive|involved|metasta(?:sis|ses|tic)|present|seen|involvement|yes)\b|\(\+\)',
                re.IGNORECASE,
            ),
            True,
            not_assessable_re,
            not_assessed_re,
            numeric_neg_re,
            negation_re,
            evidence_neg_re,
            'absent',
            'present',
            distant_mets_re,
            list_neg_ranges=list_neg_ranges,
        )

        # --- LVI (Lymphovascular Invasion) ---
        result['lvi'] = CAPRAS_Score._classify_field(
            lower,
            re.compile(r'\b(?:lymphovascular|vascular invasion|lvi|angiolymphatic)\b', re.IGNORECASE),
            re.compile(r'\b(?:present|positive|invasion|invasive|noted|identified|seen|yes)\b|\(\+\)', re.IGNORECASE),
            False,
            None,
            not_assessed_re,
            numeric_neg_re,
            negation_re,
            evidence_neg_re,
            'absent',
            'present',
            list_neg_ranges=list_neg_ranges,
        )


        return result

    @staticmethod
    def compute(psa, surgical_pathology_report):
        """Compute CAPRA-S score from PSA and surgical pathology report text.

        Returns (score, breakdown_str, imputed_list).
        Unextractable components are assigned 0 points and flagged as imputed.
        """
        comps = CAPRAS_Score._extract_from_surgical_report(surgical_pathology_report)
        breakdown = []
        imputed = []
        total = 0

        # PSA (max 3) — published CAPRA-S: <=6 -> 0, 6.01-10 -> 1, 10.01-20 -> 2, >20 -> 3
        psa_val = _safe_float(psa)
        if psa_val is None:
            pts = 0
            imputed.append("PSA")
            breakdown.append(f"PSA [+{pts}](imputed)")
        elif psa_val <= 6:
            pts = 0
            breakdown.append(f"PSA [+{pts}]")
        elif psa_val <= 10:
            pts = 1
            breakdown.append(f"PSA [+{pts}]")
        elif psa_val <= 20:
            pts = 2
            breakdown.append(f"PSA [+{pts}]")
        else:
            pts = 3
            breakdown.append(f"PSA [+{pts}]")
        total += pts

        # Pathologic Gleason (max 3) — published CAPRA-S: 2-6 -> 0, 3+4 -> 1, 4+3 -> 2, 8-10 -> 3
        gp = comps['gleason_prim']
        gs = comps['gleason_sec']
        if gp is None or gs is None:
            pts = 0
            imputed.append("pGleason")
            breakdown.append(f"pGleason [+{pts}](imputed)")
        else:
            gsum = gp + gs
            if gsum <= 6:
                pts = 0
            elif gp == 3 and gs == 4:
                pts = 1
            elif gp == 4 and gs == 3:
                pts = 2
            else:  # gsum >= 8
                pts = 3
            breakdown.append(f"pGleason [+{pts}] ({gp}+{gs})")
        total += pts

        # Surgical Margin (max 2) — published CAPRA-S: negative -> 0, positive -> 2
        margin = comps['margin']
        if margin is None:
            pts = 0
            imputed.append("Margin")
            breakdown.append(f"Margin [+{pts}](imputed)")
        elif margin == 'positive':
            pts = 2
            breakdown.append(f"Margin [+{pts}]")
        else:  # negative
            pts = 0
            breakdown.append(f"Margin [+{pts}]")
        total += pts

        # ECE (max 1) — published CAPRA-S: absent -> 0, present -> 1
        ece = comps['ece']
        if ece is None:
            pts = 0
            imputed.append("ECE")
            breakdown.append(f"ECE [+{pts}](imputed)")
        elif ece == 'present':
            pts = 1
            breakdown.append(f"ECE [+{pts}]")
        else:  # absent
            pts = 0
            breakdown.append(f"ECE [+{pts}]")
        total += pts

        # SVI (max 2) — published CAPRA-S: absent -> 0, present -> 2
        svi = comps['svi']
        if svi is None:
            pts = 0
            imputed.append("SVI")
            breakdown.append(f"SVI [+{pts}](imputed)")
        elif svi == 'present':
            pts = 2
            breakdown.append(f"SVI [+{pts}]")
        else:  # absent
            pts = 0
            breakdown.append(f"SVI [+{pts}]")
        total += pts

        # LNI (max 1) — published CAPRA-S: negative -> 0, positive -> 1
        lni = comps['lni']
        if lni is None:
            pts = 0
            imputed.append("LNI")
            breakdown.append(f"LNI [+{pts}](imputed)")
        elif lni == 'present':
            pts = 1
            breakdown.append(f"LNI [+{pts}]")
        else:  # absent
            pts = 0
            breakdown.append(f"LNI [+{pts}]")
        total += pts

        score = min(total, 12)
        return score, ", ".join(breakdown), imputed


# ---------------------------------------------------------------------------
# EmbeddingAudit
# ---------------------------------------------------------------------------

class EmbeddingAudit:
    """Compute L2 norm, mean, std, min, max from neural representation lists."""

    @staticmethod
    def audit(vectors):
        """Audit a list of vectors (list of lists of floats).

        Returns dict with: available, shape, l2_norm, mean, std, min, max.
        """
        if not vectors or not isinstance(vectors, list) or len(vectors) == 0:
            return {
                "available": False,
                "shape": "[0]",
                "l2_norm": "0",
                "mean": MISSING,
                "std": MISSING,
                "min": MISSING,
                "max": MISSING,
            }
        num_vecs = len(vectors)
        dim = len(vectors[0]) if vectors[0] else 0
        flat = []
        for vec in vectors:
            if isinstance(vec, list):
                flat.extend(float(x) for x in vec)
        n = len(flat)
        if n == 0:
            return {
                "available": False,
                "shape": f"[{num_vecs}, 0]",
                "l2_norm": "0",
                "mean": MISSING,
                "std": MISSING,
                "min": MISSING,
                "max": MISSING,
            }
        sq_sum = sum(x * x for x in flat)
        l2 = math.sqrt(sq_sum)
        mean = sum(flat) / n
        var = sum((x - mean) ** 2 for x in flat) / n
        std = math.sqrt(var)
        return {
            "available": True,
            "shape": f"[{num_vecs}, {dim}]",
            "l2_norm": f"{l2:.2f}",
            "mean": f"{mean:.4f}",
            "std": f"{std:.4f}",
            "min": f"{min(flat):.4f}",
            "max": f"{max(flat):.4f}",
        }


# ---------------------------------------------------------------------------
# BundleRenderer
# ---------------------------------------------------------------------------

class BundleRenderer:
    """Assemble the 9-View Markdown bundle from all computed components."""

    def __init__(self, case_id, task, sp, clinical, neural, cohort):
        self.case_id = case_id
        self.task = task
        self.sp = sp
        self.clinical = clinical
        self.neural = neural
        self.cohort = cohort

    def _get(self, key, default=None):
        return self.sp.get(key, default)

    def _clinical(self, key, default=None):
        return self.clinical.get(key, default) if self.clinical else default

    def _compute_psad(self):
        psa = _safe_float(self._get("psa"))
        vol = _safe_float(self._get("vol"))
        if psa is not None and vol is not None and vol > 0:
            return psa / vol
        return None

    def _eval_psa(self):
        psa = _safe_float(self._get("psa"))
        if psa is None:
            return MISSING
        if psa < 4:
            return "Normal"
        if psa <= 10:
            return "Borderline"
        return "Elevated"

    def _eval_vol(self):
        vol = _safe_float(self._get("vol"))
        if vol is None:
            return MISSING
        if vol < 30:
            return "Normal"
        return "Enlarged"

    def _eval_psad(self, psad):
        if psad is None:
            return MISSING
        if psad < 0.15:
            return "Low Risk"
        return "High Risk"

    def _eval_pirads(self):
        pirads = self._get("pirads")
        if pirads is None or str(pirads).strip() in ("", "NA", "null"):
            return MISSING
        try:
            p = int(pirads)
        except (ValueError, TypeError):
            return MISSING
        if p in (1, 2):
            return "Benign"
        if p == 3:
            return "Equivocal"
        if p in (4, 5):
            return "Malignant"
        return MISSING

    def _eval_dre_ct(self):
        dre = self._get("dre")
        ct = self._get("ct")
        ct_rank = _parse_ct(ct)
        parts = []
        if ct_rank is not None:
            if ct_rank[0] <= 2:
                parts.append("Organ Confined (cT1-cT2)")
            else:
                parts.append("Advanced (>=cT3)")
        if dre is not None and isinstance(dre, str):
            dl = dre.lower()
            if "abnormal" in dl or "nodus" in dl or "suspicious" in dl:
                parts.append("DRE: Abnormal")
            elif "normal" in dl and "abnormal" not in dl:
                parts.append("DRE: Normal")
            elif "not done" in dl:
                parts.append("DRE: Not done")
        if not parts:
            return MISSING
        return "; ".join(parts)

    def _eval_isup(self):
        isup = self._get("bx_isup")
        if isup is None:
            return MISSING
        try:
            i = int(isup)
        except (ValueError, TypeError):
            return MISSING
        if i == 1:
            return "Low"
        if i in (2, 3):
            return "Intermediate"
        if i in (4, 5):
            return "High"
        return MISSING

    def _eval_fh(self):
        fh = self._clinical("family_history")
        if fh is None:
            return MISSING
        fh_str = str(fh).strip()
        if fh_str == "Yes":
            return "Familial Risk Flag"
        if fh_str == "No":
            return "No familial risk"
        if fh_str == "Unknown":
            return MISSING
        fl = fh_str.lower()
        if "no family history" in fl or "no history" in fl:
            return "No familial risk"
        if "family history" in fl and "no" not in fl:
            return "Familial Risk Flag"
        return MISSING

    def _eval_cspca(self):
        cspca = _safe_float(self._get("cspca"))
        if cspca is None:
            return MISSING
        if cspca < 0.5:
            level = "Low"
        elif cspca <= 0.75:
            level = "Moderate"
        else:
            level = "High"
        return f"{level} (Uncalibrated Model Likelihood)"

    def _note_section(self, section_name):
        sections = self._get("note_sections")
        if not sections or not isinstance(sections, list):
            return MISSING
        for sec in sections:
            if isinstance(sec, dict) and sec.get("s") == section_name:
                t = sec.get("t")
                if t and str(t).strip():
                    return str(t)
        return MISSING

    def _format_previous_notes(self):
        pn = self._clinical("previous_notes")
        if pn is None:
            return MISSING
        if isinstance(pn, str):
            if pn.strip():
                return pn
            return MISSING
        if isinstance(pn, list):
            if len(pn) == 0:
                return MISSING
            parts = []
            for entry in pn:
                if not isinstance(entry, dict):
                    continue
                date = entry.get("date", "")
                author = entry.get("author", "")
                text = entry.get("text", "")
                parts.append(f"{date} ({author}): {text}")
            if parts:
                return "\n\n".join(parts)
            return MISSING
        return MISSING

    def _concordance(self):
        pirads = self._get("pirads")
        isup = self._get("bx_isup")
        pirads_val = None
        if pirads is not None and str(pirads).strip() not in ("", "NA", "null"):
            try:
                pirads_val = int(pirads)
            except (ValueError, TypeError):
                pass
        isup_val = None
        if isup is not None:
            try:
                isup_val = int(isup)
            except (ValueError, TypeError):
                pass
        if pirads_val is None or isup_val is None:
            return "Insufficient Data", "No significant tension — insufficient data for concordance analysis"
        if pirads_val >= 4 and isup_val >= 2:
            return "Concordant", "No significant tension — imaging and histology converge on significant disease"
        if pirads_val <= 2 and isup_val == 1:
            return "Concordant", "No significant tension — imaging and histology both indicate low-risk disease"
        if pirads_val >= 4 and isup_val == 1:
            return "Discordant", ("Imaging-histology mismatch: PI-RADS %d suggests significant disease but "
                                  "biopsy ISUP 1 shows low-grade findings. Consider targeted re-biopsy or "
                                  "MRI-US fusion to rule out sampling error." % pirads_val)
        if pirads_val <= 2 and isup_val >= 2:
            return "Discordant", ("Imaging-histology mismatch: PI-RADS %d is low but biopsy ISUP %d shows "
                                  "significant disease. Possible undersampling on MRI or missed lesion. "
                                  "Review biopsy cores and consider repeat imaging." % (pirads_val, isup_val))
        if pirads_val == 3:
            return "Indeterminate (PI-RADS 3 equivocal)", ("PI-RADS 3 is equivocal; concordance with "
                                                            "ISUP %d is indeterminate. Clinical correlation "
                                                            "and PSAD assessment advised." % isup_val)
        return "Insufficient Data", "No significant tension — concordance indeterminate"

    def _output_schema(self):
        if self.task == 1:
            return ('Binary decision "yes" (biopsy indicated) / "no" (no biopsy) '
                    "+ 10 variable_weights")
        if self.task == 2:
            return ('4-class treatment decision (active_surveillance / '
                    "nerve-sparing_prostatectomy / non-nerve-sparing_prostatectomy / "
                    "radiotherapy) + 11 variable_weights")
        if self.task == 3:
            return ('months_to_recurrence (float, months) + event (int 0=last follow-up / '
                    "1=recurrence) + free_text reasoning")
        return MISSING

    def render(self):
        lines = []
        case_id = self._get("case_id", self.case_id)
        task = self._get("task", self.task)
        psad = self._compute_psad()

        # ---- Header ----
        lines.append(f"# CHIMERA-AGENT CLINICAL CASE BUNDLE: {case_id} (Task {task})")
        lines.append("")

        # ---- VIEW 1 ----
        vitals = self._get("vitals") or {}
        pmhx = self._get("pmhx")
        allergies = self._get("allergies")
        fh = self._clinical("family_history")

        def _vitals_field(key):
            v = vitals.get(key) if isinstance(vitals, dict) else None
            return _or_missing(v)

        def _pmhx_str():
            if pmhx is None:
                return MISSING
            if isinstance(pmhx, list):
                if len(pmhx) == 0:
                    return MISSING
                return ", ".join(str(x) for x in pmhx)
            return _or_missing(pmhx)

        def _allergies_str():
            if allergies is None:
                return "None recorded"
            if isinstance(allergies, list):
                if len(allergies) == 0:
                    return "None recorded"
                return ", ".join(str(x) for x in allergies)
            return _or_missing(allergies)

        lines.append("## VIEW 1: PATIENT HEADER & DEMOGRAPHICS")
        lines.append("| Field | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| Case ID | {case_id} |")
        lines.append(f"| Task | {task} |")
        age = self._get("age")
        age_str = f"{age} years" if age is not None else MISSING
        lines.append(f"| Age | {age_str} |")
        lines.append(f"| Occupation | {_or_missing(self._get('occupation'))} |")
        lines.append(f"| Marital Status | {_or_missing(self._get('marital'))} |")
        lines.append(f"| Living Situation | {_or_missing(self._get('living'))} |")
        lines.append(f"| Vitals - BP | {_vitals_field('bp')} |")
        lines.append(f"| Vitals - HR | {_vitals_field('hr')} |")
        lines.append(f"| Vitals - BMI | {_vitals_field('bmi')} |")
        lines.append(f"| Smoking | {_vitals_field('smoking')} |")
        lines.append(f"| Comorbidities | {_pmhx_str()} |")
        lines.append(f"| Medications | {_or_missing(self._get('meds'))} |")
        lines.append(f"| Allergies | {_allergies_str()} |")
        lines.append(f"| IPSS | {_or_missing(self._get('ipss'))} |")
        lines.append(f"| Family History | {_or_missing(fh)} |")
        lines.append("")

        # ---- VIEW 2 ----
        psa = self._get("psa")
        vol = self._get("vol")
        pirads = self._get("pirads")
        dre = self._get("dre")
        ct = self._get("ct")
        bx_isup = self._get("bx_isup")
        bx_gl_prim = self._get("bx_gl_prim")
        bx_gl_sec = self._get("bx_gl_sec")
        cspca = self._get("cspca")

        psad_display = _fmt_num(psad, 4) if psad is not None else MISSING
        dre_ct_display = f"{_or_missing(dre)} / {_or_missing(ct)}"
        gl_display = f"{bx_gl_prim}+{bx_gl_sec}" if (bx_gl_prim is not None and bx_gl_sec is not None) else MISSING
        isup_display = _or_missing(bx_isup)
        fh_display = _or_missing(fh)

        lines.append("## VIEW 2: STRUCTURED CLINICAL MATRIX")
        lines.append("| Variable | Value | Normal Range / Scale | Clinical Status |")
        lines.append("| :--- | :--- | :--- | :--- |")
        lines.append(f"| Serum PSA | {_fmt_display(psa) if psa is not None else MISSING} ng/mL | 0.0 - 4.0 ng/mL | {self._eval_psa()} |")
        lines.append(f"| Prostate Volume | {_fmt_display(vol) if vol is not None else MISSING} cc | 20 - 30 cc | {self._eval_vol()} |")
        lines.append(f"| PSA Density (PSAD) | {psad_display} ng/mL/cc | < 0.15 ng/mL/cc | {self._eval_psad(psad)} |")
        lines.append(f"| PI-RADS v2.1 Score | {_or_missing(pirads)} | 1 to 5 | {self._eval_pirads()} |")
        lines.append(f"| DRE / Clinical T Stage | {dre_ct_display} | Normal / cT1c | {self._eval_dre_ct()} |")
        lines.append(f"| Biopsy ISUP Grade | {isup_display} (Gl: {gl_display}) | Group 1 - 5 | {self._eval_isup()} |")
        lines.append(f"| Family History | {fh_display} | Yes / No | {self._eval_fh()} |")
        lines.append(f"| DL csPCa Likelihood | {_fmt_display(cspca) if cspca is not None else MISSING} | 0.0 - 1.0 (Uncalibrated) | {self._eval_cspca()} |")
        lines.append("")

        # ---- VIEW 3 ----
        rad_report = self._clinical("radiology_report")
        path_report = self._clinical("pathology_report")
        surg_path = self._clinical("surgical_pathology_report")

        lines.append("## VIEW 3: CHRONOLOGICAL CLINICAL VIGNETTE")
        lines.append(f"**Presentation:** {self._note_section('Chief complaint')}")
        lines.append(f"**History of Present Illness:** {self._note_section('History')}")
        lines.append(f"**Physical Examination:** {self._note_section('Physical examination')}")
        lines.append(f"**Radiology Report:** {_or_missing(rad_report)}")
        if self.task == 1:
            lines.append(f"**Pathology Report:** {MISSING}")
        else:
            lines.append(f"**Pathology Report:** {_or_missing(path_report)}")
        if self.task == 3:
            lines.append(f"**Surgical Pathology:** {_or_missing(surg_path)}")
        else:
            lines.append(f"**Surgical Pathology:** {MISSING}")
        lines.append(f"**Previous Notes:** {self._format_previous_notes()}")
        lines.append("")

        # ---- VIEW 4 ----
        eau_tier, eau_criteria = EAURiskTier.classify(
            psa, bx_isup, ct, self._get("high_risk_patterns"))
        capra_score, capra_breakdown = CAPRAScore.compute(
            self._get("age"), psa, bx_gl_prim, bx_gl_sec,
            self._get("bx_gl_tert"), ct,
            self._get("cores_positive"), self._get("cores_total"))

        lines.append("## VIEW 4: GUIDELINE RISK STRATIFICATION")
        lines.append(f"EAU 2026 Risk Tier: {eau_tier} (Criteria: {eau_criteria})")
        lines.append(f"CAPRA Score: {capra_score}/10 (Points: {capra_breakdown})")
        if self.task == 3:
            surg_path = self._clinical("surgical_pathology_report")
            capras_score, capras_breakdown, capras_imputed = CAPRAS_Score.compute(
                psa, surg_path)
            imputed_str = ""
            if capras_imputed:
                imputed_str = f" (imputed: {', '.join(capras_imputed)})"
            lines.append(
                f"CAPRA-S (post-surgical): {capras_score}/12 "
                f"(components: {capras_breakdown}){imputed_str}")
        lines.append("")

        # ---- VIEW 5 ----
        lines.append("## VIEW 5: COHORT PERCENTILE CONTEXT (N=423)")
        lines.append("| Metric | Value | Percentile | Robust Z-Score |")
        lines.append("| :--- | :--- | :--- | :--- |")

        def _fmt_cohort(value):
            if value is None:
                return MISSING
            rounded = round(value, 4)
            if rounded == int(rounded):
                return str(int(rounded))
            return str(rounded)

        def _ordinal(n):
            if n is None:
                return MISSING
            if 10 <= n % 100 <= 20:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            return f"{n}{suffix}"

        def _cohort_row(display_name, metric_key, value, unit):
            if value is None:
                return f"| {display_name} | {MISSING} | {MISSING} | {MISSING} |"
            pct = self.cohort.percentile(metric_key, value)
            rz = self.cohort.robust_z(metric_key, value)
            pct_str = _ordinal(pct) if pct is not None else MISSING
            rz_str = f"{rz:.2f}" if rz is not None else MISSING
            return f"| {display_name} | {_fmt_cohort(value)} {unit} | {pct_str} | {rz_str} |"

        lines.append(_cohort_row("Serum PSA", "psa", _safe_float(psa), "ng/mL"))
        lines.append(_cohort_row("PSA Density", "psad", psad, "ng/mL/cc"))
        lines.append(_cohort_row("Prostate Volume", "vol", _safe_float(vol), "cc"))
        lines.append("")

        # ---- VIEW 6 ----
        concordance, tension = self._concordance()
        bx_isup_pred = self._get("bx_isup_pred")
        isup_val = bx_isup
        isup_pred_val = bx_isup_pred
        delta = None
        if isup_val is not None and isup_pred_val is not None:
            try:
                delta = int(isup_val) - int(isup_pred_val)
            except (ValueError, TypeError):
                pass

        lines.append("## VIEW 6: CROSS-MODAL CONCORDANCE")
        pirads_display = _or_missing(pirads)
        isup_display_v6 = _or_missing(bx_isup)
        lines.append(f"MRI (PI-RADS {pirads_display}) vs Histology (ISUP {isup_display_v6}): {concordance}")
        lines.append(f"- Diagnostic Tension: {tension}")
        delta_str = str(delta) if delta is not None else MISSING
        lines.append(f"- Digital Pathology AI: Human ISUP {isup_display_v6} vs AI ISUP {_or_missing(bx_isup_pred)} (Delta: {delta_str})")
        lines.append("")

        # ---- VIEW 7 ----
        psa_trend = self._clinical("psa_trend")
        kinetics = PSAKinetics(psa_trend)
        n_pts = kinetics.n_points
        span = kinetics.span_months

        if kinetics.insufficient:
            psav_str = "Insufficient data (N<3 or span<6mo)"
            psadt_str = "Insufficient data (N<3 or span<6mo)"
            traj_str = "Insufficient data (N<3 or span<6mo)"
        else:
            psav_val = kinetics.psav()
            psadt_val = kinetics.psadt()
            psav_str = f"{psav_val:.2f} ng/mL/year" if psav_val is not None else MISSING
            if psadt_val is None:
                psadt_str = "Non-rising (slope <= 0)"
            else:
                psadt_str = f"{psadt_val:.2f} months"
            traj_str = kinetics.trajectory(psadt_val)

        span_str = f"{span:.1f}" if span > 0 else "0"

        lines.append("## VIEW 7: LONGITUDINAL PSA KINETICS")
        lines.append(f"- Measurement Points: {n_pts} points spanning {span_str} months")
        lines.append(f"- PSA Velocity (PSAV): {psav_str} (Threshold for suspicion: >0.75 ng/mL/yr)")
        lines.append(f"- PSA Doubling Time (PSADT): {psadt_str} (Aggressive threshold: <12 months)")
        lines.append(f"- Trajectory Classification: {traj_str}")
        lines.append("")

        # ---- VIEW 8 ----
        lines.append("## VIEW 8: NEURAL EMBEDDING AUDIT")
        lines.append("| Modality | Status | Shape | L2 Norm | Mean | Std | Min | Max |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        modality_labels = [("MRI image", "MRI"), ("Biopsy slide", "Biopsy"), ("Prostatectomy slide", "Prostatectomy")]
        for key, label in modality_labels:
            vecs = self.neural.get(key) if self.neural else None
            audit = EmbeddingAudit.audit(vecs)
            status = "Available" if audit["available"] else "Not Available"
            lines.append(f"| {label} | {status} | {audit['shape']} | {audit['l2_norm']} | {audit['mean']} | {audit['std']} | {audit['min']} | {audit['max']} |")
        lines.append("")

        # ---- VIEW 9 ----
        lines.append("## VIEW 9: CHAIN-OF-THOUGHT REASONING CONTRACT")
        lines.append("Execute your diagnostic reasoning across the following mandatory stages:")
        lines.append("STAGE 1: Evidence Synthesis & Modality Concordance Analysis")
        lines.append("STAGE 2: Guideline Rule Verification (EAU 2026 & NCCN)")
        lines.append("STAGE 3: Risk-Benefit & Life Expectancy Trade-Off Synthesis")
        lines.append("STAGE 4: Final JSON Output Generation (Target Decision + Variable Weights + Reveal Sequence)")
        lines.append("")
        lines.append(f"Target Task: {task}")
        lines.append(f"Required Output Schema: {self._output_schema()}")
        lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def list_case_dirs(task_dir):
    if not task_dir.is_dir():
        return []
    return sorted([d for d in task_dir.iterdir() if d.is_dir()])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tasks = [1, 2, 3]
    task_dirs = {t: TRAIN_DIR / f"task{t}" for t in tasks}

    # ---- First pass: collect cohort stats ----
    cohort = CohortStats()
    case_data = {t: [] for t in tasks}  # (case_dir, sp, clinical, neural)

    for t in tasks:
        for case_dir in list_case_dirs(task_dirs[t]):
            sp = load_json(case_dir / "structured-prompt.json")
            if sp is None:
                continue
            clinical = load_json(case_dir / CLINICAL_DATA_FILE[t])
            neural = load_json(case_dir / "prostate-modality-level-neural-representations.json")

            psa = _safe_float(sp.get("psa"))
            vol = _safe_float(sp.get("vol"))
            if vol is None and t == 3 and clinical:
                rr = clinical.get("radiology_report", "")
                m = re.search(r"Prostate volume:\s*([\d.]+)", rr, re.I)
                if m:
                    try:
                        vol = float(m.group(1))
                    except (ValueError, TypeError):
                        vol = None
            psad = None
            if psa is not None and vol is not None and vol > 0:
                psad = psa / vol

            cohort.add("psa", psa)
            cohort.add("psad", psad)
            cohort.add("vol", vol)

            case_data[t].append((case_dir, sp, clinical, neural))

    cohort.finalize()

    total_cohort = sum(len(case_data[t]) for t in tasks)

    # ---- Second pass: generate bundles ----
    counts = {t: 0 for t in tasks}
    errors = []

    for t in tasks:
        out_dir = OUTPUT_DIR / f"task{t}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for case_dir, sp, clinical, neural in case_data[t]:
            case_id = sp.get("case_id", case_dir.name)
            try:
                renderer = BundleRenderer(case_id, t, sp, clinical, neural, cohort)
                markdown = renderer.render()
                out_path = out_dir / f"{case_id}.md"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(markdown)
                counts[t] += 1
            except Exception as e:
                errors.append(f"task{t}/{case_id}: {e}")

    # ---- Summary ----
    print("=" * 60)
    print("CHIMERA-AGENT PHASE A: 9-VIEW BUNDLE GENERATOR")
    print("=" * 60)
    print(f"Cohort size: {total_cohort} cases")
    print(f"  PSA distribution:   N={cohort.n('psa')}  median={cohort._median.get('psa')}")
    print(f"  PSAD distribution:  N={cohort.n('psad')}  median={cohort._median.get('psad')}")
    print(f"  Vol distribution:   N={cohort.n('vol')}  median={cohort._median.get('vol')}")
    print()
    print(f"Bundles generated:")
    for t in tasks:
        print(f"  task{t}: {counts[t]} bundles")
    print(f"  TOTAL: {sum(counts.values())} bundles")
    print()
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print("No errors encountered.")
    print("=" * 60)


if __name__ == "__main__":
    main()

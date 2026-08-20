# tests/test_biomarkers.py
"""Unit and boundary test suite for clinical biomarkers, PSA kinetics, robust z-scores, and tensor embedding statistics."""

import math
import sys
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_bundles import (
    DateParser,
    CohortStats,
    PSAKinetics,
    EmbeddingAudit,
    _safe_float,
    _fmt_num,
    MISSING,
    DAYS_PER_YEAR,
    DAYS_PER_MONTH,
    LN2,
)


class TestPSADCalculations(unittest.TestCase):
    """Tests for Prostate-Specific Antigen Density (PSAD = PSA / Volume)."""

    def test_psad_normal_values(self):
        psa = 6.0
        vol = 40.0
        psad = psa / vol
        self.assertAlmostEqual(psad, 0.15, places=4)

        psa = 10.0
        vol = 25.0
        self.assertAlmostEqual(psa / vol, 0.40, places=4)

        psa = 2.0
        vol = 50.0
        self.assertAlmostEqual(psa / vol, 0.04, places=4)

    def test_psad_boundary_and_guards(self):
        # vol = 0
        self.assertIsNone(_safe_float(None))
        self.assertIsNone(_safe_float("N/A"))
        self.assertIsNone(_safe_float(False))

        def compute_psad(psa_in, vol_in):
            p = _safe_float(psa_in)
            v = _safe_float(vol_in)
            if p is not None and v is not None and v > 0:
                return p / v
            return None

        self.assertIsNone(compute_psad(6.0, 0.0))
        self.assertIsNone(compute_psad(6.0, -10.0))
        self.assertIsNone(compute_psad(None, 40.0))
        self.assertIsNone(compute_psad(6.0, None))
        self.assertIsNone(compute_psad(6.0, "invalid"))

    def test_psad_risk_evaluation(self):
        def eval_psad(psad):
            if psad is None:
                return MISSING
            if psad < 0.15:
                return "Low Risk"
            return "High Risk"

        self.assertEqual(eval_psad(0.1499), "Low Risk")
        self.assertEqual(eval_psad(0.1500), "High Risk")
        self.assertEqual(eval_psad(0.2500), "High Risk")
        self.assertEqual(eval_psad(None), MISSING)


class TestDateParser(unittest.TestCase):
    """Tests for DateParser supporting 'DD Mon YYYY' and 'Mon YYYY'."""

    def test_parse_valid_dates(self):
        d1 = DateParser.parse("15 Jan 2022")
        self.assertEqual(d1, datetime(2022, 1, 15))

        d2 = DateParser.parse("Dec 2020")
        self.assertEqual(d2, datetime(2020, 12, 1))

        d3 = DateParser.parse(" 3 Mar 2021 ")
        self.assertEqual(d3, datetime(2021, 3, 3))

    def test_parse_invalid_dates(self):
        self.assertIsNone(DateParser.parse(None))
        self.assertIsNone(DateParser.parse(""))
        self.assertIsNone(DateParser.parse("Not A Date"))
        self.assertIsNone(DateParser.parse("32 Jan 2020"))
        self.assertIsNone(DateParser.parse(12345))


class TestPSAKinetics(unittest.TestCase):
    """Tests for PSA Velocity (PSAV) and PSA Doubling Time (PSADT)."""

    def test_carter_gating_n_points(self):
        # 0 points
        k0 = PSAKinetics([])
        self.assertTrue(k0.insufficient)
        self.assertIsNone(k0.psav())
        self.assertIsNone(k0.psadt())

        # 1 point
        k1 = PSAKinetics([{"date": "Jan 2020", "val": 4.0}])
        self.assertTrue(k1.insufficient)
        self.assertIsNone(k1.psav())
        self.assertIsNone(k1.psadt())

        # 2 points
        k2 = PSAKinetics([
            {"date": "Jan 2020", "val": 4.0},
            {"date": "Jan 2021", "val": 5.0}
        ])
        self.assertTrue(k2.insufficient)
        self.assertIsNone(k2.psav())
        self.assertIsNone(k2.psadt())

    def test_carter_gating_span_months(self):
        # 3 points but span < 6 months (e.g. 2 months)
        k_short = PSAKinetics([
            {"date": "1 Jan 2020", "val": 4.0},
            {"date": "1 Feb 2020", "val": 4.5},
            {"date": "1 Mar 2020", "val": 5.0}
        ])
        self.assertTrue(k_short.insufficient)
        self.assertLess(k_short.span_months, 6.0)
        self.assertIsNone(k_short.psav())
        self.assertIsNone(k_short.psadt())

    def test_psav_exact_linear_slope(self):
        # 3 points spanning 2 years with exact slope = 1.0 ng/mL/yr
        # Jan 2020 (t=0, v=4.0), Jan 2021 (t=1, v=5.0), Jan 2022 (t=2, v=6.0)
        trend = [
            {"date": "1 Jan 2020", "val": 4.0},
            {"date": "1 Jan 2021", "val": 5.0},
            {"date": "1 Jan 2022", "val": 6.0},
        ]
        k = PSAKinetics(trend)
        self.assertFalse(k.insufficient)
        psav = k.psav()
        self.assertIsNotNone(psav)
        self.assertAlmostEqual(psav, 1.0, places=2)

    def test_psav_zero_slope_and_negative_slope(self):
        # Flat slope = 0.0
        trend_flat = [
            {"date": "1 Jan 2020", "val": 5.0},
            {"date": "1 Jul 2020", "val": 5.0},
            {"date": "1 Jan 2021", "val": 5.0},
        ]
        k_flat = PSAKinetics(trend_flat)
        self.assertAlmostEqual(k_flat.psav(), 0.0, places=3)

        # Declining slope = -1.0
        trend_dec = [
            {"date": "1 Jan 2020", "val": 6.0},
            {"date": "1 Jan 2021", "val": 5.0},
            {"date": "1 Jan 2022", "val": 4.0},
        ]
        k_dec = PSAKinetics(trend_dec)
        self.assertAlmostEqual(k_dec.psav(), -1.0, places=2)

    def test_psadt_exact_exponential_doubling(self):
        # PSA doubles exactly every 12 months:
        # t=0: 2.0, t=12m: 4.0, t=24m: 8.0
        trend = [
            {"date": "1 Jan 2020", "val": 2.0},
            {"date": "1 Jan 2021", "val": 4.0},
            {"date": "1 Jan 2022", "val": 8.0},
        ]
        k = PSAKinetics(trend)
        self.assertFalse(k.insufficient)
        psadt = k.psadt()
        self.assertIsNotNone(psadt)
        # In months, 1 year is approx 12.0 months
        self.assertAlmostEqual(psadt, 12.0, delta=0.2)
        self.assertEqual(k.trajectory(psadt), "Moderate")

    def test_psadt_rapid_and_aggressive_trajectories(self):
        # Aggressive (< 6 mo doubling time, but span >= 6 mo)
        trend_agg = [
            {"date": "1 Jan 2020", "val": 2.0},
            {"date": "1 May 2020", "val": 8.0},
            {"date": "1 Sep 2020", "val": 32.0},
        ]
        k_agg = PSAKinetics(trend_agg)
        psadt_agg = k_agg.psadt()
        self.assertLess(psadt_agg, 6.0)
        self.assertEqual(k_agg.trajectory(psadt_agg), "Aggressive")

        # Rapid (6 to 12 mo)
        trend_rap = [
            {"date": "1 Jan 2020", "val": 2.0},
            {"date": "1 Jul 2020", "val": 3.5},
            {"date": "1 Jan 2021", "val": 6.0},
        ]
        k_rap = PSAKinetics(trend_rap)
        psadt_rap = k_rap.psadt()
        self.assertTrue(6.0 <= psadt_rap < 12.0)
        self.assertEqual(k_rap.trajectory(psadt_rap), "Rapid")

        # Indolent (>= 24 mo)
        trend_ind = [
            {"date": "1 Jan 2020", "val": 4.0},
            {"date": "1 Jan 2021", "val": 4.5},
            {"date": "1 Jan 2022", "val": 5.0},
        ]
        k_ind = PSAKinetics(trend_ind)
        psadt_ind = k_ind.psadt()
        self.assertGreaterEqual(psadt_ind, 24.0)
        self.assertEqual(k_ind.trajectory(psadt_ind), "Indolent")

    def test_psadt_non_rising_guard(self):
        # Declining PSA
        trend_dec = [
            {"date": "1 Jan 2020", "val": 6.0},
            {"date": "1 Jan 2021", "val": 5.0},
            {"date": "1 Jan 2022", "val": 4.0},
        ]
        k_dec = PSAKinetics(trend_dec)
        self.assertIsNone(k_dec.psadt())
        self.assertEqual(k_dec.trajectory(None), "Stable/Declining")

        # Flat PSA
        trend_flat = [
            {"date": "1 Jan 2020", "val": 4.0},
            {"date": "1 Jul 2020", "val": 4.0},
            {"date": "1 Jan 2021", "val": 4.0},
        ]
        k_flat = PSAKinetics(trend_flat)
        self.assertIsNone(k_flat.psadt())
        self.assertEqual(k_flat.trajectory(None), "Stable/Declining")

    def test_identical_dates_zero_denominator_guard(self):
        # All points on identical date
        trend_same = [
            {"date": "1 Jan 2020", "val": 4.0},
            {"date": "1 Jan 2020", "val": 5.0},
            {"date": "1 Jan 2020", "val": 6.0},
        ]
        k_same = PSAKinetics(trend_same)
        self.assertIsNone(k_same.psav())
        self.assertIsNone(k_same.psadt())

    def test_non_positive_psa_filtered_out(self):
        trend_zero = [
            {"date": "1 Jan 2020", "val": 0.0},
            {"date": "1 Jul 2020", "val": -2.0},
            {"date": "1 Jan 2021", "val": 4.0},
        ]
        k = PSAKinetics(trend_zero)
        # Only 1 valid point remains
        self.assertEqual(k.n_points, 1)
        self.assertTrue(k.insufficient)


class TestCohortStats(unittest.TestCase):
    """Tests for CohortStats robust z-scores and percentiles."""

    def test_percentile_and_median_linear_interpolation(self):
        stats = CohortStats()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            stats.add("psa", v)
        stats.finalize()

        self.assertEqual(stats._median["psa"], 30.0)
        self.assertEqual(stats._q1["psa"], 20.0)
        self.assertEqual(stats._q3["psa"], 40.0)
        self.assertEqual(stats._iqr["psa"], 20.0)

    def test_robust_z_score_calculation(self):
        stats = CohortStats()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            stats.add("psa", v)
        stats.finalize()

        # NIQR = 20.0 / 1.349 = 14.825796886582654
        # Median = 30.0
        # z for 30.0 -> 0.0
        self.assertAlmostEqual(stats.robust_z("psa", 30.0), 0.0, places=4)
        # z for 44.8258 -> 1.0
        self.assertAlmostEqual(stats.robust_z("psa", 30.0 + 20.0 / 1.349), 1.0, places=4)
        # z for 15.1742 -> -1.0
        self.assertAlmostEqual(stats.robust_z("psa", 30.0 - 20.0 / 1.349), -1.0, places=4)

    def test_zero_iqr_guard(self):
        stats = CohortStats()
        for _ in range(10):
            stats.add("psa", 5.0)
        stats.finalize()

        self.assertEqual(stats._iqr["psa"], 0.0)
        self.assertEqual(stats.robust_z("psa", 5.0), 0.0)
        self.assertEqual(stats.robust_z("psa", 10.0), 0.0)

    def test_percentile_rank_with_ties(self):
        stats = CohortStats()
        for v in [1, 2, 2, 3]:
            stats.add("psa", v)
        stats.finalize()

        # For 1: rank_below=0, count_equal=1 -> (0 + 0.5)/4 * 100 = 12.5 -> 12 or 13
        self.assertEqual(stats.percentile("psa", 1), 12)
        # For 2: rank_below=1, count_equal=2 -> (1 + 1.0)/4 * 100 = 50%
        self.assertEqual(stats.percentile("psa", 2), 50)
        # For 3: rank_below=3, count_equal=1 -> (3 + 0.5)/4 * 100 = 87.5 -> 88%
        self.assertEqual(stats.percentile("psa", 3), 88)


class TestEmbeddingAudit(unittest.TestCase):
    """Tests for EmbeddingAudit tensor statistics."""

    def test_embedding_statistics_valid(self):
        vecs = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        audit = EmbeddingAudit.audit(vecs)
        self.assertTrue(audit["available"])
        self.assertEqual(audit["shape"], "[2, 3]")

        # Flat: [1, 2, 3, 4, 5, 6]
        # Sum of squares = 1+4+9+16+25+36 = 91 -> sqrt(91) = 9.53939
        self.assertEqual(audit["l2_norm"], "9.54")
        self.assertEqual(audit["mean"], "3.5000")
        # Var = ((1-3.5)^2 + (2-3.5)^2 + (3-3.5)^2 + (4-3.5)^2 + (5-3.5)^2 + (6-3.5)^2)/6 = 17.5/6 = 2.916666
        # Std = sqrt(2.916666) = 1.7078
        self.assertEqual(audit["std"], "1.7078")
        self.assertEqual(audit["min"], "1.0000")
        self.assertEqual(audit["max"], "6.0000")

    def test_embedding_statistics_empty_and_missing(self):
        audit_none = EmbeddingAudit.audit(None)
        self.assertFalse(audit_none["available"])
        self.assertEqual(audit_none["l2_norm"], "0")
        self.assertEqual(audit_none["mean"], MISSING)

        audit_empty = EmbeddingAudit.audit([])
        self.assertFalse(audit_empty["available"])
        self.assertEqual(audit_empty["l2_norm"], "0")

        audit_empty_sub = EmbeddingAudit.audit([[]])
        self.assertFalse(audit_empty_sub["available"])


if __name__ == "__main__":
    unittest.main()

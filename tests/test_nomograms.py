# tests/test_nomograms.py
"""Unit and boundary test suite for clinical nomograms and risk guidelines:
- EAU 2026 5-tier guidelines + fail-closed rules
- UCSF CAPRA (0-10) pre-treatment nomogram
- UCSF CAPRA-S (0-12) post-surgical nomogram & NLP regex extractor
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_bundles import (
    _parse_ct,
    EAURiskTier,
    CAPRAScore,
    CAPRAS_Score,
    MISSING,
)


class TestCTParser(unittest.TestCase):
    """Tests for clinical T-stage parser."""

    def test_valid_ct_stages(self):
        self.assertEqual(_parse_ct("cT1a"), (1, 1))
        self.assertEqual(_parse_ct("cT1b"), (1, 2))
        self.assertEqual(_parse_ct("cT1c"), (1, 3))
        self.assertEqual(_parse_ct("cT2a"), (2, 1))
        self.assertEqual(_parse_ct("cT2b"), (2, 2))
        self.assertEqual(_parse_ct("cT2c"), (2, 3))
        self.assertEqual(_parse_ct("cT3a"), (3, 1))
        self.assertEqual(_parse_ct("cT3b"), (3, 2))
        self.assertEqual(_parse_ct("cT4"), (4, 0))

    def test_ambiguous_and_invalid_ct_stages(self):
        self.assertEqual(_parse_ct("cT2"), (2, 0))
        self.assertIsNone(_parse_ct("cTx"))
        self.assertIsNone(_parse_ct("T1"))
        self.assertIsNone(_parse_ct(""))
        self.assertIsNone(_parse_ct(None))
        self.assertIsNone(_parse_ct(123))


class TestEAU2025Guidelines(unittest.TestCase):
    """Tests for EAU 2025 5-tier risk classification (no 'Very High' tier)."""

    def test_high_risk_psa_over_20(self):
        # PSA > 20 is High (EAU 2025 — no "Very High" tier)
        tier, reason = EAURiskTier.classify(25.0, None, None, None)
        self.assertEqual(tier, "High")
        self.assertIn("PSA 25.0 > 20", reason)

        tier2, _ = EAURiskTier.classify(20.1, 1, "cT1c", None)
        self.assertEqual(tier2, "High")

    def test_locally_advanced_staging(self):
        # cT3b is Locally Advanced (cT3-4 takes precedence)
        tier, _ = EAURiskTier.classify(5.0, 5, "cT3b", None)
        self.assertEqual(tier, "Locally Advanced")

        # cT4
        tier2, _ = EAURiskTier.classify(4.0, 1, "cT4", None)
        self.assertEqual(tier2, "Locally Advanced")

        # cT3a
        tier3, _ = EAURiskTier.classify(5.0, 1, "cT3a", None)
        self.assertEqual(tier3, "Locally Advanced")

    def test_unfavorable_intermediate_psa_10_to_20(self):
        # PSA 10-20 is Unfavorable Intermediate (not High — EAU 2025)
        tier, reason = EAURiskTier.classify(10.0, None, None, None)
        self.assertEqual(tier, "Unfavorable Intermediate")
        self.assertIn("PSA 10.0 10-20", reason)

        tier2, _ = EAURiskTier.classify(15.5, 2, "cT1a", None)
        self.assertEqual(tier2, "Unfavorable Intermediate")

    def test_high_risk_isup_or_ct2c(self):
        # ISUP 4 or 5 with PSA < 10
        tier, _ = EAURiskTier.classify(6.0, 4, "cT1c", None)
        self.assertEqual(tier, "High")

        tier2, _ = EAURiskTier.classify(6.0, 5, "cT2a", None)
        self.assertEqual(tier2, "High")

        # cT2c is High (EAU 2025 — not Unfavorable Intermediate)
        tier3, _ = EAURiskTier.classify(6.0, 2, "cT2c", None)
        self.assertEqual(tier3, "High")

    def test_unfavorable_intermediate_risk(self):
        # ISUP 3
        tier, _ = EAURiskTier.classify(7.0, 3, "cT1c", None)
        self.assertEqual(tier, "Unfavorable Intermediate")

        # cT2b
        tier2, _ = EAURiskTier.classify(6.0, 1, "cT2b", None)
        self.assertEqual(tier2, "Unfavorable Intermediate")

        # ISUP 2 upgraded due to high-risk patterns (e.g. cribriform)
        tier4, reason4 = EAURiskTier.classify(6.0, 2, "cT2a", "cribriform pattern present")
        self.assertEqual(tier4, "Unfavorable Intermediate")
        self.assertIn("upgraded", reason4)

    def test_favorable_intermediate_risk(self):
        # PSA < 10 AND ISUP 2 AND cT in LOW_CT AND no high-risk patterns
        tier, _ = EAURiskTier.classify(6.0, 2, "cT1c", None)
        self.assertEqual(tier, "Favorable Intermediate")

        tier2, _ = EAURiskTier.classify(5.0, 2, "cT2a", "")
        self.assertEqual(tier2, "Favorable Intermediate")

    def test_low_risk(self):
        # PSA < 10 AND ISUP 1 AND cT in LOW_CT
        tier, _ = EAURiskTier.classify(4.0, 1, "cT1c", None)
        self.assertEqual(tier, "Low")

        tier2, _ = EAURiskTier.classify(5.5, 1, "cT2a", None)
        self.assertEqual(tier2, "Low")

    def test_fail_closed_data_not_recorded(self):
        # Both ISUP and cT missing (PSA < 10)
        tier, _ = EAURiskTier.classify(5.0, None, None, None)
        self.assertEqual(tier, MISSING)

        # Ambiguous cT2 (cannot distinguish cT2a from cT2b/c)
        tier2, _ = EAURiskTier.classify(5.0, 1, "cT2", None)
        self.assertEqual(tier2, MISSING)

        # Missing PSA and missing staging
        tier3, _ = EAURiskTier.classify(None, None, None, None)
        self.assertEqual(tier3, MISSING)


class TestCAPRAScorer(unittest.TestCase):
    """Tests for UCSF CAPRA pre-treatment nomogram (0-10)."""

    def test_capra_individual_components(self):
        # Age: <50 (0 pts), >=50 (1 pt)
        s_young, _ = CAPRAScore.compute(45, 4.0, 3, 3, None, "cT1c", 0, 12)
        s_old, _ = CAPRAScore.compute(65, 4.0, 3, 3, None, "cT1c", 0, 12)
        self.assertEqual(s_young, 0)
        self.assertEqual(s_old, 1)

        # PSA bins: <=6 (0), 6.01-10 (1), 10.01-20 (2), 20.01-30 (3), >30 (4)
        s_psa0, _ = CAPRAScore.compute(45, 5.5, 3, 3, None, "cT1c", 0, 12)
        s_psa1, _ = CAPRAScore.compute(45, 8.0, 3, 3, None, "cT1c", 0, 12)
        s_psa2, _ = CAPRAScore.compute(45, 15.0, 3, 3, None, "cT1c", 0, 12)
        s_psa3, _ = CAPRAScore.compute(45, 25.0, 3, 3, None, "cT1c", 0, 12)
        s_psa4, _ = CAPRAScore.compute(45, 35.0, 3, 3, None, "cT1c", 0, 12)
        self.assertEqual(s_psa0, 0)
        self.assertEqual(s_psa1, 1)
        self.assertEqual(s_psa2, 2)
        self.assertEqual(s_psa3, 3)
        self.assertEqual(s_psa4, 4)

        # Gleason: 3+3 (0), 3+4/3+5 (1), 4+3/4+4/4+5/5+3/5+4/5+5 (3)
        s_gl0, _ = CAPRAScore.compute(45, 4.0, 3, 3, None, "cT1c", 0, 12)
        s_gl1, _ = CAPRAScore.compute(45, 4.0, 3, 4, None, "cT1c", 0, 12)
        s_gl3, _ = CAPRAScore.compute(45, 4.0, 4, 3, None, "cT1c", 0, 12)
        self.assertEqual(s_gl0, 0)
        self.assertEqual(s_gl1, 1)
        self.assertEqual(s_gl3, 3)

        # T-Stage: cT1-cT2 (0), cT3a+ (1)
        s_t0, _ = CAPRAScore.compute(45, 4.0, 3, 3, None, "cT2a", 0, 12)
        s_t1, _ = CAPRAScore.compute(45, 4.0, 3, 3, None, "cT3a", 0, 12)
        self.assertEqual(s_t0, 0)
        self.assertEqual(s_t1, 1)

        # % Positive Cores: <34% (0), >=34% (1)
        s_core0, _ = CAPRAScore.compute(45, 4.0, 3, 3, None, "cT1c", 2, 10)  # 20%
        s_core1, _ = CAPRAScore.compute(45, 4.0, 3, 3, None, "cT1c", 4, 10)  # 40%
        self.assertEqual(s_core0, 0)
        self.assertEqual(s_core1, 1)

    def test_capra_maximum_score_and_capping(self):
        # Max: Age(1) + PSA(4) + Gleason(3) + T(1) + Cores(1) = 10
        score, breakdown = CAPRAScore.compute(70, 45.0, 4, 4, None, "cT3a", 10, 12)
        self.assertEqual(score, 10)
        self.assertIn("Age [+1]", breakdown)
        self.assertIn("PSA [+4]", breakdown)
        self.assertIn("Gleason [+3]", breakdown)
        self.assertIn("T-Stage [+1]", breakdown)
        self.assertIn("Cores [+1]", breakdown)

    def test_capra_missing_component_imputation(self):
        # All fields missing -> score = 0 with imputed labels
        score, breakdown = CAPRAScore.compute(None, None, None, None, None, None, None, None)
        self.assertEqual(score, 0)
        self.assertIn("imputed: data not recorded", breakdown)


class TestCAPRAS_Scorer(unittest.TestCase):
    """Tests for UCSF CAPRA-S post-surgical nomogram (0-12) and regex parser."""

    def test_surgical_report_regex_extraction(self):
        # High risk surgical pathology report text
        report = (
            "Radical prostatectomy specimen. Gleason 4+3 with tertiary pattern 5. "
            "Surgical margins were positive; extraprostatic extension was present. "
            "Seminal vesicles were invaded. Lymph node metastasis was present."
        )
        extracted = CAPRAS_Score._extract_from_surgical_report(report)
        self.assertEqual(extracted['gleason_prim'], 4)
        self.assertEqual(extracted['gleason_sec'], 3)
        self.assertEqual(extracted['margin'], 'positive')
        self.assertEqual(extracted['ece'], 'present')
        self.assertEqual(extracted['svi'], 'present')
        self.assertEqual(extracted['lni'], 'present')

    def test_surgical_report_negative_polarity(self):
        # Negative / benign pathology text (negative matched before positive)
        report = (
            "Gleason 3+3 (Grade Group 1). Negative margins (R0). "
            "No extraprostatic extension identified. "
            "Seminal vesicles are not invaded. No lymph node metastasis identified."
        )
        extracted = CAPRAS_Score._extract_from_surgical_report(report)
        self.assertEqual(extracted['gleason_prim'], 3)
        self.assertEqual(extracted['gleason_sec'], 3)
        self.assertEqual(extracted['margin'], 'negative')
        self.assertEqual(extracted['ece'], 'absent')
        self.assertEqual(extracted['svi'], 'absent')
        self.assertEqual(extracted['lni'], 'absent')

    def test_unremoved_lymph_nodes_imputed(self):
        report = "Gleason 3+4; margins negative; ECE absent; SVI absent; no lymph nodes were removed."
        extracted = CAPRAS_Score._extract_from_surgical_report(report)
        self.assertIsNone(extracted['lni'])

    def test_capras_point_calculation(self):
        # Test individual point components:
        # Pre-op PSA: <=6 (0), 6.01-10 (1), 10.01-20 (2), >20 (3)
        # pGleason: 2-6 (0), 3+4 (1), 4+3 (2), 8-10 (3)
        # Margin: pos (2), neg (0)
        # ECE: pos (1), neg (0)
        # SVI: pos (2), neg (0)
        # LNI: pos (1), neg (0)

        # Minimum score = 0
        min_report = "Gleason 3+3; margins negative; ECE absent; seminal vesicles negative; lymph nodes negative."
        s_min, _, imp_min = CAPRAS_Score.compute(4.0, min_report)
        self.assertEqual(s_min, 0)
        self.assertEqual(len(imp_min), 0)

        # Maximum score = 12 (3+3+2+1+2+1)
        max_report = "Gleason 4+5; margins positive; extraprostatic extension present; seminal vesicle invasion present; lymph node metastasis present."
        s_max, bd_max, imp_max = CAPRAS_Score.compute(35.0, max_report)
        self.assertEqual(s_max, 12)
        self.assertEqual(len(imp_max), 0)
        self.assertIn("PSA [+3]", bd_max)
        self.assertIn("pGleason [+3]", bd_max)
        self.assertIn("Margin [+2]", bd_max)
        self.assertIn("ECE [+1]", bd_max)
        self.assertIn("SVI [+2]", bd_max)
        self.assertIn("LNI [+1]", bd_max)

    def test_capras_missing_report_imputation(self):
        score, breakdown, imputed = CAPRAS_Score.compute(None, None)
        self.assertEqual(score, 0)
        self.assertEqual(set(imputed), {'PSA', 'pGleason', 'Margin', 'ECE', 'SVI', 'LNI'})


if __name__ == "__main__":
    unittest.main()

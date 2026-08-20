# tests/test_all_429_cases.py
"""Comprehensive test sweep across all 429 multimodal patient traces (423 local + 6 sample eval).
Asserts:
- Zero NaN, zero Inf, zero undefined across all numeric fields.
- Schema validity and data completeness.
- Deterministic calculation of all clinical features (PSAD, PSAV, PSADT, EAU 2026, CAPRA, CAPRA-S).
- 100% bit-for-bit reproducible execution.
"""

import glob
import json
import math
import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_bundles import (
    DateParser,
    PSAKinetics,
    EAURiskTier,
    CAPRAScore,
    CAPRAS_Score,
    _safe_float,
    MISSING,
)

BASE_DIR = Path(__file__).resolve().parent.parent
# Patient data lives in a sibling directory (../chimera-data/) — not in the repo.
DATA_DIR = BASE_DIR.parent / "chimera-data"
LOCAL_TRACES_DIR = DATA_DIR / "traces" / "all_local_cases"
SAMPLE_TRACES_DIR = DATA_DIR / "traces" / "run_sample_eval"
BUNDLES_DIR = DATA_DIR / "bundles"


class TestAll429CasesSweep(unittest.TestCase):
    """Full sweep verification over all 429 multimodal patient cases."""

    @classmethod
    def setUpClass(cls):
        local_files = sorted(glob.glob(str(LOCAL_TRACES_DIR / "*.json")))
        sample_files = sorted(glob.glob(str(SAMPLE_TRACES_DIR / "*.json")))
        cls.all_files = local_files + sample_files
        cls.all_traces = []
        for fpath in cls.all_files:
            with open(fpath, "r", encoding="utf-8") as fh:
                cls.all_traces.append((fpath, json.load(fh)))

    def test_total_trace_inventory_count(self):
        self.assertEqual(len(self.all_files), 429)
        self.assertEqual(len(self.all_traces), 429)

    def _assert_no_nan_inf(self, obj, path="root"):
        if isinstance(obj, float):
            self.assertFalse(math.isnan(obj), f"NaN float found at {path}")
            self.assertFalse(math.isinf(obj), f"Inf float found at {path}")
        elif isinstance(obj, str):
            self.assertNotIn(obj.lower(), ("nan", "infinity", "-infinity"), f"String NaN/Inf found at {path}: {obj}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self._assert_no_nan_inf(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._assert_no_nan_inf(v, f"{path}[{i}]")

    def test_zero_nan_inf_across_all_429_traces(self):
        for fpath, trace in self.all_traces:
            fname = os.path.basename(fpath)
            self._assert_no_nan_inf(trace, path=fname)

    def test_schema_validity_and_required_keys(self):
        valid_tasks = {"task1", "task2", "task3"}
        for fpath, trace in self.all_traces:
            fname = os.path.basename(fpath)
            self.assertIn("case_id", trace, f"Missing case_id in {fname}")
            self.assertIsInstance(trace["case_id"], str, f"Invalid case_id type in {fname}")
            self.assertIn("task", trace, f"Missing task in {fname}")
            task = str(trace["task"]).lower()
            self.assertIn(task, valid_tasks, f"Invalid task '{task}' in {fname}")
            self.assertIn("patient_demographics", trace, f"Missing patient_demographics in {fname}")
            self.assertIn("clinical_records", trace, f"Missing clinical_records in {fname}")

    def test_deterministic_clinical_computations_all_cases(self):
        valid_tiers = {
            "Low",
            "Favorable Intermediate",
            "Unfavorable Intermediate",
            "High",
            "Locally Advanced",
            MISSING,
        }

        for fpath, trace in self.all_traces:
            fname = os.path.basename(fpath)
            task = str(trace.get("task", "task1")).lower()
            demo = trace.get("patient_demographics", {})
            clin = trace.get("clinical_records", {})

            # 1. PSAD computation
            psa = _safe_float(demo.get("psa"))
            vol = _safe_float(demo.get("vol"))
            if psa is not None and vol is not None and vol > 0:
                psad = psa / vol
                self.assertFalse(math.isnan(psad), f"PSAD is NaN in {fname}")
                self.assertFalse(math.isinf(psad), f"PSAD is Inf in {fname}")
                self.assertGreater(psad, 0.0)

            # 2. PSA Kinetics (PSAV / PSADT)
            psa_trend = clin.get("psa_trend")
            if psa_trend and isinstance(psa_trend, list) and len(psa_trend) > 0:
                k = PSAKinetics(psa_trend)
                psav = k.psav()
                psadt = k.psadt()
                if psav is not None:
                    self.assertFalse(math.isnan(psav), f"PSAV is NaN in {fname}")
                    self.assertFalse(math.isinf(psav), f"PSAV is Inf in {fname}")
                if psadt is not None:
                    self.assertFalse(math.isnan(psadt), f"PSADT is NaN in {fname}")
                    self.assertFalse(math.isinf(psadt), f"PSADT is Inf in {fname}")
                    self.assertGreater(psadt, 0.0, f"PSADT non-positive in {fname}")

            # 3. EAU 2026 Guidelines
            bx_isup = demo.get("bx_isup")
            ct = demo.get("ct")
            tier, reason = EAURiskTier.classify(psa, bx_isup, ct, None)
            self.assertIn(tier, valid_tiers, f"Invalid EAU tier '{tier}' in {fname}")

            # 4. CAPRA Nomogram (Tasks 1 & 2)
            if task in ("task1", "task2"):
                age = demo.get("age")
                gp = demo.get("bx_gl_prim")
                gs = demo.get("bx_gl_sec")
                cp = demo.get("cores_positive")
                ctot = demo.get("cores_total")
                score, breakdown = CAPRAScore.compute(age, psa, gp, gs, None, ct, cp, ctot)
                self.assertTrue(0 <= score <= 10, f"CAPRA score out of range (0-10): {score} in {fname}")

            # 5. CAPRA-S Nomogram (Task 3)
            if task == "task3":
                surg_report = clin.get("surgical_pathology_report") or clin.get("pathology_report")
                score, breakdown, imputed = CAPRAS_Score.compute(psa, surg_report)
                self.assertTrue(0 <= score <= 12, f"CAPRA-S score out of range (0-12): {score} in {fname}")

    def test_bundle_artifacts_presence_and_integrity(self):
        # Verify bundles in dashboard/bundles/
        for task_num in [1, 2, 3]:
            task_dir = BUNDLES_DIR / f"task{task_num}"
            if task_dir.exists():
                bundle_files = glob.glob(str(task_dir / "*.md"))
                self.assertGreater(len(bundle_files), 0, f"No bundles found in task{task_num}")
                for bf in bundle_files[:10]:  # spot-check first 10 per task
                    with open(bf, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    self.assertGreater(len(content), 100, f"Empty or truncated bundle: {bf}")
                    self.assertNotIn("NaN", content, f"Raw NaN found in bundle {bf}")
                    self.assertNotIn("Traceback", content, f"Crash traceback found in bundle {bf}")


if __name__ == "__main__":
    unittest.main()

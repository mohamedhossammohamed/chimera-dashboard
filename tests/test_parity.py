# tests/test_parity.py
"""Direct cross-language mathematical parity tests between Python backend and JS client implementations.
Executes Node.js child processes to evaluate JavaScript code against Python standard implementations.
"""

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_bundles import (
    PSAKinetics as PyPSAKinetics,
    EAURiskTier as PyEAURiskTier,
    CAPRAScore as PyCAPRAScore,
    CAPRAS_Score as PyCAPRAS_Score,
    MISSING,
)
from scripts.cohort_analytics import (
    rank_data as py_rank_data,
    pearson_r as py_pearson_r,
    spearman_rho as py_spearman_rho,
    tukey_box as py_tukey_box,
    silverman_bandwidth as py_silverman_bandwidth,
    gaussian_kde as py_gaussian_kde,
)


def run_node_eval(script: str):
    """Execute a Node.js snippet in ES module mode and return parsed JSON stdout."""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parent.parent)
    )
    return json.loads(proc.stdout)


class TestClinicalKineticsParity(unittest.TestCase):
    """Parity tests for PSA Velocity (PSAV) and PSA Doubling Time (PSADT)."""

    def test_psav_and_psadt_parity_across_trends(self):
        test_trends = [
            # Standard rising trend
            [
                {"date": "1 Jan 2020", "val": 4.0},
                {"date": "1 Jan 2021", "val": 5.0},
                {"date": "1 Jan 2022", "val": 6.0},
            ],
            # Rapid rising trend
            [
                {"date": "1 Jan 2020", "val": 2.0},
                {"date": "1 May 2020", "val": 4.0},
                {"date": "1 Sep 2020", "val": 8.0},
            ],
            # Declining trend (k <= 0)
            [
                {"date": "1 Jan 2020", "val": 6.0},
                {"date": "1 Jan 2021", "val": 5.0},
                {"date": "1 Jan 2022", "val": 4.0},
            ],
            # Insufficient points (< 3)
            [
                {"date": "1 Jan 2020", "val": 4.0},
                {"date": "1 Jan 2021", "val": 5.0},
            ],
        ]

        js_code = f"""
        import {{ PSAKinetics }} from './dashboard/js/clinical_engine.js';
        const trends = {json.dumps(test_trends)};
        const results = trends.map(t => {{
            const k = new PSAKinetics(t);
            return {{
                psav: k.calculatePSAV(),
                psadt: k.calculatePSADT(),
                insufficient: k.isInsufficient,
                nPoints: k.nPoints
            }};
        }});
        console.log(JSON.stringify(results));
        """

        js_results = run_node_eval(js_code)

        for i, trend in enumerate(test_trends):
            py_k = PyPSAKinetics(trend)
            py_psav = py_k.psav()
            py_psadt = py_k.psadt()
            js_res = js_results[i]

            self.assertEqual(py_k.insufficient, js_res["insufficient"], f"Gating mismatch on trend {i}")
            self.assertEqual(py_k.n_points, js_res["nPoints"], f"Point count mismatch on trend {i}")

            if py_psav is None:
                self.assertIsNone(js_res["psav"], f"PSAV expected None on trend {i}")
            else:
                self.assertIsNotNone(js_res["psav"], f"PSAV expected float on trend {i}")
                self.assertAlmostEqual(py_psav, js_res["psav"], places=2, msg=f"PSAV parity mismatch on trend {i}")

            if py_psadt is None:
                self.assertIsNone(js_res["psadt"], f"PSADT expected None on trend {i}")
            else:
                self.assertIsNotNone(js_res["psadt"], f"PSADT expected float on trend {i}")
                self.assertAlmostEqual(py_psadt, js_res["psadt"], delta=0.5, msg=f"PSADT parity mismatch on trend {i}")


class TestNomogramParity(unittest.TestCase):
    """Parity tests for EAU 2026, CAPRA, and CAPRA-S nomograms."""

    def test_eau_guidelines_parity_grid(self):
        test_cases = [
            {"psa": 25.0, "isup": 1, "ct": "cT1c"},
            {"psa": 15.0, "isup": 1, "ct": "cT1c"},
            {"psa": 6.0, "isup": 5, "ct": "cT3b"},
            {"psa": 6.0, "isup": 1, "ct": "cT4"},
            {"psa": 6.0, "isup": 4, "ct": "cT1c"},
            {"psa": 6.0, "isup": 1, "ct": "cT3a"},
            {"psa": 6.0, "isup": 3, "ct": "cT1c"},
            {"psa": 6.0, "isup": 1, "ct": "cT2b"},
            {"psa": 6.0, "isup": 2, "ct": "cT1c"},
            {"psa": 4.0, "isup": 1, "ct": "cT1c"},
            {"psa": 4.0, "isup": None, "ct": None},
        ]

        js_code = f"""
        import {{ EAURiskClassifier }} from './dashboard/js/clinical_engine.js';
        const cases = {json.dumps(test_cases)};
        const results = cases.map(c => EAURiskClassifier.classify(c.psa, c.isup, c.ct).tier);
        console.log(JSON.stringify(results));
        """

        js_results = run_node_eval(js_code)

        for i, c in enumerate(test_cases):
            py_tier, _ = PyEAURiskTier.classify(c["psa"], c["isup"], c["ct"], None)
            js_tier = js_results[i]
            # Map JS 'Missing' to Python MISSING
            if js_tier in ("Missing", "Indeterminate"):
                self.assertEqual(py_tier, MISSING, f"EAU tier mismatch for case {c}")
            else:
                self.assertEqual(py_tier, js_tier, f"EAU tier mismatch for case {c}")

    def test_capra_nomogram_parity(self):
        test_cases = [
            {"age": 45, "psa": 4.0, "gp": 3, "gs": 3, "ct": "cT1c", "cp": 2, "ctot": 10},
            {"age": 65, "psa": 8.0, "gp": 3, "gs": 4, "ct": "cT1c", "cp": 4, "ctot": 10},
            {"age": 72, "psa": 15.0, "gp": 4, "gs": 3, "ct": "cT3a", "cp": 6, "ctot": 12},
            {"age": 80, "psa": 35.0, "gp": 4, "gs": 5, "ct": "cT3a", "cp": 10, "ctot": 12},
            {"age": None, "psa": None, "gp": None, "gs": None, "ct": None, "cp": None, "ctot": None},
        ]

        js_code = f"""
        import {{ CAPRAScorer }} from './dashboard/js/clinical_engine.js';
        const cases = {json.dumps(test_cases)};
        const results = cases.map(c => CAPRAScorer.calculate(c.age, c.psa, c.gp, c.gs, c.ct, c.cp, c.ctot).score);
        console.log(JSON.stringify(results));
        """

        js_results = run_node_eval(js_code)

        for i, c in enumerate(test_cases):
            py_score, _ = PyCAPRAScore.compute(c["age"], c["psa"], c["gp"], c["gs"], None, c["ct"], c["cp"], c["ctot"])
            js_score = js_results[i]
            self.assertEqual(py_score, js_score, f"CAPRA score mismatch on case {c}")

    def test_capras_nomogram_parity(self):
        test_reports = [
            (4.0, "Gleason 3+3; margins negative; ECE absent; seminal vesicles negative; lymph nodes negative."),
            (8.0, "Gleason 3+4; margins positive; extraprostatic extension present; seminal vesicles not invaded; lymph nodes negative."),
            (15.0, "Gleason 4+3; margins positive; extraprostatic extension present; seminal vesicle invasion present; lymph nodes negative."),
            (35.0, "Gleason 4+5; margins positive; extraprostatic extension present; seminal vesicle invasion present; lymph node metastasis present."),
        ]

        js_code = f"""
        import {{ CAPRASScorer }} from './dashboard/js/clinical_engine.js';
        const reports = {json.dumps(test_reports)};
        const results = reports.map(r => CAPRASScorer.calculate(r[0], r[1]).score);
        console.log(JSON.stringify(results));
        """

        js_results = run_node_eval(js_code)

        for i, (psa, text) in enumerate(test_reports):
            py_score, _, _ = PyCAPRAS_Score.compute(psa, text)
            js_score = js_results[i]
            self.assertEqual(py_score, js_score, f"CAPRA-S score mismatch on report {i}")


class TestCohortMathParity(unittest.TestCase):
    """Parity tests for Tukey boxplots, Spearman rho, and Silverman KDE."""

    def test_tukey_box_parity(self):
        datasets = [
            list(range(1, 21)),
            [2.5, 3.1, 4.8, 5.2, 5.9, 6.4, 7.8, 8.1, 9.5, 12.0, 45.0],
            [10.0] * 10,
        ]

        js_code = f"""
        import {{ tukeyBox }} from './dashboard/js/cohort_engine.js';
        const datasets = {json.dumps(datasets)};
        const results = datasets.map(d => tukeyBox(d));
        console.log(JSON.stringify(results));
        """

        js_results = run_node_eval(js_code)

        for i, d in enumerate(datasets):
            py_b = py_tukey_box(d)
            js_b = js_results[i]
            self.assertAlmostEqual(py_b["q1"], js_b["q1"], places=4)
            self.assertAlmostEqual(py_b["median"], js_b["median"], places=4)
            self.assertAlmostEqual(py_b["q3"], js_b["q3"], places=4)
            self.assertAlmostEqual(py_b["iqr"], js_b["iqr"], places=4)
            self.assertAlmostEqual(py_b["whisker_lo"], js_b["whisker_lo"], places=4)
            self.assertAlmostEqual(py_b["whisker_hi"], js_b["whisker_hi"], places=4)
            self.assertEqual(len(py_b["outliers"]), len(js_b["outliers"]))

    def test_spearman_rho_parity(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0]

        js_code = f"""
        import {{ spearmanRho }} from './dashboard/js/cohort_engine.js';
        const res = spearmanRho({json.dumps(x)}, {json.dumps(y)});
        console.log(JSON.stringify({{ rho: res }}));
        """

        js_res = run_node_eval(js_code)
        py_res = py_spearman_rho(x, y)
        self.assertAlmostEqual(py_res, js_res["rho"], places=4)

    def test_silverman_kde_parity(self):
        vals = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0]
        grid = [10.0, 15.0, 20.0, 25.0, 30.0]

        js_code = f"""
        import {{ silvermanBandwidth, gaussianKDE }} from './dashboard/js/cohort_engine.js';
        const vals = {json.dumps(vals)};
        const grid = {json.dumps(grid)};
        const h = silvermanBandwidth(vals);
        const dens = gaussianKDE(vals, h, grid);
        console.log(JSON.stringify({{ h: h, densities: dens }}));
        """

        js_res = run_node_eval(js_code)
        py_h = py_silverman_bandwidth(vals)
        py_dens = py_gaussian_kde(vals, py_h, grid)

        self.assertAlmostEqual(py_h, js_res["h"], places=4)
        for d_py, d_js in zip(py_dens, js_res["densities"]):
            self.assertAlmostEqual(d_py, d_js, places=4)


if __name__ == "__main__":
    unittest.main()

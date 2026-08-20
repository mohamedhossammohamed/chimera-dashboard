# tests/test_cohort_math.py
"""Unit and boundary test suite for cohort analytics and matrix operations:
- Covariance eigendecomposition PCA
- Midrank Spearman rank-order correlation (rho)
- Lance-Williams Ward hierarchical clustering
- Silverman rule-of-thumb bandwidth Gaussian KDE
- Tukey 5-number boxplot distributions
- Deterministic verification of 9 cohort JSON artifacts (0 NaN/Inf)
"""

import json
import math
import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.cohort_analytics import (
    rank_data,
    pearson_r,
    spearman_rho,
    ward_cluster,
    compute_pca,
    tukey_box,
    silverman_bandwidth,
    gaussian_kde,
    safe_float,
    NUMERIC_VARS,
    ROOT,
    OUT_DIR,
)


class TestRankAndCorrelation(unittest.TestCase):
    """Tests for mid-rank transformation, Pearson r, and Spearman rho."""

    def test_rank_data_unique_values(self):
        vals = [10.0, 20.0, 30.0, 40.0]
        ranks = rank_data(vals)
        self.assertEqual(ranks, [1.0, 2.0, 3.0, 4.0])

    def test_rank_data_with_ties(self):
        vals = [10.0, 20.0, 20.0, 30.0]
        ranks = rank_data(vals)
        self.assertEqual(ranks, [1.0, 2.5, 2.5, 4.0])

        vals_all_tied = [5.0, 5.0, 5.0, 5.0]
        self.assertEqual(rank_data(vals_all_tied), [2.5, 2.5, 2.5, 2.5])

    def test_pearson_r_bounds_and_cases(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pos = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.assertAlmostEqual(pearson_r(x, y_pos), 1.0, places=6)

        y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
        self.assertAlmostEqual(pearson_r(x, y_neg), -1.0, places=6)

        # N < 3 guard
        self.assertIsNone(pearson_r([1.0, 2.0], [3.0, 4.0]))

        # Zero denominator / zero variance guard
        self.assertIsNone(pearson_r([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]))

    def test_spearman_rho_monotonic_nonlinear(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y = [1.0, 8.0, 27.0, 64.0, 125.0, 216.0]  # y = x^3
        # Pearson is < 1.0, but Spearman rho is exactly 1.0
        r_pearson = pearson_r(x, y)
        r_spearman = spearman_rho(x, y)
        self.assertLess(r_pearson, 0.95)
        self.assertAlmostEqual(r_spearman, 1.0, places=6)

    def test_spearman_rho_min_observations_guard(self):
        # < 5 pairs -> None
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        self.assertIsNone(spearman_rho(x, y))

        # 5 pairs -> valid
        x5 = [1.0, 2.0, 3.0, 4.0, 5.0]
        y5 = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.assertAlmostEqual(spearman_rho(x5, y5), 1.0, places=6)

    def test_spearman_rho_missing_values(self):
        x = [1.0, 2.0, None, 4.0, 5.0, 6.0]
        y = [2.0, None, 6.0, 8.0, 10.0, 12.0]
        # Only 4 complete pairs (indices 0, 3, 4, 5) -> < 5 complete pairs -> None
        self.assertIsNone(spearman_rho(x, y))


class TestWardHierarchicalClustering(unittest.TestCase):
    """Tests for Ward agglomerative hierarchical clustering via Lance-Williams."""

    def test_ward_cluster_ordering_2_vars(self):
        rho = [
            [1.0, 0.8],
            [0.8, 1.0]
        ]
        order = ward_cluster(rho, ["var1", "var2"])
        self.assertEqual(len(order), 2)
        self.assertEqual(set(order), {0, 1})

    def test_ward_cluster_grouping(self):
        # 4 variables: (0, 1) highly correlated, (2, 3) highly correlated, others orthogonal
        rho = [
            [1.0, 0.95, 0.0, 0.0],
            [0.95, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.90],
            [0.0, 0.0, 0.90, 1.0]
        ]
        var_names = ["a1", "a2", "b1", "b2"]
        order = ward_cluster(rho, var_names)
        self.assertEqual(len(order), 4)
        self.assertEqual(set(order), {0, 1, 2, 3})


class TestPCAEigendecomposition(unittest.TestCase):
    """Tests for Covariance Eigendecomposition PCA."""

    def test_pca_variance_explained_and_projection(self):
        # Simple 2D dataset with clear variance along first dimension
        np.random.seed(42)
        x1 = np.linspace(-10, 10, 50)
        x2 = np.random.normal(0, 0.1, 50)
        vectors = np.column_stack([x1, x2]).tolist()
        case_ids = [f"C{i}" for i in range(50)]
        targets = ["yes"] * 50
        tasks = ["task1"] * 50

        res = compute_pca(vectors, case_ids, targets, tasks)
        self.assertEqual(res["n"], 50)
        self.assertEqual(len(res["points" ]), 50)
        # PC1 should explain > 95% of the variance
        self.assertGreater(res["variance_explained" ][0], 95.0)
        self.assertAlmostEqual(res["cumulative_2pc" ], 100.0, places=1)

    def test_pca_empty_guard(self):
        res = compute_pca([], [], [], [])
        self.assertEqual(res["n"], 0)
        self.assertEqual(res["points"], [])


class TestTukeyBoxplot(unittest.TestCase):
    """Tests for Tukey 5-number summary and 1.5*IQR fences."""

    def test_tukey_box_standard_dataset(self):
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # n=10, Q1 = 3.25, Med = 5.5, Q3 = 7.75, IQR = 4.5
        # lo_whisker = 3.25 - 1.5*4.5 = -3.5 -> actual = 1.0
        # hi_whisker = 7.75 + 1.5*4.5 = 14.5 -> actual = 10.0
        vals = list(range(1, 11))
        b = tukey_box(vals)
        self.assertAlmostEqual(b["q1"], 3.25, places=4)
        self.assertAlmostEqual(b["median"], 5.5, places=4)
        self.assertAlmostEqual(b["q3"], 7.75, places=4)
        self.assertAlmostEqual(b["iqr"], 4.5, places=4)
        self.assertEqual(b["whisker_lo"], 1.0)
        self.assertEqual(b["whisker_hi"], 10.0)
        self.assertEqual(b["outliers"], [])

    def test_tukey_box_with_outliers(self):
        vals = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 100.0]
        b = tukey_box(vals)
        self.assertIn(100.0, b["outliers" ])
        self.assertLessEqual(b["whisker_hi"], 15.0)

    def test_tukey_box_empty(self):
        self.assertIsNone(tukey_box([]))


class TestSilvermanKDE(unittest.TestCase):
    """Tests for Silverman bandwidth and Gaussian Kernel Density Estimation."""

    def test_silverman_bandwidth_standard_normal(self):
        # Sample standard normal
        np.random.seed(42)
        vals = np.random.normal(0, 1, 100).tolist()
        h = silverman_bandwidth(vals)
        self.assertGreater(h, 0.2)
        self.assertLess(h, 0.6)

    def test_silverman_bandwidth_zero_variance(self):
        vals = [5.0] * 50
        h = silverman_bandwidth(vals)
        self.assertEqual(h, 0.0)

    def test_gaussian_kde_density_properties(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        h = silverman_bandwidth(vals)
        eval_pts = np.linspace(0, 60, 100).tolist()
        densities = gaussian_kde(vals, h, eval_pts)

        self.assertEqual(len(densities), 100)
        # All densities >= 0
        for d in densities:
            self.assertGreaterEqual(d, 0.0)

        # Numerical integration via trapezoid rule should be close to 1.0
        dx = eval_pts[1] - eval_pts[0]
        integral = sum(densities) * dx
        self.assertGreater(integral, 0.90)
        self.assertLess(integral, 1.10)


class TestCohortJSONArtifacts(unittest.TestCase):
    """Verify all 9 precomputed JSON artifacts in dashboard/cohort/ (0 NaN/Inf, valid schema)."""

    ARTIFACT_FILES = [
        "composition.json",
        "correlation.json",
        "missingness.json",
        "pca_mri.json",
        "pca_biopsy.json",
        "raincloud_psa.json",
        "raincloud_psad.json",
        "raincloud_vol.json",
        "raincloud_age.json",
    ]

    def _assert_no_nan_inf(self, obj, path="root"):
        if isinstance(obj, float):
            self.assertFalse(math.isnan(obj), f"NaN found at {path}")
            self.assertFalse(math.isinf(obj), f"Inf found at {path}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self._assert_no_nan_inf(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._assert_no_nan_inf(v, f"{path}[{i}]")

    def test_all_cohort_artifacts_exist_and_clean(self):
        for fname in self.ARTIFACT_FILES:
            fpath = os.path.join(OUT_DIR, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing cohort artifact: {fname}")
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._assert_no_nan_inf(data, path=fname)

    def test_composition_structure(self):
        fpath = os.path.join(OUT_DIR, "composition.json")
        with open(fpath, "r", encoding="utf-8") as fh:
            comp = json.load(fh)
        self.assertIn("tasks", comp)
        self.assertIn("task1", comp["tasks"])
        self.assertIn("task2", comp["tasks"])
        self.assertIn("task3", comp["tasks"])
        total_cases = sum(comp["tasks"][t]["total"] for t in ["task1", "task2", "task3"])
        self.assertEqual(total_cases, 423)

    def test_correlation_matrix_properties(self):
        fpath = os.path.join(OUT_DIR, "correlation.json")
        with open(fpath, "r", encoding="utf-8") as fh:
            corr = json.load(fh)
        self.assertIn("variables", corr)
        self.assertIn("matrix", corr)
        k = len(corr["variables"])
        mat = corr["matrix"]
        self.assertEqual(len(mat), k)
        for i in range(k):
            self.assertEqual(len(mat[i]), k)
            self.assertEqual(mat[i][i], 1.0)  # Diagonal is 1.0


if __name__ == "__main__":
    unittest.main()

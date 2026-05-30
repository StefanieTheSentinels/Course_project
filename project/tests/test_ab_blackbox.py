"""
tests/test_ab_blackbox.py
-------------------------
Unit tests for ab_blackbox.

    pytest tests/
    python tests/test_ab_blackbox.py
"""

import os
import sys
import math
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ab_blackbox import (
    BlackBox,
    run_ab_test,
    two_proportion_ztest,
    calibrate_linear_model,
    generate_synthetic_dataset,
    get_model,
)
from ab_blackbox.model import LinearButtonModel


# Test 1: two_proportion_z-test against a known textbook example
class TestTwoProportionZTest(unittest.TestCase):
    """
    Textbook example:
        n_A=100, x_A=10  -> p_A=0.10
        n_B=100, x_B=20  -> p_B=0.20
        delta=0.10
        pooled p=0.15, SE=sqrt(0.15*0.85*2/100)=0.050498
        z = 0.10/0.050498 = 1.980
        p ~ 0.0477 (two-sided) -> significant at alpha=0.05
    """

    def setUp(self):
        self.result = two_proportion_ztest(100, 10, 100, 20)

    def test_delta(self):
        self.assertAlmostEqual(self.result["delta"], 0.10, places=9)

    def test_significant_at_alpha05(self):
        self.assertTrue(self.result["significant"])

    def test_z_value(self):
        expected_z = 0.10 / 0.050498
        self.assertAlmostEqual(self.result["z_statistic"], expected_z, delta=0.01)

    def test_ci_excludes_zero(self):
        lo, hi = self.result["ci_95"]
        self.assertGreater(lo, 0.0)

    def test_no_effect_not_significant(self):
        r = two_proportion_ztest(1000, 100, 1000, 100)
        self.assertFalse(r["significant"])
        self.assertAlmostEqual(r["delta"], 0.0, places=9)

    def test_relative_lift(self):
        # p_B/p_A - 1 = 0.20/0.10 - 1 = 1.0 (100% lift)
        self.assertAlmostEqual(self.result["relative_lift"], 1.0, places=6)


# Test 2: calibration round-trip
class TestCalibrationRoundTrip(unittest.TestCase):

    def test_fit_errors_are_zero(self):
        """Calibrated model should reproduce input CTRs exactly (zero fit error)."""
        df  = generate_synthetic_dataset(n=5000, ctr_A=0.12, ctr_B=0.17, seed=0)
        cal = calibrate_linear_model(df)
        self.assertAlmostEqual(cal["fit_quality"]["ctr_fit_error_A"], 0.0, places=6)
        self.assertAlmostEqual(cal["fit_quality"]["ctr_fit_error_B"], 0.0, places=6)

    def test_simulated_ctr_close_to_target(self):
        """
        noise_std=0, n=10000: simulated CTR should be within 4 binomial sigmas
        of the calibration target.
        """
        df  = generate_synthetic_dataset(n=5000, ctr_A=0.12, ctr_B=0.17, seed=1)
        cal = calibrate_linear_model(df)

        params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
        params_B = {"color": 1.0, "size": 0.0, "text": 0.0}

        box    = BlackBox(model=cal["model"], n_users=10_000, noise_std=0.0, seed=42)
        result = run_ab_test(box, params_A, params_B)

        tol = 4 * math.sqrt(0.15 * 0.85 / 10_000)
        self.assertAlmostEqual(
            result.control.ctr, cal["fit_quality"]["empirical_ctr_A"], delta=tol
        )
        self.assertAlmostEqual(
            result.treatment.ctr, cal["fit_quality"]["empirical_ctr_B"], delta=tol
        )

    def test_calibrated_delta_positive(self):
        """B should beat A when ctr_B > ctr_A."""
        df  = generate_synthetic_dataset(n=2000, ctr_A=0.10, ctr_B=0.20, seed=99)
        cal = calibrate_linear_model(df)
        params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
        params_B = {"color": 1.0, "size": 0.0, "text": 0.0}
        box    = BlackBox(model=cal["model"], n_users=5000, noise_std=0.0, seed=7)
        result = run_ab_test(box, params_A, params_B)
        self.assertGreater(result.treatment.ctr, result.control.ctr)


# Test 3: deterministic simulator (noise_std=0)
class TestDeterministicSimulator(unittest.TestCase):

    def test_ctr_converges_to_theoretical(self):
        """Large-n, zero-noise: observed CTR should match model p_click."""
        model  = LinearButtonModel(base_click=0.20, w_color=0.05)
        params = {"color": 1.0, "size": 0.0, "text": 0.0}
        theo   = model._eval(model.click_expr, params)   # 0.25

        box    = BlackBox(model=model, n_users=100_000, noise_std=0.0, seed=0)
        result = box(params)

        tol = 4 * math.sqrt(theo * (1 - theo) / 100_000)
        self.assertAlmostEqual(result.ctr, theo, delta=tol)

    def test_zero_noise_is_reproducible(self):
        """Same seed, same params -> identical results."""
        model  = get_model("linear")
        params = {"color": 0.5, "size": 0.5, "text": 0.5}

        r1 = BlackBox(model=model, n_users=1000, noise_std=0.0, seed=42)(params)
        r2 = BlackBox(model=model, n_users=1000, noise_std=0.0, seed=42)(params)

        self.assertEqual(r1.ctr, r2.ctr)
        self.assertEqual(r1.n_clicks, r2.n_clicks)

    def test_conversion_conditional_on_click(self):
        """Conversions <= clicks always."""
        model  = LinearButtonModel(base_click=0.30, base_conv=0.50)
        params = {"color": 0.0, "size": 0.0, "text": 0.0}
        box    = BlackBox(model=model, n_users=5000, noise_std=0.0, seed=1)
        result = box(params)
        self.assertLessEqual(result.n_conversions, result.n_clicks)

    def test_no_clicks_no_conversions(self):
        """base_click=0 -> no clicks -> no conversions."""
        model  = LinearButtonModel(base_click=0.0, w_color=0.0,
                                   base_conv=0.99, w_conv_color=0.0)
        params = {"color": 0.0, "size": 0.0, "text": 0.0}
        box    = BlackBox(model=model, n_users=10_000, noise_std=0.0, seed=5)
        result = box(params)
        self.assertEqual(result.n_clicks, 0)
        self.assertEqual(result.n_conversions, 0)

    def test_theoretical_effect_matches_eval(self):
        """theoretical_effect should agree with _eval for both arms."""
        model    = get_model("linear")
        params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
        params_B = {"color": 1.0, "size": 0.5, "text": 0.5}
        effects  = model.theoretical_effect(params_A, params_B)
        self.assertAlmostEqual(
            effects["p_click_A"],
            model._eval(model.click_expr, params_A), places=9
        )
        self.assertAlmostEqual(
            effects["p_click_B"],
            model._eval(model.click_expr, params_B), places=9
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

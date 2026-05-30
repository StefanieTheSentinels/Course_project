"""
tests/test_ab_blackbox.py
-------------------------
Unit tests for ab_blackbox (legacy + new pipeline).

Run:
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
    p_click,
    contrast_ratio,
    relative_luminance,
    f_contrast,
    f_size,
    f_scroll_decay,
    total_penalty,
    generate_full_synthetic_dataset,
    synthetic_dataset_summary,
    build_feature_vector,
    build_feature_matrix,
    FEATURE_NAMES,
    train_and_evaluate,
    TrainedMLModel,
    FullSyntheticModel,
)
from ab_blackbox.model import LinearButtonModel


class TestTwoProportionZTest(unittest.TestCase):

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
        lo, _ = self.result["ci_95"]
        self.assertGreater(lo, 0.0)

    def test_no_effect_not_significant(self):
        r = two_proportion_ztest(1000, 100, 1000, 100)
        self.assertFalse(r["significant"])
        self.assertAlmostEqual(r["delta"], 0.0, places=9)

    def test_relative_lift(self):
        self.assertAlmostEqual(self.result["relative_lift"], 1.0, places=6)





class TestGeneratingFormula(unittest.TestCase):

    def test_white_on_white_penalized(self):
        """White-on-white (contrast=1) must score below an identical
        high-contrast config: the contrast penalty bites."""
        base = {
            "btn_w": 100, "btn_h": 50, "font_size": 16,
            "text_quality": 1.0, "whitespace_ratio": 0.3,
            "scroll_to_button": 0.0, "hour": 12, "device": "desktop",
        }
        p_white = p_click({**base, "rgb_bg": (255,255,255), "rgb_text": (255,255,255)})
        p_black = p_click({**base, "rgb_bg": (255,255,255), "rgb_text": (0,0,0)})
        self.assertLess(p_white, p_black)
        self.assertLess(p_white, 0.5 * p_black)

    def test_black_on_white_high_ctr(self):
        params = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 200, "btn_h": 60, "font_size": 18,
            "text_quality": 1.0, "whitespace_ratio": 0.35,
            "scroll_to_button": 0.0, "hour": 13, "device": "desktop",
        }
        self.assertGreater(p_click(params), 0.20)

    def test_scroll_decay_monotonic(self):
        base = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 150, "btn_h": 50, "font_size": 16,
            "text_quality": 0.8, "whitespace_ratio": 0.3,
            "hour": 12, "device": "desktop",
        }
        p_top    = p_click({**base, "scroll_to_button": 0.0})
        p_mid    = p_click({**base, "scroll_to_button": 0.5})
        p_bottom = p_click({**base, "scroll_to_button": 1.0})
        self.assertGreater(p_top, p_mid)
        self.assertGreater(p_mid, p_bottom)

    def test_text_overflow_penalty(self):
        base = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 200, "text_quality": 0.8, "whitespace_ratio": 0.3,
            "scroll_to_button": 0.0, "hour": 12, "device": "desktop",
        }
        p_ok  = p_click({**base, "btn_h": 60, "font_size": 16})
        p_bad = p_click({**base, "btn_h": 20, "font_size": 40})
        self.assertGreater(p_ok, p_bad)

    def test_weights_override(self):
        params = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 200, "btn_h": 60, "font_size": 18,
            "text_quality": 1.0, "whitespace_ratio": 0.3,
            "scroll_to_button": 0.0, "hour": 12, "device": "desktop",
        }
        p_low  = p_click(params, weights={"beta_text": 0.1})
        p_high = p_click(params, weights={"beta_text": 2.0})
        self.assertGreater(p_high, p_low)


class TestContrastRatio(unittest.TestCase):

    def test_white_white_is_one(self):
        self.assertAlmostEqual(contrast_ratio((255,255,255), (255,255,255)),
                               1.0, places=3)

    def test_black_white_is_21(self):
        self.assertAlmostEqual(contrast_ratio((255,255,255), (0,0,0)),
                               21.0, places=1)

    def test_symmetric(self):
        cr1 = contrast_ratio((100, 100, 100), (200, 200, 200))
        cr2 = contrast_ratio((200, 200, 200), (100, 100, 100))
        self.assertAlmostEqual(cr1, cr2, places=6)


class TestDatasetGenerator(unittest.TestCase):

    def test_dataset_shape(self):
        df = generate_full_synthetic_dataset(n=200, seed=0)
        self.assertEqual(len(df), 200)
        for col in ["click", "p_true", "device", "btn_w", "scroll_to_button"]:
            self.assertIn(col, df.columns)

    def test_ctr_within_expected_range(self):
        df = generate_full_synthetic_dataset(n=2000, seed=0)
        s  = synthetic_dataset_summary(df)
        self.assertGreater(s["ctr"], 0.01)
        self.assertLess(s["ctr"], 0.30)

    def test_clicks_are_binary(self):
        df = generate_full_synthetic_dataset(n=300, seed=0)
        self.assertTrue(((df["click"] == 0) | (df["click"] == 1)).all())

    def test_p_true_consistent_with_clicks(self):
        df = generate_full_synthetic_dataset(n=5000, seed=0, extra_noise=0.0)
        s  = synthetic_dataset_summary(df)
        # mean(click) converges to E[p_true]; account for variance of p across data
        self.assertAlmostEqual(s["ctr"], s["mean_p_true"], delta=0.02)

    def test_custom_pclick_fn(self):
        df = generate_full_synthetic_dataset(
            n=300, seed=0, p_click_fn=lambda params: 0.5
        )
        self.assertAlmostEqual(df["p_true"].mean(), 0.5, places=6)


class TestVectorizedFeatures(unittest.TestCase):

    def test_matrix_matches_vector(self):
        df = generate_full_synthetic_dataset(n=100, seed=3)
        X_mat = build_feature_matrix(df)
        for i in range(10):
            r = df.iloc[i]
            params = {
                "rgb_bg":   (int(r["bg_r"]),   int(r["bg_g"]),   int(r["bg_b"])),
                "rgb_text": (int(r["text_r"]), int(r["text_g"]), int(r["text_b"])),
                "btn_w": r["btn_w"], "btn_h": r["btn_h"],
                "font_size": r["font_size"],
                "text_quality": r["text_quality"],
                "whitespace_ratio": r["whitespace_ratio"],
                "scroll_to_button": r["scroll_to_button"],
                "hour": r["hour"], "device": r["device"],
            }
            v = build_feature_vector(params)
            np.testing.assert_allclose(X_mat[i], v, rtol=1e-6)


class TestFeatureBuilder(unittest.TestCase):

    def test_feature_count(self):
        params = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 100, "btn_h": 50, "font_size": 16,
            "text_quality": 0.5, "whitespace_ratio": 0.2,
            "scroll_to_button": 0.3, "hour": 14, "device": "mobile",
        }
        self.assertEqual(len(build_feature_vector(params)), len(FEATURE_NAMES))

    def test_device_mobile_binary(self):
        base = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 100, "btn_h": 50, "font_size": 16,
            "text_quality": 0.5, "whitespace_ratio": 0.2,
            "scroll_to_button": 0.3, "hour": 14,
        }
        idx = FEATURE_NAMES.index("device_mobile")
        self.assertEqual(build_feature_vector({**base, "device": "mobile"})[idx], 1.0)
        self.assertEqual(build_feature_vector({**base, "device": "desktop"})[idx], 0.0)


class TestTrainingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = generate_full_synthetic_dataset(n=5000, seed=42)
        cls.tr = train_and_evaluate(cls.df, cv_folds=3, verbose=False)

    def test_all_models_trained(self):
        for name in ["Logistic", "Logistic_L2"]:
            self.assertIn(name, self.tr["results"])

    def test_auc_above_random(self):
        for name, info in self.tr["results"].items():
            self.assertGreater(info["cv_auc_mean"], 0.55,
                               f"{name} AUC too low: {info['cv_auc_mean']}")

    def test_best_model_predicts_proba(self):
        params = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 200, "btn_h": 60, "font_size": 18,
            "text_quality": 0.9, "whitespace_ratio": 0.3,
            "scroll_to_button": 0.1, "hour": 13, "device": "desktop",
        }
        X = build_feature_vector(params).reshape(1, -1)
        proba = self.tr["best_model"].predict_proba(X)[0, 1]
        self.assertGreaterEqual(proba, 0.0)
        self.assertLessEqual(proba, 1.0)


class TestTrainedMLBlackBox(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = generate_full_synthetic_dataset(n=5000, seed=42)
        tr = train_and_evaluate(df, cv_folds=3, verbose=False)
        cls.ml  = TrainedMLModel(classifier=tr["best_model"],
                                 feature_builder=build_feature_vector)
        cls.box = BlackBox(model=cls.ml, n_users=2000, noise_std=0.0, seed=0)

    def test_blackbox_returns_valid_result(self):
        params = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 200, "btn_h": 60, "font_size": 18,
            "text_quality": 0.9, "whitespace_ratio": 0.3,
            "scroll_to_button": 0.1, "hour": 13, "device": "desktop",
        }
        res = self.box(params)
        self.assertEqual(res.n_users, 2000)
        self.assertGreaterEqual(res.ctr, 0.0)
        self.assertLessEqual(res.ctr, 1.0)

    def test_good_config_beats_bad(self):
        good = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 200, "btn_h": 60, "font_size": 18,
            "text_quality": 0.95, "whitespace_ratio": 0.35,
            "scroll_to_button": 0.0, "hour": 13, "device": "desktop",
        }
        bad = {
            "rgb_bg": (200, 200, 200), "rgb_text": (180, 180, 180),
            "btn_w": 30, "btn_h": 20, "font_size": 20,
            "text_quality": 0.1, "whitespace_ratio": 0.02,
            "scroll_to_button": 0.9, "hour": 3, "device": "mobile",
        }
        self.assertGreater(self.box(good).ctr, self.box(bad).ctr)


class TestModelEdgeCases(unittest.TestCase):

    def test_single_class_predict_proba(self):
        """TrainedMLModel must not crash when classifier saw one class.
        RandomForest can fit a single-class target (LogisticRegression cannot),
        and its predict_proba returns shape (n, 1)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        df = generate_full_synthetic_dataset(n=50, seed=1)
        X  = build_feature_matrix(df)
        y  = np.zeros(len(df), dtype=int)
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=10,
                                                        random_state=0))])
        pipe.fit(X, y)
        ml = TrainedMLModel(classifier=pipe, feature_builder=build_feature_vector)
        params = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 200, "btn_h": 60, "font_size": 18,
            "text_quality": 0.9, "whitespace_ratio": 0.3,
            "scroll_to_button": 0.1, "hour": 13, "device": "desktop",
        }
        p = ml.predict_proba(params)  # must not raise
        self.assertEqual(p, 0.0)  # only class 0 seen -> p(click)=0


class TestOracle(unittest.TestCase):

    def test_oracle_matches_p_click(self):
        params = {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 150, "btn_h": 50, "font_size": 16,
            "text_quality": 0.8, "whitespace_ratio": 0.3,
            "scroll_to_button": 0.2, "hour": 12, "device": "desktop",
        }
        oracle = FullSyntheticModel()
        self.assertAlmostEqual(oracle.predict_proba(params),
                               p_click(params), places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)

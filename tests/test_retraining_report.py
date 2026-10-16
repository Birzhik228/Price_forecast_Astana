import json
import unittest

import numpy as np
import pandas as pd

from retraining_report import build_segments, compare_predictions, render_html


def frame(n=4):
    return pd.DataFrame({"record_id": [f"id-{i}" for i in range(n)], "price": np.arange(n) * 100.0 + 1000,
                         "area": 50., "rooms": 2., "floor": 3., "total_floors": 9., "year_built": 2020.,
                         "district": "Нура", "building_type": "монолитный", "condition": "ремонт",
                         "residential_complex": "Тест", "house_key": [f"house-{i}" for i in range(n)],
                         "group": [f"apartment-{i}" for i in range(n)]})


def predictions(data, offset=10.):
    return {name: data.price.to_numpy() + offset for name in ("current", "repair_only", "trained_geo")}


class RetrainingReportTests(unittest.TestCase):
    def test_identity_and_constant_improvement_with_reproducible_intervals(self):
        data = frame()
        pred = predictions(data)
        pred["trained_geo"] = data.price.to_numpy() + 5
        groups = data.house_key
        result = compare_predictions(data, pred, groups)
        self.assertEqual(result, compare_predictions(data, pred, groups))
        self.assertEqual(result["paired"]["repair_only"]["mae_delta_ci95_tenge"], [0., 0.])
        self.assertEqual(result["paired"]["repair_only"]["tied_rows"], len(data))
        improved = result["paired"]["trained_geo"]
        self.assertEqual(improved["mae_delta_tenge"], -5.)
        self.assertEqual(improved["mae_delta_pct"], -50.)
        self.assertEqual(improved["mae_delta_ci95_tenge"], [-5., -5.])
        self.assertEqual(improved["improved_rows"], len(data))
        json.dumps(result, allow_nan=False)

    def test_bootstrap_samples_whole_unequal_clusters_and_keeps_row_weighting(self):
        data = frame(100)
        pred = predictions(data, 20.)
        pred["repair_only"] = pred["current"] + np.r_[np.repeat(10., 99), -10.]
        labels = pd.Series(["large"] * 99 + ["small"], index=data.index)
        result = compare_predictions(data, pred, labels)
        paired = result["paired"]["repair_only"]
        self.assertAlmostEqual(paired["mae_delta_tenge"], 9.8)
        # A row bootstrap cannot recover this interval: entire houses move together.
        self.assertEqual(paired["mae_delta_ci95_tenge"], [-10., 10.])
        self.assertEqual(result["bootstrap"]["clusters"], 2)

    def test_zero_baseline_and_constant_target_have_explicit_undefined_metrics(self):
        data = frame()
        data["price"] = 1000.
        result = compare_predictions(data, predictions(data, 0.), data.house_key)
        self.assertIsNone(result["overall"]["current"]["r2"])
        self.assertIsNone(result["paired"]["repair_only"]["mae_delta_pct"])
        self.assertIsNone(result["paired"]["repair_only"]["mae_delta_ci95_pct"])
        json.dumps(result, allow_nan=False)

    def test_metrics_use_tenge_and_strict_twenty_percent_boundary(self):
        data = frame(2)
        data["price"] = [100., 200.]
        pred = {key: np.array([120., 250.]) for key in ("current", "repair_only", "trained_geo")}
        score = compare_predictions(data, pred, data.house_key)["overall"]["current"]
        self.assertEqual(score["mae_tenge"], 35.)
        self.assertEqual(score["median_ape_pct"], 22.5)
        self.assertEqual(score["over_20_n"], 1)
        self.assertEqual(score["over_20_pct"], 50.)
        self.assertAlmostEqual(score["rmse_tenge"], np.sqrt(1450))
        self.assertAlmostEqual(score["r2"], .42)

    def test_bad_predictions_and_targets_fail_instead_of_broadcasting(self):
        data = frame()
        for invalid in (np.zeros((len(data), 1)), np.zeros(len(data)-1), np.array([1., 2., 3., np.nan]),
                        np.array([1., 2., 3., np.inf]), np.ones(len(data), dtype=bool)):
            with self.subTest(shape=invalid.shape):
                pred = predictions(data)
                pred["repair_only"] = invalid
                with self.assertRaises(ValueError):
                    compare_predictions(data, pred, data.house_key)
        for price in (0., -1., np.nan, np.inf):
            bad = data.copy()
            bad.loc[0, "price"] = price
            with self.assertRaises(ValueError):
                compare_predictions(bad, predictions(data), data.house_key)
        with self.assertRaises(ValueError):
            compare_predictions(data, {"current": np.zeros(len(data))}, data.house_key)

    def test_duplicate_identity_or_index_is_rejected(self):
        data = frame()
        bad = data.copy()
        bad.loc[1, "record_id"] = bad.loc[0, "record_id"]
        with self.assertRaises(ValueError):
            compare_predictions(bad, predictions(data), bad.house_key)
        bad = data.copy()
        bad.index = [0, 0, 1, 2]
        with self.assertRaises(ValueError):
            compare_predictions(bad, predictions(data), bad.house_key)

    def test_cluster_and_series_prediction_indices_cannot_be_silently_realigned(self):
        data = frame()
        with self.assertRaises(ValueError):
            compare_predictions(data, predictions(data), data.house_key.iloc[::-1])
        missing = data.house_key.copy()
        missing.iloc[0] = None
        with self.assertRaises(ValueError):
            compare_predictions(data, predictions(data), missing)
        pred = predictions(data)
        pred["trained_geo"] = pd.Series(pred["trained_geo"], index=data.index[::-1])
        with self.assertRaises(ValueError):
            compare_predictions(data, pred, data.house_key)

    def test_segments_share_membership_and_respect_non_default_index(self):
        data = frame(31)
        data.index = pd.Index(np.arange(31) + 100)
        data.loc[100, "district"] = "Алматы"
        data.loc[100, "area"] = 100.
        data.loc[100, "condition"] = None
        pred = predictions(data)
        pred["repair_only"] = data.price.to_numpy() + 5
        geography = pd.Series("Нет пригодных координат", index=data.index)
        segments = build_segments(data, pred, {"тест"}, geography)
        counts = segments.pivot(index=["dimension", "segment"], columns="model", values="n")
        self.assertTrue(counts.eq(counts.current, axis=0).all().all())
        district = segments[segments.dimension.eq("district")]
        self.assertTrue(district[district.segment.eq("Алматы")].small_group.all())
        self.assertFalse(district[district.segment.eq("Нура")].small_group.any())
        self.assertTrue(segments[segments.model.eq("repair_only")].mae_delta_tenge.eq(-5.).all())
        with self.assertRaises(ValueError):
            build_segments(data, pred, {"тест"}, geography.iloc[::-1])

    def test_html_escapes_data_and_reports_reuse_without_missing_values(self):
        data = frame()
        data["district"] = '<script>alert("x")</script>'
        pred = predictions(data)
        result = compare_predictions(data, pred, data.house_key)
        segments = build_segments(data, pred, {"тест"}, pd.Series("Нет координат", index=data.index))
        html = render_html({"comparison": result, "model_labels": {"current": "<current>"},
                            "training_rows": 19269, "holdout_rows": len(data),
                            "limitations": ["Runner limitation"]}, segments)
        self.assertIn("4 одинаковых архивных объявлений", html)
        self.assertIn("ранее изученном holdout", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("NaN", html)
        self.assertNotIn("None", html)
        self.assertIn("2000 повторов", html)
        self.assertIn("&lt;current&gt;", html)
        self.assertIn("Runner limitation", html)


if __name__ == "__main__":
    unittest.main()

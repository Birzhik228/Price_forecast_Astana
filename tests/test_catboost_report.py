import json
import unittest

import numpy as np
import pandas as pd

import retraining_report
from catboost_report import build_segments, compare_predictions, render_html


def frame(n=4):
    return pd.DataFrame({"record_id": [f"id-{i}" for i in range(n)], "price": np.arange(n) * 100. + 1000.,
                         "area": 50., "rooms": 2., "floor": 3., "total_floors": 9., "year_built": 2020.,
                         "district": "Нура", "building_type": "монолитный", "condition": "ремонт",
                         "residential_complex": "Тест", "house_key": [f"h-{i}" for i in range(n)],
                         "group": [f"a-{i}" for i in range(n)]})


def predictions(data, rf=False):
    keys = ["current", "catboost_mae", "catboost_log"] + (["rf_control"] if rf else [])
    return {key: data.price.to_numpy() + 20. for key in keys}


class CatBoostReportTests(unittest.TestCase):
    def test_optional_rf_and_constant_improvement_without_global_mutation(self):
        data = frame()
        original_keys = retraining_report.MODEL_KEYS
        for rf in (False, True):
            pred = predictions(data, rf)
            pred["catboost_mae"] -= 5.
            result = compare_predictions(data, pred, data.house_key)
            self.assertEqual(set(result["overall"]), set(pred))
            self.assertEqual(set(result["paired"]), set(pred) - {"current"})
            self.assertEqual(result["paired"]["catboost_mae"]["mae_delta_tenge"], -5.)
            self.assertEqual(result["paired"]["catboost_mae"]["mae_delta_ci95_tenge"], [-5., -5.])
            self.assertEqual(result["paired"]["catboost_log"]["tied_rows"], len(data))
            self.assertEqual(result, compare_predictions(data, pred, data.house_key))
            json.dumps(result, allow_nan=False)
        self.assertEqual(retraining_report.MODEL_KEYS, original_keys)

    def test_cluster_bootstrap_keeps_entire_houses_and_row_weights(self):
        data = frame(100)
        pred = predictions(data, True)
        pred["catboost_mae"] += np.r_[np.repeat(10., 99), -10.]
        pred["rf_control"] = pred["catboost_mae"].copy()
        clusters = pd.Series(["large"] * 99 + ["small"], index=data.index)
        result = compare_predictions(data, pred, clusters)
        self.assertEqual(result["paired"]["catboost_mae"], result["paired"]["rf_control"])
        self.assertAlmostEqual(result["paired"]["catboost_mae"]["mae_delta_tenge"], 9.8)
        self.assertEqual(result["paired"]["catboost_mae"]["mae_delta_ci95_tenge"], [-10., 10.])

    def test_wrong_keys_arrays_targets_and_duplicate_ids_fail(self):
        data = frame()
        for value in (np.zeros((4, 1)), np.ones(3), np.array([1., 2., np.nan, 4.]),
                      np.array([1., 2., np.inf, 4.]), np.ones(4, dtype=bool)):
            pred = predictions(data)
            pred["catboost_log"] = value
            with self.assertRaises(ValueError):
                compare_predictions(data, pred, data.house_key)
        for key in ("other", "repair_only"):
            pred = predictions(data)
            pred[key] = np.ones(len(data))
            with self.assertRaises(ValueError):
                compare_predictions(data, pred, data.house_key)
        pred = predictions(data)
        del pred["catboost_log"]
        with self.assertRaises(ValueError):
            compare_predictions(data, pred, data.house_key)
        bad = data.copy()
        bad.loc[0, "price"] = 0.
        with self.assertRaises(ValueError):
            compare_predictions(bad, predictions(data), data.house_key)
        bad = data.copy()
        bad.loc[1, "record_id"] = bad.loc[0, "record_id"]
        with self.assertRaises(ValueError):
            compare_predictions(bad, predictions(data), data.house_key)

    def test_alignment_is_checked_and_all_models_share_segment_membership(self):
        data = frame(31)
        data.index = np.arange(31) + 100
        data.loc[100, "district"] = "Алматы"
        pred = predictions(data, True)
        geography = pd.Series("Нет координат", index=data.index)
        segments = build_segments(data, pred, {"тест"}, geography)
        pivot = segments.pivot(index=["dimension", "segment"], columns="model", values="n")
        self.assertTrue(pivot.eq(pivot.current, axis=0).all().all())
        self.assertTrue(segments[segments.segment.eq("Алматы")].small_group.all())
        self.assertFalse(segments[segments.segment.eq("Нура")].small_group.any())
        with self.assertRaises(ValueError):
            compare_predictions(data, pred, data.house_key.iloc[::-1])
        pred["catboost_log"] = pd.Series(pred["catboost_log"], index=data.index[::-1])
        with self.assertRaises(ValueError):
            compare_predictions(data, pred, data.house_key)
        with self.assertRaises(ValueError):
            build_segments(data, predictions(data), {"тест"}, geography.iloc[::-1])

    def test_metrics_are_currency_scale_and_zero_baseline_is_json_safe(self):
        data = frame(2)
        data["price"] = [100., 200.]
        pred = {key: np.array([120., 250.]) for key in predictions(data)}
        result = compare_predictions(data, pred, data.house_key)
        self.assertEqual(result["overall"]["catboost_log"]["mae_tenge"], 35.)
        self.assertEqual(result["overall"]["catboost_log"]["median_ape_pct"], 22.5)
        self.assertEqual(result["overall"]["catboost_log"]["over_20_n"], 1)
        data["price"] = 100.
        pred = {key: data.price.to_numpy() for key in predictions(data)}
        result = compare_predictions(data, pred, data.house_key)
        self.assertIsNone(result["overall"]["current"]["r2"])
        self.assertIsNone(result["paired"]["catboost_log"]["mae_delta_pct"])
        json.dumps(result, allow_nan=False)

    def test_html_uses_locked_cv_choice_not_holdout_winner_and_escapes_text(self):
        data = frame()
        data["district"] = '<script>alert("x")</script>'
        pred = predictions(data, True)
        pred["catboost_log"] = data.price.to_numpy()  # Holdout winner is deliberately not selected.
        result = compare_predictions(data, pred, data.house_key)
        segments = build_segments(data, pred, {"тест"}, pd.Series("Нет координат", index=data.index))
        report = {"comparison": result, "training_rows": 19269, "holdout_rows": len(data),
                  "model_labels": {"catboost_mae": "Chosen <MAE>"}, "selected_variant": "catboost_mae",
                  "nested_cv": {"outer_folds": 3, "inner_folds": 3,
                                "summary": {"selected_procedure": {"n": 19269, "mae_tenge": 42.}},
                                "folds": [{"outer_fold": 1, "variant": "catboost_mae", "selected_params_id": "p1", "n": 10, "metrics": {"mae_tenge": 43.}}]},
                  "selection_cv": {"chosen_variant": "catboost_mae", "results": [
                      {"variant": "catboost_mae", "params_id": "p1", "selected": True, "mae_tenge": 10.}]},
                  "limitations": ["Дополнительное ограничение"]}
        html = render_html(report, segments)
        self.assertIn("<strong>Chosen &lt;MAE&gt;</strong>", html)
        self.assertIn("4 одинаковых контрольных объявлений", html)
        self.assertIn("Внутренний выбор CatBoost", html)
        self.assertIn("восемь исходных признаков и ЖК, без координат", html)
        self.assertIn("Дополнительное ограничение", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("None", html)
        report["selection_cv"]["chosen_variant"] = "catboost_log"
        with self.assertRaises(ValueError):
            render_html(report, segments)


if __name__ == "__main__":
    unittest.main()

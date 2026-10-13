"""Проверки границ ошибок, маскирования и соответствия групп строкам."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from error_analysis import mask_features, metric_rows, summarize, summarize_groups
from features import FEATURE_COLUMNS


class ErrorMetricTests(unittest.TestCase):
    def test_relative_and_absolute_thresholds_are_strict_and_independent(self) -> None:
        actual = np.array([50e6, 50e6, 50e6, 50e6, 100e6, 20e6])
        predicted = np.array([60e6, 40e6, 60e6 + 1, 40e6 - 1, 115e6, 25e6])

        rows = metric_rows(actual, predicted, half_width=10e6)

        self.assertEqual(rows.over_20.tolist(), [False, False, True, True, False, True])
        self.assertEqual(rows.over_10m.tolist(), [False, False, True, True, True, False])
        self.assertEqual(rows.under_20.tolist(), [False, False, False, True, False, False])
        self.assertEqual(rows.overprediction_20.tolist(), [False, False, True, False, False, True])

    def test_bias_preserves_direction_and_is_not_mean_absolute_error(self) -> None:
        rows = metric_rows([10e6, 20e6, 30e6], [12e6, 16e6, 33e6], half_width=4e6)

        result = summarize(rows)

        np.testing.assert_array_equal(rows.signed_error, [2e6, -4e6, 3e6])
        self.assertAlmostEqual(result["bias_tenge"], 1e6 / 3)
        self.assertEqual(result["mae_tenge"], 3e6)
        self.assertAlmostEqual(result["rmse_tenge"], np.sqrt(29 / 3) * 1e6)

    def test_coverage_includes_both_interval_boundaries(self) -> None:
        rows = metric_rows(
            [15e6, 25e6, 15e6 - 1, 25e6 + 1],
            [20e6, 20e6, 20e6, 20e6],
            half_width=5e6,
        )

        self.assertEqual(rows.covered.tolist(), [True, True, False, False])

    def test_invalid_metric_inputs_are_rejected(self) -> None:
        cases = [
            ([], [], 1),
            ([1, 2], [1], 1),
            ([[1]], [[1]], 1),
            ([0], [1], 1),
            ([np.nan], [1], 1),
            ([1], [np.inf], 1),
            ([1], [1], -1),
            ([1], [1], np.nan),
        ]
        for actual, predicted, width in cases:
            with self.subTest(actual=actual, predicted=predicted, width=width):
                with self.assertRaises(ValueError):
                    metric_rows(actual, predicted, width)


class MaskFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "area": [45.0, 70.0], "rooms": [1, 3], "floor": [2.0, np.nan],
            "total_floors": [9, 12], "year_built": [2000, 2020],
            "district": ["Есиль", "Нура"], "building_type": ["кирпичный", "монолитный"],
            "condition": ["хорошее", "черновая отделка"], "price": [20e6, 45e6],
        }, index=pd.Index([90, 11], name="listing"))

    def test_mask_keeps_mandatory_fields_index_and_source_unchanged(self) -> None:
        original = self.frame.copy(deep=True)

        masked = mask_features(self.frame, ("area", "rooms"))

        self.assertEqual(tuple(masked.columns), FEATURE_COLUMNS)
        pd.testing.assert_index_equal(masked.index, original.index)
        pd.testing.assert_frame_equal(masked[["area", "rooms"]], original[["area", "rooms"]])
        self.assertTrue(masked.drop(columns=["area", "rooms"]).isna().all().all())
        pd.testing.assert_frame_equal(self.frame, original)
        masked.loc[90, "area"] = 999
        pd.testing.assert_frame_equal(self.frame, original)

    def test_full_mask_preserves_original_missing_values_and_excludes_target(self) -> None:
        masked = mask_features(self.frame, FEATURE_COLUMNS)

        pd.testing.assert_frame_equal(masked, self.frame.loc[:, list(FEATURE_COLUMNS)])
        self.assertNotIn("price", masked.columns)

    def test_masks_cannot_remove_mandatory_fields_or_keep_unknown_fields(self) -> None:
        for kept in [("area",), ("rooms",), ("area", "rooms", "price")]:
            with self.subTest(kept=kept):
                with self.assertRaises(ValueError):
                    mask_features(self.frame, kept)


class GroupAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = metric_rows([10e6, 20e6, 30e6], [12e6, 16e6, 33e6], half_width=4e6)
        self.rows.index = pd.Index([90, 11, 70], name="listing")

    def test_nondefault_indices_and_missing_labels_keep_the_right_errors(self) -> None:
        labels = pd.Series(["Есиль", None, "Есиль"], index=self.rows.index)

        groups = {row["group"]: row for row in summarize_groups(self.rows, labels)}

        self.assertEqual(set(groups), {"Есиль", "Не указано"})
        self.assertEqual(groups["Есиль"]["n"], 2)
        self.assertEqual(groups["Есиль"]["mae_tenge"], 2.5e6)
        self.assertEqual(groups["Есиль"]["bias_tenge"], 2.5e6)
        self.assertEqual(groups["Не указано"]["n"], 1)
        self.assertEqual(groups["Не указано"]["bias_tenge"], -4e6)
        self.assertAlmostEqual(groups["Есиль"]["error_share_pct"], 500 / 9)
        self.assertAlmostEqual(sum(row["error_share_pct"] for row in groups.values()), 100)

    def test_reordered_or_different_row_indices_are_rejected(self) -> None:
        labels = pd.Series(["Есиль", "Нура", "Есиль"], index=self.rows.index)
        cases = [labels.iloc[::-1], labels.reset_index(drop=True), labels.iloc[:2]]
        for wrong_labels in cases:
            with self.subTest(indices=wrong_labels.index.tolist()):
                with self.assertRaises(ValueError):
                    summarize_groups(self.rows, wrong_labels)


if __name__ == "__main__":
    unittest.main()

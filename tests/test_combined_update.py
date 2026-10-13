"""Проверки сравнения моделей и автономного отчёта без обучения."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from combined_report import render_report
from features import FEATURE_COLUMNS
from update_combined import paired_mae_difference, segment_comparisons, write_csv_once


class PairedDifferenceTests(unittest.TestCase):
    def test_identical_predictions_have_zero_difference_and_interval(self) -> None:
        actual = np.array([10, 20, 30, 40]) * 1_000_000
        predictions = actual + [1_000_000, -2_000_000, 3_000_000, -4_000_000]
        result = paired_mae_difference(actual, predictions, predictions, iterations=100)
        self.assertEqual(result["new_minus_january_mae_tenge"], 0)
        self.assertEqual(result["ci95_low_tenge"], 0)
        self.assertEqual(result["ci95_high_tenge"], 0)

    def test_constant_improvement_keeps_tenge_scale_and_sign(self) -> None:
        actual = np.array([10, 20, 30, 40]) * 1_000_000
        result = paired_mae_difference(actual, actual + 7_000_000, actual - 2_000_000, iterations=100)
        for key in ("new_minus_january_mae_tenge", "ci95_low_tenge", "ci95_high_tenge"):
            self.assertEqual(result[key], -5_000_000)
        self.assertEqual(result["iterations"], 100)

    def test_bootstrap_is_deterministic(self) -> None:
        arguments = ([10, 20, 30, 40], [11, 25, 28, 35], [10, 18, 33, 42])
        self.assertEqual(paired_mae_difference(*arguments, iterations=100),
                         paired_mae_difference(*arguments, iterations=100))

    def test_invalid_inputs_fail_before_comparison(self) -> None:
        cases = [([], [], [], 100), ([1], [1], [1], 100),
                 ([1, 2], [1], [1, 2], 100),
                 ([[1, 2]], [[1, 2]], [[1, 2]], 100),
                 ([1, np.nan], [1, 2], [1, 2], 100),
                 ([1, 2], [1, np.inf], [1, 2], 100),
                 ([1, 2], [1, 2], [np.nan, 2], 100),
                 ([1, 2], [1, 2], [1, 2], 99)]
        for actual, old, new, iterations in cases:
            with self.subTest(actual=actual, old=old, new=new, iterations=iterations):
                with self.assertRaises(ValueError):
                    paired_mae_difference(actual, old, new, iterations=iterations)


class SegmentComparisonTests(unittest.TestCase):
    def test_masks_follow_row_order_with_nonconsecutive_indices(self) -> None:
        frame = pd.DataFrame({
            "price": np.array([10, 20, 30, 40]) * 1_000_000,
            "area": [40, 70, 120, 121], "rooms": [1, 2, 3, 4],
            "floor": [3] * 4, "total_floors": [9] * 4, "year_built": [2015] * 4,
            "district": ["A", "B", "A", None], "building_type": ["кирпичный"] * 4,
            "condition": ["хорошее", "хорошее", "хорошее", None],
        }, index=[91, 4, 32, 7])
        predictions = {"january": np.array([12, 25, 24, 44]) * 1_000_000,
                       "combined": np.array([11, 22, 39, 36]) * 1_000_000}
        result = segment_comparisons(frame, predictions)
        districts = {row["group"]: row for row in result["district"]}
        self.assertEqual(districts["A"]["n"], 2)
        self.assertEqual(districts["A"]["january_mae_tenge"], 4_000_000)
        self.assertEqual(districts["A"]["combined_mae_tenge"], 5_000_000)
        self.assertEqual(districts["A"]["mae_change_tenge"], 1_000_000)
        # Exactly 20% is not an over-20% error.
        self.assertEqual(districts["A"]["january_over_20_pct"], 0)
        self.assertEqual(districts["A"]["combined_over_20_pct"], 50)
        self.assertEqual(districts["B"]["january_over_20_pct"], 100)
        self.assertEqual(districts["B"]["combined_over_20_pct"], 0)
        self.assertEqual(districts["Не указан"]["n"], 1)
        for entries in result.values():
            self.assertEqual(sum(entry["n"] for entry in entries), len(frame))
            self.assertTrue(all(entry["small_group"] for entry in entries))
            for model, expected in (("january", 4_250_000), ("combined", 4_000_000)):
                weighted = sum(entry["n"] * entry[f"{model}_mae_tenge"] for entry in entries) / len(frame)
                self.assertEqual(weighted, expected)
        completeness = {entry["group"]: entry["n"] for entry in result["completeness"]}
        self.assertEqual(completeness, {"8 из 8": 3, "6 из 8": 1})
        self.assertEqual([entry["n"] for entry in result["area"]], [1, 1, 1, 1])
        # Relabelling the index cannot alter the same rows' segment results.
        self.assertEqual(result, segment_comparisons(frame.reset_index(drop=True), predictions))
        self.assertEqual(len(FEATURE_COLUMNS), 8)


class ImmutableCsvTests(unittest.TestCase):
    def test_repeated_identical_write_is_allowed_but_changed_data_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.csv"
            original = pd.DataFrame({"source_row": [1, 2], "district": ["Есиль", "Сарайшык"]})
            write_csv_once(original, path)
            content = path.read_bytes()
            write_csv_once(original.copy(), path)
            self.assertEqual(path.read_bytes(), content)
            changed = original.copy()
            changed.loc[0, "district"] = "Алматы"
            with self.assertRaisesRegex(ValueError, "отличается"):
                write_csv_once(changed, path)
            self.assertEqual(path.read_bytes(), content)


class CombinedReportTests(unittest.TestCase):
    @staticmethod
    def report_fixture(change: float, low: float, high: float) -> dict:
        comparison = [{"model": model, "algorithm": "Tree <candidate>",
                       "mae_tenge": 2_500_000, "rmse_tenge": 5_500_000, "r2": .75,
                       "over_20_pct": 25, "bias_tenge": -500_000}
                      for model in ("january", "march", "combined")]
        segment = {"group": "Район <A>", "n": 37, "small_group": False,
                   "january_mae_tenge": 3_000_000, "combined_mae_tenge": 2_500_000,
                   "mae_change_tenge": -500_000}
        return {"created_at_utc": "2026-10-12T10:00:00+00:00",
                "counts": {"input_rows": 1000, "unique_rows": 975, "removed_rows": 25,
                           "train_rows": 938, "test_rows": 37, "reference_test_rows": 41,
                           "train_sources": {"2026-01": 800, "2026-03": 138}},
                "mae_improvement_pct": change,
                "paired_mae_difference": {"new_minus_january_mae_tenge": -500_000,
                                          "ci95_low_tenge": low, "ci95_high_tenge": high},
                "common_holdout_comparison": comparison, "original_holdout_comparison": comparison,
                "segments": {dimension: [segment] for dimension in ("district", "area", "completeness")}}

    def test_report_counts_are_dynamic_and_no_placeholders_remain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            render_report(self.report_fixture(12.5, -1_000_000, 100_000), path)
            rendered = path.read_text(encoding="utf-8")
        self.assertIn("<strong>37 квартир</strong>", rendered)
        self.assertIn("37 представителей из прежних 41 объявлений", rendered)
        self.assertIn("включая 4 дополнительных строк", rendered)
        self.assertIn("Прогнозы на прежних 41 строках", rendered)
        self.assertNotIn("594", rendered)
        self.assertNotIn("590", rendered)
        self.assertIn("снизилась на 12,5%", rendered)
        self.assertIn("Интервал включает ноль", rendered)
        self.assertIn("Сравнение диагностическое", rendered)
        self.assertIn("Tree &lt;candidate&gt;", rendered)
        self.assertIn("Район &lt;A&gt;", rendered)
        self.assertNotRegex(rendered, r"\{(?:c\[|report\[|interval\[|fmt\(|comparison_table\(|segment_table\()")

    def test_regression_and_nonzero_interval_are_described_accurately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            render_report(self.report_fixture(-10, 100_000, 900_000), path)
            rendered = path.read_text(encoding="utf-8")
        self.assertIn("выросла на 10,0%", rendered)
        self.assertIn("Интервал не включает ноль", rendered)
        self.assertIn("повторное использование test", rendered)


if __name__ == "__main__":
    unittest.main()

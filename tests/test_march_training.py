"""Контракты March/combined: группы, защищённый test и замороженный baseline."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

import march_training
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_pipeline
from prediction import load_artifacts


class AuditPipeline(Pipeline):
    fit_rows: list[tuple[int, ...]] = []

    def fit(self, X, y=None, **params):
        type(self).fit_rows.append(tuple(X.index))
        return super().fit(X, y, **params)


def synthetic_frame() -> pd.DataFrame:
    positions = np.arange(24)
    area = 35.0 + positions * 2
    return pd.DataFrame({
        "area": area, "rooms": 1 + positions % 3, "floor": 1 + positions % 8,
        "total_floors": 12, "year_built": 2000 + positions % 15,
        "district": np.where(positions % 2, "Нура", "Есиль"),
        "building_type": "кирпичный", "condition": "хорошее",
        "price": 5e6 + area * 500_000, "source_row": positions + 101,
    })


def quick_models(config):
    # Реальные дешёвые оценщики проверяют Pipeline/CV/сериализацию без обучения деревьев проекта.
    estimators = {"LinearRegression": LinearRegression(), "RandomForest": DummyRegressor(),
                  "XGBoost": DecisionTreeRegressor(max_depth=1, random_state=42)}
    return {name: AuditPipeline(build_pipeline(estimator).steps) for name, estimator in estimators.items()}


class MarchTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_frame()
        self.groups = pd.Series([f"flat-{i // 2}" for i in self.frame.index])
        self.train_ids = np.arange(18)
        self.test_ids = np.arange(18, 24)
        self.old_pipeline = build_pipeline(LinearRegression()).fit(
            self.frame.loc[:, list(FEATURE_COLUMNS)], self.frame.price * .8)
        self.old_metadata = {
            "schema_version": 1, "currency": "KZT", "target": "asking_price",
            "feature_columns": list(FEATURE_COLUMNS), "model_name": "synthetic January",
            "test_metrics": {"mae_tenge": 1e6}, "approximate_range": {"half_width_tenge": 1e6},
            "categories_from_train": {c: sorted(self.frame[c].unique()) for c in CATEGORICAL_FEATURES},
            "numeric_ranges_from_train": {}, "model_sha256": "0" * 64,
        }

    def test_partition_rejects_group_leakage_and_misaligned_groups(self) -> None:
        leaked = self.groups.copy()
        leaked.iloc[18] = leaked.iloc[0]
        for groups in (leaked, self.groups.iloc[::-1]):
            with self.subTest(groups=groups.iloc[:2].tolist()):
                with self.assertRaises(ValueError):
                    march_training._validate_partition(self.frame, self.train_ids, self.test_ids, groups)

    def test_partition_rejects_duplicate_missing_and_overlapping_rows(self) -> None:
        invalid = [
            (np.r_[self.train_ids, 0], self.test_ids),
            (self.train_ids[:-1], self.test_ids),
            (self.train_ids, np.r_[17, self.test_ids]),
            (self.train_ids.astype(float), self.test_ids),
        ]
        for train_ids, test_ids in invalid:
            with self.subTest(train_ids=train_ids.tolist(), test_ids=test_ids.tolist()):
                with self.assertRaises(ValueError):
                    march_training._validate_partition(self.frame, train_ids, test_ids, self.groups)

    def test_group_cv_rejects_single_row_validation_folds(self) -> None:
        x, y = march_training._feature_target(self.frame.iloc[:6])
        groups = pd.Series(["large"] * 4 + ["one", "two"])
        with self.assertRaisesRegex(ValueError, "две строки"):
            march_training._checked_folds(x, y, groups)

    def test_prediction_csv_preserves_exact_float32_values_as_float64(self) -> None:
        ids = np.array([18, 19, 20])
        predicted = np.array([347_840_972, 40_000_004, 70_000_008], dtype=np.float32)
        expected = predicted.astype(float)
        exported = march_training._prediction_frame(
            self.frame, self.frame.loc[ids, list(FEATURE_COLUMNS)], self.frame.loc[ids, "price"], predicted)

        restored = pd.read_csv(io.StringIO(exported.to_csv(index=False)))

        self.assertEqual(exported.predicted_price.dtype, np.dtype(float))
        np.testing.assert_array_equal(restored.predicted_price, expected)
        np.testing.assert_array_equal(restored.absolute_error, np.abs(self.frame.loc[ids, "price"].to_numpy() - expected))
        np.testing.assert_array_equal(restored.row_index, ids)

    def test_csv_mismatch_stops_before_any_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "march.csv"
            self.frame.to_csv(input_path, index=False)
            changed = self.frame.copy()
            changed.loc[0, "price"] += 100
            with patch("march_training.make_models") as factory:
                with self.assertRaisesRegex(ValueError, "не совпадает"):
                    march_training.train_march(
                        changed, self.train_ids, self.test_ids, self.groups, root / "models",
                        root / "reports", input_path, self.old_pipeline, self.old_metadata)
                factory.assert_not_called()

    def test_runner_keeps_groups_and_test_separate_and_never_refits_old_model(self) -> None:
        self._check_runner(march_training.train_march, march_training.MARCH_CONTEXT)

    def test_combined_context_and_selection_do_not_leak_protected_holdout(self) -> None:
        reports = self._check_runner(march_training.train_combined, march_training.COMBINED_CONTEXT)
        for report in reports:
            self.assertEqual(report["data_period"], "2026-01_to_2026-03")
            self.assertIn("including the former January holdout", report["training_scope"])
            self.assertIn("protected March holdout excluded", report["training_scope"])
            self.assertTrue(report["holdout_previously_analyzed"])
            self.assertIn("Previously analyzed protected March holdout", report["holdout_description"])
            self.assertTrue(any("not fresh independent validation" in text for text in report["limitations"]))
            self.assertTrue(any("no longer validates this model" in text for text in report["limitations"]))
            self.assertFalse(any("March-only dataset" in text for text in report["limitations"]))
            candidates = [row for row in report["comparison"] if row["role"] != "january_frozen_reference"]
            self.assertTrue(all(row["role"] == "combined_candidate" for row in candidates))
            self.assertNotIn("cv_on_march_train", report["old_model"])

    def _check_runner(self, runner, context):
        AuditPipeline.fit_rows.clear()
        old_before = self.old_pipeline.predict(self.frame.loc[self.test_ids, list(FEATURE_COLUMNS)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = []
            saved_predictions = []
            for run in range(2):
                frame = self.frame.copy()
                if run:
                    # Меняется только закрытый target; CV и сохранённые прогнозы обязаны остаться теми же.
                    frame.loc[self.test_ids, "price"] *= 3
                input_path = root / f"march_{run}.csv"
                frame.to_csv(input_path, index=False)
                frame = pd.read_csv(input_path)
                models_dir, reports_dir = root / f"models_{run}", root / f"reports_{run}"
                with patch("march_training.make_models", side_effect=quick_models), \
                     patch("march_training.save_figures"), \
                     patch.object(self.old_pipeline, "fit", side_effect=AssertionError("old model must stay frozen")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    report = runner(
                        frame, self.train_ids, self.test_ids, self.groups, models_dir,
                        reports_dir, input_path, self.old_pipeline, self.old_metadata)
                reports.append(report)
                loaded, metadata = load_artifacts(models_dir / "best_model.joblib", models_dir / "model_metadata.json")
                predicted = pd.read_csv(reports_dir / "test_predictions.csv")
                old_predicted = pd.read_csv(reports_dir / "old_model_test_predictions.csv")
                saved_predictions.append(predicted.predicted_price.to_numpy())
                np.testing.assert_array_equal(predicted.row_index, self.test_ids)
                np.testing.assert_array_equal(old_predicted.row_index, predicted.row_index)
                np.testing.assert_array_equal(predicted.source_row, frame.loc[self.test_ids, "source_row"])
                np.testing.assert_allclose(old_predicted.predicted_price, old_before)
                np.testing.assert_allclose(loaded.predict(frame.loc[self.test_ids, list(FEATURE_COLUMNS)]),
                                           predicted.predicted_price, rtol=1e-12, atol=.01)
                manifest = pd.read_csv(reports_dir / "split_manifest.csv")
                self.assertEqual(manifest.groupby("group").subset.nunique().max(), 1)
                self.assertEqual(manifest.query("subset == 'train'").groupby("group").cv_validation_fold.nunique().max(), 1)
                self.assertEqual(set(manifest.query("subset == 'train'").cv_validation_fold), {1, 2, 3})
                self.assertTrue(manifest.query("subset == 'test'").cv_validation_fold.eq(0).all())
                self.assertEqual(metadata["split_manifest_sha256"], march_training.file_sha256(reports_dir / "split_manifest.csv"))
                self.assertEqual(metadata["input_sha256"], march_training.file_sha256(input_path))
                self.assertEqual(metadata["cv_strategy"], "GroupKFold")
                self.assertEqual(metadata["test_rows"], 6)
                self.assertEqual(metadata["train_rows"], 18)
                self.assertEqual(metadata["data_period"], context.data_period)
                self.assertEqual(metadata["data_period_label"], context.data_period_label)
                self.assertEqual(metadata["training_scope"], context.training_scope)
                self.assertEqual(metadata["candidate_role"], context.candidate_role)
                self.assertEqual(metadata["holdout_description"], context.holdout_description)
                self.assertEqual(metadata["holdout_previously_analyzed"], context.holdout_previously_analyzed)
                self.assertEqual(metadata["feature_columns"], list(FEATURE_COLUMNS))
                self.assertEqual(metadata["cv_folds"], 3)
                self.assertEqual(metadata["approximate_range"]["half_width_tenge"], metadata["test_metrics"]["mae_tenge"])
                old_row = next(row for row in report["comparison"] if row["role"] == "january_frozen_reference")
                self.assertIsNone(old_row["cv_mae_tenge"])
                self.assertFalse(old_row["selected"])
                self.assertFalse(report["old_model"]["refitted"])
                json.loads((reports_dir / "training_report.json").read_text(encoding="utf-8"))
            self.assertEqual(reports[0]["model_name"], reports[1]["model_name"])
            self.assertEqual(reports[0]["selected_cv_mae_tenge"], reports[1]["selected_cv_mae_tenge"])
            np.testing.assert_array_equal(saved_predictions[0], saved_predictions[1])
            self.assertNotEqual(reports[0]["test_metrics"]["mae_tenge"], reports[1]["test_metrics"]["mae_tenge"])
        self.assertTrue(AuditPipeline.fit_rows)
        for fitted_ids in AuditPipeline.fit_rows:
            self.assertTrue(set(fitted_ids).issubset(self.train_ids))
            counts = self.groups.loc[list(fitted_ids)].value_counts()
            self.assertTrue(counts.eq(2).all(), "CV must retain or exclude whole groups")
        np.testing.assert_array_equal(self.old_pipeline.predict(self.frame.loc[self.test_ids, list(FEATURE_COLUMNS)]), old_before)
        return reports


if __name__ == "__main__":
    unittest.main()

"""Проверки ML-пайплайна на искусственных данных только для тестов."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_pipeline,
    build_preprocessor,
)
from train import compute_metrics, split_data


def synthetic_listings(count: int = 20) -> pd.DataFrame:
    """Вымышленные квартиры: эти строки не сохраняются в рабочие данные."""
    positions = np.arange(count)
    area = 30.0 + positions * 3.0
    return pd.DataFrame(
        {
            "area": area,
            "rooms": 1 + positions % 4,
            "floor": 1 + positions % 8,
            "total_floors": np.full(count, 12),
            "year_built": 2000 + positions % 20,
            "district": np.where(positions % 2 == 0, "Есиль", "Нура"),
            "building_type": np.where(positions % 3 == 0, "кирпичный", "монолитный"),
            "condition": np.where(positions % 2 == 0, "хорошее", "черновая отделка"),
            "price": 5_000_000.0 + area * 500_000.0,
            "price_per_m2": (5_000_000.0 + area * 500_000.0) / area,
            "url": [f"https://example.invalid/apartment/{i}" for i in positions],
        },
        index=pd.Index(1000 + positions, name="source_index"),
    )


class FeaturePipelineTests(unittest.TestCase):
    def test_feature_schema_contains_only_user_inputs(self) -> None:
        self.assertEqual(
            set(NUMERIC_FEATURES),
            {"area", "rooms", "floor", "total_floors", "year_built"},
        )
        self.assertEqual(set(CATEGORICAL_FEATURES), {"district", "building_type", "condition"})
        self.assertEqual(tuple(FEATURE_COLUMNS), tuple(NUMERIC_FEATURES) + tuple(CATEGORICAL_FEATURES))
        self.assertEqual(TARGET_COLUMN, "price")

    def test_unknown_categories_and_missing_values_produce_finite_predictions(self) -> None:
        frame = synthetic_listings()
        # Пропуски на обучении и на предсказании обрабатывает один пайплайн.
        frame.loc[frame.index[0], "floor"] = np.nan
        frame.loc[frame.index[1], "condition"] = np.nan
        pipeline = build_pipeline(LinearRegression())
        pipeline.fit(frame[list(FEATURE_COLUMNS)], frame[TARGET_COLUMN])
        incoming = frame.iloc[[0, 1]].loc[:, list(FEATURE_COLUMNS)].copy()
        incoming.loc[incoming.index[0], list(CATEGORICAL_FEATURES)] = "новая категория"
        incoming.loc[incoming.index[1], list(CATEGORICAL_FEATURES)] = np.nan
        incoming.loc[incoming.index[1], list(NUMERIC_FEATURES)] = np.nan

        prediction = pipeline.predict(incoming)

        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_target_and_derived_price_columns_do_not_enter_preprocessor(self) -> None:
        frame = synthetic_listings()
        preprocessor = build_preprocessor()
        preprocessor.fit(frame)
        expected = preprocessor.transform(frame[list(FEATURE_COLUMNS)])
        corrupted = frame.copy()
        corrupted["price"] = -999_999_999_999.0
        corrupted["price_per_m2"] = np.inf
        corrupted["url"] = "совершенно другое значение"

        np.testing.assert_allclose(preprocessor.transform(corrupted), expected)
        names = preprocessor.get_feature_names_out()
        self.assertFalse(any("price" in name or "url" in name for name in names))

    def test_numeric_imputation_is_learned_only_from_training_rows(self) -> None:
        frame = synthetic_listings(6)
        frame["area"] = [20.0, 40.0, np.nan, 60.0, 80.0, 100.0]
        preprocessor = build_preprocessor()
        preprocessor.fit(frame[list(FEATURE_COLUMNS)])
        numeric = preprocessor.named_transformers_["numeric"]
        imputer = numeric.named_steps["imputer"]
        area_position = list(NUMERIC_FEATURES).index("area")
        self.assertEqual(imputer.statistics_[area_position], 60.0)
        original_statistics = imputer.statistics_.copy()
        incoming = frame.iloc[[0, 1]].loc[:, list(FEATURE_COLUMNS)].copy()
        incoming["area"] = [np.nan, 1_000_000.0]

        preprocessor.transform(incoming)

        np.testing.assert_allclose(imputer.statistics_, original_statistics)
        imputed = imputer.transform(incoming[list(NUMERIC_FEATURES)])
        self.assertEqual(imputed[0, area_position], 60.0)

    def test_entirely_missing_numeric_feature_keeps_its_position(self) -> None:
        frame = synthetic_listings()
        frame["year_built"] = np.nan
        pipeline = build_pipeline(LinearRegression())
        pipeline.fit(frame[list(FEATURE_COLUMNS)], frame[TARGET_COLUMN])
        numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        imputer = numeric.named_steps["imputer"]
        incoming = frame.iloc[[0]].loc[:, list(FEATURE_COLUMNS)].copy()
        incoming["year_built"] = 2020.0
        imputed = imputer.transform(incoming[list(NUMERIC_FEATURES)])

        self.assertTrue(imputer.keep_empty_features)
        self.assertEqual(len(imputer.statistics_), len(NUMERIC_FEATURES))
        self.assertEqual(imputed.shape[1], len(NUMERIC_FEATURES) + 1)
        self.assertEqual(imputed[0, list(NUMERIC_FEATURES).index("year_built")], 2020.0)
        self.assertTrue(np.isfinite(pipeline.predict(incoming)).all())

    def test_joblib_round_trip_keeps_preprocessing_and_predictions(self) -> None:
        frame = synthetic_listings()
        pipeline = build_pipeline(LinearRegression())
        pipeline.fit(frame[list(FEATURE_COLUMNS)], frame[TARGET_COLUMN])
        incoming = frame.iloc[[0, 2]].loc[:, list(FEATURE_COLUMNS)].copy()
        incoming.loc[incoming.index[0], "district"] = "новая категория"
        incoming.loc[incoming.index[1], "floor"] = np.nan
        expected = pipeline.predict(incoming)

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "test_pipeline.joblib"
            joblib.dump(pipeline, model_path)
            restored = joblib.load(model_path)
            actual = restored.predict(incoming)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


class TrainingContractTests(unittest.TestCase):
    def test_metrics_are_in_tenge_and_r_squared_is_dimensionless(self) -> None:
        actual = np.array([10_000_000.0, 20_000_000.0, 30_000_000.0])
        predicted = np.array([12_000_000.0, 18_000_000.0, 30_000_000.0])

        metrics = compute_metrics(actual, predicted)

        self.assertAlmostEqual(metrics["mae_tenge"], 4_000_000.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse_tenge"], np.sqrt(8_000_000_000_000.0 / 3.0))
        self.assertAlmostEqual(metrics["r2"], 0.96)

    def test_split_is_reproducible_disjoint_and_excludes_target_derived_fields(self) -> None:
        frame = synthetic_listings(50)
        first = split_data(frame, random_state=42, test_size=0.2)
        second = split_data(frame, random_state=42, test_size=0.2)
        x_train, x_test, y_train, y_test = first

        self.assertEqual((len(x_train), len(x_test)), (40, 10))
        self.assertFalse(set(x_train.index) & set(x_test.index))
        self.assertEqual(set(x_train.index) | set(x_test.index), set(frame.index))
        for features in (x_train, x_test):
            self.assertEqual(tuple(features.columns), tuple(FEATURE_COLUMNS))
        pd.testing.assert_series_equal(y_train, frame.loc[x_train.index, TARGET_COLUMN])
        pd.testing.assert_series_equal(y_test, frame.loc[x_test.index, TARGET_COLUMN])
        for first_part, second_part in zip(first, second):
            if isinstance(first_part, pd.DataFrame):
                pd.testing.assert_frame_equal(first_part, second_part)
            else:
                pd.testing.assert_series_equal(first_part, second_part)
        different = split_data(frame, random_state=17, test_size=0.2)
        self.assertNotEqual(set(x_test.index), set(different[1].index))


if __name__ == "__main__":
    unittest.main()

"""Проверки ввода, расчётов и пользовательского сценария Streamlit."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from config import MODEL_METADATA_PATH, MODEL_PIPELINE_PATH, PROJECT_ROOT, REFERENCE_YEAR
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from prediction import load_artifacts, predict_apartment, prepare_input


def metadata_fixture() -> dict:
    """Небольшие метаданные для тестов; рабочие артефакты не изменяются."""
    return {
        "schema_version": 1,
        "target": "asking_price",
        "currency": "KZT",
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categories_from_train": {
            "district": ["Есиль", "Нура"],
            "building_type": ["кирпичный", "монолитный"],
            "condition": ["свежий ремонт", "черновая отделка"],
        },
        "numeric_ranges_from_train": {
            "area": {"min": 20.0, "max": 300.0},
            "rooms": {"min": 1.0, "max": 8.0},
            "floor": {"min": 1.0, "max": 30.0},
            "total_floors": {"min": 2.0, "max": 40.0},
            "year_built": {"min": 1960.0, "max": 2028.0},
        },
        "test_metrics": {"mae_tenge": 1_000_000.0, "rmse_tenge": 2_000_000.0, "r2": 0.8},
        "approximate_range": {
            "half_width_tenge": 1_000_000.0,
            "source": "holdout_test_mae",
            "guaranteed_coverage": None,
        },
        "model_sha256": "0" * 64,
    }


def apartment_fixture() -> dict:
    return {
        "area": 60.0,
        "rooms": 2,
        "floor": 5,
        "total_floors": 9,
        "year_built": 2015,
        "district": "Есиль",
        "building_type": "монолитный",
        "condition": "свежий ремонт",
    }


class InputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = metadata_fixture()
        self.values = apartment_fixture()

    def test_valid_input_has_exact_training_schema_and_does_not_mutate_source(self) -> None:
        original_values = copy.deepcopy(self.values)
        original_metadata = copy.deepcopy(self.metadata)

        frame, warnings = prepare_input(self.values, self.metadata)

        self.assertEqual(frame.shape, (1, 8))
        self.assertEqual(list(frame.columns), list(FEATURE_COLUMNS))
        self.assertEqual(frame.loc[0, "area"], 60.0)
        self.assertFalse(warnings)
        self.assertEqual(self.values, original_values)
        self.assertEqual(self.metadata, original_metadata)

    def test_unknown_optional_values_reach_pipeline_as_missing(self) -> None:
        for feature in ("floor", "total_floors", "year_built", *CATEGORICAL_FEATURES):
            self.values[feature] = None

        frame, _ = prepare_input(self.values, self.metadata)

        self.assertTrue(frame.loc[0, ["floor", "total_floors", "year_built"]].isna().all())
        self.assertTrue(frame.loc[0, list(CATEGORICAL_FEATURES)].isna().all())
        self.assertEqual(frame.loc[0, "area"], 60.0)

    def test_area_and_rooms_are_required(self) -> None:
        for feature in ("area", "rooms"):
            with self.subTest(feature=feature):
                values = self.values | {feature: None}
                with self.assertRaises(ValueError):
                    prepare_input(values, self.metadata)

    def test_nonfinite_numbers_are_rejected(self) -> None:
        for feature in NUMERIC_FEATURES:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(feature=feature, value=value):
                    with self.assertRaises(ValueError):
                        prepare_input(self.values | {feature: value}, self.metadata)

    def test_fractional_counts_and_year_are_rejected(self) -> None:
        for feature in ("rooms", "floor", "total_floors", "year_built"):
            with self.subTest(feature=feature):
                value = float(self.values[feature]) + 0.5
                with self.assertRaises(ValueError):
                    prepare_input(self.values | {feature: value}, self.metadata)

    def test_physical_limits_are_enforced(self) -> None:
        invalid_values = {
            "area": (9.9, 1000.1),
            "rooms": (0, 21),
            "floor": (0, 101),
            "total_floors": (0, 101),
            "year_built": (1849, REFERENCE_YEAR + 6),
        }
        for feature, values in invalid_values.items():
            for value in values:
                with self.subTest(feature=feature, value=value):
                    with self.assertRaises(ValueError):
                        prepare_input(self.values | {feature: value}, self.metadata)

    def test_inclusive_limits_are_allowed_with_extrapolation_warning(self) -> None:
        boundary_values = [
            {"area": 10.0, "rooms": 1, "floor": 1, "total_floors": 1, "year_built": 1850},
            {"area": 1000.0, "rooms": 20, "floor": 100, "total_floors": 100,
             "year_built": REFERENCE_YEAR + 5},
        ]
        for values in boundary_values:
            with self.subTest(values=values):
                frame, warnings = prepare_input(self.values | values, self.metadata)
                self.assertEqual(frame.loc[0, "area"], values["area"])
                self.assertTrue(warnings)

    def test_floor_cannot_exceed_building_height(self) -> None:
        with self.assertRaises(ValueError):
            prepare_input(self.values | {"floor": 10, "total_floors": 9}, self.metadata)
        # Если этажность неизвестна, противоречие установить нельзя.
        frame, _ = prepare_input(self.values | {"floor": 10, "total_floors": None}, self.metadata)
        self.assertEqual(frame.loc[0, "floor"], 10)

    def test_unrecognised_categories_are_rejected(self) -> None:
        for feature in CATEGORICAL_FEATURES:
            with self.subTest(feature=feature):
                with self.assertRaises(ValueError):
                    prepare_input(self.values | {feature: "неизвестная категория"}, self.metadata)


class PredictionCalculationTests(unittest.TestCase):
    def test_price_per_area_and_mae_range_use_unrounded_prediction(self) -> None:
        pipeline = Mock()
        pipeline.predict.return_value = np.array([42_123_456.78])

        result = predict_apartment(pipeline, metadata_fixture(), apartment_fixture())

        self.assertAlmostEqual(result.price_tenge, 42_123_456.78)
        self.assertAlmostEqual(result.price_per_m2, 42_123_456.78 / 60.0)
        self.assertAlmostEqual(result.lower_tenge, 41_123_456.78)
        self.assertAlmostEqual(result.upper_tenge, 43_123_456.78)
        self.assertIsInstance(result.warnings, tuple)
        self.assertEqual(pipeline.predict.call_count, 1)
        frame = pipeline.predict.call_args.args[0]
        self.assertEqual(list(frame.columns), list(FEATURE_COLUMNS))

    def test_lower_bound_is_not_negative(self) -> None:
        pipeline = Mock()
        pipeline.predict.return_value = np.array([500_000.0])
        result = predict_apartment(pipeline, metadata_fixture(), apartment_fixture())
        self.assertEqual(result.lower_tenge, 0.0)
        self.assertEqual(result.upper_tenge, 1_500_000.0)

    def test_invalid_model_outputs_do_not_become_visible_prices(self) -> None:
        for value in (np.nan, np.inf, -np.inf, 0.0, -1.0):
            with self.subTest(value=value):
                pipeline = Mock()
                pipeline.predict.return_value = np.array([value])
                with self.assertRaises(ValueError):
                    predict_apartment(pipeline, metadata_fixture(), apartment_fixture())

    def test_bad_input_does_not_call_model(self) -> None:
        pipeline = Mock()
        with self.assertRaises(ValueError):
            predict_apartment(pipeline, metadata_fixture(), apartment_fixture() | {"area": 0})
        pipeline.predict.assert_not_called()

    def test_extrapolation_warning_survives_prediction(self) -> None:
        pipeline = Mock()
        pipeline.predict.return_value = np.array([100_000_000.0])
        result = predict_apartment(
            pipeline, metadata_fixture(), apartment_fixture() | {"area": 400.0},
        )
        self.assertTrue(result.warnings)


class ArtifactValidationTests(unittest.TestCase):
    def test_missing_model_gives_file_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                load_artifacts(path / "missing.joblib", path / "missing.json")

    def test_hash_mismatch_is_detected_before_deserializing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            model_path = path / "model.joblib"
            metadata_path = path / "model.json"
            model_path.write_bytes(b"damaged model")
            metadata_path.write_text(json.dumps(metadata_fixture()), encoding="utf-8")
            with patch("prediction.joblib.load") as load:
                with self.assertRaises(ValueError):
                    load_artifacts(model_path, metadata_path)
                load.assert_not_called()

    def test_incompatible_feature_schema_is_detected_before_deserializing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            model_path = path / "model.joblib"
            metadata_path = path / "model.json"
            content = b"not a pipeline"
            model_path.write_bytes(content)
            metadata = metadata_fixture()
            metadata["model_sha256"] = hashlib.sha256(content).hexdigest()
            metadata["feature_columns"].append("price_per_m2")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with patch("prediction.joblib.load") as load:
                with self.assertRaises(ValueError):
                    load_artifacts(model_path, metadata_path)
                load.assert_not_called()


@unittest.skipUnless(
    MODEL_PIPELINE_PATH.exists() and MODEL_METADATA_PATH.exists(),
    "Для интеграционных проверок нужны артефакты этапа 5.",
)
class RealModelApplicationTests(unittest.TestCase):
    def test_saved_pipeline_prediction_agrees_with_application_calculation(self) -> None:
        pipeline, metadata = load_artifacts(MODEL_PIPELINE_PATH, MODEL_METADATA_PATH)
        values = apartment_fixture()
        frame, _ = prepare_input(values, metadata)
        raw_price = float(pipeline.predict(frame)[0])

        result = predict_apartment(pipeline, metadata, values)

        self.assertTrue(np.isfinite(result.price_tenge))
        self.assertGreater(result.price_tenge, 0)
        self.assertAlmostEqual(result.price_tenge, raw_price)
        self.assertAlmostEqual(result.price_per_m2, raw_price / values["area"])
        self.assertAlmostEqual(result.upper_tenge - raw_price, metadata["test_metrics"]["mae_tenge"])

    def test_form_submission_then_invalid_input_does_not_leave_stale_price(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        for key in NUMERIC_FEATURES:
            self.assertIsNotNone(app.number_input(key=key))
            app.number_input(key=key).set_value(apartment_fixture()[key])
        for key in CATEGORICAL_FEATURES:
            self.assertIsNotNone(app.selectbox(key=key))
        submit = next(button for button in app.button if button.label == "Рассчитать стоимость")

        submit.click().run()

        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        labels = {metric.label for metric in app.metric}
        self.assertTrue({"Прогноз цены", "Цена за м²"}.issubset(labels))
        self.assertTrue(app.info)

        app.number_input(key="floor").set_value(10)
        app.number_input(key="total_floors").set_value(9)
        next(button for button in app.button if button.label == "Рассчитать стоимость").click().run()

        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)

    def test_empty_required_fields_show_validation_error(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        self.assertIsNone(app.number_input(key="area").value)
        self.assertIsNone(app.number_input(key="rooms").value)

        next(button for button in app.button if button.label == "Рассчитать стоимость").click().run()

        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)

    def test_clearing_optional_numbers_keeps_them_missing_after_resubmission(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        for key in NUMERIC_FEATURES:
            app.number_input(key=key).set_value(apartment_fixture()[key])
        next(button for button in app.button if button.label == "Рассчитать стоимость").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.metric)

        optional_numbers = ("floor", "total_floors", "year_built")
        for key in optional_numbers:
            app.number_input(key=key).set_value(None)
        next(button for button in app.button if button.label == "Рассчитать стоимость").click().run()

        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(app.metric)
        snapshot = app.session_state["last_input"]
        for key in optional_numbers:
            self.assertIsNone(snapshot[key])
        frame, _ = prepare_input(snapshot, metadata_fixture())
        self.assertTrue(frame.loc[0, list(optional_numbers)].isna().all())
        warning_text = " ".join(warning.value for warning in app.warning).lower()
        for label in ("этаж", "этажность дома", "год постройки"):
            self.assertIn(label, warning_text)

    def test_clearing_required_area_removes_previous_prediction(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        app.number_input(key="area").set_value(60.0)
        app.number_input(key="rooms").set_value(2)
        next(button for button in app.button if button.label == "Рассчитать стоимость").click().run()
        self.assertTrue(app.metric)

        app.number_input(key="area").set_value(None)
        next(button for button in app.button if button.label == "Рассчитать стоимость").click().run()

        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()

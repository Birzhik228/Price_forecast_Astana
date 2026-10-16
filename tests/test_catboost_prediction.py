"""Application contract for the trained MAE candidate, using tiny synthetic fits."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from catboost_model import CATBOOST_MODEL_COLUMNS, CatBoostPriceRegressor
from location_model import LOCATION_INPUT_COLUMNS, prepare_location_input
from location_prediction import LocationPredictionResult, load_app_artifacts, predict_app_apartment
from test_app import apartment_fixture, metadata_fixture


MAE = 5_580_000.0


def catboost_metadata_fixture() -> dict:
    metadata = metadata_fixture()
    metadata.update({
        "schema_version": 3,
        "variant": "catboost_mae",
        "model_label": "CatBoost с MAE · ЖК",
        "feature_columns": list(LOCATION_INPUT_COLUMNS),
        "actual_model_features": list(CATBOOST_MODEL_COLUMNS),
        "target_mode": "price_mae",
        "uses_coordinates": False,
        "known_complexes_from_train": ["тестовый жк", "второй жк"],
        "data_period_label": "января и конца марта 2026 года",
        "test_metrics": {"mae_tenge": MAE, "rmse_tenge": 10_000_000., "r2": .9},
        "approximate_range": {"half_width_tenge": MAE, "source": "reused_march_holdout", "guaranteed_coverage": None},
        "selected_by_cv": False,
        "cv_selected_variant": "catboost_log",
        "activation_reason": "explicit_user_request",
    })
    return metadata


def synthetic_catboost(target_mode: str = "price_mae") -> CatBoostPriceRegressor:
    rows = [apartment_fixture() | {
        "area": float(45 + index * 2),
        "district": "Есиль" if index < 12 else "Нура",
        "residential_complex": "Тестовый ЖК" if index < 12 else "Второй ЖК",
        "house_key": f"synthetic-house-{index % 4}",
        "raw_latitude": 51.12 + .01 * (index % 4),
        "raw_longitude": 71.43 + .01 * (index % 4),
        "raw_coordinate_status": "unverified",
    } for index in range(24)]
    frame = pd.DataFrame(rows, columns=list(LOCATION_INPUT_COLUMNS))
    target = frame.area * 500_000 + frame.district.eq("Есиль") * 1_000_000
    return CatBoostPriceRegressor(target_mode=target_mode, iterations=3, depth=2, thread_count=1).fit(frame, target)


class CatBoostApplicationPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fitted = synthetic_catboost()
        cls.log_fitted = synthetic_catboost("log_rmse")

    def setUp(self) -> None:
        self.model = copy.deepcopy(self.fitted)
        self.metadata = catboost_metadata_fixture()
        self.values = apartment_fixture()
        self.location = {"complex_name": "Тестовый ЖК", "address_normalized": "Тестовая 1", "location_district": "Есиль"}

    def test_adapter_matches_direct_prediction_without_training_or_mutation(self) -> None:
        expected = prepare_location_input(pd.DataFrame([self.values | {
            "residential_complex": "тестовый жк", "house_key": None,
            "raw_latitude": np.nan, "raw_longitude": np.nan, "raw_coordinate_status": "missing",
        }]))
        expected_price = float(self.model.predict(expected)[0])
        original = copy.deepcopy((self.values, self.location, self.metadata))
        with patch.object(CatBoostPriceRegressor, "fit", side_effect=AssertionError("App must not train")):
            result = predict_app_apartment(self.model, self.metadata, self.values, self.location)
        self.assertIsInstance(result, LocationPredictionResult)
        self.assertEqual(result.price_tenge, expected_price)
        self.assertEqual(result.price_per_m2, expected_price / self.values["area"])
        self.assertEqual(result.lower_tenge, max(0., expected_price - MAE))
        self.assertEqual(result.upper_tenge, expected_price + MAE)
        self.assertEqual(result.warnings, ())
        self.assertTrue(any("Учтён ЖК" in note for note in result.location_notes))
        self.assertEqual((self.values, self.location, self.metadata), original)

    def test_address_and_catalog_coordinates_do_not_change_price_or_native_features(self) -> None:
        injected = self.location | {
            "address_normalized": "Другая улица 999", "house_key": "forged",
            "latitude": -70., "longitude": -130., "raw_latitude": 51.12,
            "raw_longitude": 71.43, "raw_coordinate_status": "verified",
            "coordinate_status": "plausible", "building_id": "pretend-known",
        }
        normal = predict_app_apartment(self.model, self.metadata, self.values, self.location)
        with patch.object(self.model, "predict", wraps=self.model.predict) as spy:
            changed = predict_app_apartment(self.model, self.metadata, self.values, injected)
        self.assertEqual(changed.price_tenge, normal.price_tenge)
        frame = spy.call_args.args[0]
        self.assertEqual(list(frame.columns), list(LOCATION_INPUT_COLUMNS))
        self.assertTrue(frame[["raw_latitude", "raw_longitude"]].isna().all().all())
        self.assertEqual(frame.raw_coordinate_status.iloc[0], "missing")
        native = self.model.transform_features(frame)
        self.assertEqual(list(native.columns), list(CATBOOST_MODEL_COLUMNS))
        self.assertNotIn("house_key", native)
        self.assertNotIn("latitude", native)
        notes = " ".join(changed.location_notes).lower()
        self.assertNotIn("координаты выбранного дома учтены", notes)
        self.assertNotIn("координаты учтены по точному", notes)

    def test_missing_or_unknown_complex_keeps_prediction_available_and_explains_it(self) -> None:
        missing = predict_app_apartment(self.model, self.metadata, self.values)
        self.assertGreater(missing.price_tenge, 0)
        self.assertTrue(any("ЖК не указан" in note for note in missing.location_notes))
        unknown = predict_app_apartment(self.model, self.metadata, self.values, {"complex_name": "Несуществующий ЖК"})
        self.assertGreater(unknown.price_tenge, 0)
        self.assertTrue(any("не встречался" in warning for warning in unknown.warnings))
        for null in (None, np.nan, pd.NA):
            with self.subTest(null=null):
                result = predict_app_apartment(self.model, self.metadata, self.values, {"complex_name": null})
                self.assertEqual(result, missing)

    def test_district_conflict_is_rejected_before_prediction_but_alias_is_allowed(self) -> None:
        with patch.object(self.model, "predict", side_effect=AssertionError("Reject before predicting")):
            with self.assertRaisesRegex(ValueError, "Район квартиры"):
                predict_app_apartment(self.model, self.metadata, self.values | {"district": "Нура"}, self.location)
            with self.assertRaisesRegex(ValueError, "противоречиво"):
                predict_app_apartment(self.model, self.metadata, self.values, self.location | {"district": "Нура"})
        alias = predict_app_apartment(self.model, self.metadata, self.values, self.location | {"location_district": "Есильский"})
        self.assertEqual(alias, predict_app_apartment(self.model, self.metadata, self.values, self.location))

    def test_invalid_model_outputs_are_rejected(self) -> None:
        for output in ([np.nan], [np.inf], [0], [-1], [1, 2], [[20e6]]):
            with self.subTest(output=output), patch.object(self.model, "predict", return_value=np.asarray(output)):
                with self.assertRaisesRegex(ValueError, "положительную конечную"):
                    predict_app_apartment(self.model, self.metadata, self.values, self.location)

    def test_loader_roundtrip_and_schema_hash_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path, metadata_path = Path(temporary) / "model.joblib", Path(temporary) / "metadata.json"
            joblib.dump(self.model, model_path)
            metadata = self.metadata | {"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}

            def save(value):
                metadata_path.write_text(json.dumps(value), encoding="utf-8")

            save(metadata)
            model, loaded_metadata = load_app_artifacts(model_path, metadata_path)
            self.assertEqual(predict_app_apartment(model, loaded_metadata, self.values, self.location),
                             predict_app_apartment(self.model, metadata, self.values, self.location))
            for changes in (
                {"target_mode": "log_rmse"}, {"variant": "catboost_log"}, {"uses_coordinates": True},
                {"actual_model_features": [*CATBOOST_MODEL_COLUMNS, "latitude", "longitude"]},
                {"feature_columns": list(CATBOOST_MODEL_COLUMNS)}, {"model_sha256": "f" * 64},
                {"approximate_range": {"half_width_tenge": MAE + 1_000_000}},
            ):
                with self.subTest(changes=changes):
                    save(metadata | changes)
                    with self.assertRaises(ValueError):
                        load_app_artifacts(model_path, metadata_path)

    def test_loader_rejects_real_log_model_even_with_mae_metadata_and_matching_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path, metadata_path = Path(temporary) / "model.joblib", Path(temporary) / "metadata.json"
            joblib.dump(self.log_fitted, model_path)
            metadata = self.metadata | {"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_app_artifacts(model_path, metadata_path)

    def test_loader_rejects_renamed_native_features_even_with_matching_sha(self) -> None:
        self.model.model_.set_feature_names([*CATBOOST_MODEL_COLUMNS[:-1], "wrong_complex_field"])
        with tempfile.TemporaryDirectory() as temporary:
            model_path, metadata_path = Path(temporary) / "model.joblib", Path(temporary) / "metadata.json"
            joblib.dump(self.model, model_path)
            metadata = self.metadata | {"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_app_artifacts(model_path, metadata_path)


if __name__ == "__main__":
    unittest.main()

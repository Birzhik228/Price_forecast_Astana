"""Публичный адаптер приложения: схема 2 и неизменность старого пути прогнозов."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from building_catalog import canonical_house_key
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, build_pipeline
from location_model import LOCATION_INPUT_COLUMNS, make_location_pipeline, prepare_location_input
from location_prediction import LocationPredictionResult, load_app_artifacts, predict_app_apartment
from prediction import PredictionResult, predict_apartment


def fixture() -> tuple[Pipeline, dict, pd.DataFrame, pd.Series]:
    rows = []
    for i in range(24):
        first = i < 12
        name, district, address = ("Первый ЖК", "Есиль", "Тестовая 1") if first else ("Второй ЖК", "Нура", "Тестовая 2")
        rows.append({"area": float(40 + i), "rooms": 2., "floor": 5., "total_floors": 9., "year_built": 2015.,
                     "district": district, "building_type": "кирпичный", "condition": "хорошее",
                     "residential_complex": name, "house_key": canonical_house_key(address, district, name),
                     "raw_latitude": 51.12 if first else 51.15, "raw_longitude": 71.43 if first else 71.46,
                     "raw_coordinate_status": "unverified"})
    frame = pd.DataFrame(rows)
    target = frame.area * 500_000 + frame.district.eq("Есиль") * 1_000_000
    model = make_location_pipeline(RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42), "complex_coords").fit(frame, target)
    metadata = {"schema_version": 2, "variant": "complex_coords", "feature_columns": list(LOCATION_INPUT_COLUMNS),
                "target": "asking_price", "currency": "KZT", "model_sha256": "0" * 64,
                "categories_from_train": {column: sorted(frame[column].unique()) for column in CATEGORICAL_FEATURES},
                "numeric_ranges_from_train": {column: {"min": float(frame[column].min()), "max": float(frame[column].max())} for column in NUMERIC_FEATURES},
                "test_metrics": {"mae_tenge": 2_000_000.}, "approximate_range": {"half_width_tenge": 2_000_000.},
                "selected_by_cv": False, "activation_reason": "explicit_user_request"}
    return model, metadata, frame, target


class LocationPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.metadata, self.training, self.target = fixture()
        self.values = {column: self.training.iloc[5][column] for column in FEATURE_COLUMNS}
        self.location = {"complex_name": "Первый ЖК", "address_normalized": "Тестовая 1", "location_district": "Есиль"}

    def test_known_house_exact_input_schema_prediction_and_notes_without_fit(self) -> None:
        expected = prepare_location_input(pd.DataFrame([{
            **self.values, "residential_complex": "Первый ЖК",
            "house_key": canonical_house_key("Тестовая 1", "Есиль", "Первый ЖК"),
            "raw_latitude": np.nan, "raw_longitude": np.nan, "raw_coordinate_status": "missing",
        }]))
        expected_price = float(self.model.predict(expected)[0])
        before_map = copy.deepcopy(self.model.named_steps["location_features"].house_coordinate_map_)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No application fit")):
            with patch.object(self.model, "predict", wraps=self.model.predict) as spy:
                result = predict_app_apartment(self.model, self.metadata, self.values, self.location)
        self.assertIsInstance(result, LocationPredictionResult)
        self.assertIsInstance(result, PredictionResult)
        pd.testing.assert_frame_equal(spy.call_args.args[0], expected)
        self.assertEqual(result.price_tenge, expected_price)
        self.assertEqual(result.price_per_m2, expected_price / self.values["area"])
        self.assertEqual(result.lower_tenge, max(0, expected_price - 2e6))
        self.assertEqual(result.upper_tenge, expected_price + 2e6)
        self.assertEqual(result.warnings, ())
        self.assertTrue(any("Учтён ЖК" in note for note in result.location_notes))
        self.assertTrue(any("Координаты выбранного дома учтены" in note for note in result.location_notes))
        self.assertEqual(before_map, self.model.named_steps["location_features"].house_coordinate_map_)

    def test_catalog_coordinates_and_supplied_house_key_cannot_inject_model_geography(self) -> None:
        injected = {"house_key": self.training.house_key.iloc[0], "latitude": 51.12, "longitude": 71.43,
                    "raw_latitude": 51.12, "raw_longitude": 71.43, "raw_coordinate_status": "unverified",
                    "building_id": "pretend-known", "coordinate_status": "plausible"}
        with patch.object(self.model, "predict", wraps=self.model.predict) as spy:
            result = predict_app_apartment(self.model, self.metadata, self.values, injected)
        actual = spy.call_args.args[0]
        self.assertIsNone(actual.house_key.iloc[0])
        self.assertTrue(actual[["raw_latitude", "raw_longitude"]].isna().all().all())
        self.assertEqual(actual.raw_coordinate_status.iloc[0], "missing")
        self.assertTrue(self.model.named_steps["location_features"].transform(actual).latitude.isna().all())
        self.assertTrue(any("недоступны" in note for note in result.location_notes))
        known = predict_app_apartment(self.model, self.metadata, self.values, self.location)
        tainted = predict_app_apartment(self.model, self.metadata, self.values, self.location | injected)
        self.assertEqual(known, tainted)

    def test_missing_location_is_optional_and_unknown_complex_is_explained(self) -> None:
        missing = predict_app_apartment(self.model, self.metadata, self.values)
        self.assertEqual(missing.warnings, ())
        self.assertTrue(any("ЖК не указан" in note for note in missing.location_notes))
        self.assertTrue(any("недоступны" in note for note in missing.location_notes))
        unknown = predict_app_apartment(self.model, self.metadata, self.values, {"residential_complex": "Новый ЖК"})
        self.assertTrue(any("не встречался" in warning for warning in unknown.warnings))
        for null in (None, np.nan, pd.NA):
            current = predict_app_apartment(self.model, self.metadata, self.values,
                                            {"complex_name": null, "address_normalized": null, "location_district": null})
            self.assertEqual(current, missing)

    def test_conflicting_year_disables_coordinates_and_district_conflict_rejects(self) -> None:
        with patch.object(self.model, "predict", wraps=self.model.predict) as spy:
            result = predict_app_apartment(self.model, self.metadata, self.values | {"year_built": 2020}, self.location)
        transformed = self.model.named_steps["location_features"].transform(spy.call_args.args[0])
        self.assertTrue(transformed.latitude.isna().all())
        self.assertTrue(any("координаты не использованы" in warning for warning in result.warnings))
        with patch.object(self.model, "predict", side_effect=AssertionError("Must reject before predict")):
            with self.assertRaisesRegex(ValueError, "Район квартиры"):
                predict_app_apartment(self.model, self.metadata, self.values | {"district": "Нура"}, self.location)
            with self.assertRaisesRegex(ValueError, "противоречиво"):
                predict_app_apartment(self.model, self.metadata, self.values, self.location | {"district": "Нура"})
        alias = predict_app_apartment(self.model, self.metadata, self.values, self.location | {"location_district": "Есильский"})
        expected = predict_app_apartment(self.model, self.metadata, self.values, self.location)
        self.assertEqual(alias, expected)

    def test_invalid_outputs_are_rejected(self) -> None:
        for output in ([np.nan], [np.inf], [0], [-1], [1, 2], [[20e6]]):
            with self.subTest(output=output), patch.object(self.model, "predict", return_value=np.asarray(output)):
                with self.assertRaisesRegex(ValueError, "положительную конечную"):
                    predict_app_apartment(self.model, self.metadata, self.values)

    def test_loader_checks_sha_variant_and_legacy_projection_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            model_path, metadata_path = path / "model.joblib", path / "metadata.json"
            joblib.dump(self.model, model_path)
            metadata = self.metadata | {"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
            def save(value):
                metadata_path.write_text(json.dumps(value), encoding="utf-8")
            save(metadata)
            restored, loaded_metadata = load_app_artifacts(model_path, metadata_path)
            self.assertEqual(predict_app_apartment(restored, loaded_metadata, self.values, self.location),
                             predict_app_apartment(self.model, metadata, self.values, self.location))
            for changes in ({"variant": "complex"}, {"feature_columns": list(FEATURE_COLUMNS)},
                            {"approximate_range": {"half_width_tenge": 123}}, {"model_sha256": "f" * 64}):
                with self.subTest(changes=changes):
                    save(metadata | changes)
                    with self.assertRaises(ValueError):
                        load_app_artifacts(model_path, metadata_path)
            save(metadata)
            with model_path.open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "не совпадают"):
                load_app_artifacts(model_path, metadata_path)

    def test_legacy_schema_load_and_predict_delegate_without_behavior_change(self) -> None:
        model = build_pipeline(RandomForestRegressor(n_estimators=3, random_state=42)).fit(self.training[list(FEATURE_COLUMNS)], self.target)
        metadata = self.metadata | {"schema_version": 1, "feature_columns": list(FEATURE_COLUMNS)}
        expected = predict_apartment(model, metadata, self.values)
        actual = predict_app_apartment(model, metadata, self.values, {"location_district": "Нура"})
        self.assertEqual(actual, expected)
        self.assertNotIsInstance(actual, LocationPredictionResult)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            joblib.dump(model, path / "legacy.joblib")
            metadata["model_sha256"] = hashlib.sha256((path / "legacy.joblib").read_bytes()).hexdigest()
            (path / "legacy.json").write_text(json.dumps(metadata), encoding="utf-8")
            restored, restored_meta = load_app_artifacts(path / "legacy.joblib", path / "legacy.json")
            self.assertEqual(predict_app_apartment(restored, restored_meta, self.values), expected)


if __name__ == "__main__":
    unittest.main()

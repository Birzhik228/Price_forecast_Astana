"""Отсутствие утечки в location-признаках и совместимость отдельного pipeline."""

from __future__ import annotations

import copy
import io
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from features import FEATURE_COLUMNS, build_pipeline
from location_model import (
    COMPLEX_MAX_CATEGORIES, LOCATION_INPUT_COLUMNS, LOCATION_SCHEMA_VERSION,
    LOCATION_VARIANTS, LocationFeatureTransformer, make_location_pipeline,
    prepare_location_input,
)


def row(house: str | None = "house-a", **changes) -> dict:
    return {
        "area": 60.0, "rooms": 2, "floor": 5, "total_floors": 9, "year_built": 2015,
        "district": "Есиль", "building_type": "кирпичный", "condition": "хорошее",
        "residential_complex": "ЖК Пример", "house_key": house,
        "raw_latitude": 51.12, "raw_longitude": 71.43, "raw_coordinate_status": "unverified",
        **changes,
    }


def training_frame() -> pd.DataFrame:
    return pd.DataFrame([
        row(area=40 + i, floor=1 + i % 6) if i < 12 else
        row("house-b", area=40 + i, floor=1 + i % 6, raw_latitude=51.14,
            raw_longitude=71.45, residential_complex="Другой ЖК")
        for i in range(24)
    ], index=pd.Index(np.arange(100, 124), name="source"))


def estimator() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=7, max_depth=4, min_samples_leaf=1,
                                 max_features=0.8, random_state=42, n_jobs=1)


class LocationMapTests(unittest.TestCase):
    def test_validation_points_never_change_fit_map_or_promote_unseen_or_singleton_house(self) -> None:
        train = pd.DataFrame([row(), row(), row("singleton", raw_latitude=51.16, raw_longitude=71.47)])
        transformer = LocationFeatureTransformer("coords").fit(train)
        before_map, before_audit = copy.deepcopy(transformer.house_coordinate_map_), transformer.coordinate_audit_.copy()
        validation = pd.DataFrame([
            row(raw_latitude=0, raw_longitude=0, raw_coordinate_status="invalid"),
            row("singleton", raw_latitude=51.16, raw_longitude=71.47),
            row("unseen"), row("unseen"),
        ], index=[80, 30, 20, 10])
        transformed = transformer.transform(validation)
        self.assertEqual(transformer.house_coordinate_map_, before_map)
        pd.testing.assert_frame_equal(transformer.coordinate_audit_, before_audit)
        pd.testing.assert_index_equal(transformed.index, validation.index)
        np.testing.assert_allclose(transformed.loc[80, ["latitude", "longitude"]].to_numpy(float), [51.12, 71.43])
        self.assertTrue(transformed.loc[[30, 20, 10], ["latitude", "longitude"]].isna().all().all())
        changed = validation.copy()
        changed[["raw_latitude", "raw_longitude"]] = [51.3, 71.8]
        changed["raw_coordinate_status"] = "unverified"
        pd.testing.assert_frame_equal(transformer.transform(changed), transformed)

    def test_conflicts_are_learned_only_on_fit_and_query_conflict_is_row_local(self) -> None:
        train = pd.DataFrame([row(), row()])
        transformer = LocationFeatureTransformer("complex_coords").fit(train)
        query = pd.DataFrame([row(), row(year_built=2020), row(total_floors=20),
                              row(residential_complex="Другое название"),
                              row(year_built=np.nan, total_floors=np.nan, residential_complex=None)])
        result = transformer.transform(query)
        self.assertTrue(result.loc[[0, 4], "latitude"].notna().all())
        self.assertTrue(result.loc[[1, 2, 3], "latitude"].isna().all())
        for index in query.index:
            pd.testing.assert_frame_equal(transformer.transform(query.loc[[index]]), result.loc[[index]])
        # Если конфликт попал именно в fit, дом уже не даёт координат никому.
        for column, value in (("year_built", 2020), ("total_floors", 20),
                              ("residential_complex", "Другое название")):
            with self.subTest(column=column):
                corrupted = train.copy()
                corrupted.loc[1, column] = value
                fitted = LocationFeatureTransformer("coords").fit(corrupted)
                self.assertEqual(fitted.house_coordinate_map_, {})
                self.assertEqual(fitted.coordinate_audit_.coordinate_status.iloc[0], "trait_conflict")

    def test_raw_coordinate_quality_rejects_invalid_spread_singletons_and_reused_markers(self) -> None:
        cases = {
            "invalid": [row(), row(), row(raw_coordinate_status="invalid", raw_latitude=np.nan)],
            "out_of_bounds": [row(), row(raw_latitude=0)],
            "bad_number": [row(), row(raw_latitude="bad")],
            "incomplete": [row(), row(raw_longitude=None)],
            "singleton": [row()],
            "spread": [row(), row(raw_latitude=51.16)],
            "reused": [row(), row(), row("house-b"), row("house-b")],
        }
        for case, records in cases.items():
            with self.subTest(case=case):
                fitted = LocationFeatureTransformer("coords").fit(pd.DataFrame(records))
                self.assertEqual(fitted.house_coordinate_map_, {})
        valid = LocationFeatureTransformer("coords").fit(pd.DataFrame([row(), row(raw_latitude=51.1201)]))
        np.testing.assert_allclose(valid.house_coordinate_map_["house-a"], [51.12005, 71.43])
        self.assertFalse(valid.training_geo_summary_["externally_verified"])

    def test_missing_raw_points_may_inherit_only_the_training_house_consensus(self) -> None:
        train = pd.DataFrame([row(), row(), row(raw_latitude=None, raw_longitude=None,
                                               raw_coordinate_status="missing")])
        fitted = LocationFeatureTransformer("coords").fit(train)
        march = pd.DataFrame([row(raw_latitude=None, raw_longitude=None, raw_coordinate_status="missing"),
                              row("new-house", raw_latitude=None, raw_longitude=None, raw_coordinate_status="missing")])
        result = fitted.transform(march)
        self.assertEqual(result.latitude.iloc[0], 51.12)
        self.assertTrue(np.isnan(result.latitude.iloc[1]))
        self.assertEqual(fitted.coordinate_audit_.coordinate_observations.iloc[0], 2)


class LocationPipelineTests(unittest.TestCase):
    def test_schema_excludes_target_ids_and_full_catalog_geography(self) -> None:
        frame = training_frame()
        frame["price"] = frame.area * 500_000
        frame["price_per_m2"] = 1e6
        frame["latitude"], frame["longitude"] = 99, 99
        frame["coordinate_status"], frame["building_id"] = "plausible", "all-data-id"
        frame["record_id"], frame["group"] = "not-a-feature", "not-a-feature"
        prepared = prepare_location_input(frame)
        self.assertEqual(LOCATION_SCHEMA_VERSION, 2)
        self.assertEqual(tuple(prepared.columns), LOCATION_INPUT_COLUMNS)
        self.assertEqual(len(prepared.columns), 13)
        original = frame.copy(deep=True)
        transformer = LocationFeatureTransformer().fit(frame, frame["price"])
        expected = transformer.transform(frame)
        frame[["price", "price_per_m2", "latitude", "longitude"]] = -1e20
        frame["coordinate_status"] = "invalid"
        actual = transformer.transform(frame)
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(prepare_location_input(original), prepared)
        self.assertEqual(set(actual.columns), set(FEATURE_COLUMNS) | {"residential_complex", "latitude", "longitude"})

    def test_original_baseline_preprocessing_and_regressor_parameters_are_preserved(self) -> None:
        frame = training_frame()
        y = frame.area * 500_000 + frame.floor * 100_000
        base = estimator()
        old = build_pipeline(clone(base).set_params(n_jobs=2)).fit(frame[list(FEATURE_COLUMNS)], y)
        fresh = make_location_pipeline(base, "baseline8").fit(frame, y)
        self.assertEqual(base.n_jobs, 1)
        expected_parameters = base.get_params() | {"n_jobs": 2}
        self.assertEqual(fresh.named_steps["model"].get_params(), expected_parameters)
        np.testing.assert_array_equal(fresh[:-1].transform(frame), old[:-1].transform(frame))
        np.testing.assert_array_equal(fresh.predict(frame), old.predict(frame))
        self.assertFalse(hasattr(clone(fresh).named_steps["location_features"], "house_coordinate_map_"))

    def test_all_variants_handle_unseen_complex_and_missing_geo_without_batch_effects(self) -> None:
        frame = training_frame()
        y = frame.area * 500_000
        query = pd.DataFrame([row("unseen", residential_complex="Новый ЖК", raw_coordinate_status="missing",
                                  raw_latitude=None, raw_longitude=None),
                              row("house-a", residential_complex=None, floor=np.nan)])
        for variant in LOCATION_VARIANTS:
            with self.subTest(variant=variant):
                pipeline = make_location_pipeline(estimator(), variant).fit(frame, y)
                predicted = pipeline.predict(query)
                self.assertTrue(np.isfinite(predicted).all())
                individual = np.concatenate([pipeline.predict(query.iloc[[index]]) for index in range(len(query))])
                np.testing.assert_array_equal(predicted, individual)
                modified = query.copy()
                modified["raw_latitude"], modified["raw_longitude"] = 51.2, 71.5
                modified["raw_coordinate_status"] = "unverified"
                np.testing.assert_array_equal(pipeline.predict(modified), predicted)
                names = pipeline[:-1].get_feature_names_out()
                self.assertFalse(any("price" in name or "house_key" in name or "raw_" in name for name in names))

    def test_joblib_roundtrip_preserves_fitted_maps_encoders_and_predictions(self) -> None:
        frame = training_frame()
        pipeline = make_location_pipeline(estimator(), "complex_coords").fit(frame, frame.area * 500_000)
        incoming = pd.DataFrame([row(), row("unknown", residential_complex="Новый ЖК")])
        stream = io.BytesIO()
        joblib.dump(pipeline, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        self.assertEqual(restored.named_steps["location_features"].house_coordinate_map_,
                         pipeline.named_steps["location_features"].house_coordinate_map_)
        np.testing.assert_array_equal(restored.predict(incoming), pipeline.predict(incoming))

    def test_high_cardinality_complex_encoding_is_dense_and_bounded(self) -> None:
        frame = pd.DataFrame([row(residential_complex=f"ЖК {index // 10}") for index in range(3000)])
        preprocessing = make_location_pipeline(estimator(), "complex")[:-1]
        transformed = preprocessing.fit_transform(frame)
        self.assertFalse(sparse.issparse(transformed))
        encoder = preprocessing.named_steps["preprocess"].named_transformers_["complex"].named_steps["encoder"]
        self.assertLessEqual(len(encoder.get_feature_names_out()), COMPLEX_MAX_CATEGORIES)
        # Новая категория не расширяет и не переобучает словарь.
        incoming = pd.DataFrame([row(residential_complex="Неизвестный комплекс")])
        self.assertEqual(preprocessing.transform(incoming).shape[1], transformed.shape[1])

    def test_all_missing_coordinates_keep_stable_prediction_shape(self) -> None:
        frame = training_frame()
        frame[["raw_latitude", "raw_longitude"]] = np.nan
        frame["raw_coordinate_status"] = "missing"
        frame["residential_complex"] = pd.NA
        model = make_location_pipeline(estimator(), "complex_coords").fit(frame, frame.area * 500_000)
        predicted = model.predict(pd.DataFrame([row("unseen", residential_complex=None)]))
        self.assertEqual(predicted.shape, (1,))
        self.assertTrue(np.isfinite(predicted).all())
        self.assertEqual(model.named_steps["location_features"].house_coordinate_map_, {})

    def test_rejects_inherited_coordinate_status_or_incomplete_raw_schema(self) -> None:
        frame = training_frame()
        for variant in ("unknown", ""):
            with self.assertRaises(ValueError):
                make_location_pipeline(estimator(), variant)
        with self.assertRaises(ValueError):
            prepare_location_input(frame.drop(columns=["raw_latitude"]))
        frame["raw_coordinate_status"] = "plausible"
        with self.assertRaisesRegex(ValueError, "исходные точки"):
            prepare_location_input(frame)


if __name__ == "__main__":
    unittest.main()

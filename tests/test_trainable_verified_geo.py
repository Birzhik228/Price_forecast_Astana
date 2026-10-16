"""External coordinates during training, fold isolation and portable serialization."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

from features import FEATURE_COLUMNS
from location_model import LOCATION_INPUT_COLUMNS
from trainable_verified_geo import TrainableVerifiedGeoTransformer, make_trainable_verified_pipeline
from verified_geo_model import load_verified_registry


def key(number: int) -> str:
    return f"astana|есиль|тестовая {number}"


def row(house: str | None, **changes) -> dict:
    return {"area": 60., "rooms": 2., "floor": 5., "total_floors": 9., "year_built": 2015.,
            "district": "Есиль", "building_type": "кирпичный", "condition": "хорошее",
            "residential_complex": "Пример", "house_key": house, "raw_latitude": np.nan,
            "raw_longitude": np.nan, "raw_coordinate_status": "missing", **changes}


class TrainableVerifiedGeoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.registry_path = Path(self.temporary.name) / "registry.json"
        self.registry_path.write_text(json.dumps({"schema_version": 1, "entries": {
            key(number): {"latitude": 51.15 + number * .001, "longitude": 71.46,
                          "source_status": "osm_exact_address", "osm_url": f"https://www.openstreetmap.org/way/{number}"}
            for number in (1, 2, 3, 99)
        }}), encoding="utf-8")
        self.registry = load_verified_registry(self.registry_path)

    def test_fit_transform_includes_registry_points_preserves_base_points_and_original_input(self):
        frame = pd.DataFrame([
            row(key(1), raw_latitude=51.12, raw_longitude=71.42, raw_coordinate_status="unverified"),
            row(key(1), raw_latitude=51.12, raw_longitude=71.42, raw_coordinate_status="unverified"),
            row(key(2)), row(None),
        ], index=[8, 2, 7, 4])
        before = frame.copy(deep=True)
        transformer = TrainableVerifiedGeoTransformer(self.registry)
        result = transformer.fit_transform(frame, y=np.array([1., 9e9, 8., 7.]))
        self.assertEqual(result.latitude.loc[8], 51.12)  # fixed registry never replaces fitted points
        self.assertEqual(result.latitude.loc[7], 51.152)
        self.assertTrue(pd.isna(result.latitude.loc[4]))
        pd.testing.assert_frame_equal(result, transformer.transform(frame))
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(list(transformer.feature_names_in_), list(LOCATION_INPUT_COLUMNS))
        self.assertEqual(list(transformer.get_feature_names_out()), [*FEATURE_COLUMNS, "residential_complex", "latitude", "longitude"])
        summary = transformer.training_geo_summary_
        self.assertEqual(summary["fit_rows"], 4)
        self.assertEqual(summary["training_consensus_rows"], 2)
        self.assertEqual(summary["external_coordinate_train_rows"], 1)
        self.assertEqual(summary["available_coordinate_train_rows"], 3)
        self.assertEqual(summary["fit_rows_without_house_key"], 1)
        self.assertEqual(summary["external_houses_not_present_in_train"], 2)
        self.assertEqual(set(transformer.coordinate_audit_.house_key), {key(1), key(2)})

    def test_heldout_points_and_traits_do_not_learn_or_change_another_row(self):
        train = pd.DataFrame([row(key(2)), row(key(4), raw_latitude=51.2, raw_longitude=71.5, raw_coordinate_status="unverified")])
        transformer = TrainableVerifiedGeoTransformer(self.registry).fit(train)
        before_maps = copy.deepcopy(transformer.house_coordinate_map_)
        before_traits = copy.deepcopy(transformer.house_traits_)
        before_audit = transformer.coordinate_audit_.copy(deep=True)
        validation = pd.DataFrame([row(key(2)), row(key(2), year_built=2020),
                                   row(key(4), raw_latitude=51.2, raw_longitude=71.5, raw_coordinate_status="unverified"),
                                   row(key(5), raw_latitude=51.3, raw_longitude=71.6, raw_coordinate_status="unverified"),
                                   row(key(99))], index=[50, 10, 40, 30, 20])
        result = transformer.transform(validation)
        self.assertTrue(result.latitude.loc[[10, 40, 30]].isna().all())  # own trait conflict, singleton, unseen raw point
        self.assertTrue(result.latitude.loc[[50, 20]].notna().all())  # fixed external map applies to unseen address too
        changed = validation.copy()
        changed[["raw_latitude", "raw_longitude"]] = [0., 0.]
        changed["raw_coordinate_status"] = "invalid"
        changed["price"] = 1e12  # forbidden target column is ignored by the schema
        pd.testing.assert_frame_equal(transformer.transform(changed), result)
        pd.testing.assert_frame_equal(transformer.transform(validation.loc[[50]]), result.loc[[50]])
        self.assertEqual(transformer.house_coordinate_map_, before_maps)
        self.assertEqual(transformer.house_traits_, before_traits)
        pd.testing.assert_frame_equal(transformer.coordinate_audit_, before_audit)
        self.assertNotIn(key(99), transformer.house_traits_)

    def test_train_conflict_blocks_external_map_and_refit_does_not_retain_previous_fold(self):
        transformer = TrainableVerifiedGeoTransformer(self.registry)
        frame = pd.DataFrame([row(key(3)), row(key(3), total_floors=20), row(key(2))])
        transformer.fit(frame)
        self.assertTrue(transformer.transform(frame).latitude.iloc[:2].isna().all())
        self.assertEqual(transformer.provenance(key(3))["source_status"], "blocked_trait_conflict")
        self.assertEqual(transformer.training_geo_summary_["registry_houses_blocked_by_train_traits"], 1)
        transformer.fit(pd.DataFrame([row(key(3))]))
        self.assertEqual(set(transformer.house_traits_), {key(3)})
        self.assertTrue(transformer.transform(pd.DataFrame([row(key(3))])).latitude.notna().all())
        self.assertEqual(transformer.training_geo_summary_["registry_houses_blocked_by_train_traits"], 0)

    def test_withdrawn_missing_keys_cannot_be_recreated_from_address_or_catalog_coords(self):
        frame = pd.DataFrame([row(None), row(None), row(None)])
        frame["house_key"] = [None, np.nan, pd.NA]
        frame["address_normalized"] = "Тестовая 2"
        frame["candidate_building_id"] = key(2)
        frame[["latitude", "longitude", "raw_latitude", "raw_longitude"]] = [51.2, 71.4, 51.2, 71.4]
        frame["raw_coordinate_status"] = "unverified"
        transformer = TrainableVerifiedGeoTransformer(self.registry)
        result = transformer.fit_transform(frame)
        self.assertTrue(result.latitude.isna().all())
        self.assertFalse(transformer.coordinate_provenance(frame).coordinate_available.any())
        self.assertEqual(transformer.training_geo_summary_["houses_seen"], 0)

    def test_clone_is_unfitted_isolated_and_uses_same_registry_content(self):
        original = TrainableVerifiedGeoTransformer(self.registry).fit(pd.DataFrame([row(key(2))]))
        copied = clone(original)
        self.assertIsNot(copied.registry, original.registry)
        self.assertEqual(copied.registry.entries, original.registry.entries)
        self.assertEqual(copied.registry.sha256, original.registry.sha256)
        with self.assertRaises(NotFittedError):
            copied.transform(pd.DataFrame([row(key(2))]))
        copied.fit(pd.DataFrame([row(key(3), year_built=2020)]))
        self.assertEqual(set(original.house_traits_), {key(2)})
        self.assertEqual(set(copied.house_traits_), {key(3)})
        original.registry.entries[key(2)]["latitude"] = 0.0
        self.assertEqual(original.transform(pd.DataFrame([row(key(2))])).latitude.iloc[0], 51.152)

    def test_pipeline_trains_on_external_values_and_joblib_reloads_without_registry_file(self):
        frame = pd.DataFrame([row(key(2) if i < 12 else key(3), area=float(40 + i)) for i in range(24)])
        target = frame.area * 300_000 + frame.house_key.eq(key(3)) * 2e6
        pipeline = make_trainable_verified_pipeline(RandomForestRegressor(n_estimators=5, random_state=42), self.registry)
        cloned = clone(pipeline)
        cloned.fit(frame, target)
        self.assertEqual(cloned.named_steps["model"].n_jobs, 2)
        self.assertEqual(list(cloned.feature_names_in_), list(LOCATION_INPUT_COLUMNS))
        features = cloned.named_steps["location_features"].transform(frame)
        # StandardScaler.mean_ verifies the downstream training step saw the OSM
        # coordinates; a predict-only overlay would instead train on imputed zeros.
        numeric = cloned.named_steps["preprocess"].named_transformers_["numeric"]
        self.assertAlmostEqual(numeric.named_steps["scaler"].mean_[5], float(features.latitude.mean()))
        self.assertAlmostEqual(numeric.named_steps["scaler"].mean_[6], 71.46)
        buffer = io.BytesIO()
        joblib.dump(cloned, buffer)
        self.registry_path.unlink()
        buffer.seek(0)
        restored = joblib.load(buffer)
        with joblib.parallel_backend("sequential"):
            np.testing.assert_array_equal(cloned.predict(frame), restored.predict(frame))
        pd.testing.assert_frame_equal(restored.named_steps["location_features"].coordinate_provenance(frame),
                                      cloned.named_steps["location_features"].coordinate_provenance(frame))
        self.assertEqual(restored.named_steps["location_features"].registry_sha256_, self.registry.sha256)
        self.assertEqual(restored.named_steps["location_features"].training_geo_summary_, cloned.named_steps["location_features"].training_geo_summary_)


if __name__ == "__main__":
    unittest.main()

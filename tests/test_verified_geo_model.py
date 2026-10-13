"""External coordinates preserve the frozen predictor and its identity guards."""

from __future__ import annotations

import copy
import io
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

from location_model import LOCATION_INPUT_COLUMNS, make_location_pipeline
from verified_geo_model import build_verified_candidate, load_verified_registry


def house(name: str) -> str:
    return f"astana|есиль|тестовая {name}"


def point(latitude=51.15, longitude=71.46, **extra) -> dict:
    return {"latitude": latitude, "longitude": longitude, "source_status": "osm_exact_address",
            "osm_url": "https://www.openstreetmap.org/way/12345", **extra}


def fixture() -> tuple[Pipeline, pd.DataFrame]:
    rows = []
    for i in range(32):
        key = house("1" if i < 10 else "2" if i < 20 else "3" if i < 26 else "4")
        known = i < 20
        rows.append({"record_id": f"test:{i}", "area": float(45 + i % 10), "rooms": 2., "floor": 4.,
                     "total_floors": 10. if i == 31 else 9., "year_built": 2015.,
                     "district": "Есиль", "building_type": "кирпичный", "condition": "хорошее",
                     "residential_complex": "Пример", "house_key": key,
                     "raw_latitude": (51.12 if i < 10 else 51.2) if known else np.nan,
                     "raw_longitude": (71.4 if i < 10 else 71.5) if known else np.nan,
                     "raw_coordinate_status": "unverified" if known else "missing",
                     "price": 20e6 + (8e6 if 10 <= i < 20 else 0) + (i % 10) * 200_000})
    frame = pd.DataFrame(rows, index=pd.Index(range(100, 132), name="source"))
    model = make_location_pipeline(RandomForestRegressor(n_estimators=9, random_state=42, max_depth=5),
                                   "complex_coords").fit(frame, frame.price)
    return model, frame


def registry(directory: Path, entries: dict):
    path = directory / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": entries}, ensure_ascii=False), encoding="utf-8")
    return load_verified_registry(path)


class VerifiedGeoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.base, self.frame = fixture()

    def test_adds_exact_new_houses_without_touching_fitted_coordinates_or_estimators(self):
        before = copy.deepcopy(self.base.named_steps["location_features"].house_coordinate_map_)
        original_model = self.base.named_steps["model"]
        snapshot = registry(self.directory, {house("1"): point(), house("3"): point(), house("99"): point()})
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No fit allowed")):
            candidate = build_verified_candidate(self.base, snapshot)
            transformed = candidate.named_steps["location_features"].transform(self.frame)
        self.assertEqual(self.base.named_steps["location_features"].house_coordinate_map_, before)
        self.assertEqual(candidate.named_steps["model"].get_params(), original_model.get_params())
        # sklearn tree pickles contain structure padding; compare learned values,
        # not a byte hash whose padding can change on an otherwise exact deepcopy.
        for original, copied in zip(original_model.estimators_, candidate.named_steps["model"].estimators_, strict=True):
            for attribute in ("children_left", "children_right", "feature", "threshold", "value", "n_node_samples"):
                np.testing.assert_array_equal(getattr(original.tree_, attribute), getattr(copied.tree_, attribute))
        self.assertIsNot(candidate.named_steps["model"], self.base.named_steps["model"])
        self.assertEqual(list(candidate.feature_names_in_), list(LOCATION_INPUT_COLUMNS))
        self.assertEqual(transformed.loc[100, "latitude"], 51.12)
        self.assertEqual(transformed.loc[120, "latitude"], 51.15)
        pd.testing.assert_frame_equal(transformed.drop(columns=["latitude", "longitude"]),
                                      self.base.named_steps["location_features"].transform(self.frame).drop(columns=["latitude", "longitude"]))
        self.assertEqual(candidate.named_steps["location_features"].provenance(house("1"))["source_status"], "training_consensus")

    def test_training_identity_conflicts_and_own_query_conflicts_stay_blocked(self):
        candidate = build_verified_candidate(self.base, registry(self.directory, {house("3"): point(), house("4"): point()}))
        transformer = candidate.named_steps["location_features"]
        queries = pd.concat([self.frame.loc[[120]]] * 5, ignore_index=True)
        queries.loc[1, "year_built"] = 2020
        queries.loc[2, "total_floors"] = 20
        queries.loc[3, "residential_complex"] = "Другой ЖК"
        queries.loc[4, ["year_built", "total_floors", "residential_complex"]] = [np.nan, np.nan, None]
        result = transformer.transform(queries)
        self.assertTrue(result.loc[[0, 4], "latitude"].notna().all())
        self.assertTrue(result.loc[[1, 2, 3], "latitude"].isna().all())
        self.assertTrue(transformer.transform(self.frame.loc[[126]]).latitude.isna().all())
        self.assertEqual(transformer.provenance(house("4"))["source_status"], "blocked_trait_conflict")
        self.assertEqual(transformer.coordinate_provenance(queries).source_status.iloc[1], "query_trait_conflict")

    def test_raw_gps_rejection_can_be_replaced_by_external_address_but_raw_query_cannot(self):
        training = self.frame.copy()
        training.loc[training.house_key.eq(house("3")), "raw_coordinate_status"] = "invalid"
        model = make_location_pipeline(RandomForestRegressor(n_estimators=3, random_state=42), "complex_coords").fit(training, training.price)
        candidate = build_verified_candidate(model, registry(self.directory, {house("3"): point()}))
        queries = training.loc[[120, 126]].copy()
        queries[["raw_latitude", "raw_longitude"]] = [51.3, 71.8]
        queries["raw_coordinate_status"] = "unverified"
        features = candidate.named_steps["location_features"].transform(queries)
        self.assertEqual(features.latitude.iloc[0], 51.15)
        self.assertTrue(pd.isna(features.latitude.iloc[1]))

    def test_query_batch_mutations_do_not_learn_map_and_serialization_is_stable(self):
        candidate = build_verified_candidate(self.base, registry(self.directory, {house("3"): point()}))
        transformer = candidate.named_steps["location_features"]
        before = copy.deepcopy(transformer.house_coordinate_map_)
        expected = transformer.transform(self.frame)
        changed = self.frame.copy()
        changed.loc[120, ["raw_latitude", "raw_longitude", "price"]] = [0, 0, 1e12]
        pd.testing.assert_frame_equal(transformer.transform(changed), expected)
        pd.testing.assert_frame_equal(transformer.transform(self.frame.loc[[120]]), expected.loc[[120]])
        self.assertEqual(transformer.house_coordinate_map_, before)
        buffer = io.BytesIO()
        joblib.dump(candidate, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        with joblib.parallel_backend("sequential"):
            np.testing.assert_array_equal(candidate.predict(self.frame), restored.predict(self.frame))
        with self.assertRaisesRegex(RuntimeError, "fit запрещён"):
            restored.fit(self.frame, self.frame.price)

    def test_loader_rejects_unsafe_unverified_nonfinite_and_duplicate_registry(self):
        malformed = [point(source_status="plausible"), point(osm_url="https://openstreetmap.org.evil/way/12345"),
                     point(osm_url="javascript:alert(1)"), point(latitude=float("nan")), point(longitude=0), point(latitude=True),
                     point(eligible_for_enrichment=False)]
        for entry in malformed:
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                registry(self.directory, {house("3"): entry})
        path = self.directory / "registry.json"
        path.write_text('{"schema_version":1,"entries":{},"entries":{}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Повторный"):
            load_verified_registry(path)
        snapshot = registry(self.directory, {house("3"): point()})
        snapshot.entries[house("3")]["latitude"] = 51.2
        with self.assertRaisesRegex(ValueError, "изменён"):
            build_verified_candidate(self.base, snapshot)


if __name__ == "__main__":
    unittest.main()

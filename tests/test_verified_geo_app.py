"""The UI must distinguish external address evidence from old training points."""

import json
from pathlib import Path
import tempfile
import unittest

from features import FEATURE_COLUMNS
from location_prediction import predict_app_apartment
from tests.test_location_prediction import fixture
from verified_geo_model import build_verified_candidate, load_verified_registry


class VerifiedGeoAppTests(unittest.TestCase):
    def test_source_link_is_shown_only_when_external_coordinate_was_used(self):
        model, metadata, frame, _ = fixture()
        key = frame.house_key.iloc[0]
        # Simulate a fitted address with traits but no usable original point.
        model.named_steps["location_features"].house_coordinate_map_.pop(key)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps({"schema_version": 1, "entries": {key: {
                "latitude": 51.15, "longitude": 71.4, "source_status": "osm_exact_address",
                "osm_url": "https://www.openstreetmap.org/way/123", "eligible_for_enrichment": True,
            }}}), encoding="utf-8")
            candidate = build_verified_candidate(model, load_verified_registry(path))
        values = frame.iloc[0][list(FEATURE_COLUMNS)].to_dict()
        location = {"complex_name": "Первый ЖК", "address_normalized": "Тестовая 1", "location_district": "Есиль"}
        result = predict_app_apartment(candidate, metadata, values, location)
        self.assertTrue(any("https://www.openstreetmap.org/way/123" in note for note in result.location_notes))
        self.assertTrue(any("ODbL" in note for note in result.location_notes))
        blocked = predict_app_apartment(candidate, metadata, {**values, "year_built": 2020}, location)
        self.assertFalse(any("https://www.openstreetmap.org/way/123" in note for note in blocked.location_notes))
        self.assertTrue(any("координаты не использованы" in warning for warning in blocked.warnings))


if __name__ == "__main__":
    unittest.main()

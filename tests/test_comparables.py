"""Правила поиска аналогов и целостность локального каталога без обучения модели."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from comparables import (
    BUILDING_COLUMNS, LISTING_COLUMNS, RESULT_COLUMNS,
    find_comparables, load_catalog, safe_listing_url,
)


def listing(index: int, **changes) -> dict:
    return {
        "record_id": f"march:{index}", "group": f"group-{index}",
        "url": f"https://krisha.kz/a/show/{index}",
        "price": 40_000_000.0, "area": 60.0, "rooms": 2,
        "floor": 5, "total_floors": 9, "year_built": 2015,
        "district": "Есиль", "building_type": "кирпичный", "condition": "хорошее",
        "building_id": None, "complex_id": None, "complex_name": None,
        "address_normalized": None, "latitude": np.nan, "longitude": np.nan,
        "coordinate_status": "missing", "parking_type": None,
        "date_label": "Март 2026 (по описанию источника)", **changes,
    }


class ComparableTests(unittest.TestCase):
    def test_location_priority_and_explicit_district_fallback(self) -> None:
        frame = pd.DataFrame([
            listing(1, building_id="house-a", area=72),
            listing(2, complex_id="complex-a"),
            listing(3, district="Другой", latitude=51.121, longitude=71.43, coordinate_status="plausible"),
            listing(4, latitude=51.3, longitude=71.7, coordinate_status="plausible"),
            listing(5, district="Другой"),
        ])
        original = frame.copy(deep=True)
        result = find_comparables(frame, {
            "area": 60, "rooms": 2, "building_id": "house-a", "complex_id": "complex-a",
            "latitude": 51.12, "longitude": 71.43, "coordinate_status": "plausible", "district": "Есиль",
        })
        self.assertEqual(result.record_id.tolist(), ["march:1", "march:2", "march:3", "march:4"])
        self.assertEqual(result.location_match.tolist(), ["building", "complex", "nearby", "district"])
        self.assertTrue(np.isnan(result.iloc[0].distance_m))
        self.assertAlmostEqual(result.iloc[2].distance_m, 111.195, delta=0.1)
        self.assertIn("координаты не проверены", result.iloc[2].similarity_reason)
        self.assertIn("близость домов не установлена", result.iloc[3].similarity_reason)
        self.assertEqual(result.iloc[0].price_per_m2, 40_000_000 / 72)
        pd.testing.assert_frame_equal(frame, original)

    def test_exact_room_and_inclusive_area_limits_with_usable_prices_and_links(self) -> None:
        rows = [listing(1, area=48), listing(2, area=72), listing(3, area=47.999),
                listing(4, area=72.001), listing(5, rooms=3), listing(6, price=0),
                listing(7, price=np.inf), listing(8, price=np.nan),
                listing(9, url="javascript:alert(1)"), listing(10, area=np.nan)]
        result = find_comparables(pd.DataFrame(rows), {"area": 60, "rooms": 2, "district": "Есиль"})
        self.assertEqual(set(result.record_id), {"march:1", "march:2"})
        fractional = find_comparables(
            pd.DataFrame([listing(1, area=54.8), listing(2, area=82.2)]),
            {"area": 68.5, "rooms": 2, "district": "Есиль"},
        )
        self.assertEqual(len(fractional), 2)

    def test_missing_location_does_not_match_missing_ids_or_return_citywide_results(self) -> None:
        frame = pd.DataFrame([listing(1), listing(2, building_id="known")])
        for optional in ({}, {"building_id": pd.NA, "complex_id": np.nan, "district": None},
                         {"latitude": 51.12, "longitude": 71.43, "coordinate_status": "unverified"},
                         {"building_id": "unknown-house"}):
            with self.subTest(optional=optional):
                result = find_comparables(frame, {"area": 60, "rooms": 2, **optional})
                self.assertTrue(result.empty)
                self.assertTrue(set(RESULT_COLUMNS).issubset(result.columns))

    def test_bad_or_unverified_coordinates_only_allow_independent_district_match(self) -> None:
        frame = pd.DataFrame([
            listing(1, coordinate_status="plausible", latitude=51.12, longitude=71.43),
            listing(2, coordinate_status="unverified", latitude=51.12, longitude=71.43),
            listing(3, coordinate_status="plausible", latitude=71.43, longitude=51.12),
            listing(4, coordinate_status="plausible", latitude=np.inf, longitude=71.43),
            listing(5, coordinate_status="plausible", latitude=True, longitude=71.43),
        ])
        query = {"area": 60, "rooms": 2, "latitude": 51.12, "longitude": 71.43,
                 "coordinate_status": "plausible"}
        nearby = find_comparables(frame, query)
        self.assertEqual(nearby.record_id.tolist(), ["march:1"])
        fallback = find_comparables(frame, {**query, "district": "  ЕСИЛЬ "})
        self.assertEqual(fallback.location_match.tolist(), ["nearby"] + ["district"] * 4)
        self.assertTrue(fallback.iloc[1:].distance_m.isna().all())
        for invalid_query in ({"latitude": 0}, {"longitude": "bad"}, {"coordinate_status": "missing"}):
            self.assertTrue(find_comparables(frame, {**query, **invalid_query}).empty)

    def test_transitive_duplicates_and_self_aliases_are_excluded_before_filtering(self) -> None:
        rows = [listing(1, group="a", rooms=3),
                listing(2, group="a"),
                listing(3, group="b", url="http://www.krisha.kz/a/show/2/?utm_source=test"),
                listing(4, group="b"), listing(5)]
        query = {"area": 60, "rooms": 2, "district": "Есиль"}
        frame = pd.DataFrame(rows)
        result = find_comparables(frame, query)
        self.assertEqual(result.record_id.tolist(), ["march:2", "march:5"])
        for exclusion in ({"exclude_record_id": "march:1"}, {"exclude_group": "a"},
                          {"exclude_url": "https://krisha.kz/a/show/1?alias=true"}):
            with self.subTest(exclusion=exclusion):
                result = find_comparables(frame, {**query, **exclusion})
                self.assertEqual(result.record_id.tolist(), ["march:5"])

    def test_rank_does_not_depend_on_price_prediction_or_input_row_order(self) -> None:
        frame = pd.DataFrame([listing(1, area=59), listing(2, area=61), listing(3, area=60)])
        frame.index = [40, 2, 77]
        query = {"area": 60, "rooms": 2, "district": "Есиль", "prediction": 1}
        first = find_comparables(frame, query)
        changed = frame.copy()
        changed["price"] = [1e9, 1, 1e12]
        second = find_comparables(changed.iloc[::-1], {**query, "prediction": 1e20})
        self.assertEqual(first.record_id.tolist(), ["march:3", "march:1", "march:2"])
        self.assertEqual(first.record_id.tolist(), second.record_id.tolist())

    def test_nearby_prefers_distance_then_area_and_respects_two_kilometres(self) -> None:
        frame = pd.DataFrame([
            listing(1, latitude=51.125, longitude=71.43, coordinate_status="plausible"),
            listing(2, latitude=51.121, longitude=71.43, coordinate_status="plausible", area=70),
            listing(3, latitude=51.15, longitude=71.43, coordinate_status="plausible"),
        ])
        query = {"area": 60, "rooms": 2, "latitude": 51.12, "longitude": 71.43,
                 "coordinate_status": "plausible"}
        self.assertEqual(find_comparables(frame, query).record_id.tolist(), ["march:2", "march:1"])
        self.assertEqual(find_comparables(frame, query, limit=1).record_id.tolist(), ["march:2"])
        self.assertTrue(find_comparables(frame, query, limit=0).empty)

    def test_query_validation_and_missing_optional_characteristics(self) -> None:
        frame = pd.DataFrame([listing(1, floor=pd.NA, condition=pd.NA, year_built=np.nan)])
        query = {"area": 60, "rooms": 2, "district": "Есиль", "floor": 5, "condition": "хорошее"}
        self.assertEqual(len(find_comparables(frame, query)), 1)
        for changes in ({"area": None}, {"area": True}, {"area": np.inf}, {"area": -1},
                        {"rooms": 2.5}, {"rooms": np.nan}, {"rooms": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                find_comparables(frame, {**query, **changes})
        for limit in (-1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                find_comparables(frame, query, limit=limit)


class CatalogTests(unittest.TestCase):
    def test_safe_url_only_accepts_actual_krisha_listing_routes(self) -> None:
        self.assertEqual(safe_listing_url("HTTP://www.krisha.kz/a/show/123/?utm=a#b"),
                         "https://krisha.kz/a/show/123")
        for value in (None, np.nan, pd.NA, "javascript:alert(1)", "https://krisha.kz.evil/a/show/1",
                      "https://krisha.kz@evil.test/a/show/1", "https://krisha.kz:443/a/show/1",
                      "https://evil.test/a/show/1", "https://krisha.kz/a/show/1/other",
                      "https://krisha.kz/a/show/1\n", "https://kri\nsha.kz/a/show/1"):
            with self.subTest(value=value):
                self.assertIsNone(safe_listing_url(value))

    def write_catalog(self, root: Path) -> dict:
        rows = pd.DataFrame([listing(1, building_id="b1")])
        buildings = pd.DataFrame([{column: None for column in BUILDING_COLUMNS}])
        buildings.loc[0, ["building_id", "listing_count", "display_label"]] = ["b1", 1, "Дом 1"]
        manifest = {"schema_version": 1, "files": {}}
        for key, frame in (("listings", rows), ("buildings", buildings)):
            path = root / f"{key}.csv"
            frame.to_csv(path, index=False)
            manifest["files"][key] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "rows": len(frame), "columns": frame.columns.tolist()}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_catalog_loads_both_tables_with_identity_strings_and_numeric_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_catalog(root)
            catalog = load_catalog(root)
            self.assertEqual(catalog.listings.record_id.iloc[0], "march:1")
            self.assertEqual(catalog.buildings.building_id.iloc[0], "b1")
            self.assertTrue(pd.api.types.is_numeric_dtype(catalog.listings.area))
            self.assertEqual(load_catalog(root / "manifest.json").manifest["schema_version"], 1)

    def test_both_csv_hashes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for table in ("listings", "buildings"):
                with self.subTest(table=table):
                    self.write_catalog(root)
                    with (root / f"{table}.csv").open("a", encoding="utf-8") as stream:
                        stream.write("\n")
                    with self.assertRaisesRegex(ValueError, "Контрольная сумма"):
                        load_catalog(root)

    def test_invalid_manifest_versions_paths_schema_and_counts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mutation in ("version", "bool_version", "path", "sha", "columns", "rows", "missing_table"):
                with self.subTest(mutation=mutation):
                    manifest = self.write_catalog(root)
                    spec = manifest["files"]["listings"]
                    if mutation == "version":
                        manifest["schema_version"] = 2
                    elif mutation == "bool_version":
                        manifest["schema_version"] = True
                    elif mutation == "path":
                        spec["path"] = "../listings.csv"
                    elif mutation == "sha":
                        spec["sha256"] = "invalid"
                    elif mutation == "columns":
                        spec["columns"] = ["wrong"]
                    elif mutation == "rows":
                        spec["rows"] = 2
                    else:
                        del manifest["files"]["buildings"]
                    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_catalog(root)

    def test_valid_hash_cannot_hide_missing_schema_or_broken_building_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mutation in ("missing_column", "missing_building", "duplicate_record"):
                with self.subTest(mutation=mutation):
                    manifest = self.write_catalog(root)
                    path = root / "listings.csv"
                    frame = pd.read_csv(path)
                    if mutation == "missing_column":
                        frame = frame.drop(columns=[next(iter(LISTING_COLUMNS))])
                    elif mutation == "missing_building":
                        frame["building_id"] = "nonexistent"
                    else:
                        frame = pd.concat([frame, frame], ignore_index=True)
                    frame.to_csv(path, index=False)
                    manifest["files"]["listings"] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_catalog(root)


if __name__ == "__main__":
    unittest.main()

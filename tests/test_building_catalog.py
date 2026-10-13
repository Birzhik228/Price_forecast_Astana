"""Проверки происхождения, границ нормализации и честной неопределённости."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_catalog import build_catalog
from building_catalog import build_property_catalog, canonical_house_key, normalize_address, normalize_complex_name


def example(period: str, source: int, **changes: object) -> dict:
    row = {"dataset_period": period, "original_source_row": source,
           "original_row_index": source - 1, "source_row": 10_000 + source,
           "record_id": f"{period}:{source}", "group": f"{period}:g{source}",
           "url": f"https://krisha.kz/a/show/{source}", "price": 40_000_000,
           "area": 65.0, "rooms": 2, "floor": 3, "total_floors": 9,
           "year_built": 2015, "building_type": "кирпичный", "condition": None,
           "district": "Есиль", "address": "Улица Пример 10/1", "residential_complex": "ЖК Пример",
           "collected_at": "2026-01-04T10:01:02" if period == "2026-01" else None}
    row.update(changes)
    return row


def raw_january(count: int = 2, **changes: object) -> pd.DataFrame:
    rows = [{"url": f"https://krisha.kz/a/show/{i}", "latitude": "51.10",
             "longitude": str(71.40 + (i - 1) * .0001), "parking": "",
             "residential_complex_name": "ЖК Пример"} for i in range(1, count + 1)]
    for row in rows:
        row.update(changes)
    return pd.DataFrame(rows, columns=["url", "latitude", "longitude", "parking", "residential_complex_name"])


def raw_march(count: int = 1) -> pd.DataFrame:
    return pd.DataFrame([{"url": f"https://krisha.kz/a/show/{i}", "Парковка": "",
                          "Жилой комплекс": "ЖК Пример"} for i in range(1, count + 1)],
                        columns=["url", "Парковка", "Жилой комплекс"])


class BuildingCatalogTests(unittest.TestCase):
    def build(self, rows: list[dict], jan: pd.DataFrame | None = None, mar: pd.DataFrame | None = None) -> dict:
        return build_property_catalog(pd.DataFrame(rows), jan if jan is not None else raw_january(),
                                      mar if mar is not None else raw_march())

    def test_provenance_uses_original_position_not_global_source_or_dataframe_index(self) -> None:
        jan = raw_january(3)
        jan.index = [44, 55, 66]
        jan.loc[66, "parking"] = "паркинг"
        result = self.build([example("2026-01", 3, source_row=1)], jan)
        self.assertEqual(result["listings"].iloc[0].parking_type, "паркинг")
        self.assertAlmostEqual(result["listings"].iloc[0].longitude, 71.4002)

    def test_reordered_raw_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "URL provenance mismatch"):
            self.build([example("2026-01", 1)], raw_january().iloc[::-1])

    def test_fractional_out_of_range_and_mismatched_ids_rejected(self) -> None:
        for changes in [{"original_source_row": 1.5}, {"original_source_row": 0},
                        {"original_source_row": 300}, {"record_id": "2026-03:1"},
                        {"dataset_period": "2026-04"}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.build([example("2026-01", 1, **changes)])

    def test_raw_tracking_query_matches_cleaned_url(self) -> None:
        jan = raw_january()
        jan.loc[0, "url"] += "?utm_source=archive"
        self.assertEqual(len(self.build([example("2026-01", 1)], jan)["listings"]), 1)

    def test_house_suffixes_and_streets_never_collapse(self) -> None:
        addresses = ["Улица Пример 10/1", "Улица Пример 101", "Улица Пример 10 корпус 1",
                     "Улица Пример 10 корпус 2", "Улица Пример 10а", "Другая улица 10/1"]
        rows = [example("2026-01", i, address=address) for i, address in enumerate(addresses, 1)]
        result = self.build(rows, raw_january(len(rows)))
        self.assertEqual(result["listings"].building_id.nunique(), len(rows))
        self.assertEqual(result["listings"].complex_id.nunique(), 1)

    def test_conservative_addresses_and_complex_only_never_create_house(self) -> None:
        for address in ["байтурсынулы 39,39/1,39/2,39/3", "мкр уркер, е-128 127",
                        "Сыганак — Кабанбай батыра", "Туран", "Улица 10 — срочно", "мкр уркер 1"]:
            with self.subTest(address=address):
                self.assertNotEqual(normalize_address(address)[1], "house")
        self.assertEqual(normalize_address("ЖК Пример 2", "ЖК Пример")[1], "ambiguous")
        result = self.build([example("2026-01", 1, address="Туран"),
                             example("2026-01", 2, district=None)])
        self.assertTrue(result["listings"].building_id.isna().all())
        self.assertTrue(result["buildings"].empty)

    def test_case_and_spacing_normalize_but_building_ids_ignore_price(self) -> None:
        first = example("2026-01", 1, address="  УЛИЦА   Пример 10/1  ")
        second = example("2026-01", 2, address="улица пример 10/1", price=999_000_000)
        result = self.build([first, second])
        self.assertEqual(result["listings"].building_id.nunique(), 1)
        self.assertEqual(result["buildings"].iloc[0].listing_count, 2)

    def test_unknown_amenities_preserved_and_no_majority_resolves_conflict(self) -> None:
        jan = raw_january(3)
        jan.loc[0, "parking"] = "паркинг"
        jan.loc[1, "parking"] = "гараж"
        result = self.build([example("2026-01", i) for i in range(1, 4)], jan)
        self.assertTrue(pd.isna(result["listings"].iloc[2].parking_type))
        self.assertEqual(result["listings"].iloc[2].parking_status, "unknown")
        self.assertTrue(pd.isna(result["buildings"].iloc[0].parking_type))
        self.assertEqual(result["buildings"].iloc[0].parking_status, "conflict")
        self.assertTrue(result["listings"][["developer", "elevator"]].isna().all().all())

    def test_conflicting_autofill_fields_and_complex_leave_individual_records(self) -> None:
        result = self.build([example("2026-01", 1), example("2026-01", 2, year_built=2016,
                            total_floors=10, building_type="панельный", residential_complex="Другой ЖК")])
        self.assertTrue(result["buildings"].empty)
        self.assertTrue(result["listings"].building_id.isna().all())
        self.assertEqual(result["listings"].candidate_building_id.nunique(), 1)
        self.assertTrue(result["listings"].building_match_status.eq("ambiguous_conflict").all())
        self.assertEqual(result["listings"].year_built.tolist(), [2015, 2016])
        self.assertEqual(result["summary"]["candidate_conflicts"]["year_built_conflict"], 1)
        self.assertEqual(len(result["conflict_audit"]), 2)

    def test_single_coordinate_not_usable_for_distances(self) -> None:
        result = self.build([example("2026-01", 1), example("2026-03", 1)])
        self.assertEqual(result["buildings"].iloc[0].coordinate_status, "unverified")
        self.assertEqual(result["listings"].coordinate_status.tolist(), ["unverified", "missing"])
        self.assertTrue(pd.isna(result["listings"].iloc[1].latitude))

    def test_consistent_coordinates_inherit_only_for_same_exact_house(self) -> None:
        result = self.build([example("2026-01", 1), example("2026-01", 2), example("2026-03", 1)])
        self.assertTrue(result["listings"].coordinate_status.eq("plausible").all())
        march = result["listings"].iloc[2]
        self.assertEqual(march.coordinate_source, "building_consensus")
        self.assertAlmostEqual(march.longitude, 71.40005)
        self.assertEqual(result["summary"]["coordinate_inherited_count"], 1)

    def test_outside_city_and_nonfinite_coordinates_invalid(self) -> None:
        for latitude, longitude in [(43.49, 77.03), (None, 71.4), ("inf", 71.4), ("oops", 71.4)]:
            with self.subTest(latitude=latitude, longitude=longitude):
                result = self.build([example("2026-01", 1)], raw_january(latitude=latitude, longitude=longitude))
                self.assertEqual(result["listings"].iloc[0].coordinate_status, "invalid")

    def test_internal_spread_blocks_all_distances_and_inheritance(self) -> None:
        jan = raw_january()
        jan.loc[1, "longitude"] = "71.45"
        result = self.build([example("2026-01", 1), example("2026-01", 2), example("2026-03", 1)], jan)
        self.assertEqual(result["listings"].coordinate_status.tolist(), ["unverified", "unverified", "missing"])
        self.assertIn("coordinate_spread", result["summary"]["candidate_conflicts"])
        self.assertTrue(result["listings"].building_id.isna().all())

    def test_invalid_outlier_not_silently_discarded_from_house_consensus(self) -> None:
        jan = raw_january(3)
        jan.loc[2, "latitude"] = "43.0"
        result = self.build([example("2026-01", i) for i in range(1, 4)], jan)
        self.assertTrue(result["buildings"].empty)
        self.assertTrue(result["listings"].building_id.isna().all())
        self.assertEqual(result["listings"].coordinate_status.tolist(), ["unverified", "unverified", "invalid"])

    def test_identical_marker_across_different_houses_not_house_precision(self) -> None:
        jan = raw_january(4, longitude="71.4")
        rows = [example("2026-01", i, address="Улица Пример " + ("1" if i <= 2 else "2")) for i in range(1, 5)]
        result = self.build(rows, jan)
        self.assertTrue(result["buildings"].coordinate_status.eq("unverified").all())
        self.assertEqual(result["summary"]["issue_counts"]["coordinate_reused_across_houses"], 2)

    def test_dates_collection_versus_declared_period(self) -> None:
        result = self.build([example("2026-01", 1), example("2026-03", 1)])
        jan, march = result["listings"].date_label.tolist()
        self.assertIn("Сбор: 2026-01-04", jan)
        self.assertIn("часовой пояс не указан", jan)
        self.assertIn("по заявлению источника", march)
        self.assertIn("дата объявления неизвестна", march)
        self.assertTrue(pd.isna(result["listings"].iloc[1].collected_at))

    def test_inputs_and_original_columns_unchanged_deterministic(self) -> None:
        combined = pd.DataFrame([example("2026-01", 1), example("2026-01", 2)])
        jan, mar = raw_january(), raw_march()
        snapshots = [item.copy(deep=True) for item in (combined, jan, mar)]
        first = build_property_catalog(combined, jan, mar)
        second = build_property_catalog(combined.iloc[::-1].reset_index(drop=True), jan, mar)
        for actual, original in zip((combined, jan, mar), snapshots):
            pd.testing.assert_frame_equal(actual, original)
        pd.testing.assert_frame_equal(first["listings"][combined.columns], combined)
        pd.testing.assert_frame_equal(first["buildings"], second["buildings"])
        self.assertEqual(first["summary"], second["summary"])

    def test_cli_function_writes_verified_hashes_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / name for name in ["combined.csv", "january.csv", "march.csv"]]
            for frame, path in zip([pd.DataFrame([example("2026-01", 1)]), raw_january(), raw_march()], paths):
                frame.to_csv(path, index=False)
            before = [path.read_bytes() for path in paths]
            manifest = build_catalog(*paths, root / "catalog", root / "report")
            self.assertEqual(manifest["schema_version"], 1)
            for spec in manifest["files"].values():
                self.assertEqual(spec["sha256"], hashlib.sha256((root / "catalog" / spec["path"]).read_bytes()).hexdigest())
            self.assertEqual([path.read_bytes() for path in paths], before)
            self.assertEqual(json.loads((root / "report/quality.json").read_text(encoding="utf-8"))["summary"], manifest["summary"])
            self.assertTrue((root / "report/issues.csv").exists())
            self.assertTrue((root / "report/normalization_audit.csv").exists())
            self.assertTrue((root / "report/conflict_audit.csv").exists())

    def test_suffix_aliases_apply_only_house_or_corpus_tokens_and_are_idempotent(self) -> None:
        for original, expected in [
            ("Туран 44a", "туран 44а"), ("Туран 44A", "туран 44а"),
            ("A-52 44a/2c", "a-52 44а/2с"), ("Туран 44 корпус 2a", "туран 44 корпус 2а"),
            ("Туран 44 блок A", "туран 44 блок а"),
            ("Aстана 44b", "aстана 44b"),
        ]:
            with self.subTest(original=original):
                self.assertEqual(normalize_address(original), (expected, "house"))
                self.assertEqual(normalize_address(expected), (expected, "house"))
        self.assertNotEqual(normalize_complex_name("Amanat"), normalize_complex_name("Аmanat"))
        self.assertNotEqual(canonical_house_key("A-52 44a", "Есиль"), canonical_house_key("А-52 44а", "Есиль"))

    def test_accepted_alias_merge_audit_includes_both_original_ids(self) -> None:
        rows = [example("2026-01", 1, address="Туран 44a"), example("2026-01", 2, address="Туран 44а")]
        result = self.build(rows)
        self.assertEqual(result["listings"].building_id.nunique(), 1)
        self.assertEqual(result["listings"].old_building_id.nunique(), 2)
        self.assertEqual(result["listings"].address.tolist(), ["Туран 44a", "Туран 44а"])
        audit = result["normalization_audit"]
        self.assertEqual(len(audit), 2)
        self.assertTrue(audit.decision.eq("accepted_merge").all())
        self.assertEqual(audit.suffix_changed.sum(), 1)
        self.assertEqual(result["summary"]["normalization"]["accepted_merge_candidates"], 1)

    def test_suffix_change_without_district_is_audited_without_merge(self) -> None:
        result = self.build([example("2026-01", 1, address="Туран 44a", district=None)])
        audit = result["normalization_audit"]
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit.iloc[0].decision, "rejected_missing_district")
        self.assertTrue(pd.isna(audit.iloc[0].new_building_id))
        self.assertTrue(pd.isna(result["listings"].iloc[0].house_key))
        self.assertEqual(result["summary"]["normalization"]["changed_address_rows"], 1)

    def test_rejected_alias_merge_conflicts_not_resolved_by_majority_or_missing_fields(self) -> None:
        for conflicting in [{"residential_complex": "Другой ЖК"}, {"year_built": 2020},
                            {"total_floors": 20}, {"building_type": "панельный"}]:
            with self.subTest(conflicting=conflicting):
                rows = [example("2026-01", 1, address="Туран 44a"),
                        example("2026-01", 2, address="Туран 44а", **conflicting),
                        example("2026-03", 1, address="Туран 44a", year_built=None)]
                result = self.build(rows)
                self.assertTrue(result["listings"].building_id.isna().all())
                self.assertEqual(len(result["listings"]), 3)
                self.assertTrue(result["normalization_audit"].decision.eq("rejected_ambiguous").all())
                self.assertEqual(result["summary"]["normalization"]["rejected_merge_candidates"], 1)

    def test_house_key_and_raw_coordinates_are_pointwise_before_global_decisions(self) -> None:
        base = example("2026-01", 1, address="Туран 44a")
        alone = self.build([base])["listings"].iloc[0]
        combined = self.build([base, example("2026-01", 2, address="Туран 44а", year_built=2020)])
        conflicted = combined["listings"].iloc[0]
        for field in ["house_key", "candidate_building_id", "raw_latitude", "raw_longitude", "raw_coordinate_status"]:
            self.assertEqual(alone[field], conflicted[field])
        self.assertTrue(pd.isna(conflicted.building_id))
        inherited = self.build([example("2026-01", 1), example("2026-01", 2), example("2026-03", 1)])["listings"].iloc[2]
        self.assertEqual(inherited.coordinate_status, "plausible")
        self.assertEqual(inherited.raw_coordinate_status, "missing")
        self.assertTrue(pd.isna(inherited.raw_latitude))
        self.assertTrue(pd.isna(inherited.raw_longitude))

    def test_conflicting_candidate_markers_still_block_other_house_centroid(self) -> None:
        jan = raw_january(4, longitude="71.4")
        rows = [example("2026-01", 1, address="Пример 1"), example("2026-01", 2, address="Пример 1"),
                example("2026-01", 3, address="Пример 2"), example("2026-01", 4, address="Пример 2", year_built=2020)]
        result = self.build(rows, jan)
        self.assertEqual(result["buildings"].iloc[0].coordinate_status, "unverified")
        self.assertTrue(result["listings"].iloc[2:].building_id.isna().all())

    def test_old_stage_archived_and_repeat_outputs_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, reports = root / "catalog", root / "reports"
            output.mkdir(); reports.mkdir()
            old_manifest = {"schema_version": 1, "summary": {"listing_count": 123, "building_count": 45}}
            (output / "manifest.json").write_text(json.dumps(old_manifest), encoding="utf-8")
            (output / "listings.csv").write_text("old original\n", encoding="utf-8")
            (reports / "index.html").write_text("old report", encoding="utf-8")
            paths = [root / name for name in ["combined.csv", "january.csv", "march.csv"]]
            for frame, path in zip([pd.DataFrame([example("2026-01", 1)]), raw_january(), raw_march()], paths):
                frame.to_csv(path, index=False)
            first = build_catalog(*paths, output, reports)
            archive = Path(first["previous_stage_archives"]["catalog"]["path"])
            self.assertEqual((archive / "listings.csv").read_text(encoding="utf-8"), "old original\n")
            outputs = {name: (output / name).read_bytes() for name in ["listings.csv", "buildings.csv"]}
            audits = {name: (reports / name).read_bytes() for name in ["normalization_audit.csv", "conflict_audit.csv"]}
            second = build_catalog(*paths, output, reports)
            self.assertEqual(first["previous_stage_archives"], second["previous_stage_archives"])
            for name, before in outputs.items():
                self.assertEqual((output / name).read_bytes(), before)
            for name, before in audits.items():
                self.assertEqual((reports / name).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

"""Synthetic invariants for the reviewed archive; no model or real data access."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import repair_archived_data as repair


def fixture(count=3):
    records, sources = [], []
    for number in range(1, count + 1):
        url = f"https://krisha.kz/a/show/{1000 + number}"
        records.append({
            "source_row": number + 100, "price": 30_000_000 + number,
            "area": 60.0, "rooms": 2, "floor": 4, "total_floors": 9,
            "year_built": 2020, "building_type": "кирпичный", "condition": pd.NA,
            "district": pd.NA, "residential_complex": pd.NA,
            "address": "нура р-н", "url": url,
            "collected_at": "2026-01-04T10:30:00", "dataset_period": "2026-01",
            "original_row_index": number - 1, "original_source_row": number,
            "record_id": f"2026-01:{number}", "group": f"group_{number}",
        })
        sources.append({"url": url + "?source=archive", "address": "Нура р-н",
                        "district": "", "residential_complex_name": ""})
    combined = pd.DataFrame(records)
    january = pd.DataFrame(sources)
    split = pd.DataFrame({"record_id": combined.record_id,
                          "subset": ["test" if i == count - 1 else "train" for i in range(count)]})
    return combined, january, split


class DistrictEvidenceTests(unittest.TestCase):
    def test_requires_whole_label_and_explicit_marker(self):
        accepted = {"Нура р-н": "Нура", "  район Байконыр  ": "Байконур",
                    "Сарайшык район": "Сарайшык", "р-н Есильский": "Есиль",
                    "АЛМАТЫ Р-Н": "Алматы", "Сарыаркинский район": "Сарыарка"}
        for value, expected in accepted.items():
            with self.subTest(value=value):
                self.assertEqual(repair.explicit_district_address(value), expected)
        for value in [None, pd.NA, "", "Сарайшык", "Сарайшык 7", "район ТЦ Евразия",
                      "Нура р-н, Туран 5", "возле района Нура", "Астана, Нура р-н",
                      "Новый район", "Нура/Есиль р-н", "Нура р-н — центр"]:
            with self.subTest(value=value):
                self.assertIsNone(repair.explicit_district_address(value))

    def test_recovers_only_missing_values_and_keeps_every_other_column(self):
        combined, january, split = fixture()
        january.loc[1, "address"] = "Сарайшык р-н"
        combined.loc[1, "address"] = "сарайшык р-н"
        combined.loc[2, "district"] = "Алматы"
        before, raw_before, split_before = [x.copy(deep=True) for x in (combined, january, split)]
        revised, changes, queue = repair.correct_source_fields(combined, january, split)
        self.assertEqual(revised.district.tolist(), ["Нура", "Сарайшык", "Алматы"])
        self.assertEqual(changes.record_id.tolist(), ["2026-01:1", "2026-01:2"])
        self.assertEqual(changes.field.tolist(), ["district", "district"])
        self.assertEqual(changes.columns.tolist(), repair.CHANGE_COLUMNS)
        self.assertEqual(changes.source_field.tolist(), ["january_raw.address"] * 2)
        self.assertEqual(changes.source_value.tolist(), ["Нура р-н", "Сарайшык р-н"])
        self.assertEqual(queue.record_id.tolist(), ["2026-01:3"])
        pd.testing.assert_frame_equal(before.drop(columns=["district", "residential_complex"]),
                                      revised.drop(columns=["district", "residential_complex"]))
        for actual, expected in [(combined, before), (january, raw_before), (split, split_before)]:
            pd.testing.assert_frame_equal(actual, expected)

    def test_street_name_unknown_district_and_existing_value_not_corrected(self):
        combined, january, split = fixture()
        january.loc[0, "address"] = "Сарайшык"
        january.loc[1, "address"] = "Новый р-н"
        combined.loc[2, "district"] = "неизвестный"
        revised, changes, queue = repair.correct_source_fields(combined, january, split)
        self.assertEqual(len(changes), 0)
        self.assertTrue(pd.isna(revised.loc[0, "district"]))
        self.assertTrue(pd.isna(revised.loc[1, "district"]))
        self.assertEqual(revised.loc[2, "district"], "неизвестный")
        self.assertEqual(queue.record_id.tolist(), ["2026-01:3"])

    def test_conflicting_raw_district_fails_without_selecting_value(self):
        combined, january, split = fixture(1)
        january.loc[0, "district"] = "Алматы р-н"
        with self.assertRaises(ValueError):
            repair.correct_source_fields(combined, january, split)
        self.assertTrue(pd.isna(combined.loc[0, "district"]))

    def test_missing_district_with_agreeing_raw_district_is_recoverable(self):
        combined, january, split = fixture(1)
        january.loc[0, "district"] = "Нура р-н"
        revised, changes, _ = repair.correct_source_fields(combined, january, split)
        self.assertEqual(revised.loc[0, "district"], "Нура")
        self.assertEqual(len(changes), 1)

    def test_repeated_correction_is_idempotent(self):
        combined, january, split = fixture(1)
        revised, _, _ = repair.correct_source_fields(combined, january, split)
        again, changes, queue = repair.correct_source_fields(revised, january, split)
        pd.testing.assert_frame_equal(revised, again)
        self.assertTrue(changes.empty)
        self.assertTrue(queue.empty)

    def test_march_row_is_never_bound_to_january_source(self):
        combined, january, split = fixture(1)
        combined.loc[0, ["record_id", "dataset_period"]] = ["2026-03:1", "2026-03"]
        split.loc[0, "record_id"] = "2026-03:1"
        revised, changes, _ = repair.correct_source_fields(combined, january, split)
        pd.testing.assert_frame_equal(combined, revised)
        self.assertTrue(changes.empty)


class ComplexEvidenceTests(unittest.TestCase):
    def test_missing_complex_uses_only_own_full_label_and_keeps_address(self):
        combined, january, split = fixture(1)
        january.loc[0, "address"] = "Жк София"
        combined.loc[0, "address"] = "жк софия"
        before = combined.copy(deep=True)
        revised, changes, queue = repair.correct_source_fields(combined, january, split)
        self.assertEqual(revised.loc[0, "residential_complex"], "софия")
        self.assertTrue(pd.isna(revised.loc[0, "district"]))
        self.assertEqual(changes.field.tolist(), ["residential_complex"])
        self.assertEqual(changes.new_value.tolist(), ["софия"])
        self.assertEqual(changes.source_value.tolist(), ["Жк София"])
        self.assertEqual(changes.subset.tolist(), ["test"])
        pd.testing.assert_frame_equal(before.drop(columns="residential_complex"),
                                      revised.drop(columns="residential_complex"))
        self.assertTrue(queue.empty)

    def test_nearby_names_placeholder_and_incomplete_labels_are_not_filled(self):
        labels = ["возле ЖК София", "напротив ЖК София", "рядом с ЖК София",
                  "ЖК неизвестно", "ЖК нет", "ЖК не указан", "ЖК -", "ЖК",
                  "ЖК София 2", "ЖК София — рядом школа", "ЖК София, Туран 7"]
        for label in labels:
            with self.subTest(label=label):
                combined, january, split = fixture(1)
                january.loc[0, "address"] = label
                revised, changes, _ = repair.correct_source_fields(combined, january, split)
                self.assertTrue(pd.isna(revised.loc[0, "residential_complex"]))
                self.assertTrue(changes.empty)

    def test_nonmissing_complex_never_overwritten(self):
        combined, january, split = fixture(1)
        january.loc[0, "address"] = "ЖК София"
        combined.loc[0, "residential_complex"] = "другой жк"
        revised, changes, _ = repair.correct_source_fields(combined, january, split)
        self.assertEqual(revised.loc[0, "residential_complex"], "другой жк")
        self.assertTrue(changes.empty)


class ProvenanceGuardTests(unittest.TestCase):
    def test_url_mismatch_is_rejected_before_any_correction(self):
        combined, january, split = fixture(1)
        january.loc[0, "url"] = "https://krisha.kz/a/show/999999"
        with self.assertRaises(ValueError):
            repair.correct_source_fields(combined, january, split)

    def test_source_record_identity_mismatch_is_rejected(self):
        combined, january, split = fixture(2)
        combined.loc[0, "original_source_row"] = 2
        with self.assertRaises(ValueError):
            repair.correct_source_fields(combined, january, split)

    def test_invalid_source_positions_are_rejected(self):
        for position in [0, -1, 2, 0.5, np.nan, np.inf]:
            with self.subTest(position=position):
                combined, january, split = fixture(1)
                combined["original_source_row"] = float(position)
                with self.assertRaises(ValueError):
                    repair.correct_source_fields(combined, january, split)

    def test_duplicate_or_incomplete_split_identity_fails(self):
        combined, january, split = fixture(2)
        for bad in [pd.concat([split, split.iloc[[0]]], ignore_index=True), split.iloc[:1],
                    split.assign(record_id=["2026-01:1", "2026-01:99"])]:
            with self.subTest(ids=bad.record_id.tolist()):
                with self.assertRaises(ValueError):
                    repair.correct_source_fields(combined, january, bad)

    def test_unknown_or_missing_subset_fails(self):
        for value in ["validation", "Train", "", pd.NA]:
            with self.subTest(value=value):
                combined, january, split = fixture(1)
                split.loc[0, "subset"] = value
                with self.assertRaises(ValueError):
                    repair.correct_source_fields(combined, january, split)

    def test_duplicate_combined_identity_fails(self):
        combined, january, split = fixture(2)
        combined.loc[1, "record_id"] = combined.loc[0, "record_id"]
        with self.assertRaises(ValueError):
            repair.correct_source_fields(combined, january, split)


class DuplicateReviewTests(unittest.TestCase):
    def test_equal_features_different_urls_remain_records_and_ignore_price(self):
        combined, _, _ = fixture(3)
        combined["district"] = "Алматы"
        combined["address"] = ["Абылай-хана 20A", "Абылай-хана 20а", "Абылай-хана 20а"]
        combined.loc[1, "price"] = 99_000_000
        combined.loc[2, "floor"] = np.nan
        before = combined.copy(deep=True)
        queue = repair.possible_duplicates(combined)
        self.assertEqual(queue.record_id.tolist(), ["2026-01:1", "2026-01:2"])
        self.assertEqual(queue.review_group.nunique(), 1)
        self.assertTrue(queue.decision.eq("retain_distinct_urls_identity_unconfirmed").all())
        pd.testing.assert_frame_equal(combined, before)

    def test_different_house_previous_deletion_is_review_only(self):
        combined, _, _ = fixture(2)
        combined["district"] = "Нура"
        combined["address"] = ["Е-36 4", "Е-36 5"]
        replay = combined.set_index("record_id")
        audit = pd.DataFrame([{"record_id": "2026-01:1", "kept_record_id": "2026-01:2",
                               "kept": False, "group": "group_a", "original_subset": "train",
                               "match_type": "signature"}])
        before = replay.copy(deep=True)
        queue = repair.removed_address_conflicts(replay, audit)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.loc[0, "removed_record_id"], "2026-01:1")
        self.assertNotEqual(queue.loc[0, "removed_house_key"], queue.loc[0, "kept_house_key"])
        self.assertIn("do_not_restore", queue.loc[0, "decision"])
        pd.testing.assert_frame_equal(replay, before)

    def test_house_suffix_alias_or_cross_district_is_not_address_conflict(self):
        for addresses, districts in [(["Туран 44a", "Туран 44а"], ["Нура", "Нура"]),
                                     (["Туран 4", "Туран 5"], ["Нура", "Есиль"]),
                                     (["Туран", "Туран 5"], ["Нура", "Нура"])]:
            with self.subTest(addresses=addresses, districts=districts):
                combined, _, _ = fixture(2)
                combined["address"], combined["district"] = addresses, districts
                audit = pd.DataFrame([{"record_id": "2026-01:1", "kept_record_id": "2026-01:2",
                                       "kept": False, "group": "g", "original_subset": "train",
                                       "match_type": "signature"}])
                self.assertTrue(repair.removed_address_conflicts(combined.set_index("record_id"), audit).empty)

    def test_same_listing_url_update_is_not_flagged_as_different_house_merge(self):
        combined, _, _ = fixture(2)
        combined["district"] = "Нура"
        combined["address"] = ["Е-36 4", "Е-36 5"]
        combined["url"] = ["https://krisha.kz/a/show/1001?old=1", "https://krisha.kz/a/show/1001"]
        audit = pd.DataFrame([{"record_id": "2026-01:1", "kept_record_id": "2026-01:2",
                               "kept": False, "group": "g", "original_subset": "train",
                               "match_type": "url"}])
        self.assertTrue(repair.removed_address_conflicts(combined.set_index("record_id"), audit).empty)


def street_fixture():
    listings, _, _ = fixture(3)
    bad, good = "astana|нура|е 915", "astana|нура|туран 5"
    listings["address"] = ["е 915", "е 915", "туран 5"]
    listings["raw_address"] = ["Е 915", "Е 915", "Туран 5"]
    listings["house_key"] = [bad, bad, good]
    listings["building_id"] = ["bad_house", "bad_house", "good_house"]
    listings["candidate_building_id"] = listings.building_id
    listings["address_status"] = "house"
    listings["building_match_status"] = "resolved"
    listings["building_conflict_reasons"] = "year_built_conflict"
    listings["coordinate_spread_m"] = 10.0
    listings["latitude"] = 51.15
    listings["longitude"] = 71.45
    listings["coordinate_status"] = "plausible"
    listings["coordinate_source"] = "house_consensus"
    listings["raw_latitude"] = [51.05, np.nan, 51.15]
    listings["raw_longitude"] = [71.42, np.nan, 71.45]
    listings["raw_coordinate_status"] = ["unverified", "missing", "unverified"]
    listings["raw_coordinate_source"] = ["january_raw", "january_raw", "january_raw"]
    buildings = pd.DataFrame({"house_key": [bad, good], "building_id": ["bad_house", "good_house"],
                              "listing_count": [2, 1], "address_normalized": ["е 915", "туран 5"]})
    evidence = pd.DataFrame({"record_id": listings.record_id.iloc[:2], "old_house_key": [bad, bad]})
    return listings, buildings, evidence


class StreetOnlyWithdrawalTests(unittest.TestCase):
    def test_withdraws_only_derived_identity_and_preserves_source_values(self):
        listings, buildings, evidence = street_fixture()
        before = [frame.copy(deep=True) for frame in (listings, buildings, evidence)]
        revised, houses = repair.withdraw_street_only_house_keys(listings, buildings, evidence)
        self.assertEqual(revised.record_id.tolist(), listings.record_id.tolist())
        self.assertTrue(revised.loc[:1, ["house_key", "building_id", "candidate_building_id"]].isna().all().all())
        self.assertTrue(revised.loc[:1, "address_status"].eq("street_only_osm_evidence").all())
        self.assertTrue(revised.loc[:1, "building_match_status"].eq("address_unresolved").all())
        self.assertTrue(revised.loc[:1, "building_conflict_reasons"].isna().all())
        self.assertTrue(revised.loc[:1, "coordinate_spread_m"].isna().all())
        for derived, raw in [("latitude", "raw_latitude"), ("longitude", "raw_longitude"),
                             ("coordinate_status", "raw_coordinate_status"), ("coordinate_source", "raw_coordinate_source")]:
            pd.testing.assert_series_equal(revised.loc[:1, derived], listings.loc[:1, raw], check_names=False)
        self.assertTrue(pd.isna(revised.loc[1, "latitude"]))
        changed = ["house_key", "building_id", "candidate_building_id", "address_status", "building_match_status",
                   "building_conflict_reasons", "coordinate_spread_m", "latitude", "longitude", "coordinate_status", "coordinate_source"]
        pd.testing.assert_frame_equal(revised.drop(columns=changed), listings.drop(columns=changed))
        pd.testing.assert_series_equal(revised.loc[2], listings.loc[2])
        pd.testing.assert_frame_equal(houses, buildings.iloc[[1]])
        for actual, expected in zip((listings, buildings, evidence), before):
            pd.testing.assert_frame_equal(actual, expected)

    def test_mismatched_duplicate_or_unknown_evidence_is_rejected(self):
        listings, buildings, evidence = street_fixture()
        bad_key = evidence.copy()
        bad_key.loc[0, "old_house_key"] = "astana|нура|другой 1"
        unknown = evidence.copy()
        unknown.loc[0, "record_id"] = "2026-01:999"
        duplicates = pd.concat([evidence, evidence.iloc[[0]]], ignore_index=True)
        for bad in [bad_key, unknown, duplicates]:
            with self.subTest(evidence=bad.to_dict("records")), self.assertRaises(ValueError):
                repair.withdraw_street_only_house_keys(listings, buildings, bad)

    def test_mixed_key_cannot_remove_an_unreviewed_listing_identity(self):
        listings, buildings, evidence = street_fixture()
        with self.assertRaises(ValueError):
            repair.withdraw_street_only_house_keys(listings, buildings, evidence.iloc[[0]])
        self.assertTrue(listings.house_key.notna().all())
        self.assertEqual(len(buildings), 2)

    def test_empty_evidence_is_a_no_op(self):
        listings, buildings, evidence = street_fixture()
        revised, houses = repair.withdraw_street_only_house_keys(listings, buildings, evidence.iloc[:0])
        pd.testing.assert_frame_equal(revised, listings)
        pd.testing.assert_frame_equal(houses, buildings)


def top_error_fixture():
    errors = pd.DataFrame([{
        "record_id": "2026-03:1", "dataset_period": "2026-03", "original_source_row": 1,
        "url": "https://krisha.kz/a/show/7001", "actual_price": 100_000_000,
        "predicted_price": 60_000_000, "absolute_error": 40_000_000, "signed_error": -40_000_000,
        "ape_pct": 40.0, "district": "Нура", "area": 130.0, "residential_complex": "софия",
        "rooms": 3, "floor": 4, "total_floors": 9, "year_built": 2020,
        "building_type": "кирпичный", "condition": pd.NA, "address": "туран 5",
        "filled_base_fields": 7, "coordinate_available": False, "building_conflict_reasons": pd.NA,
    }])
    march = pd.DataFrame([{"url": "https://krisha.kz/a/show/7001?source=test", "title": "Исходный заголовок",
                            "price": "100000000〒", "Площадь": "130 м²", "Этаж": "4 из 9",
                            "Год постройки": "2020", "Город": "Астана, Нура р-н", "Жилой комплекс": "София"}])
    catalog = pd.DataFrame({"record_id": ["2026-01:1", "2026-01:2", "2026-03:1"],
                            "residential_complex": ["софия", "софия", "софия"]})
    catalog["price"] = errors.loc[0, "actual_price"]
    for field in ["area", "rooms", "floor", "total_floors", "year_built", "district", "building_type", "condition", "address"]:
        catalog[field] = errors.loc[0, field]
    split = pd.DataFrame({"record_id": catalog.record_id, "subset": ["train", "train", "test"]})
    return errors, march, catalog, split


class TopErrorReviewTests(unittest.TestCase):
    def test_links_original_fields_and_counts_complex_in_train_only(self):
        errors, march, catalog, split = top_error_fixture()
        before = [frame.copy(deep=True) for frame in (errors, march, catalog, split)]
        review = repair.review_top_errors(errors, march, catalog, split)
        self.assertEqual(review.loc[0, "train_complex_rows"], 2)
        self.assertEqual(review.loc[0, "raw_title"], "Исходный заголовок")
        self.assertEqual(review.loc[0, "raw_price"], "100000000〒")
        self.assertEqual(review.loc[0, "actual_price"], 100_000_000)
        self.assertEqual(set(review.loc[0, "risk_flags"].split(";")),
                         {"area_over_120m2", "incomplete_base_fields", "no_usable_geography", "complex_less_than_30_training_rows"})
        self.assertIn("no_confirmed_value_replacement", review.loc[0, "decision"])
        for actual, expected in zip((errors, march, catalog, split), before):
            pd.testing.assert_frame_equal(actual, expected)

    def test_rejects_source_url_mismatch(self):
        errors, march, catalog, split = top_error_fixture()
        march.loc[0, "url"] = "https://krisha.kz/a/show/7002"
        with self.assertRaises(ValueError):
            repair.review_top_errors(errors, march, catalog, split)

    def test_rejects_other_period_in_march_source_review(self):
        errors, march, catalog, split = top_error_fixture()
        errors.loc[0, "dataset_period"] = "2026-01"
        with self.assertRaises(ValueError):
            repair.review_top_errors(errors, march, catalog, split)

    def test_rejects_forged_record_id_even_when_url_matches(self):
        errors, march, catalog, split = top_error_fixture()
        errors.loc[0, "record_id"] = "2026-03:999"
        with self.assertRaisesRegex(ValueError, "record_id/source row"):
            repair.review_top_errors(errors, march, catalog, split)

    def test_rejects_invalid_raw_position_including_zero(self):
        for position in [0, -1, 2, 0.5, np.inf, np.nan]:
            with self.subTest(position=position):
                errors, march, catalog, split = top_error_fixture()
                errors["original_source_row"] = float(position)
                with self.assertRaisesRegex(ValueError, "record_id/source row"):
                    repair.review_top_errors(errors, march, catalog, split)

    def test_rejects_forged_price_or_feature_against_catalog(self):
        mutations = {"actual_price": 99_000_000, "area": 129.0, "rooms": 2, "floor": 3,
                     "total_floors": 10, "year_built": 2021, "district": "Есиль",
                     "building_type": "монолитный", "condition": "свежий ремонт",
                     "residential_complex": "другой", "address": "туран 6"}
        for field, value in mutations.items():
            with self.subTest(field=field):
                errors, march, catalog, split = top_error_fixture()
                errors.loc[0, field] = value
                with self.assertRaisesRegex(ValueError, f"source mismatch: {field}"):
                    repair.review_top_errors(errors, march, catalog, split)

    def test_rejects_missing_or_duplicate_catalog_identity(self):
        errors, march, catalog, split = top_error_fixture()
        for invalid in [catalog.iloc[:2], pd.concat([catalog, catalog.iloc[[2]]], ignore_index=True)]:
            with self.subTest(rows=len(invalid)), self.assertRaisesRegex(ValueError, "exactly one catalog"):
                repair.review_top_errors(errors, march, invalid, split)


class OutputProtectionTests(unittest.TestCase):
    def test_protected_input_and_parent_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            protected = ["data/raw/new", "models/new", "data/processed/property_catalog/new",
                         "data/processed/combined_2026/new", "data", "data/processed", ""]
            for name in protected:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    repair.require_new_outputs(root, [root / name, root / "reports/new_review"])

    def test_existing_output_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            data = root / "data/processed/reviewed"
            data.mkdir(parents=True)
            sentinel = data / "listings.csv"
            sentinel.write_bytes(b"preserve-this")
            with self.assertRaises(ValueError):
                repair.require_new_outputs(root, [data, root / "reports/new_review"])
            self.assertEqual(sentinel.read_bytes(), b"preserve-this")

    def test_outside_or_overlapping_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            data = root / "data/processed/reviewed"
            for outputs in [[root.parent / "outside", root / "reports/reviewed"],
                            [data, data], [data, data / "reports"], [data / "nested", data]]:
                with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                    repair.require_new_outputs(root, outputs)

    def test_separate_empty_destinations_are_allowed_without_creating_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            data, reports = root / "data/processed/reviewed", root / "reports/reviewed"
            data.mkdir(parents=True)
            repair.require_new_outputs(root, [data, reports])
            self.assertEqual(list(data.iterdir()), [])
            self.assertFalse(reports.exists())


if __name__ == "__main__":
    unittest.main()

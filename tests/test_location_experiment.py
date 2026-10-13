"""Контракт разбиения эксперимента; обучение и рабочие артефакты не затрагиваются."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from building_catalog import canonical_house_key
from features import FEATURE_COLUMNS, build_pipeline
from location_experiment import apartment_groups, house_validation_groups, metrics, prepare_partition
from location_model import prepare_location_input
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit


def listing(number: int, **changes) -> dict:
    row = {
        "source_row": number + 1, "original_source_row": number + 101,
        "original_row_index": number, "dataset_period": "2026-03",
        "record_id": f"2026-03:{number + 101}", "group": f"original_{number}",
        "price": 30_000_000, "area": 60.0, "rooms": 2, "floor": 3,
        "total_floors": 9, "year_built": 2015, "district": "Есиль",
        "building_type": "кирпичный", "condition": "свежий ремонт",
        "address": f"Улица {number + 1}", "residential_complex": None,
        "url": f"https://krisha.kz/a/show/{number + 1000}", "collected_at": None,
    }
    row.update(changes)
    return row


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    result["house_key"] = [canonical_house_key(row.address, row.district, row.residential_complex)
                           for row in frame.itertuples()]
    result["raw_latitude"] = np.nan
    result["raw_longitude"] = np.nan
    result["raw_coordinate_status"] = "missing"
    return result


def manifest(frame: pd.DataFrame, train_count: int = 3) -> pd.DataFrame:
    # Exact legacy CSV schema: record_id is not present in the frozen manifest.
    return pd.DataFrame({"row_index": frame.index,
                         "subset": ["train"] * train_count + ["test"] * (len(frame) - train_count),
                         "cv_validation_fold": [1] * train_count + [0] * (len(frame) - train_count),
                         "group": frame.group, "source_row": frame.source_row})


class ApartmentEquivalenceTests(unittest.TestCase):
    def test_house_suffix_alias_links_different_prices_without_using_targets(self) -> None:
        frame = pd.DataFrame([listing(0, address="Туран 44a", price=20_000_000),
                              listing(1, address="Туран 44а", price=70_000_000), listing(2)])
        enriched = enrich(frame)
        groups = apartment_groups(enriched)
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])
        changed = enriched.copy()
        changed["price"] = [700_000_000, 1, 900_000_000]
        changed["condition"] = [None, "черновая отделка", "требует ремонта"]
        pd.testing.assert_series_equal(groups, apartment_groups(changed))

    def test_incomplete_signatures_do_not_merge_distinct_old_groups(self) -> None:
        for field in ("area", "rooms", "floor", "total_floors", "year_built", "house_key"):
            with self.subTest(field=field):
                frame = enrich(pd.DataFrame([listing(0, address="Туран 44а"), listing(1, address="Туран 44а")]))
                frame[field] = None
                self.assertEqual(apartment_groups(frame).nunique(), 2)

    def test_original_group_and_signature_links_are_transitive(self) -> None:
        frame = enrich(pd.DataFrame([
            listing(0, address="Туран 44a"),
            listing(1, address="Туран 44а", group="bridge"),
            listing(2, address=None, floor=None, group="bridge"),
        ]))
        self.assertEqual(apartment_groups(frame).nunique(), 1)

    def test_identity_is_stable_under_input_row_reordering(self) -> None:
        frame = enrich(pd.DataFrame([listing(0, address="Туран 44a"),
                                     listing(1, address="Туран 44а"), listing(2)]))
        before = dict(zip(frame.record_id, apartment_groups(frame)))
        reordered = frame.iloc[[2, 1, 0]].copy()
        self.assertEqual(before, dict(zip(reordered.record_id, apartment_groups(reordered))))


class ExperimentPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame([listing(number) for number in range(8)])

    def test_legacy_manifest_and_reordered_catalog_preserve_every_source_value(self) -> None:
        enriched, split = enrich(self.frame), manifest(self.frame)
        original = [part.copy(deep=True) for part in (self.frame, enriched, split)]
        result = prepare_partition(self.frame, enriched.iloc[::-1], split.iloc[::-1])
        pd.testing.assert_frame_equal(result["frame"][self.frame.columns], self.frame)
        np.testing.assert_array_equal(result["train_ids"], [0, 1, 2])
        np.testing.assert_array_equal(result["test_ids"], [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(result["evaluation_ids"], result["test_ids"])
        self.assertFalse(result["exclusions"])
        for part, unchanged in zip((self.frame, enriched, split), original):
            pd.testing.assert_frame_equal(part, unchanged)

    def test_changed_source_columns_are_rejected(self) -> None:
        changes = {"price": 31_000_000, "address": "Другой адрес 99", "source_row": 999,
                   "dataset_period": "2026-01", "year_built": 2020}
        for column, value in changes.items():
            with self.subTest(column=column):
                enriched = enrich(self.frame)
                enriched.loc[0, column] = value
                with self.assertRaises(ValueError):
                    prepare_partition(self.frame, enriched, manifest(self.frame))

    def test_incomplete_duplicate_or_misaligned_split_is_rejected(self) -> None:
        good = manifest(self.frame)
        bad_cases = [good.iloc[:-1], pd.concat([good, good.iloc[[0]]], ignore_index=True)]
        for column, value in (("source_row", 999), ("group", "wrong"), ("subset", "invalid")):
            changed = good.copy()
            changed.loc[0, column] = value
            bad_cases.append(changed)
        changed = good.copy()
        changed["record_id"] = self.frame.record_id
        changed.loc[0, "record_id"] = self.frame.loc[1, "record_id"]
        bad_cases.append(changed)
        for split in bad_cases:
            with self.subTest(split=split.to_dict("list")):
                with self.assertRaises(ValueError):
                    prepare_partition(self.frame, enrich(self.frame), split)

    def test_missing_or_duplicate_catalog_records_are_rejected(self) -> None:
        enriched = enrich(self.frame)
        for invalid in (enriched.iloc[:-1], pd.concat([enriched, enriched.iloc[[0]]], ignore_index=True)):
            with self.subTest(rows=len(invalid)):
                with self.assertRaises(ValueError):
                    prepare_partition(self.frame, invalid, manifest(self.frame))

    def test_cross_train_alias_and_test_duplicate_exclusions_are_price_independent(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "address"] = "Туран 44a"
        frame.loc[3, "address"] = "Туран 44а"
        frame.loc[[4, 5], "address"] = "Кунаева 500"
        frame["price"] = np.arange(1, 9) * 10_000_000
        result = prepare_partition(frame, enrich(frame), manifest(frame))
        np.testing.assert_array_equal(result["train_ids"], [0, 1, 2])
        np.testing.assert_array_equal(result["test_ids"], [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(result["evaluation_ids"], [4, 6, 7])
        self.assertEqual({item["row_index"]: item["reason"] for item in result["exclusions"]},
                         {3: "new_alias_matches_training", 5: "new_alias_duplicate_in_test"})
        changed = frame.copy()
        changed["price"] = changed.price.to_numpy()[::-1]
        again = prepare_partition(changed, enrich(changed).iloc[::-1], manifest(changed).iloc[::-1])
        np.testing.assert_array_equal(result["evaluation_ids"], again["evaluation_ids"])
        self.assertEqual(result["exclusions"], again["exclusions"])
        train_groups = set(result["groups"].loc[result["train_ids"]])
        self.assertTrue(train_groups.isdisjoint(result["groups"].loc[result["evaluation_ids"]]))


class ExperimentMetricsTests(unittest.TestCase):
    def test_metrics_keep_tenge_scale_and_strict_twenty_percent_threshold(self) -> None:
        result = metrics([10_000_000, 20_000_000], [12_000_000, 25_000_000])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mae_tenge"], 3_500_000)
        self.assertAlmostEqual(result["rmse_tenge"], np.sqrt(14.5) * 1_000_000)
        self.assertAlmostEqual(result["r2"], .42)
        self.assertEqual(result["over_20_pct"], 50)

    def test_invalid_shapes_and_nonfinite_values_are_rejected(self) -> None:
        for actual, predicted in (([1, 2], [1]), ([[1, 2]], [[1, 2]]),
                                  ([1, np.nan], [1, 2]), ([1, 2], [1, np.inf])):
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    metrics(actual, predicted)


class HouseValidationGroupTests(unittest.TestCase):
    def test_house_split_keeps_equivalent_apartments_with_unknown_addresses_together(self) -> None:
        records = [listing(number) for number in range(30)]
        records[0].update(address="Туран 44a", group="bridge")
        records[1].update(address=None, group="bridge")
        records[2].update(address="Туран 44а", area=80)
        frame = enrich(pd.DataFrame(records))
        apartments = apartment_groups(frame)
        houses = house_validation_groups(frame, apartments)
        self.assertEqual(houses.iloc[0], houses.iloc[1])
        self.assertEqual(houses.iloc[0], houses.iloc[2])
        fit, valid = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(frame, groups=houses))
        self.assertTrue(set(apartments.iloc[fit]).isdisjoint(apartments.iloc[valid]))
        fit_houses, valid_houses = frame.iloc[fit].house_key.dropna(), frame.iloc[valid].house_key.dropna()
        self.assertTrue(set(fit_houses).isdisjoint(valid_houses))

    def test_unrelated_missing_houses_stay_separate_and_index_alignment_is_required(self) -> None:
        frame = enrich(pd.DataFrame([listing(0, address=None), listing(1, address=None)]))
        apartments = apartment_groups(frame)
        self.assertEqual(house_validation_groups(frame, apartments).nunique(), 2)
        with self.assertRaises(ValueError):
            house_validation_groups(frame, apartments.iloc[::-1])


class FrozenModelInputTests(unittest.TestCase):
    def test_missing_string_district_is_compatible_with_legacy_random_forest(self) -> None:
        training = enrich(pd.DataFrame([
            listing(number, area=40.0 + number * 5, price=20_000_000 + number * 2_000_000,
                    district="Есиль" if number % 2 else "Алматы")
            for number in range(8)
        ]))
        pipeline = build_pipeline(RandomForestRegressor(
            n_estimators=3, max_depth=2, random_state=42, n_jobs=1,
        ))
        pipeline.fit(prepare_location_input(training).loc[:, list(FEATURE_COLUMNS)], training.price)
        query = training.iloc[:2].copy(deep=True)
        query["district"] = pd.Series([pd.NA, "Есиль"], index=query.index, dtype="string")
        original = query.copy(deep=True)
        prepared = prepare_location_input(query).loc[:, list(FEATURE_COLUMNS)]
        self.assertEqual(prepared.district.dtype, object)
        self.assertIsNot(prepared.district.iloc[0], pd.NA)
        self.assertTrue(np.isnan(prepared.district.iloc[0]))
        self.assertEqual(prepared.district.iloc[1], "Есиль")
        expected_input = query.loc[:, list(FEATURE_COLUMNS)].copy()
        expected_input["district"] = pd.Series([np.nan, "Есиль"], index=query.index, dtype=object)
        predicted = pipeline.predict(prepared)
        self.assertTrue(np.isfinite(predicted).all())
        np.testing.assert_array_equal(predicted, pipeline.predict(expected_input))
        pd.testing.assert_frame_equal(query, original)


if __name__ == "__main__":
    unittest.main()

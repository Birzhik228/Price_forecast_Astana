"""Synthetic checks that retraining cannot silently change its holdout."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from retraining_data import validate_retraining_partition


def fixture():
    rows = []
    for i in range(6):
        period = "2026-01" if i < 3 else "2026-03"
        number = i + 1 if i < 3 else i - 2
        rows.append({"record_id": f"{period}:{number}", "source_row": 100 + i,
                     "original_source_row": number, "original_row_index": number - 1,
                     "dataset_period": period, "group": f"group_{i}",
                     "price": 30_000_000 + i, "area": 60.0, "rooms": 2,
                     "floor": 4, "total_floors": 9, "year_built": 2020,
                     "district": "Нура", "building_type": "кирпичный", "condition": pd.NA,
                     "residential_complex": "софия", "address": f"туран {i + 1}",
                     "url": f"https://krisha.kz/a/show/{1000 + i}", "collected_at": pd.NA})
    original = pd.DataFrame(rows)
    corrected = original.copy(deep=True)
    # A confirmed training-only correction is allowed.
    original.loc[0, "district"] = pd.NA
    catalog = corrected.copy(deep=True)
    catalog["house_key"] = [f"astana|нура|туран {i + 1}" for i in range(6)]
    catalog["raw_latitude"] = np.nan
    catalog["raw_longitude"] = np.nan
    catalog["raw_coordinate_status"] = "missing"
    combined_split = pd.DataFrame({"row_index": range(6), "source_row": corrected.source_row,
                                    "group": corrected.group, "subset": ["train"] * 3 + ["test"] * 3})
    location_split = combined_split.drop(columns="source_row").copy()
    location_split["record_id"] = corrected.record_id
    location_split["primary_evaluation"] = location_split.subset.eq("test")
    # Historical keys intentionally disagree. They must never replace corrected ones.
    location_split["house_key"] = "obsolete-key"
    errors = corrected.loc[3:, ["record_id", "price"]].rename(columns={"price": "actual_price"}).reset_index(drop=True)
    return original, corrected, catalog, combined_split, location_split, errors


def validate(frames, **kwargs):
    return validate_retraining_partition(*frames, expected_holdout_rows=3, expected_train_rows=3, **kwargs)


class FrozenPartitionTests(unittest.TestCase):
    def test_valid_correction_preserves_holdout_and_uses_corrected_keys(self):
        frames = fixture()
        before = [frame.copy(deep=True) for frame in frames]
        bundle = validate(frames)
        self.assertEqual(bundle["train_ids"].tolist(), [0, 1, 2])
        self.assertEqual(bundle["test_ids"].tolist(), [3, 4, 5])
        self.assertEqual(bundle["evaluation_ids"].tolist(), [3, 4, 5])
        self.assertEqual(bundle["exclusions"], [])
        self.assertEqual(bundle["verified_train_record_ids"], frames[1].record_id.iloc[:3].tolist())
        self.assertEqual(bundle["verified_evaluation_record_ids"], frames[5].record_id.tolist())
        self.assertEqual(bundle["frame"].house_key.tolist(), frames[2].house_key.tolist())
        self.assertFalse(set(bundle["groups"].iloc[:3]) & set(bundle["groups"].iloc[3:]))
        for actual, expected in zip(frames, before):
            pd.testing.assert_frame_equal(actual, expected)

    def test_withdrawn_house_key_is_not_restored_from_reference(self):
        frames = list(fixture())
        frames[2].loc[3, "house_key"] = pd.NA
        bundle = validate(frames)
        self.assertTrue(pd.isna(bundle["frame"].loc[3, "house_key"]))
        self.assertEqual(len(bundle["evaluation_ids"]), 3)

    def test_target_change_fails_in_train_and_test(self):
        for position in [0, 3]:
            with self.subTest(position=position):
                frames = list(fixture())
                frames[1].loc[position, "price"] += 1
                frames[2].loc[position, "price"] += 1
                with self.assertRaisesRegex(ValueError, "targets changed"):
                    validate(frames)

    def test_row_removal_reordering_or_identity_change_fails(self):
        for mutation in ["remove", "reorder", "replace_id"]:
            with self.subTest(mutation=mutation):
                frames = list(fixture())
                if mutation == "remove":
                    frames[1] = frames[1].iloc[:-1].reset_index(drop=True)
                elif mutation == "reorder":
                    frames[1] = frames[1].iloc[[1, 0, 2, 3, 4, 5]].reset_index(drop=True)
                else:
                    frames[1].loc[0, "record_id"] = "2026-01:999"
                with self.assertRaisesRegex(ValueError, "identity, order or provenance"):
                    validate(frames)

    def test_source_provenance_change_fails(self):
        for field, value in [("original_source_row", 99), ("source_row", 999),
                             ("url", "https://krisha.kz/a/show/9999"), ("address", "туран 999")]:
            with self.subTest(field=field):
                frames = list(fixture())
                frames[1].loc[0, field] = value
                frames[2].loc[0, field] = value
                with self.assertRaisesRegex(ValueError, "identity, order or provenance"):
                    validate(frames)

    def test_holdout_base_features_cannot_change(self):
        for field, value in [("district", "Есиль"), ("area", 61.0), ("rooms", 3),
                             ("floor", 5), ("total_floors", 10), ("year_built", 2021),
                             ("building_type", "монолитный"), ("condition", "свежий ремонт")]:
            with self.subTest(field=field):
                frames = list(fixture())
                frames[1].loc[3, field] = value
                frames[2].loc[3, field] = value
                with self.assertRaisesRegex(ValueError, "holdout base features"):
                    validate(frames)

    def test_catalog_value_mismatch_or_missing_record_is_rejected(self):
        for mutation in ["value", "missing", "duplicate"]:
            with self.subTest(mutation=mutation):
                frames = list(fixture())
                if mutation == "value":
                    frames[2].loc[0, "area"] += 1
                elif mutation == "missing":
                    frames[2] = frames[2].iloc[:-1]
                else:
                    frames[2] = pd.concat([frames[2], frames[2].iloc[[0]]], ignore_index=True)
                with self.assertRaises(ValueError):
                    validate(frames)

    def test_corrected_alias_cannot_silently_drop_training_overlap(self):
        frames = list(fixture())
        frames[2].loc[3, "house_key"] = frames[2].loc[0, "house_key"]
        with self.assertRaisesRegex(ValueError, "silently exclude"):
            validate(frames)

    def test_corrected_alias_cannot_silently_drop_duplicate_test_row(self):
        frames = list(fixture())
        frames[2].loc[4, "house_key"] = frames[2].loc[3, "house_key"]
        with self.assertRaisesRegex(ValueError, "silently exclude"):
            validate(frames)

    def test_test_only_repeated_rows_are_not_added_to_training(self):
        frames = list(fixture())
        frames[3].loc[3, "subset"] = "train"
        with self.assertRaisesRegex(ValueError, "splits disagree"):
            validate(frames)

    def test_historical_primary_exclusions_are_rejected(self):
        frames = list(fixture())
        frames[4].loc[3, "primary_evaluation"] = False
        with self.assertRaisesRegex(ValueError, "entire frozen test"):
            validate(frames)

    def test_non_boolean_primary_flags_are_rejected(self):
        frames = list(fixture())
        frames[4]["primary_evaluation"] = ["False"] * 3 + ["True"] * 3
        with self.assertRaisesRegex(ValueError, "boolean flags"):
            validate(frames)

    def test_duplicate_or_incomplete_split_rows_fail(self):
        for split_index in [3, 4]:
            for mutation in ["duplicate", "missing", "fractional"]:
                with self.subTest(split=split_index, mutation=mutation):
                    frames = list(fixture())
                    if mutation == "duplicate":
                        frames[split_index] = pd.concat([frames[split_index], frames[split_index].iloc[[0]]])
                    elif mutation == "missing":
                        frames[split_index] = frames[split_index].iloc[:-1]
                    else:
                        frames[split_index]["row_index"] = frames[split_index].row_index.astype(float)
                        frames[split_index].loc[0, "row_index"] = 0.5
                    with self.assertRaises(ValueError):
                        validate(frames)

    def test_location_split_record_mapping_must_match_original(self):
        frames = list(fixture())
        frames[4].loc[3, "record_id"] = "2026-03:999"
        with self.assertRaisesRegex(ValueError, "location identity"):
            validate(frames)

    def test_saved_errors_ids_order_and_targets_are_fixed(self):
        for mutation in ["order", "price", "missing", "duplicate"]:
            with self.subTest(mutation=mutation):
                frames = list(fixture())
                if mutation == "order":
                    frames[5] = frames[5].iloc[::-1].reset_index(drop=True)
                elif mutation == "price":
                    frames[5].loc[0, "actual_price"] += 1
                elif mutation == "missing":
                    frames[5] = frames[5].iloc[:-1]
                else:
                    frames[5] = pd.concat([frames[5], frames[5].iloc[[0]]])
                with self.assertRaises(ValueError):
                    validate(frames)

    def test_expected_counts_are_enforced_and_can_be_disabled(self):
        frames = fixture()
        with self.assertRaisesRegex(ValueError, "train row count"):
            validate_retraining_partition(*frames)
        with self.assertRaisesRegex(ValueError, "holdout row count"):
            validate_retraining_partition(*frames, expected_train_rows=3, expected_holdout_rows=4)
        bundle = validate_retraining_partition(*frames, expected_train_rows=None, expected_holdout_rows=None)
        self.assertEqual(len(bundle["evaluation_ids"]), 3)

    def test_shuffled_catalog_and_split_storage_align_by_verified_ids(self):
        frames = list(fixture())
        for position in [2, 3, 4]:
            frames[position] = frames[position].iloc[::-1].reset_index(drop=True)
        bundle = validate(frames)
        self.assertEqual(bundle["frame"].record_id.tolist(), frames[1].record_id.tolist())
        self.assertEqual(bundle["evaluation_ids"].tolist(), [3, 4, 5])


if __name__ == "__main__":
    unittest.main()

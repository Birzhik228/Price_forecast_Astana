"""Synthetic identity and nesting guards; no model training or real artifacts."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from catboost_validation import export_fold_manifest, make_nested_folds


def fixture(n: int = 18):
    # Noncontiguous, nonmonotonic labels expose accidental positional indexing.
    index = pd.Index([100 + ((i * 7) % n) * 13 for i in range(n)], name="original_row")
    frame = pd.DataFrame({"record_id": [f"record-{i}" for i in range(n)],
                          "house_key": [f"astana|нура|туран {i}" for i in range(n)],
                          "price": np.arange(n) * 1_000_000 + 20_000_000}, index=index)
    apartments = pd.Series([f"apartment-{i}" for i in range(n)], index=index)
    return frame, apartments


def memberships(bundle):
    return [(phase, parent, fold["fold"], tuple(fold["train_ids"]), tuple(fold["validation_ids"]))
            for phase, parent, entries in [
                ("outer", None, bundle["outer"]),
                *[("inner", fold["fold"], fold["inner"]) for fold in bundle["outer"]],
                ("selection", None, bundle["selection"]),
            ] for fold in entries]


class NestedValidationTests(unittest.TestCase):
    def test_global_original_ids_cover_each_level_without_group_leakage(self):
        frame, apartments = fixture()
        frame_before, apartments_before = frame.copy(deep=True), apartments.copy(deep=True)
        bundle = make_nested_folds(frame, apartments)
        pd.testing.assert_index_equal(bundle["groups"].index, frame.index)
        for scope, entries in [(set(frame.index), bundle["outer"]),
                               (set(frame.index), bundle["selection"]),
                               *[(set(fold["train_ids"]), fold["inner"]) for fold in bundle["outer"]]]:
            valid_seen = []
            for fold in entries:
                train, valid = set(fold["train_ids"]), set(fold["validation_ids"])
                self.assertFalse(train & valid)
                self.assertEqual(train | valid, scope)
                self.assertTrue(set(bundle["groups"].loc[list(train)]).isdisjoint(
                    bundle["groups"].loc[list(valid)]))
                valid_seen.extend(valid)
            self.assertEqual(len(valid_seen), len(set(valid_seen)))
            self.assertEqual(set(valid_seen), scope)
        pd.testing.assert_frame_equal(frame, frame_before)
        pd.testing.assert_series_equal(apartments, apartments_before)

    def test_transitive_house_apartment_bridges_and_missing_house(self):
        frame, apartments = fixture()
        # house A joins rows0,1; apartment bridge joins1,2; house B joins2,3.
        frame.iloc[1, frame.columns.get_loc("house_key")] = frame.house_key.iloc[0]
        apartments.iloc[2] = apartments.iloc[1]
        frame.iloc[3, frame.columns.get_loc("house_key")] = frame.house_key.iloc[2]
        frame.iloc[4, frame.columns.get_loc("house_key")] = pd.NA
        apartments.iloc[4] = apartments.iloc[3]
        frame.iloc[5:7, frame.columns.get_loc("house_key")] = pd.NA
        bundle = make_nested_folds(frame, apartments)
        groups = bundle["groups"]
        self.assertEqual(groups.iloc[:5].nunique(), 1)
        self.assertNotEqual(groups.iloc[5], groups.iloc[6])
        for _, _, _, train, valid in memberships(bundle):
            self.assertFalse(set(apartments.loc[list(train)]) & set(apartments.loc[list(valid)]))

    def test_target_changes_and_other_columns_cannot_change_membership(self):
        frame, apartments = fixture()
        first = make_nested_folds(frame, apartments)
        frame.price = [object() for _ in range(len(frame))]
        frame["prediction"] = np.nan
        frame["absolute_error"] = 1e30
        second = make_nested_folds(frame, apartments)
        third = make_nested_folds(frame.drop(columns=["price", "prediction", "absolute_error"]), apartments)
        self.assertEqual(memberships(first), memberships(second))
        self.assertEqual(memberships(first), memberships(third))
        pd.testing.assert_series_equal(first["groups"], second["groups"])

    def test_deterministic_even_when_input_rows_are_reordered(self):
        frame, apartments = fixture()
        first = make_nested_folds(frame, apartments)
        second = make_nested_folds(frame, apartments)
        self.assertEqual(memberships(first), memberships(second))
        reverse = make_nested_folds(frame.iloc[::-1], apartments.iloc[::-1])
        left = {(phase, parent, n): set(valid) for phase, parent, n, _, valid in memberships(first)}
        right = {(phase, parent, n): set(valid) for phase, parent, n, _, valid in memberships(reverse)}
        self.assertEqual(left, right)

    def test_selection_split_count_follows_inner_splits(self):
        frame, apartments = fixture()
        bundle = make_nested_folds(frame, apartments, outer_splits=4, inner_splits=2)
        self.assertEqual(len(bundle["outer"]), 4)
        self.assertEqual(len(bundle["selection"]), 2)
        self.assertTrue(all(len(fold["inner"]) == 2 for fold in bundle["outer"]))

    def test_string_index_labels_are_preserved(self):
        frame, apartments = fixture()
        frame.index = pd.Index([f"row_{i * 10}" for i in range(len(frame))])
        apartments.index = frame.index
        bundle = make_nested_folds(frame, apartments)
        manifest = export_fold_manifest(frame, bundle)
        self.assertEqual(set(manifest.row_index), set(frame.index))

    def test_misaligned_or_missing_apartment_groups_fail(self):
        frame, apartments = fixture()
        for changed in [apartments.iloc[::-1], apartments.iloc[:-1],
                        apartments.mask(apartments.index == frame.index[0]),
                        apartments.map(lambda _: ""), apartments.map(lambda _: 1)]:
            with self.subTest(groups=changed.tolist()[:2]):
                with self.assertRaisesRegex(ValueError, "Groups"):
                    make_nested_folds(frame, changed)

    def test_duplicate_or_missing_row_identity_fails(self):
        for mutation in ["duplicate_index", "missing_index", "duplicate_record", "missing_record", "blank_record"]:
            frame, apartments = fixture()
            if mutation == "duplicate_index":
                frame.index = pd.Index([0] * len(frame))
                apartments.index = frame.index
            elif mutation == "missing_index":
                frame.index = pd.Index([np.nan, *frame.index[1:]])
                apartments.index = frame.index
            elif mutation == "duplicate_record":
                frame.iloc[1, 0] = frame.iloc[0, 0]
            else:
                frame.iloc[0, 0] = None if mutation == "missing_record" else " "
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                make_nested_folds(frame, apartments)

    def test_missing_house_column_or_empty_inputs_fail(self):
        frame, apartments = fixture()
        for changed, groups in [(frame.drop(columns="house_key"), apartments),
                                (frame.iloc[:0], apartments.iloc[:0])]:
            with self.assertRaises(ValueError):
                make_nested_folds(changed, groups)

    def test_invalid_counts_and_insufficient_inner_components_fail(self):
        frame, apartments = fixture()
        for count in [True, 1, 0, 2.5, "3"]:
            for name in ["outer_splits", "inner_splits"]:
                with self.subTest(name=name, count=count), self.assertRaises(ValueError):
                    make_nested_folds(frame, apartments, **{name: count})
        tiny, groups = fixture(3)
        with self.assertRaisesRegex(ValueError, "inner.*insufficient"):
            make_nested_folds(tiny, groups)
        with self.assertRaisesRegex(ValueError, "outer.*insufficient"):
            make_nested_folds(frame, apartments.map(lambda _: "one-apartment"))

    def test_manifest_reconstructs_all_splits_without_targets(self):
        frame, apartments = fixture()
        bundle = make_nested_folds(frame, apartments)
        manifest = export_fold_manifest(frame, bundle)
        self.assertEqual(len(manifest), 4 * len(frame))
        self.assertEqual(list(manifest), ["phase", "outer_fold", "fold", "row_index", "record_id", "group"])
        self.assertFalse(manifest.duplicated(["phase", "outer_fold", "row_index"]).any())
        for phase, parent, number, train, valid in memberships(bundle):
            mask = manifest.phase.eq(phase) & manifest.fold.eq(number)
            mask &= manifest.outer_fold.isna() if parent is None else manifest.outer_fold.eq(parent)
            part = manifest.loc[mask]
            self.assertEqual(part.row_index.tolist(), list(valid))
            scope = frame.index if parent is None else pd.Index(bundle["outer"][parent - 1]["train_ids"])
            self.assertEqual(scope[~scope.isin(part.row_index)].tolist(), list(train))
            self.assertEqual(part.record_id.tolist(), frame.loc[part.row_index, "record_id"].tolist())
            self.assertEqual(part.group.tolist(), bundle["groups"].loc[part.row_index].tolist())

    def test_export_rejects_tampered_duplicates_unknown_rows_and_nesting(self):
        frame, apartments = fixture()
        original = make_nested_folds(frame, apartments)
        for mutation in ["duplicate", "unknown", "outer_validation_in_inner", "repeat_validation", "missing_inner"]:
            bundle = copy.deepcopy(original)
            outer = bundle["outer"][0]
            if mutation == "duplicate":
                outer["train_ids"][1] = outer["train_ids"][0]
            elif mutation == "unknown":
                outer["train_ids"][0] = -999
            elif mutation == "outer_validation_in_inner":
                outer["inner"][0]["train_ids"][0] = outer["validation_ids"][0]
            elif mutation == "repeat_validation":
                replacement = copy.deepcopy(bundle["selection"][0])
                replacement["fold"] = 2
                bundle["selection"][1] = replacement
            else:
                outer.pop("inner")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                export_fold_manifest(frame, bundle)

    def test_export_rejects_group_leakage_even_with_complete_row_partitions(self):
        frame, apartments = fixture()
        frame.iloc[1, frame.columns.get_loc("house_key")] = frame.house_key.iloc[0]
        bundle = make_nested_folds(frame, apartments)
        first, second = frame.index[:2]
        outer = next(fold for fold in bundle["outer"] if first in fold["validation_ids"])
        other = outer["train_ids"][0]
        outer["validation_ids"][outer["validation_ids"] == second] = other
        outer["train_ids"][outer["train_ids"] == other] = second
        with self.assertRaisesRegex(ValueError, "group leakage"):
            export_fold_manifest(frame, bundle)


if __name__ == "__main__":
    unittest.main()

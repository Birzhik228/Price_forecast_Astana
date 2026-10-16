"""Audit alignment, pre-imputation missingness and small-group interpretation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from active_error_analysis import (
    MISSING, _verify_hashes, align_holdout, assemble_errors, count_filled_base_fields,
    segment_labels, summarize_segments, top_errors,
)
from error_analysis import metric_rows, sha256
from features import FEATURE_COLUMNS
from tests.test_verified_geo_model import fixture, house, point, registry
from verified_geo_model import build_verified_candidate


def tables():
    combined = pd.DataFrame({
        "record_id": ["a", "b", "c", "d", "e"], "source_row": [40, 12, 81, 90, 32],
        "price": [20e6, 30e6, 40e6, 50e6, 60e6], "group": ["ga", "gb", "gc", "gd", "ge"],
        "district": ["Есиль", "Нура", "Есиль", None, "Нура"], "url": [f"https://krisha.kz/a/show/{i}" for i in range(5)],
    })
    catalog = combined.assign(house_key=["ha", "hb", "hc", None, "he"], area=[40., 60., 80., 100., 120.])
    split = pd.DataFrame({"row_index": range(5), "record_id": combined.record_id,
                          "group": combined.group, "experiment_group": ["ea", "eb", "ec", "ed", "ee"],
                          "house_key": catalog.house_key, "subset": ["train", "test", "train", "test", "test"],
                          "primary_evaluation": [False, True, False, True, True]})
    saved = catalog.loc[[1, 3, 4], ["record_id", "price", "group", "house_key", "district"]].reset_index(drop=True)
    return combined, catalog, split, saved


class AuditAlignmentTests(unittest.TestCase):
    def test_catalog_and_manifest_can_be_reordered_without_changing_test_identity(self):
        combined, catalog, split, saved = tables()
        catalog["district"] = catalog.district.astype("string")
        result = align_holdout(combined, catalog.iloc[[4, 0, 3, 1, 2]], split.iloc[[3, 2, 1, 4, 0]], saved, expected_rows=3)
        self.assertEqual(result.record_id.tolist(), ["b", "d", "e"])
        self.assertEqual(result.source_row.tolist(), [12, 90, 32])
        self.assertEqual(result.row_index.tolist(), [1, 3, 4])
        self.assertEqual(result.price.tolist(), saved.price.tolist())

    def test_rejects_duplicate_identity_source_row_corruption_and_price_changes(self):
        for column, value in (("record_id", "a"), ("source_row", 999), ("price", 999.)):
            combined, catalog, split, saved = tables()
            catalog.loc[1, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                align_holdout(combined, catalog, split, saved, expected_rows=3)
        combined, catalog, split, saved = tables()
        with self.assertRaisesRegex(ValueError, "порядок"):
            align_holdout(combined, catalog, split, saved.iloc[::-1], expected_rows=3)

    def test_rejects_wrong_split_alias_overlap_and_truthy_string_booleans(self):
        for kind in ("record", "alias_leak", "train_in_primary", "string_boolean"):
            combined, catalog, split, saved = tables()
            if kind == "record":
                split.loc[1, "record_id"] = "e"
            elif kind == "alias_leak":
                split.loc[0, "experiment_group"] = split.loc[1, "experiment_group"]
            elif kind == "train_in_primary":
                split.loc[0, "primary_evaluation"] = True
            else:
                split["primary_evaluation"] = split.primary_evaluation.astype(str)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                align_holdout(combined, catalog, split, saved, expected_rows=3)

    def test_hash_check_rejects_modified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.csv"
            path.write_text("price\n123\n", encoding="utf-8")
            hashes = {path.name: sha256(path)}
            _verify_hashes(root, hashes)
            path.write_text("price\n456\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                _verify_hashes(root, hashes)


class AuditSegmentationTests(unittest.TestCase):
    def test_missingness_is_original_eight_fields_and_labels_handle_none_nan_blank(self):
        _, frame = fixture()
        frame = frame.iloc[:4].copy()
        frame.loc[frame.index[0], "condition"] = "   "
        frame.loc[frame.index[1], "district"] = pd.NA
        frame.loc[frame.index[2], "year_built"] = np.nan
        frame.loc[frame.index[3], ["floor", "building_type"]] = [np.nan, None]
        frame["residential_complex"] = [None, "   ", " НОВЫЙ ЖК ", "Пример"]
        before = frame.copy(deep=True)
        self.assertEqual(count_filled_base_fields(frame).tolist(), [7, 7, 7, 6])
        geography = pd.Series("Нет пригодных координат", index=frame.index)
        labels = segment_labels(frame, {"пример"}, geography)
        self.assertEqual(labels["complex_familiarity"].tolist(), ["ЖК не указан", "ЖК не указан", "ЖК не встречался в train", "ЖК встречался в train"])
        self.assertEqual(labels["district"].iloc[1], MISSING)
        self.assertEqual(labels["completeness"].tolist(), ["7/8", "7/8", "7/8", "6/8"])
        pd.testing.assert_frame_equal(frame, before)

    def test_bins_are_right_closed_and_missing_values_remain_visible(self):
        _, frame = fixture()
        frame = frame.iloc[:5].copy()
        frame["area"] = [40., 60., 90., 120., np.nan]
        frame["price"] = [30e6, 50e6, 80e6, 120e6, 120e6 + 1]
        labels = segment_labels(frame, set(), pd.Series("Нет пригодных координат", index=frame.index))
        self.assertEqual(labels["area_band"].tolist(), ["≤40 м²", "40–60 м²", "60–90 м²", "90–120 м²", MISSING])
        self.assertEqual(labels["price_band"].tolist(), ["≤30 млн ₸", "30–50 млн ₸", "50–80 млн ₸", "80–120 млн ₸", ">120 млн ₸"])

    def test_small_worst_segment_gets_no_rank_and_signed_bias_is_preserved(self):
        actual = np.repeat(100., 89)
        predicted = np.concatenate([np.repeat(150., 29), np.repeat(120., 30), np.repeat(90., 30)])
        errors = metric_rows(actual, predicted, 10.)
        labels = {"district": pd.Series(["small"] * 29 + ["large_over"] * 30 + ["large_under"] * 30)}
        result = summarize_segments(errors, labels).set_index("segment")
        self.assertEqual(result.loc["small", "n"], 29)
        self.assertTrue(result.loc["small", "small_group"])
        self.assertFalse(result.loc["small", "eligible_for_ranking"])
        self.assertTrue(pd.isna(result.loc["small", "rank_mae"]))
        self.assertTrue(pd.isna(result.loc["small", "rank_over_20"]))
        self.assertEqual(result.loc["large_over", "rank_mae"], 1)
        self.assertEqual(result.loc["large_over", "over_20_n"], 0)
        self.assertEqual(result.loc["large_over", "overprediction_pct"], 100)
        self.assertEqual(result.loc["large_under", "underprediction_pct"], 100)
        self.assertEqual(result.loc["large_under", "bias_tenge"], -10)
        self.assertAlmostEqual(result.error_share_pct.sum(), 100.)
        with self.assertRaisesRegex(ValueError, "порядком"):
            summarize_segments(errors, {"district": labels["district"].iloc[::-1]})

    def test_top_errors_use_distinct_absolute_and_relative_rankings_with_stable_ties(self):
        errors = pd.DataFrame({"record_id": ["z", "b", "a"], "absolute_error": [100., 20., 20.], "ape_pct": [10., 40., 40.]})
        self.assertEqual(top_errors(errors, "absolute_error", 2).record_id.tolist(), ["z", "a"])
        self.assertEqual(top_errors(errors, "ape_pct", 2).record_id.tolist(), ["a", "b"])

    def test_export_uses_consumed_geography_not_catalog_and_does_not_fit_or_fill_originals(self):
        base, frame = fixture()
        with tempfile.TemporaryDirectory() as directory:
            candidate = build_verified_candidate(base, registry(Path(directory), {house("3"): point()}))
        selected = frame.iloc[[0, 20, 27]].copy()
        selected["source_row"] = [10, 20, 30]
        selected["latitude"] = 0.0  # full-catalog values must never become model geography
        selected["longitude"] = 0.0
        selected.loc[selected.index[1], "year_built"] = np.nan
        before = selected.copy(deep=True)
        with joblib.parallel_backend("sequential"):
            prediction = candidate.predict(selected)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Audit cannot fit")):
            errors, segments, summary = assemble_errors(selected, candidate, prediction, 1e6)
        self.assertEqual(errors.record_id.tolist(), selected.record_id.tolist())
        self.assertEqual(errors.coordinate_source.tolist(), ["training_consensus", "osm_exact_address", "blocked_trait_conflict"])
        self.assertEqual(errors.catalog_latitude.tolist(), [0., 0., 0.])
        self.assertEqual(errors.latitude.iloc[1], 51.15)
        self.assertTrue(pd.isna(errors.latitude.iloc[2]))
        self.assertTrue(pd.isna(errors.year_built.iloc[1]))
        self.assertEqual(errors.filled_base_fields.iloc[1], 7)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["mae_tenge"], float(np.abs(prediction - selected.price).mean()))
        self.assertTrue(segments.rank_mae.isna().all())
        pd.testing.assert_frame_equal(selected, before)


if __name__ == "__main__":
    unittest.main()

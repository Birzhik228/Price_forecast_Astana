"""Protect frozen experiment evidence while allowing the two app integration edits."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from activate_catboost_model import (
    ACTIVATION_DIR, SNAPSHOT_SOURCES, _digest, _metrics, _partition, verify_sources,
)


class SourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.expected = {}
        snapshot = self.root / ACTIVATION_DIR / "source_snapshot"
        snapshot.mkdir(parents=True)
        for name in sorted(SNAPSHOT_SOURCES):
            raw = f"original {name}\r\n".encode()
            (snapshot / name).write_bytes(raw)
            (self.root / name).write_bytes(raw)
            self.expected[name] = _digest(self.root / name)
        (self.root / "model.joblib").write_bytes(b"frozen artifact")
        self.expected["model.joblib"] = _digest(self.root / "model.joblib")

    def test_app_edits_use_only_exact_original_snapshots(self):
        for name in SNAPSHOT_SOURCES:
            (self.root / name).write_bytes(b"new application integration")
        evidence = verify_sources(self.root, self.expected)
        self.assertTrue(evidence["config.py"]["uses_original_snapshot"])
        self.assertFalse(evidence["model.joblib"]["uses_original_snapshot"])
        self.assertEqual(evidence["model.joblib"]["verified_path"], "model.joblib")

    def test_changed_snapshot_is_rejected_even_if_live_file_matches(self):
        (self.root / ACTIVATION_DIR / "source_snapshot/config.py").write_bytes(b"not original")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            verify_sources(self.root, self.expected)

    def test_missing_snapshot_is_rejected(self):
        (self.root / ACTIVATION_DIR / "source_snapshot/config.py").unlink()
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            verify_sources(self.root, self.expected)

    def test_any_other_frozen_change_is_rejected(self):
        (self.root / "model.joblib").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model.joblib"):
            verify_sources(self.root, self.expected)

    def test_both_application_hashes_are_required(self):
        self.expected.pop("location_prediction.py")
        with self.assertRaisesRegex(ValueError, "Missing application source"):
            verify_sources(self.root, self.expected)

    def test_external_source_path_is_rejected(self):
        self.expected["../outside.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Путь вне проекта"):
            verify_sources(self.root, self.expected)


class FrozenPartitionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"record_id": ["a", "b", "c"]}, index=[3, 8, 12])
        self.split = pd.DataFrame({"row_index": [12, 3, 8], "record_id": ["c", "a", "b"],
                                   "subset": ["train", "train", "test"]})

    def test_noncontiguous_ids_preserve_frame_order(self):
        self.assertEqual(_partition(self.frame, self.split), ([3, 12], [8]))

    def test_swapped_record_ids_are_rejected(self):
        self.split["record_id"] = ["a", "c", "b"]
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            _partition(self.frame, self.split)

    def test_duplicate_or_missing_indices_are_rejected(self):
        self.split["row_index"] = [3, 3, 8]
        with self.assertRaisesRegex(ValueError, "partition"):
            _partition(self.frame, self.split)


class ActivationMetricsTests(unittest.TestCase):
    def test_prices_are_scored_in_tenge_with_strict_twenty_percent_threshold(self):
        scores = _metrics([100, 200, 400], [120, 150, 500])
        self.assertAlmostEqual(scores["mae_tenge"], 170 / 3)
        self.assertEqual(scores["over_20_n"], 2)
        self.assertEqual(scores["median_ae_tenge"], 50)
        self.assertEqual(scores["median_ape_pct"], 25)

    def test_nonfinite_prediction_and_invalid_price_are_rejected(self):
        for actual, predicted in (([1, 2], [1, np.nan]), ([0, 2], [1, 2]), ([1, 2], [1])):
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaisesRegex(ValueError, "Invalid holdout"):
                    _metrics(actual, predicted)


if __name__ == "__main__":
    unittest.main()

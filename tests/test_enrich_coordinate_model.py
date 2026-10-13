"""One predeclared frozen comparison: aligned rows, strict gate and no fitting."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from enrich_coordinate_model import BASE_METADATA, BASE_MODEL, _scores, activation_gate, evaluate_candidates, run_enrichment
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from location_model import LOCATION_INPUT_COLUMNS
from location_prediction import load_app_artifacts
from tests.test_verified_geo_model import fixture, house, point, registry
from verified_geo_model import build_verified_candidate


class EnrichmentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base, self.frame = fixture()
        self.candidate = build_verified_candidate(self.base, registry(Path(self.temporary.name), {house("3"): point()}))

    def test_identical_rows_new_geo_and_unaffected_exact_predictions_without_fit(self):
        shuffled = self.frame.iloc[[20, 2, 27, 10, 21, 8]].copy()
        with joblib.parallel_backend("sequential"):
            archived = self.base.predict(shuffled)
        model_hash = joblib.hash(self.base)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No fit allowed")):
            report, predictions = evaluate_candidates(self.base, self.candidate, shuffled, archived)
        self.assertEqual(report["record_ids"], shuffled.record_id.tolist())
        self.assertEqual(predictions.price.tolist(), shuffled.price.tolist())
        self.assertEqual(report["new_geo_rows"], 2)
        self.assertEqual(report["unchanged_rows"], 4)
        self.assertTrue(report["gate"]["unchanged_rows_identical"])
        self.assertEqual(report["fit_calls"], 0)
        self.assertEqual(joblib.hash(self.base), model_hash)
        self.assertEqual(report["comparison"]["candidate"]["mae_tenge"],
                         float(np.abs(predictions.price - predictions.candidate_prediction).mean()))
        np.testing.assert_array_equal(predictions.loc[~predictions.new_geo_available, "candidate_prediction"],
                                      predictions.loc[~predictions.new_geo_available, "base_prediction"])

    def test_strict_twenty_percent_threshold_and_gate_each_condition(self):
        scores = _scores(np.array([100., 100., 100.]), np.array([120., 79.999, 80.]))
        self.assertEqual(scores["over_20_count"], 1)
        base = {"mae_tenge": 5., "over_20_count": 2}
        self.assertTrue(activation_gate(base, dict(base), True)["passed"])
        self.assertFalse(activation_gate(base, {"mae_tenge": 5.000001, "over_20_count": 1}, True)["passed"])
        self.assertFalse(activation_gate(base, {"mae_tenge": 4., "over_20_count": 3}, True)["passed"])
        self.assertFalse(activation_gate(base, dict(base), False)["passed"])

    def test_misaligned_archive_and_invalid_ids_or_targets_reject(self):
        with joblib.parallel_backend("sequential"):
            archived = self.base.predict(self.frame)
        with self.assertRaises(AssertionError):
            evaluate_candidates(self.base, self.candidate, self.frame, archived[::-1])
        for changes in ({"record_id": "same"}, {"price": 0}, {"price": np.nan}):
            malformed = self.frame.assign(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate_candidates(self.base, self.candidate, malformed)
        with self.assertRaises(ValueError):
            evaluate_candidates(self.base, self.candidate, self.frame.iloc[:1])

    def test_empty_external_coverage_is_reported_without_invented_segment_metrics(self):
        report, predictions = evaluate_candidates(self.base, self.candidate, self.frame.iloc[:10])
        self.assertEqual(report["new_geo_rows"], 0)
        self.assertIsNone(report["new_geo_segment"])
        self.assertEqual(report["changed_prediction_rows"], 0)
        self.assertTrue(report["gate"]["passed"])

    def test_runner_freezes_before_prediction_reloads_and_writes_distinct_metadata(self):
        root = Path(self.temporary.name)
        frame = pd.concat([self.frame] * 19, ignore_index=True).iloc[:590].copy()
        frame["record_id"] = [f"holdout:{i}" for i in range(590)]
        frame["group"] = frame.record_id
        for filename in ("features.py", "location_prediction.py", "comparables.py", "clean.py", "config.py"):
            (root / filename).write_text("# immutable test source\n", encoding="utf-8")
        for path in (BASE_MODEL.parent, BASE_METADATA.parent, Path("data/processed/combined_2026"),
                     Path("reports/combined_update"), Path("reports/location_experiment")):
            (root / path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.base, root / BASE_MODEL)
        model_sha = hashlib.sha256((root / BASE_MODEL).read_bytes()).hexdigest()
        metadata = {"schema_version": 2, "variant": "complex_coords", "feature_columns": list(LOCATION_INPUT_COLUMNS),
                    "currency": "KZT", "target": "asking_price", "model_sha256": model_sha,
                    "source_sha256": {BASE_MODEL.as_posix(): model_sha},
                    "categories_from_train": {column: sorted(frame[column].unique()) for column in CATEGORICAL_FEATURES},
                    "numeric_ranges_from_train": {column: {"min": float(frame[column].min()), "max": float(frame[column].max())}
                                                   for column in NUMERIC_FEATURES},
                    "test_metrics": {"mae_tenge": 12345.}, "approximate_range": {"half_width_tenge": 12345.}}
        (root / BASE_METADATA).write_text(json.dumps(metadata), encoding="utf-8")
        before_metadata = (root / BASE_METADATA).read_bytes()
        frame.to_csv(root / "data/processed/combined_2026/listings_clean.csv", index=False)
        pd.DataFrame({"row_index": range(590)}).to_csv(root / "reports/combined_update/split_manifest.csv", index=False)
        with joblib.parallel_backend("sequential"):
            archived = self.base.predict(frame)
        pd.DataFrame({"record_id": frame.record_id, "price": frame.price, "primary_evaluation": True,
                      "complex_coords_prediction": archived}).to_csv(root / "reports/location_experiment/holdout_predictions.csv", index=False)
        snapshot = registry(root, {house("3"): point()})
        def compare_after_freeze(*args):
            protocol_path = root / "reports/result/protocol.json"
            self.assertTrue(protocol_path.is_file())
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            self.assertEqual(protocol["registry_sha256"], snapshot.sha256)
            self.assertEqual(protocol["record_ids"], frame.record_id.tolist())
            return evaluate_candidates(*args)
        with patch("comparables.load_catalog", return_value=SimpleNamespace(listings=frame)), \
                patch("location_experiment.prepare_partition", return_value={"frame": frame, "evaluation_ids": np.arange(590)}), \
                patch("enrich_coordinate_model.evaluate_candidates", side_effect=compare_after_freeze), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("Never fit during enrichment")):
            report = run_enrichment(Path(snapshot.path), project_root=root,
                                    models_dir=Path("models/result"), reports_dir=Path("reports/result"))
        candidate, new_metadata = load_app_artifacts(root / "models/result/complex_coords.joblib", root / "models/result/complex_coords.json")
        self.assertTrue(new_metadata["geo_enrichment"])
        self.assertEqual(new_metadata["base_model_sha256"], model_sha)
        self.assertEqual(new_metadata["test_metrics"]["mae_tenge"], report["comparison"]["candidate"]["mae_tenge"])
        self.assertNotEqual(new_metadata["test_metrics"]["mae_tenge"], metadata["test_metrics"]["mae_tenge"])
        self.assertEqual(new_metadata["approximate_range"]["half_width_tenge"], new_metadata["test_metrics"]["mae_tenge"])
        self.assertFalse(new_metadata["active"])
        self.assertFalse(report["active_changed"])
        self.assertTrue(report["serialized_replay_exact"])
        self.assertEqual((root / BASE_METADATA).read_bytes(), before_metadata)
        self.assertEqual(hashlib.sha256((root / BASE_MODEL).read_bytes()).hexdigest(), model_sha)
        self.assertEqual(hashlib.sha256((root / "reports/result/protocol.json").read_bytes()).hexdigest(), report["protocol_sha256"])


if __name__ == "__main__":
    unittest.main()

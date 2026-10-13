"""Активация на изолированных файлах: никакого fit и настоящих артефактов."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import activate_location_model as activation
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from location_model import LOCATION_INPUT_COLUMNS


def save(root: Path, name: str, value: dict) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def checksum(root: Path, name: str) -> str:
    return hashlib.sha256((root / name).read_bytes()).hexdigest()


def fixture(root: Path) -> None:
    frame = pd.DataFrame({
        "area": [40., 60., 80., 100.], "rooms": [1, 2, 3, 4], "floor": [2, 3, 4, 5],
        "total_floors": [9, 12, 15, 18], "year_built": [2000, 2010, 2020, 2025],
        "district": ["Есиль", "Нура", "Алматы", "Сарыарка"], "building_type": ["кирпичный"] * 4,
        "condition": ["свежий ремонт"] * 4, "record_id": ["r1", "r2", "r3", "r4"],
        "price": [100., 200., 300., 500.],
    })
    (root / activation.DATA).parent.mkdir(parents=True)
    frame.to_csv(root / activation.DATA, index=False)
    split = pd.DataFrame({"row_index": range(4), "subset": ["train", "train", "test", "test"],
                          "group": ["g1", "g2", "g3", "g4"]})
    (root / activation.COMBINED_SPLIT).parent.mkdir(parents=True)
    split.to_csv(root / activation.COMBINED_SPLIT, index=False)
    location = split.assign(record_id=frame.record_id, experiment_group=["e1", "e2", "e3", "e4"],
                            primary_evaluation=[False, False, True, True])
    (root / activation.EXPERIMENT).mkdir(parents=True)
    location.to_csv(root / activation.EXPERIMENT / "split_manifest.csv", index=False)
    pd.DataFrame({"record_id": ["r3", "r4"], "price": [300., 500.], "primary_evaluation": [True, True],
                  "complex_coords_prediction": [330., 400.]}).to_csv(root / activation.EXPERIMENT / "holdout_predictions.csv", index=False)
    for name in (activation.MODEL, "models/best_model.joblib", "models/combined_2026/best_model.joblib"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"fixture: deliberately not a joblib; generation never unpickles or fits")
    save(root, "models/model_metadata.json", {"model_sha256": checksum(root, "models/best_model.joblib")})
    train = frame.iloc[:2]
    combined = {
        "schema_version": 1, "feature_columns": list(FEATURE_COLUMNS), "train_rows": 2, "test_rows": 2,
        "input_sha256": checksum(root, activation.DATA), "split_manifest_sha256": checksum(root, activation.COMBINED_SPLIT),
        "categories_from_train": {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES},
        "numeric_ranges_from_train": {c: {"min": float(train[c].min()), "max": float(train[c].max())} for c in NUMERIC_FEATURES},
        "data_period_label": "января и конца марта 2026 года (март — по описанию источника)",
    }
    save(root, activation.COMBINED_METADATA, combined)
    names = [activation.COMBINED_METADATA, activation.COMBINED_SPLIT, activation.DATA,
             "models/best_model.joblib", "models/model_metadata.json"]
    plan = {"sources": {name: checksum(root, name) for name in names}, "variants": ["complex", "complex_coords"]}
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["signature"] = signature
    save(root, activation.EXPERIMENT + "experiment_plan.json", plan)
    save(root, activation.ORIGINAL_METADATA, {
        "schema_version": 2, "variant": "complex_coords", "feature_columns": list(LOCATION_INPUT_COLUMNS),
        "target": "asking_price", "currency": "KZT", "geo_training_only": True,
        "selected_by_cv": False, "experiment_signature": signature, "train_rows": 2,
        "model_sha256": checksum(root, activation.MODEL), "activated": False,
    })
    metrics = {"n": 2, "mae_tenge": 65., "rmse_tenge": (5450.) ** .5, "r2": .455, "over_20_pct": 0.}
    save(root, activation.EXPERIMENT + "experiment_report.json", {
        "signature": signature, "selected_variant": "complex", "training_rows": 2, "original_test_rows": 2,
        "test_rows": 2, "comparison": [{"model": "complex_coords", **metrics}], "limitations": ["Reused holdout."],
    })
    names = [activation.MODEL, activation.ORIGINAL_METADATA,
             *(activation.EXPERIMENT + name for name in ("experiment_plan.json", "experiment_report.json",
                                                        "split_manifest.csv", "holdout_predictions.csv"))]
    save(root, activation.EXPERIMENT + "verification.json", {
        "status": "passed", "experiment_signature": signature, "selected_by_recomputed_oof": "complex",
        "primary_holdout_rows": 2,
        "candidates": {"complex_coords": {"model_sha256": checksum(root, activation.MODEL)}},
        "recomputed_holdout_metrics": {"complex_coords": metrics},
        "source_sha256": plan["sources"], "verified_artifact_sha256": {name: checksum(root, name) for name in names},
    })


class ActivationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)

    def load(self, name: str) -> dict:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def refresh_verified_hash(self, name: str) -> None:
        verification = self.load(activation.EXPERIMENT + "verification.json")
        verification["verified_artifact_sha256"][name] = checksum(self.root, name)
        save(self.root, activation.EXPERIMENT + "verification.json", verification)

    def test_correct_metrics_ranges_provenance_and_no_original_writes(self) -> None:
        originals = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        report = activation.prepare_activation(self.root)
        metadata = self.load(activation.DEPLOYMENT)
        self.assertEqual(metadata["test_metrics"]["mae_tenge"], 65.)
        self.assertEqual(metadata["approximate_range"]["half_width_tenge"], 65.)
        self.assertEqual(metadata["numeric_ranges_from_train"]["area"]["max"], 60.)
        self.assertEqual(metadata["feature_columns"], list(LOCATION_INPUT_COLUMNS))
        self.assertEqual(metadata["categories_from_train"]["district"], ["Есиль", "Нура"])
        self.assertFalse(metadata["selected_by_cv"])
        self.assertFalse(metadata["fresh_validation_completed"])
        self.assertEqual(metadata["activation_reason"], "explicit_user_request")
        self.assertEqual(report["fit_calls"], 0)
        self.assertEqual(report["deployment_metadata"]["sha256"], checksum(self.root, activation.DEPLOYMENT))
        for name, content in originals.items():
            self.assertEqual((self.root / name).read_bytes(), content, name)

    def test_repeat_is_byte_identical_and_keeps_timestamp(self) -> None:
        first = activation.prepare_activation(self.root)
        before = [(self.root / name).read_bytes() for name in (activation.DEPLOYMENT, activation.ACTIVATION)]
        self.assertEqual(first, activation.prepare_activation(self.root))
        self.assertEqual(before, [(self.root / name).read_bytes() for name in (activation.DEPLOYMENT, activation.ACTIVATION)])

    def test_changed_source_rejected_before_any_output(self) -> None:
        (self.root / activation.MODEL).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "замороженный"):
            activation.prepare_activation(self.root)
        self.assertFalse((self.root / activation.DEPLOYMENT).exists())

    def test_invalid_plan_signature_rejected(self) -> None:
        name = activation.EXPERIMENT + "experiment_plan.json"
        plan = self.load(name)
        plan["variants"].append("changed")
        save(self.root, name, plan)
        with self.assertRaisesRegex(ValueError, "подпись"):
            activation.prepare_activation(self.root)

    def test_source_changed_during_preparation_is_not_published(self) -> None:
        serialize = activation._bytes

        def changed_after_reads(value):
            (self.root / activation.MODEL).write_bytes(b"concurrent change")
            return serialize(value)

        with patch.object(activation, "_bytes", side_effect=changed_after_reads):
            with self.assertRaisesRegex(ValueError, "во время подготовки"):
                activation.prepare_activation(self.root)
        self.assertFalse((self.root / activation.DEPLOYMENT).exists())
        self.assertFalse((self.root / activation.ACTIVATION).exists())

    def test_same_train_size_but_different_members_rejected(self) -> None:
        name = activation.EXPERIMENT + "split_manifest.csv"
        split = pd.read_csv(self.root / name)
        split.loc[[1, 2], "subset"] = ["test", "train"]
        split.to_csv(self.root / name, index=False)
        self.refresh_verified_hash(name)
        with self.assertRaisesRegex(ValueError, "Обучающие строки"):
            activation.prepare_activation(self.root)

    def test_holdout_group_leak_rejected(self) -> None:
        name = activation.EXPERIMENT + "split_manifest.csv"
        split = pd.read_csv(self.root / name)
        split.loc[2, "experiment_group"] = split.loc[0, "experiment_group"]
        split.to_csv(self.root / name, index=False)
        self.refresh_verified_hash(name)
        with self.assertRaisesRegex(ValueError, "пересекают"):
            activation.prepare_activation(self.root)

    def test_wrong_published_metrics_rejected_even_with_updated_report_hash(self) -> None:
        name = activation.EXPERIMENT + "experiment_report.json"
        report = self.load(name)
        report["comparison"][0]["mae_tenge"] = 1.
        save(self.root, name, report)
        self.refresh_verified_hash(name)
        with self.assertRaisesRegex(ValueError, "Метрики"):
            activation.prepare_activation(self.root)

    def test_existing_different_output_never_overwritten(self) -> None:
        activation.prepare_activation(self.root)
        name = activation.ACTIVATION
        (self.root / name).write_text("manual report", encoding="utf-8")
        before = (self.root / activation.DEPLOYMENT).read_bytes()
        with self.assertRaisesRegex(ValueError, "перезаписи"):
            activation.prepare_activation(self.root)
        self.assertEqual((self.root / name).read_text(encoding="utf-8"), "manual report")
        self.assertEqual((self.root / activation.DEPLOYMENT).read_bytes(), before)

    def test_recovery_after_only_metadata_was_published(self) -> None:
        result = activation.prepare_activation(self.root)
        (self.root / activation.ACTIVATION).unlink()
        self.assertEqual(result, activation.prepare_activation(self.root))


if __name__ == "__main__":
    unittest.main()

"""Оценка только искусственных строк с временными моделями; реальные артефакты не используются."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

import fresh_evaluation
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_pipeline
from fresh_evaluation import FrozenEvaluationConfig, evaluate_frozen, evaluation_metrics
from location_model import LOCATION_INPUT_COLUMNS, make_location_pipeline


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "area": [40., 60., 80., 100.], "rooms": [1, 2, 3, 4], "floor": [2., 3., np.nan, 5.],
        "total_floors": [9., 9., np.nan, 12.], "year_built": [2010., 2011., 2012., 2013.],
        "district": ["Есиль", "Нура", pd.NA, "Есиль"], "building_type": ["кирпичный"] * 4,
        "condition": ["хорошее", pd.NA, "хорошее", "хорошее"],
        "price": [20e6, 40e6, 70e6, 70e6 + 1],
        "record_id": ["synthetic:new:7", "synthetic:new:2", "synthetic:new:9", "synthetic:new:1"],
        "url": [f"https://krisha.kz/a/show/{900 + i}" for i in range(4)],
        "collected_at": ["2026-10-13T00:00:00Z"] * 4,
        "residential_complex": ["Первый ЖК", None, "Новый ЖК", "Второй ЖК"],
        "address": ["улица 1", "улица 2", "улица 3", "улица 4"],
        "primary_eligible": [True] * 4,
    }, index=[11, 4, 19, 2])


def build_fixture(root: Path) -> None:
    training = sample().loc[:, list(FEATURE_COLUMNS)].copy()
    for column in CATEGORICAL_FEATURES:
        training[column] = training[column].astype(object).where(training[column].notna(), np.nan)
    content = training.to_csv(index=False).encode()
    for name in ("data/processed/listings_clean.csv", "data/processed/january_2026/listings_clean.csv",
                 "data/processed/combined_2026/listings_clean.csv"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    split = root / "reports/combined_update/split_manifest.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("row_index,subset\n0,train\n", encoding="utf-8")
    for directory, constant in (("models", 24e6), ("models/combined_2026", 22e6)):
        path = root / directory / "best_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        pipeline = build_pipeline(DummyRegressor(strategy="constant", constant=constant)).fit(training, [1, 2, 3, 4])
        joblib.dump(pipeline, path)
        metadata = {"schema_version": 1, "currency": "KZT", "target": "asking_price",
                    "feature_columns": list(FEATURE_COLUMNS), "test_metrics": {"mae_tenge": 1},
                    "approximate_range": {"half_width_tenge": 1}, "numeric_ranges_from_train": {},
                    "categories_from_train": {column: sorted(training[column].dropna().unique()) for column in CATEGORICAL_FEATURES},
                    "model_sha256": digest(path), "input_sha256": hashlib.sha256(content).hexdigest(),
                    "split_manifest_sha256": digest(split)}
        save_json(path.with_name("model_metadata.json"), metadata)
    source_names = ["models/best_model.joblib", "models/model_metadata.json",
                    "models/combined_2026/best_model.joblib", "models/combined_2026/model_metadata.json",
                    "data/processed/combined_2026/listings_clean.csv", "reports/combined_update/split_manifest.csv"]
    plan = {"sources": {name: digest(root / name) for name in source_names}, "variants": ["complex"]}
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["signature"] = signature
    save_json(root / "reports/location_experiment/experiment_plan.json", plan)
    location = training.assign(residential_complex=["Первый ЖК", "Первый ЖК", "Второй ЖК", "Второй ЖК"],
                               house_key=None, raw_latitude=np.nan, raw_longitude=np.nan, raw_coordinate_status="missing")
    candidate = make_location_pipeline(DummyRegressor(strategy="constant", constant=20e6), "complex").fit(location, [1, 2, 3, 4])
    model_path = root / "models/location_2026/complex.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, model_path)
    save_json(model_path.with_suffix(".json"), {
        "schema_version": 2, "variant": "complex", "feature_columns": list(LOCATION_INPUT_COLUMNS),
        "selected_by_cv": True, "currency": "KZT", "target": "asking_price",
        "experiment_signature": signature, "model_sha256": digest(model_path),
    })


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        build_fixture(self.root)
        self.config = FrozenEvaluationConfig(root=self.root, bootstrap_repetitions=200, seed=42)
        self.provenance = {"source": "synthetic unit-test fixture", "date_evidence": "not real listings"}

    def evaluate(self, frame=None, directory="result") -> dict:
        return evaluate_frozen(sample() if frame is None else frame, output_dir=self.root / directory,
                               provenance=self.provenance, config=self.config)

    def test_same_order_metrics_segments_and_artifacts_without_any_fit(self) -> None:
        frame = sample()
        original = frame.copy(deep=True)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Evaluator must not fit")):
            report = self.evaluate(frame)
        self.assertEqual(report["fit_calls"], 0)
        self.assertEqual(report["record_ids"], frame.record_id.tolist())
        self.assertEqual(report["artifact_hashes_before"], report["artifact_hashes_after"])
        self.assertEqual(report["active_model"], "january")
        self.assertFalse(report["activated"])
        predictions = pd.read_csv(self.root / "result/predictions.csv")
        self.assertEqual(predictions.record_id.tolist(), frame.record_id.tolist())
        np.testing.assert_array_equal(predictions.price, frame.price)
        self.assertEqual(predictions.complex_status.tolist(), ["known", "missing", "new", "known"])
        self.assertEqual(predictions.completeness.tolist(), [8, 7, 5, 8])
        self.assertEqual(predictions.price_band.tolist(), list(fresh_evaluation.PRICE_BANDS))
        for metric in report["comparison"]:
            expected = evaluation_metrics(frame.price.to_numpy(), predictions[metric["model"] + "_prediction"].to_numpy())
            self.assertEqual({key: metric[key] for key in expected}, expected)
        self.assertEqual(len(report["paired_bootstrap"]), 3)
        for name, expected in report["output_sha256"].items():
            self.assertEqual(digest(self.root / "result" / name), expected)
        pd.testing.assert_frame_equal(frame, original)

    def test_threshold_is_strict_and_r2_undefined_cases_are_null(self) -> None:
        score = evaluation_metrics(np.array([50e6, 50e6]), np.array([60e6, 60e6 + 1]))
        self.assertEqual(score["over_20_pct"], 50.0)
        self.assertIsNone(score["r2"])
        self.assertIsNone(evaluation_metrics(np.array([10.]), np.array([9.]))["r2"])

    def test_bootstrap_is_reproducible_and_existing_output_is_not_replaced(self) -> None:
        first, second = self.evaluate(directory="a"), self.evaluate(directory="b")
        self.assertEqual(first["paired_bootstrap"], second["paired_bootstrap"])
        saved = digest(self.root / "a/evaluation_report.json")
        with self.assertRaises(FileExistsError):
            self.evaluate(directory="a")
        self.assertEqual(saved, digest(self.root / "a/evaluation_report.json"))

    def test_invalid_or_noneligible_rows_fail_before_model_load(self) -> None:
        cases = []
        for field, value in (("price", 0), ("area", np.nan), ("rooms", 1.5), ("floor", np.inf),
                             ("rooms", True), ("record_id", ""), ("primary_eligible", False)):
            changed = sample().astype({field: object})
            changed.iloc[0, changed.columns.get_loc(field)] = value
            cases.append(changed)
        duplicate_id, duplicate_url = sample(), sample()
        duplicate_id.iloc[1, duplicate_id.columns.get_loc("record_id")] = duplicate_id.record_id.iloc[0]
        duplicate_url.iloc[1, duplicate_url.columns.get_loc("url")] = duplicate_url.url.iloc[0] + "?alias=1"
        cases.extend([sample().iloc[:0], sample().iloc[:1], duplicate_id, duplicate_url])
        with patch("fresh_evaluation._load_models", side_effect=AssertionError("validation must precede load")):
            for index, frame in enumerate(cases):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    self.evaluate(frame)

    def test_rejects_unselected_or_tampered_complex_artifact(self) -> None:
        metadata_path = self.root / "models/location_2026/complex.json"
        original = json.loads(metadata_path.read_text(encoding="utf-8"))
        save_json(metadata_path, original | {"selected_by_cv": False})
        with self.assertRaisesRegex(ValueError, "выбранной модели"):
            self.evaluate()
        save_json(metadata_path, original)
        with (self.root / "models/location_2026/complex.joblib").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "выбранной модели"):
            self.evaluate()

    def test_training_data_hash_mismatch_prevents_evaluation(self) -> None:
        path = self.root / "data/processed/combined_2026/listings_clean.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "замороженный вход"):
            self.evaluate()
        self.assertFalse((self.root / "result").exists())

    def test_changes_during_predict_are_detected_before_reports_are_written(self) -> None:
        original_loader = fresh_evaluation._load_models
        def load_then_mutate(root):
            models, metadata, hashes = original_loader(root)
            original_predict = models["january"].predict
            def malicious_predict(frame):
                target = root / "data/processed/listings_clean.csv"
                target.write_bytes(target.read_bytes() + b"\n")
                return original_predict(frame)
            models["january"].predict = malicious_predict
            return models, metadata, hashes
        with patch("fresh_evaluation._load_models", side_effect=load_then_mutate):
            with self.assertRaisesRegex(RuntimeError, "изменились"):
                self.evaluate()
        self.assertFalse((self.root / "result").exists())

    def test_html_escapes_provenance_and_empty_price_segments_remain_defined(self) -> None:
        self.provenance["source"] = "<script>bad()</script>"
        frame = sample()
        frame["price"] = [90e6, 100e6, 110e6, 120e6]
        report = self.evaluate(frame)
        page = (self.root / "result/report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        empty = [row for row in report["segments"] if row["dimension"] == "price_band" and row["n"] == 0]
        self.assertEqual(len(empty), 9)
        self.assertTrue(all(row["mae_tenge"] is None for row in empty))


if __name__ == "__main__":
    unittest.main()

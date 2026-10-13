"""Source evidence and orchestration checks; synthetic records are never reports."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import fresh_validation as validation


def source(directory: Path) -> tuple[Path, dict]:
    csv = directory / "export.csv"
    csv.write_text("price\n30000000\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "source_id": "synthetic-test-only", "source_url": "https://example.test/export",
        "csv_path": csv.name, "csv_sha256": validation.sha256(csv), "city": "astana",
        "market": "sale_apartment", "currency": "KZT", "target": "asking_price",
        "date_basis": "source_observed_at", "date_evidence": "Synthetic fixture observation dates",
        "usage_basis": "Self-authored synthetic test fixture",
    }
    path = directory / "source.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


class FreshValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_source_sha_is_verified_before_use(self):
        path, manifest = source(self.directory)
        self.assertEqual(validation.read_source_manifest(path)[0], manifest)
        (self.directory / "export.csv").write_text("price\n31000000\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validation.read_source_manifest(path)

    def test_wrong_population_and_unsubstantiated_dates_rejected(self):
        for key, value in (("date_basis", "downloaded_at"), ("city", "almaty"),
                           ("target", "transaction_price"), ("date_evidence", ""),
                           ("usage_basis", "REPLACE_license"), ("currency", "USD")):
            with self.subTest(key=key):
                path, manifest = source(self.directory)
                manifest[key] = value
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validation.read_source_manifest(path)

    def test_existing_outputs_never_overwritten(self):
        path, _ = source(self.directory)
        output = self.directory / "run"
        output.mkdir()
        result = output / "status.json"
        result.write_text("original", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            validation.run_validation(path, as_of="2026-10-13", output_dir=output)
        self.assertEqual(result.read_text(encoding="utf-8"), "original")

    def test_missing_columns_stop_before_model_loading(self):
        path, _ = source(self.directory)
        with patch.object(validation, "evaluate_frozen", side_effect=AssertionError("must not evaluate")):
            with self.assertRaisesRegex(ValueError, "Missing export columns"):
                validation.run_validation(path, as_of="2026-10-13", output_dir=self.directory / "run")
        self.assertFalse((self.directory / "run").exists())

    def test_empty_primary_and_changes_during_preparation(self):
        for mutation in (None, "source", "reference", "manifest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                path, manifest = source(directory)
                root = directory / "project"
                protected = (*validation.REFERENCE_FILES.values(), *validation.RAW_REFERENCE_FILES,
                             *validation.MODEL_FILES, *validation.CODE_FILES,
                             "data/processed/combined_2026/listings_clean.csv",
                             "reports/combined_update/split_manifest.csv", "reports/location_experiment/experiment_plan.json")
                for relative in protected:
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text("url\nhttps://krisha.kz/a/show/1000000\n", encoding="utf-8")
                columns = [*validation.FEATURE_COLUMNS, "price", "url", "collected_at", "address", "residential_complex"]
                pd.DataFrame(columns=columns).to_csv(directory / "export.csv", index=False)
                manifest["csv_sha256"] = validation.sha256(directory / "export.csv")
                path.write_text(json.dumps(manifest), encoding="utf-8")
                empty = pd.DataFrame({"primary_eligible": pd.Series(dtype=bool), "record_id": pd.Series(dtype=str)})

                def prepare(*args, **kwargs):
                    if mutation:
                        target = {"source": directory / "export.csv", "manifest": path,
                                  "reference": root / validation.REFERENCE_FILES["january"]}[mutation]
                        with target.open("a", encoding="utf-8") as stream:
                            stream.write("\n")
                    return {"frame": empty, "rejected": pd.DataFrame(), "audit": pd.DataFrame(), "summary": {"rows": 0}}

                with patch.object(validation, "prepare_fresh_holdout", side_effect=prepare), patch.object(
                    validation, "evaluate_frozen", side_effect=AssertionError("must not evaluate")
                ):
                    if mutation:
                        with self.assertRaisesRegex(RuntimeError, "Protected inputs changed"):
                            validation.run_validation(path, as_of="2026-10-13", output_dir=directory / "run", root=root)
                        self.assertFalse((directory / "run").exists())
                        continue
                    status = validation.run_validation(path, as_of="2026-10-13", output_dir=directory / "run", root=root)
                self.assertEqual(status["status"], "insufficient_independent_rows")
                self.assertFalse(status["evaluation_run"])
                self.assertTrue(status["protected_inputs_unchanged"])
                self.assertFalse((directory / "run/evaluation").exists())
                protocol = json.loads((directory / "run/protocol.json").read_text(encoding="utf-8"))
                self.assertEqual(protocol["primary_record_ids"], [])
                self.assertFalse(protocol["fit_allowed"])


if __name__ == "__main__":
    unittest.main()

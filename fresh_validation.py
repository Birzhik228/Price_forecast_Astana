"""Evaluate an evidence-backed local export without collecting or training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROJECT_ROOT
from features import FEATURE_COLUMNS
from fresh_evaluation import FrozenEvaluationConfig, evaluate_frozen
from fresh_validation_data import prepare_fresh_holdout


REFERENCE_FILES = {
    "january": "data/processed/january_2026/listings_clean.csv",
    "march": "data/processed/march_2026/listings_clean.csv",
}
RAW_REFERENCE_FILES = (
    "data/raw/kaggle_astana/astana_apartments.csv",
    "data/raw/turarr_march_2026/astana_apartments_raw.csv",
)
MODEL_FILES = (
    "models/best_model.joblib", "models/model_metadata.json",
    "models/combined_2026/best_model.joblib", "models/combined_2026/model_metadata.json",
    "models/location_2026/complex.joblib", "models/location_2026/complex.json",
)
CODE_FILES = (
    "fresh_validation.py", "fresh_validation_data.py", "fresh_evaluation.py",
    "clean.py", "building_catalog.py", "update_march.py", "location_model.py", "features.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_source_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    """Check declared provenance; these checks cannot authenticate a publisher."""
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    required_values = {
        "schema_version": 1, "city": "astana", "market": "sale_apartment",
        "currency": "KZT", "target": "asking_price", "date_basis": "source_observed_at",
    }
    for key, expected in required_values.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Source manifest requires {key}={expected!r}.")
    for key in ("source_id", "source_url", "csv_path", "csv_sha256", "date_evidence", "usage_basis"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip() or value.startswith("REPLACE_"):
            raise ValueError(f"Source manifest needs a real {key}.")
    # A download date or repository commit date is not an observation date.
    if manifest.get("observed_at_column", "collected_at") != "collected_at":
        raise ValueError("Normalize source observation dates into collected_at, preserving originals in the export.")
    csv_path = (path.parent / manifest["csv_path"]).resolve()
    if sha256(csv_path) != manifest["csv_sha256"]:
        raise ValueError("Source CSV SHA-256 differs from its manifest.")
    return manifest, csv_path


def run_validation(
    manifest_path: Path, *, as_of: str, output_dir: Path, root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    manifest_hash = sha256(manifest_path)
    manifest, csv_path = read_source_manifest(manifest_path)
    if sha256(manifest_path) != manifest_hash:
        raise RuntimeError("Source manifest changed while reading it.")
    source_hash = manifest["csv_sha256"]
    # Never overwrite an earlier frozen comparison or an existing dataset.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty; prior runs are immutable.")
    protected = (*REFERENCE_FILES.values(), *RAW_REFERENCE_FILES, *MODEL_FILES, *CODE_FILES,
                 "data/processed/combined_2026/listings_clean.csv",
                 "reports/combined_update/split_manifest.csv", "reports/location_experiment/experiment_plan.json")
    hashes = {relative: sha256(root / relative) for relative in protected}

    def verify_inputs() -> None:
        if (any(sha256(root / relative) != digest for relative, digest in hashes.items())
                or sha256(csv_path) != source_hash or sha256(manifest_path) != manifest_hash):
            raise RuntimeError("Protected inputs changed during validation; do not use this run.")

    # Snapshot every input before parsing, and verify again before any prediction.
    raw = pd.read_csv(csv_path, low_memory=False)
    required = {*FEATURE_COLUMNS, "price", "url", "collected_at", "address", "residential_complex"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing export columns: {sorted(missing)}")
    references = {name: pd.read_csv(root / relative, low_memory=False)
                  for name, relative in REFERENCE_FILES.items()}
    raw_urls: list[str] = []
    for relative in RAW_REFERENCE_FILES:
        reference = pd.read_csv(root / relative, usecols=["url"])
        raw_urls.extend(reference["url"].dropna().astype(str))
    result = prepare_fresh_holdout(
        raw, reference_frames=references, raw_reference_urls=raw_urls,
        as_of=as_of, cutoff="2026-03-31", source_observed_column="collected_at",
    )
    frame = result["frame"]
    primary = frame.loc[frame["primary_eligible"].astype(bool)].copy()
    verify_inputs()
    protocol = {
        "schema_version": 1, "as_of": as_of, "cutoff": "2026-03-31",
        "source_manifest": manifest, "source_manifest_sha256": manifest_hash,
        "source_csv_sha256": source_hash, "protected_hashes": hashes,
        "models": ["january", "combined", "complex"], "fit_allowed": False,
        "selection": "valid observation date, clean apartment, valid URL and complete address signature; no historical identity match; stable first representative",
        "deduplication": "URL + legacy apartment key + canonical house signature; transitive; no price in identity keys",
        "metrics": ["mae_tenge", "rmse_tenge", "r2", "absolute_percentage_error_gt_20_percent"],
        "segments": ["district", "fixed_price_band", "feature_completeness", "complex_known_missing_new"],
        "activation": "No model activation or training in this workflow",
        "primary_record_ids": primary["record_id"].astype(str).tolist(),
        "preparation_summary": result["summary"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "protocol.json", protocol)
    for key, filename in (("frame", "candidates.csv"), ("rejected", "rejected.csv"), ("audit", "deduplication_audit.csv")):
        result[key].to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    primary.to_csv(output_dir / "primary_holdout.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "preparation.json", result["summary"])
    if len(primary) < 2:
        status = {"status": "insufficient_independent_rows", "primary_rows": len(primary),
                  "evaluation_run": False, "fit_calls": 0,
                  "reason": "At least two independently eligible rows are required; no fresh metrics were calculated."}
    else:
        provenance = {
            "source_manifest": manifest, "source_csv_sha256": source_hash,
            "source_manifest_sha256": manifest_hash,
            "protocol_sha256": sha256(output_dir / "protocol.json"),
            "preparation_summary": result["summary"], "as_of": as_of, "cutoff": "2026-03-31",
            "limitations": [
                "Publisher provenance is declared in the reviewed source manifest, not authenticated by this program.",
                "URL and feature matching are conservative heuristics, not proof of unique physical apartments.",
                "Primary metrics cover only identifiable eligible rows; exclusions can change market representation.",
                "Targets are asking prices, not transaction prices; this is not a six-month price forecast.",
            ],
        }
        evaluate_frozen(primary, output_dir=output_dir / "evaluation", provenance=provenance,
                        config=FrozenEvaluationConfig(root=root))
        status = {"status": "evaluated", "primary_rows": len(primary), "evaluation_run": True, "fit_calls": 0}
    verify_inputs()
    status["protected_inputs_unchanged"] = True
    write_json(output_dir / "status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--as-of", required=True, help="Explicit latest allowed observation date, YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty run directory")
    args = parser.parse_args()
    status = run_validation(args.source_manifest, as_of=args.as_of, output_dir=args.output_dir)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["evaluation_run"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

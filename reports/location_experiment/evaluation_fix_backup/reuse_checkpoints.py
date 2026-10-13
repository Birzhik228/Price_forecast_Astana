"""Audited one-time reuse: change only frozen-model inference null handling.

Run only against the preserved pre-fix snapshot. Never fits or writes joblibs.
The full training procedure and every other pinned source must be unchanged.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKUP = Path(__file__).resolve().parent
REPORTS = BACKUP.parent
MODELS = ROOT / "models/location_2026"
VARIANTS = ("baseline8", "complex", "coords", "complex_coords")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    old = (BACKUP / "location_experiment.py").read_bytes()
    revised = old
    for name in ("january", "combined"):
        previous = f'{name}.predict(frame.loc[test_ids, list(FEATURE_COLUMNS)])'.encode()
        fixed = f'{name}.predict(x.loc[test_ids, list(FEATURE_COLUMNS)])'.encode()
        assert revised.count(previous) == 1
        revised = revised.replace(previous, fixed)
    assert revised == (ROOT / "location_experiment.py").read_bytes(), "Change is not limited to the two frozen-model inference calls"
    original = read(BACKUP / "experiment_plan.json")
    assert read(REPORTS / "experiment_plan.json") == original, "Already migrated or changed plan"
    assert sha(BACKUP / "location_experiment.py") == original["sources"]["location_experiment.py"]
    for name, checksum in original["sources"].items():
        if name != "location_experiment.py":
            assert sha(ROOT / name) == checksum, f"Training source changed: {name}"
    assert not (REPORTS / "experiment_report.json").exists(), "Final report must not exist before the fix"
    plan = {key: value for key, value in original.items() if key != "signature"}
    plan["sources"] = {**plan["sources"], "location_experiment.py": sha(ROOT / "location_experiment.py")}
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["signature"] = signature
    changes, hashes = [], {}
    for variant in VARIANTS:
        cv_path = REPORTS / f"cv_{variant}.json"
        cv = read(cv_path)
        assert cv == read(BACKUP / cv_path.name) and cv["signature"] == original["signature"]
        cv["fitted_under_signature"] = cv["signature"]
        cv["signature"] = signature
        changes.append((cv_path, cv))
        metadata_path = MODELS / f"{variant}.json"
        metadata = read(metadata_path)
        assert metadata == read(BACKUP / f"{variant}_metadata.json")
        assert metadata["experiment_signature"] == original["signature"]
        hashes[variant] = sha(MODELS / f"{variant}.joblib")
        assert hashes[variant] == metadata["model_sha256"]
        metadata["fitted_under_experiment_signature"] = metadata["experiment_signature"]
        metadata["experiment_signature"] = signature
        changes.append((metadata_path, metadata))
    for path, content in changes:
        write(path, content)
    write(REPORTS / "experiment_plan.json", plan)
    assert all(sha(MODELS / f"{name}.joblib") == value for name, value in hashes.items())
    write(REPORTS / "checkpoint_reuse.json", {
        "status": "completed", "at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Frozen January/combined inference requires numpy NaN instead of pandas NA; reuse already prepared eight feature columns.",
        "original_signature": original["signature"], "current_signature": signature,
        "only_code_change": "Two frozen predict inputs: frame.loc -> x.loc, after all CV and final candidate fits.",
        "training_source_code_identical": True, "all_other_pinned_sources_unchanged": True,
        "model_joblibs_unchanged_sha256": hashes, "fit_calls": 0,
        "original_artifacts_directory": "evaluation_fix_backup",
        "candidate_and_cv_metadata_retain_original_fit_signature": True,
    })
    print("Inference-only checkpoint reuse recorded; joblibs unchanged.")


if __name__ == "__main__":
    main()

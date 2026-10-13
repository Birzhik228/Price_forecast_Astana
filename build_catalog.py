"""Построить локальный каталог архивных домов/объявлений без сетевых запросов."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from building_catalog import CATALOG_POLICY_VERSION, build_property_catalog


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_previous_catalog(output_dir: Path, report_dir: Path) -> dict:
    """Preserve the entire prior stage once; never overwrite an existing snapshot."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("summary", {}).get("catalog_policy_version", 1) >= CATALOG_POLICY_VERSION:
        return previous.get("previous_stage_archives", {})
    result = {}
    for label, source in [("catalog", output_dir), ("reports", report_dir)]:
        if not source.exists():
            continue
        checksums = {str(path.relative_to(source)): _sha256(path)
                     for path in sorted(source.rglob("*")) if path.is_file()}
        destination = source.parent / "archive" / f"{source.name}_before_alias_v2"
        if destination.exists():
            if any(not (destination / rel).is_file() or _sha256(destination / rel) != checksum
                   for rel, checksum in checksums.items()):
                suffix = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()[:12]
                destination = destination.with_name(destination.name + "_" + suffix)
        if not destination.exists():
            shutil.copytree(source, destination)
        for rel, checksum in checksums.items():
            if not (destination / rel).is_file() or _sha256(destination / rel) != checksum:
                raise ValueError(f"Не удалось подтвердить архив: {destination / rel}")
        snapshot = destination / "snapshot_checksums.json"
        if not snapshot.exists():
            snapshot.write_text(json.dumps({"source": str(source.resolve()), "files": checksums},
                                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result[label] = {"path": str(destination.resolve()), "files": checksums}
    return result


def build_catalog(
    combined_path: Path, january_path: Path, march_path: Path,
    output_dir: Path, report_dir: Path,
) -> dict:
    """Сохранить две CSV, manifest и отчёт качества; вернуть manifest."""
    archives = _snapshot_previous_catalog(output_dir, report_dir)
    inputs = {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in [
        ("combined", combined_path), ("january_raw", january_path), ("march_raw", march_path),
    ]}
    result = build_property_catalog(pd.read_csv(combined_path),
                                    pd.read_csv(january_path, dtype=str, keep_default_na=False),
                                    pd.read_csv(march_path, dtype=str, keep_default_na=False))
    # Detect a source replacement while reading; provenance hashes cover exact bytes used.
    for name, path in [("combined", combined_path), ("january_raw", january_path), ("march_raw", march_path)]:
        if _sha256(path) != inputs[name]["sha256"]:
            raise ValueError(f"Вход {name} изменился во время сборки.")
    targets = [output_dir / "listings.csv", output_dir / "buildings.csv", output_dir / "manifest.json",
               report_dir / "quality.json", report_dir / "issues.csv",
               report_dir / "normalization_audit.csv", report_dir / "conflict_audit.csv",
               report_dir / "normalization_report.json"]
    source_paths = {path.resolve() for path in (combined_path, january_path, march_path)}
    if any(path.resolve() in source_paths for path in targets):
        raise ValueError("Каталог не должен перезаписывать исходные CSV.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("listings", "buildings"):
        path = output_dir / f"{name}.csv"
        result[name].to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        files[name] = {"path": path.name, "sha256": _sha256(path), "rows": len(result[name])}
    for name in ("issues", "normalization_audit", "conflict_audit"):
        result[name].to_csv(report_dir / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n")
    previous_summary = {}
    if archives.get("catalog"):
        previous_manifest = Path(archives["catalog"]["path"]) / "manifest.json"
        previous_summary = json.loads(previous_manifest.read_text(encoding="utf-8")).get("summary", {})
    normalization_report = {
        "schema_version": 1, "catalog_policy_version": CATALOG_POLICY_VERSION,
        "normalization": result["summary"]["normalization"],
        "candidate_conflicts": result["summary"]["candidate_conflicts"],
        "ambiguous_candidate_count": result["summary"]["ambiguous_candidate_count"],
        "ambiguous_listing_count": result["summary"]["ambiguous_listing_count"],
        "comparison_to_previous": {field: {"previous": previous_summary.get(field), "current": result["summary"][field]}
                                   for field in ["listing_count", "building_count", "listings_with_building_id",
                                                 "coordinate_status_counts", "coordinate_inherited_count"]},
        "archives": archives, "policy": result["summary"]["policy"],
        "audit_files": {name: {"path": f"{name}.csv", "sha256": _sha256(report_dir / f"{name}.csv"), "rows": len(result[name])}
                        for name in ("normalization_audit", "conflict_audit")},
    }
    (report_dir / "normalization_report.json").write_text(json.dumps(normalization_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quality = {"schema_version": 1, "summary": result["summary"], "inputs": inputs,
               "previous_stage_archives": archives}
    (report_dir / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                "files": files, "inputs": inputs, "summary": result["summary"],
                "previous_stage_archives": archives,
                "row_local_training_contract": {
                    "house_key": "Pointwise canonical_house_key(address,district,residential_complex); not global building_id.",
                    "raw_latitude/raw_longitude": "Own raw January coordinates only; March missing; no inheritance or aggregation.",
                    "raw_coordinate_status": "Pointwise bbox validity: unverified/invalid/missing. Never plausible from global consensus.",
                    "raw_*_text/raw_address/raw_title/raw_residential_complex/raw_parking": "Preserved source values; combined original columns unchanged.",
                    "forbidden_for_cv": ["building_id", "building_match_status", "building_conflict_reasons", "latitude", "longitude", "coordinate_status"],
                },
                "reports": {name: {"path": str((report_dir / name).resolve()), "sha256": _sha256(report_dir / name)}
                            for name in ("normalization_audit.csv", "conflict_audit.csv", "normalization_report.json", "quality.json", "issues.csv")}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, default=ROOT / "data/processed/combined_2026/listings_clean.csv")
    parser.add_argument("--january-raw", type=Path, default=ROOT / "data/raw/kaggle_astana/astana_apartments.csv")
    parser.add_argument("--march-raw", type=Path, default=ROOT / "data/raw/turarr_march_2026/astana_apartments_raw.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/property_catalog")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/property_catalog")
    args = parser.parse_args()
    manifest = build_catalog(args.combined, args.january_raw, args.march_raw, args.output_dir, args.report_dir)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

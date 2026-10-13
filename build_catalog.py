"""Построить локальный каталог архивных домов/объявлений без сетевых запросов."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from building_catalog import build_property_catalog


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_catalog(
    combined_path: Path, january_path: Path, march_path: Path,
    output_dir: Path, report_dir: Path,
) -> dict:
    """Сохранить две CSV, manifest и отчёт качества; вернуть manifest."""
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
               report_dir / "quality.json", report_dir / "issues.csv"]
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
    result["issues"].to_csv(report_dir / "issues.csv", index=False, encoding="utf-8", lineterminator="\n")
    quality = {"schema_version": 1, "summary": result["summary"], "inputs": inputs}
    (report_dir / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                "files": files, "inputs": inputs, "summary": result["summary"]}
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

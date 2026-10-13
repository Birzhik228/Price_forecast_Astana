"""Build a separately versioned address registry from a pinned local OSM snapshot.

No prices, model fitting, original catalogue changes, or network access here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from verified_addresses import match_addresses


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "coordinate_enrichment_2026_10_13"
SOURCE = ROOT / "data" / "raw" / "osm_2026_10_13"
OUTPUT = ROOT / "data" / "processed" / "verified_coordinates_2026_10_13"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    """Idempotent only for identical bytes; never silently replace an audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Existing artifact differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def make_registry(matches: pd.DataFrame, provenance: dict) -> dict:
    """Only explicit, unique, eligible exact matches may reach inference."""
    entries = {}
    for row in json.loads(matches.to_json(orient="records", force_ascii=False, double_precision=15)):
        if row.get("source_status") != "osm_exact_address":
            raise ValueError("All coordinate matches must identify the exact OSM address method")
        if row.get("eligible_for_enrichment") is not True:
            continue
        key = row.get("house_key")
        if not isinstance(key, str) or key in entries:
            raise ValueError("Missing or duplicate matched house key")
        entries[key] = {column: value for column, value in row.items() if column != "house_key"}
    return {"schema_version": 1, "created_at_utc": provenance["created_at_utc"],
            "method": "Unique exact street and house number on an individual OSM building footprint",
            "limitations": ["OSM address matching is not cadastral or on-site verification.",
                            "Points represent buildings, not entrances or apartment positions.",
                            "Original model coordinates and unresolved house identity conflicts are not overridden.",
                            "October map data do not establish when a building or address existed."],
            "license": "ODbL-1.0", "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
            "attribution": "© OpenStreetMap contributors",
            "attribution_url": "https://www.openstreetmap.org/copyright",
            "provenance": provenance, "entries": dict(sorted(entries.items()))}


def run(*, root: Path = ROOT, reports: Path = REPORTS, source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    candidate_path = reports / "address_candidates.csv"
    source_path = source / "astana_addresses.json"
    extraction = json.loads((source / "astana_addresses.source.json").read_text(encoding="utf-8"))
    source_manifest_path = source / "kazakhstan-261012.source.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if sha256(source_path) != extraction["output_sha256"]:
        raise ValueError("OSM extraction SHA mismatch")
    if extraction["input_sha256"] != source_manifest["sha256"]:
        raise ValueError("OSM PBF/extraction source mismatch")
    if sha256(source / source_manifest["snapshot_filename"]) != source_manifest["sha256"]:
        raise ValueError("OSM PBF SHA mismatch")
    audit = json.loads((reports / "audit_summary.json").read_text(encoding="utf-8"))
    if sha256(candidate_path) != audit["candidate_export"]["sha256"]:
        raise ValueError("Address candidates differ from the audited export")
    for relative, expected in audit["source_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Original artifact changed since audit: {relative}")
    paths = [candidate_path, source_path, source_manifest_path, source / "astana_addresses.source.json",
             reports / "audit_summary.json", root / "verified_addresses.py", root / "enrich_coordinates.py"]
    signatures = {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in paths}
    protocol_path = reports / "address_matching_protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol["source_sha256"] != signatures:
            raise ValueError("Inputs or matcher changed after protocol freeze; use a new versioned output directory")
    else:
        protocol = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_sha256": signatures,
                    "selection_uses_price_or_prediction_error": False,
                    "policy": ["Exact street + house number on one building footprint within Astana.",
                               "Street language aliases require evidence on the same OSM road object.",
                               "No fuzzy addresses, district guessing, ambiguous building footprints or unreviewed renamings.",
                               "Keep all refusals and confirmed-but-ineligible addresses in audit files."]}
        write_new(protocol_path, json_bytes(protocol))
    candidates = pd.read_csv(candidate_path)
    if len(candidates) != audit["candidate_export"]["rows"]:
        raise ValueError("Audited candidate row count changed")
    if any(column in candidates for column in ("price", "price_per_m2", "prediction", "absolute_error")):
        raise ValueError("Address matching input must not contain price or prediction labels")
    snapshot = json.loads(source_path.read_text(encoding="utf-8"))
    result = match_addresses(candidates, snapshot)
    registry = make_registry(result["matches"], protocol)
    from verified_geo_model import _validate_entries
    _validate_entries(registry)
    # Catch input changes during the local pass before committing its outputs.
    if signatures != {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in paths}:
        raise ValueError("Matching inputs changed during run")
    for relative, expected in audit["source_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Original artifact changed during matching: {relative}")
    write_new(output / "registry.json", json_bytes(registry))
    write_new(reports / "matched_addresses.csv", result["matches"].to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(reports / "address_matching_audit.csv", result["audit"].to_csv(index=False, lineterminator="\n").encode("utf-8"))
    summary = {"schema_version": 1, "matching": result["summary"], "registry_entries": len(registry["entries"]),
               "registry_path": str((output / "registry.json").relative_to(root)).replace("\\", "/"),
               "registry_sha256": sha256(output / "registry.json"), "protocol_sha256": sha256(protocol_path),
               "artifacts_sha256": {name: sha256(reports / name) for name in ("matched_addresses.csv", "address_matching_audit.csv")},
               "original_artifacts_unchanged": True}
    write_new(reports / "matching_summary.json", json_bytes(summary))
    write_new(output / "ATTRIBUTION.txt", (
        "© OpenStreetMap contributors. OpenStreetMap-derived address and coordinate database, ODbL-1.0.\n"
        "https://www.openstreetmap.org/copyright\nhttps://opendatacommons.org/licenses/odbl/1-0/\n"
        "Source: https://download.geofabrik.de/asia/kazakhstan-261012.osm.pbf\n"
        "The registry contains a representative point inside a uniquely matched building footprint; "
        "it does not establish cadastral, apartment or entrance accuracy.\n"
    ).encode("utf-8"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(reports=args.reports_dir.resolve(), source=args.source_dir.resolve(), output=args.output_dir.resolve()),
                     ensure_ascii=False, indent=2))

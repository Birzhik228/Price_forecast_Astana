"""Reproduce the local coordinate audit without fitting or changing old files.

The original October audit was run interactively; this retains its calculations
as a readable program. An existing output is never replaced. Use --output-dir
with a new directory for a later audit; timestamps then describe that new run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from building_catalog import ASTANA_BBOX, MAX_COORDINATE_RADIUS_M, canonical_house_key
from location_model import MIN_COORDINATE_OBSERVATIONS, prepare_location_input


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports/coordinate_enrichment_2026_10_13"
INPUTS = [
    "models/location_2026/complex_coords.joblib", "models/deployment/complex_coords.json",
    "models/location_2026/complex_coords.json", "data/processed/property_catalog/listings.csv",
    "data/processed/property_catalog/buildings.csv", "data/processed/property_catalog/manifest.json",
    "data/processed/combined_2026/listings_clean.csv", "reports/location_experiment/split_manifest.csv",
    "reports/location_experiment/training_coordinate_audit.csv", "data/raw/kaggle_astana/astana_apartments.csv",
    "data/raw/turarr_march_2026/astana_apartments_raw.csv", "building_catalog.py", "location_model.py",
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_values(series: pd.Series) -> list:
    return sorted(series.dropna().unique().tolist(), key=str)


def packed(series: pd.Series) -> str:
    return json.dumps(unique_values(series), ensure_ascii=False, allow_nan=False)


def nullable(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def counts(series: pd.Series) -> dict:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def coverage(part: pd.DataFrame) -> dict:
    return {"rows": len(part), "house_keys": int(part.house_key.nunique()),
            "rows_with_house_key": int(part.house_key.notna().sum()), "rows_without_house_key": int(part.house_key.isna().sum()),
            "model_mapped_house_rows": int(part.model_house_mapped.sum()), "actual_model_geo_rows": int(part.model_geo_available.sum()),
            "actual_model_geo_pct": float(part.model_geo_available.mean() * 100) if len(part) else 0.,
            "query_trait_conflict_rows": int(part.model_query_trait_conflict.sum()), "valid_raw_coordinate_rows": int(part.raw_bbox_valid.sum()),
            "model_house_status_row_counts": counts(part.model_house_status)}


def build_candidates(frame: pd.DataFrame, transformer) -> pd.DataFrame:
    audit = transformer.coordinate_audit_.set_index("house_key")
    rows = []
    for key, part in frame.dropna(subset=["house_key"]).groupby("house_key", sort=True):
        item = audit.loc[key] if key in audit.index else None
        valid = part.loc[part.raw_bbox_valid]
        points = valid[["raw_latitude", "raw_longitude"]].drop_duplicates().sort_values(["raw_latitude", "raw_longitude"])
        conflicts = sorted({reason for value in part.building_conflict_reasons.dropna().astype(str)
                            for reason in value.split(";") if reason})
        review = bool(conflicts or part.same_address_multiple_districts.any())
        mapped = key in transformer.house_coordinate_map_
        tier = "existing_model_coordinate_check" if mapped else ("verify_house_identity_first" if review else "address_ready_coordinate_lookup")
        sources = [{"record_id": str(row.record_id), "dataset_period": str(row.dataset_period),
                    "original_source_row": int(row.original_source_row), "subset": str(row.subset)}
                   for row in part.sort_values(["dataset_period", "original_source_row"]).itertuples()]
        rows.append({
            "house_key": key, "address_normalized": key.split("|", 2)[2], "district": key.split("|", 2)[1],
            "city": "Астана", "candidate_tier": tier, "external_verification_status": "not_started",
            "listing_count": len(part), "train_listing_count": int(part.train_row.sum()), "test_listing_count": int(part.test_row.sum()),
            "dataset_periods_json": packed(part.dataset_period), "source_records_json": json.dumps(sources, ensure_ascii=False),
            "listing_urls_json": packed(part.url), "address_originals_json": packed(part.address_original),
            "raw_addresses_json": packed(part.raw_address), "raw_titles_json": packed(part.raw_title),
            "complex_names_json": packed(part.complex_name), "year_built_values_json": packed(part.year_built),
            "total_floors_values_json": packed(part.total_floors), "building_type_values_json": packed(part.building_type),
            "resolved_building_ids_json": packed(part.building_id), "catalog_building_statuses_json": packed(part.building_match_status),
            "catalog_conflict_reasons_json": json.dumps(conflicts, ensure_ascii=False),
            "same_normalized_address_multiple_districts": bool(part.same_address_multiple_districts.any()),
            "catalog_coordinate_statuses_json": packed(part.coordinate_status), "raw_coordinate_statuses_json": packed(part.raw_coordinate_status),
            "valid_raw_coordinate_rows": len(valid), "distinct_valid_raw_coordinate_pairs": len(points),
            "raw_coordinate_pairs_json": json.dumps(points.to_numpy().tolist(), ensure_ascii=False),
            "model_house_status": str(item.coordinate_status) if item is not None else "unseen_house",
            "model_house_mapped": mapped, "actual_model_geo_rows": int(part.model_geo_available.sum()),
            "model_query_trait_conflict_rows": int(part.model_query_trait_conflict.sum()),
            "train_coordinate_observations": int(item.coordinate_observations) if item is not None else 0,
            "train_coordinate_spread_m": nullable(item.coordinate_spread_m) if item is not None else None,
            "train_has_invalid_coordinate": bool(item.has_invalid) if item is not None else False,
            "train_has_reused_marker": bool(item.has_reused_marker) if item is not None else False,
            "train_conflicting_traits": str(item.conflicting_traits) if item is not None else "",
            "model_latitude": transformer.house_coordinate_map_[key][0] if mapped else None,
            "model_longitude": transformer.house_coordinate_map_[key][1] if mapped else None,
        })
    result = pd.DataFrame(rows)
    assert result.house_key.is_unique and len(result) == frame.house_key.nunique()
    assert result.listing_count.sum() == frame.house_key.notna().sum()
    return result


def run(*, root: Path = ROOT, output: Path = OUTPUT) -> dict:
    csv_path, json_path = output / "address_candidates.csv", output / "audit_summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Existing audit outputs must be preserved; choose a new --output-dir.")
    hashes = {name: digest(root / name) for name in INPUTS}
    metadata = json.loads((root / "models/deployment/complex_coords.json").read_text(encoding="utf-8"))
    assert hashes[INPUTS[0]] == metadata["model_sha256"]
    manifest = json.loads((root / "data/processed/property_catalog/manifest.json").read_text(encoding="utf-8"))
    for name in ("listings", "buildings"):
        assert hashes[f"data/processed/property_catalog/{name}.csv"] == manifest["files"][name]["sha256"]
    transformer = joblib.load(root / INPUTS[0]).named_steps["location_features"]
    frame = pd.read_csv(root / "data/processed/property_catalog/listings.csv", low_memory=False)
    buildings = pd.read_csv(root / "data/processed/property_catalog/buildings.csv", low_memory=False)
    split = pd.read_csv(root / "reports/location_experiment/split_manifest.csv").set_index("record_id")
    assert frame.record_id.is_unique and split.index.is_unique and set(frame.record_id) == set(split.index)
    frame["subset"] = frame.record_id.map(split.subset)
    audit = transformer.coordinate_audit_.copy()
    saved = pd.read_csv(root / "reports/location_experiment/training_coordinate_audit.csv")
    pd.testing.assert_frame_equal(audit.reset_index(drop=True), saved[audit.columns].fillna({"conflicting_traits": ""}).reset_index(drop=True),
                                  check_dtype=False, rtol=1e-10, atol=1e-10)
    assert set(audit.loc[audit.coordinate_status.eq("plausible"), "house_key"]) == set(transformer.house_coordinate_map_)
    computed = pd.Series([canonical_house_key(row.address, row.district, row.residential_complex) for row in frame.itertuples()])
    mismatch = ~computed.fillna("").eq(frame.house_key.fillna(""))
    assert not mismatch.any()
    frame["model_house_status"] = frame.house_key.map(audit.set_index("house_key").coordinate_status).fillna("unseen_house")
    frame.loc[frame.house_key.isna(), "model_house_status"] = "house_key_missing"
    frame["model_house_mapped"] = frame.house_key.isin(transformer.house_coordinate_map_)
    features = prepare_location_input(frame)
    frame["model_geo_available"] = transformer.transform(features)[["latitude", "longitude"]].notna().all(axis=1)
    frame["model_query_trait_conflict"] = frame.model_house_mapped & ~frame.model_geo_available
    frame["raw_bbox_valid"] = (frame.raw_coordinate_status.eq("unverified")
                              & frame.raw_latitude.between(ASTANA_BBOX["latitude_min"], ASTANA_BBOX["latitude_max"])
                              & frame.raw_longitude.between(ASTANA_BBOX["longitude_min"], ASTANA_BBOX["longitude_max"]))
    frame["train_row"], frame["test_row"] = frame.subset.eq("train"), frame.subset.eq("test")
    raw = pd.read_csv(root / "data/raw/kaggle_astana/astana_apartments.csv", low_memory=False)
    january = frame.loc[frame.dataset_period.eq("2026-01")]
    source = raw.iloc[january.original_source_row.astype(int).to_numpy() - 1]
    for column in ("latitude", "longitude"):
        assert np.allclose(january[f"raw_{column}"].to_numpy(float), pd.to_numeric(source[column], errors="coerce").to_numpy(float),
                           rtol=0, atol=1e-12, equal_nan=True)
    assert frame.loc[frame.dataset_period.eq("2026-03"), ["raw_latitude", "raw_longitude"]].isna().all().all()
    valid = frame.loc[frame.train_row & frame.house_key.notna() & frame.raw_bbox_valid].copy()
    valid["rounded_latitude"], valid["rounded_longitude"] = valid.raw_latitude.round(6), valid.raw_longitude.round(6)
    markers = valid.groupby(["rounded_latitude", "rounded_longitude"]).agg(rows=("record_id", "size"), houses=("house_key", "nunique"),
                                                                          normalized_addresses=("address_normalized", "nunique"), districts=("district", "nunique"))
    reused = markers.loc[markers.houses.gt(1)]
    valid["reused_marker"] = pd.Series(list(zip(valid.rounded_latitude, valid.rounded_longitude)), index=valid.index).isin(set(reused.index))
    reused_house_keys = set(valid.loc[valid.reused_marker, "house_key"])
    assert reused_house_keys == set(audit.loc[audit.has_reused_marker, "house_key"])
    raw_markers = raw.groupby(["latitude", "longitude"]).agg(rows=("url", "size"), addresses=("address", "nunique"),
                                                            complexes=("residential_complex_name", "nunique")).sort_values("rows", ascending=False)
    top_raw = []
    for (latitude, longitude), item in raw_markers.head(20).iterrows():
        entries = raw.loc[raw.latitude.eq(latitude) & raw.longitude.eq(longitude)]
        top_raw.append({"latitude": float(latitude), "longitude": float(longitude), "rows": int(item["rows"]),
                        "address_texts": int(item.addresses), "complex_names": unique_values(entries.residential_complex_name),
                        "address_examples": unique_values(entries.address)[:12]})
    same_address = frame.dropna(subset=["house_key"]).groupby("address_normalized").agg(
        house_keys=("house_key", "nunique"), districts=("district", "nunique"), rows=("record_id", "size"))
    cross_district = same_address.loc[same_address.districts.gt(1)]
    frame["same_address_multiple_districts"] = frame.address_normalized.isin(set(cross_district.index)) & frame.house_key.notna()
    candidates = build_candidates(frame, transformer)
    missing = frame.loc[frame.house_key.isna()]
    missing_examples = []
    for _, part in missing.groupby(["address_original", "address_status", "district"], dropna=False, sort=False):
        first = part.iloc[0]
        missing_examples.append({"address": nullable(first.address_original), "address_status": str(first.address_status),
                                 "district": nullable(first.district), "rows": len(part), "record_id_example": str(first.record_id),
                                 "dataset_periods": unique_values(part.dataset_period)})
    missing_examples.sort(key=lambda item: (-item["rows"], str(item["address"])))
    trait_rejections = []
    for row in frame.loc[frame.model_query_trait_conflict].itertuples():
        known, prepared = transformer.house_traits_[row.house_key], features.loc[row.Index]
        reasons = []
        for field in ("residential_complex", "year_built", "total_floors"):
            value = nullable(prepared[field])
            if known.get(field) is not None and value is not None and known[field] != value:
                reasons.append({"field": field, "train_value": known[field], "query_value": value})
        trait_rejections.append({"record_id": row.record_id, "house_key": row.house_key, "conflicting_traits": reasons})
    districts = []
    for district, part in frame.groupby("district", dropna=False):
        local = audit.loc[audit.house_key.str.split("|").str[1].eq(str(district).casefold())]
        districts.append({"district": nullable(district), **coverage(part), "training_house_status_counts": counts(local.coordinate_status),
                          "by_subset": {subset: coverage(group) for subset, group in part.groupby("subset")}})
    flags = {"trait_conflict_houses": int(audit.conflicting_traits.ne("").sum()), "invalid_evidence_houses": int(audit.has_invalid.sum()),
             "reused_marker_houses": int(audit.has_reused_marker.sum()), "zero_valid_points_houses": int(audit.coordinate_observations.eq(0).sum()),
             "one_valid_point_houses": int(audit.coordinate_observations.eq(1).sum()),
             "spread_above_125m_houses": int(audit.coordinate_spread_m.gt(MAX_COORDINATE_RADIUS_M).sum())}
    summary = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read_only_existing_complex_coords_and_catalog; no fitting or external lookups",
        "active_model": {"variant": "complex_coords", "model_path": INPUTS[0], "sha256": metadata["model_sha256"],
                         "training_geo_summary": transformer.training_geo_summary_}, "source_sha256": hashes,
        "policy": {"minimum_coordinate_observations": MIN_COORDINATE_OBSERVATIONS, "maximum_coordinate_radius_m": MAX_COORDINATE_RADIUS_M,
                   "bbox": ASTANA_BBOX, "reused_marker_rounding_decimals": 6, "external_coordinate_source_used": False,
                   "status_precedence": ["trait_conflict", "invalid", "reused_marker", "missing", "insufficient_observations", "coordinate_spread", "plausible"]},
        "coverage": {"catalog": coverage(frame), "by_subset": {name: coverage(part) for name, part in frame.groupby("subset")},
                     "by_dataset_period": {name: coverage(part) for name, part in frame.groupby("dataset_period")}},
        "training_house_status_counts": counts(audit.coordinate_status), "nonexclusive_training_rejection_flags": flags,
        "training_trait_conflict_combinations": counts(audit.loc[audit.coordinate_status.eq("trait_conflict"), "conflicting_traits"]),
        "districts": districts, "catalog_coordinate_status_rows": counts(frame.coordinate_status),
        "catalog_resolved_building_coordinate_status_counts": counts(buildings.coordinate_status),
        "catalog_and_model_are_distinct": {"catalog_resolved_buildings": len(buildings), "model_train_house_keys": len(audit),
            "all_catalog_candidate_house_keys": len(candidates), "model_plausible_houses": len(transformer.house_coordinate_map_),
            "catalog_plausible_resolved_buildings": int(buildings.coordinate_status.eq("plausible").sum()),
            "note": "Catalog resolves all-sample identity including building_type; model uses train-only candidate keys and traits complex/year/floors. Counts need not coincide."},
        "query_trait_rejections": trait_rejections,
        "missing_house_key": {"rows": len(missing), "by_address_status": counts(missing.address_status),
            "by_period": {name: counts(part.address_status) for name, part in missing.groupby("dataset_period")},
            "top_address_examples": missing_examples[:40],
            "candidate_csv_exclusion_reason": "No canonical house key; requires address resolution first, not a coordinate-only lookup."},
        "marker_reuse": {"raw_january_rows": len(raw), "raw_distinct_exact_points": len(raw_markers),
            "max_exact_point_rows": int(raw_markers.rows.max()), "max_exact_point_share_of_raw_pct": float(raw_markers.rows.max() / len(raw) * 100),
            "top_exact_raw_points": top_raw, "valid_train_known_house_marker_count": len(markers), "reused_rounded_marker_count": len(reused),
            "reused_rounded_marker_rows": int(reused.rows.sum()), "reused_marker_house_count": len(reused_house_keys),
            "reused_markers_with_identical_normalized_address_but_multiple_districts": int((reused.normalized_addresses.eq(1) & reused.districts.gt(1)).sum()),
            "inference": "Top repeated points each belong to one complex and many address texts; compatible with complex-level map pins, not proof of an application-wide default. Raw provenance cannot prove the original geocoding method."},
        "address_normalization_audit": {"recomputed_key_mismatches": int(mismatch.sum()), "same_normalized_address_multiple_districts_count": len(cross_district),
            "affected_rows": int(frame.same_address_multiple_districts.sum()), "affected_house_keys": int(frame.loc[frame.same_address_multiple_districts, "house_key"].nunique()),
            "top_cross_district_addresses": cross_district.sort_values("rows", ascending=False).head(30).reset_index().to_dict("records"),
            "finding": "No pointwise key recomputation error found. Cross streets, multiple house numbers, repeated number text, and district differences need independent identity verification. No aliases were merged."},
        "candidate_export": {"path": str(csv_path.relative_to(root)).replace("\\", "/") if csv_path.is_relative_to(root) else str(csv_path),
            "rows": len(candidates), "listing_rows_represented": int(candidates.listing_count.sum()), "tiers": counts(candidates.candidate_tier),
            "price_or_target_columns": False, "source_row_semantics": "1-based original raw input position, separated by dataset_period; source_records_json preserves every linked listing."},
        "checks": {"active_model_hash_matches_deployment": True, "catalog_csv_hashes_match_manifest": True,
            "serialized_model_audit_matches_saved_training_audit": True, "raw_january_coordinate_provenance_matches": True,
            "march_has_no_raw_coordinates": True, "house_keys_recomputed_pointwise_match": True, "train_test_record_ids_match_split": True,
            "fit_calls": 0, "prediction_calls": 0, "transform_calls": 1},
        "recommendations": ["Confirm address, house number/corpus and district against an independent house-level map record before importing coordinates.",
            "Prioritize consistent explicit addresses with insufficient or missing raw coordinates; trait or district conflicts require identity review first.",
            "Record source object ID, exact address, source URL, access date, match method and precision; do not substitute complex-center pins for house coordinates.",
            "An enrichment table does not modify the frozen model house map; any model feature change requires a separate version and evaluation."],
        "limitations": ["Plausible means internally consistent archive evidence, not externally verified geolocation.",
            "Two observations may repeat an upstream map pin; the model policy is conservative but cannot prove independent measurements.",
            "House keys missing from training cannot receive coordinates through this frozen transformer.",
            "This audit does not resolve address ambiguities and does not prove that the 197 houses are all geographically correct."],
    }
    assert hashes == {name: digest(root / name) for name in INPUTS}
    output.mkdir(parents=True, exist_ok=True)
    csv_bytes = candidates.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
    summary["candidate_export"]["sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    summary["checks"]["source_hashes_unchanged_after_audit"] = True
    json_bytes = (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with csv_path.open("xb") as stream:
        stream.write(csv_bytes)
    with json_path.open("xb") as stream:
        stream.write(json_bytes)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(output=arguments.output_dir.resolve())
    print(json.dumps({"candidate_house_keys": result["candidate_export"]["rows"], "coverage": result["coverage"]}, ensure_ascii=False, indent=2))

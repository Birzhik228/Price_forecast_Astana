"""Read-only evidence for street-only addresses mistaken for complete houses.

This module never supplies a house number, modifies the frozen catalog parser,
or resolves building traits. A matched road name supports withdrawing an unsafe
house identity. It does not establish the location of the advertised apartment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

import pandas as pd

_NUMBERED_STREET = re.compile(r"([a-zа-яёәғқңөұүһі])(?:[\s\-‑–]*)([0-9]+)")
_NAME_FIELDS = ("name:ru", "name:kk", "name", "official_name", "alt_name", "short_name")
EVIDENCE_COLUMNS = [
    "record_id", "old_house_key", "address_normalized", "road_type", "road_id",
    "road_name", "road_name_field", "match_type", "osm_url", "source_url",
    "source_sha256", "extract_sha256", "road_evidence_json",
]


def _street_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = " ".join(value.lower().split())
    text = re.sub(r"^(?:улица|ул\.?|көше|street)\s+", "", text)
    return re.sub(r"\s+(?:көшесі|street)$", "", text)


def _numbered_street(value: object) -> tuple[str, tuple[str, str]] | None:
    text = _street_text(value)
    match = _NUMBERED_STREET.fullmatch(text or "")
    if match is None:
        return None
    # Scripts and all number digits (including leading zeros) remain unchanged.
    return text, (match[1], match[2])


def audit_numbered_street_addresses(
    catalog: pd.DataFrame,
    elements: Iterable[Mapping],
    *,
    source_url: str = "",
    source_sha256: str = "",
    extract_sha256: str = "",
) -> pd.DataFrame:
    """Return one deterministic evidence row per unsafe existing house key row.

    A complete address such as ``Е 535 2`` cannot match. Only case, whitespace,
    explicit street labels, and separators between one letter and a number may
    differ. Latin and Cyrillic letters are never substituted. All supporting
    road objects/names are retained in ``road_evidence_json``. Repeated OSM road
    segments do not create repeated catalog rows. This function is pure.
    """
    required = {"record_id", "house_key", "address_normalized"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
    if catalog.record_id.isna().any() or catalog.record_id.duplicated().any():
        raise ValueError("Catalog record_id must be present and unique")
    road_index: dict[tuple[str, str], list[dict]] = {}
    for element in elements:
        tags = element.get("tags", {})
        if element.get("type") != "way" or not isinstance(tags, Mapping) or not tags.get("highway"):
            continue
        try:
            road_id = int(element["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if road_id <= 0:
            continue
        for field in _NAME_FIELDS:
            if not isinstance(tags.get(field), str):
                continue
            for name in tags[field].split(";"):
                parsed = _numbered_street(name)
                if parsed is None:
                    continue
                road_index.setdefault(parsed[1], []).append({
                    "road_type": "way", "road_id": road_id, "road_name": name.strip(),
                    "road_name_field": field, "normalized_road_name": parsed[0],
                    "osm_url": f"https://www.openstreetmap.org/way/{road_id}",
                })
    result = []
    for row in catalog.itertuples(index=False):
        if pd.isna(row.house_key) or not str(row.house_key).strip():
            continue
        parsed = _numbered_street(row.address_normalized)
        if parsed is None or parsed[1] not in road_index:
            continue
        evidence = []
        seen = set()
        for road in road_index[parsed[1]]:
            unique = (road["road_id"], road["road_name_field"], road["road_name"])
            if unique in seen:
                continue
            seen.add(unique)
            evidence.append(dict(road, match_type=("exact_name" if parsed[0] == road["normalized_road_name"]
                                                  else "number_separator_only")))
        evidence.sort(key=lambda item: (
            item["match_type"] != "exact_name", item["road_id"],
            _NAME_FIELDS.index(item["road_name_field"]), item["road_name"],
        ))
        first = evidence[0]
        result.append({
            "record_id": row.record_id, "old_house_key": row.house_key,
            "address_normalized": row.address_normalized,
            **{column: first[column] for column in ("road_type", "road_id", "road_name", "road_name_field", "match_type", "osm_url")},
            "source_url": source_url, "source_sha256": source_sha256,
            "extract_sha256": extract_sha256,
            "road_evidence_json": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(result, columns=EVIDENCE_COLUMNS).sort_values("record_id", ignore_index=True)


def load_numbered_street_evidence(catalog: pd.DataFrame, source_manifest_path: str | Path) -> pd.DataFrame:
    """Read the exact local extract identified and hashed by its source manifest."""
    manifest_path = Path(source_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(manifest["output_path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    contents = path.read_bytes()
    actual_sha = hashlib.sha256(contents).hexdigest()
    if actual_sha != manifest["output_sha256"]:
        raise ValueError("OSM extract SHA256 differs from its source manifest")
    if "output_bytes" in manifest and len(contents) != manifest["output_bytes"]:
        raise ValueError("OSM extract byte count differs from its source manifest")
    payload = json.loads(contents)
    elements = payload["elements"]
    if not isinstance(elements, list):
        raise ValueError("OSM extract elements must be a list")
    if "element_count" in manifest and len(elements) != manifest["element_count"]:
        raise ValueError("OSM extract object count differs from its source manifest")
    snapshot = manifest["snapshot"]
    return audit_numbered_street_addresses(
        catalog, elements, source_url=snapshot["source_url"],
        source_sha256=snapshot["source_sha256"], extract_sha256=actual_sha,
    )

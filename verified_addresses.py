"""Exact address matching to one OSM building footprint, without fuzzy geocoding.

OSM is an independent volunteered map, not an official address registry. This
module neither edits the catalog/model nor imports matches into a fitted model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
import unicodedata
from typing import Any

import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, box, shape

from building_catalog import ASTANA_BBOX, SUFFIX_HOMOGLYPHS, canonical_house_key


_HOUSE = r"\d+[а-яёa-z]?(?:[/-]\d+[а-яёa-z]?)*(?:\s+(?:корпус|корп\.?|к\.?|блок|строение|стр\.?)\s+[\dа-яёa-z]+)?"
_ADDRESS = re.compile(rf"^(?P<street>.+?)\s+(?P<house>{_HOUSE})$")
_DESIGNATORS = {"улица", "ул", "проспект", "пр", "даңғылы", "көшесі"}
_RESIDENTIAL_BUILDINGS = {"yes", "apartments", "residential", "house", "detached", "semidetached_house", "terrace", "dormitory"}
MATCH_COLUMNS = [
    "house_key", "address_normalized", "district", "latitude", "longitude",
    "coordinate_status", "source_status", "match_method", "matched_osm_street", "matched_osm_housenumber",
    "osm_type", "osm_id", "osm_version", "osm_timestamp", "source_url", "osm_url",
    "equivalent_osm_objects_json", "street_alias_evidence_json", "boundary_check",
    "snapshot_source_url", "snapshot_source_sha256", "source_accessed_at", "source_snapshot_at",
    "source_license", "source_attribution", "existing_model_coordinate", "identity_review_required",
    "eligible_for_enrichment", "review_reasons_json",
]
AUDIT_COLUMNS = ["house_key", "address_normalized", "district", "decision", "reason", "source_object_count",
                 "distinct_footprint_count", "osm_objects_json", "existing_model_coordinate", "identity_review_required"]


def _text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split()) if isinstance(value, str) else ""


def normalize_street(value: Any) -> str | None:
    """Drop only explicit street-designator tokens; retain spelling and order."""
    text = _text(value)
    if not text or re.search(r"[,;|!?():—]", text):
        return None
    words = [word for word in text.split() if word.rstrip(".") not in _DESIGNATORS]
    return " ".join(words) or None


def normalize_house_number(value: Any) -> str | None:
    text = _text(value)
    if not re.fullmatch(_HOUSE, text):
        return None
    text = re.sub(r"(?<=\d)[a-z]", lambda match: match[0].translate(SUFFIX_HOMOGLYPHS), text)
    if " " in text:
        prefix, token = text.rsplit(" ", 1)
        if re.fullmatch(r"(?:\d+)?[a-zа-яё]", token):
            text = prefix + " " + token.translate(SUFFIX_HOMOGLYPHS)
    return text


def _address(value: Any) -> tuple[str, str] | None:
    match = _ADDRESS.fullmatch(_text(value))
    if not match:
        return None
    street, number = normalize_street(match["street"]), normalize_house_number(match["house"])
    return (street, number) if street and number else None


def _ring(points: Any) -> list[tuple[float, float]]:
    if not isinstance(points, list) or len(points) < 4:
        raise ValueError("invalid_geometry")
    result = []
    for point in points:
        longitude, latitude = float(point["lon"]), float(point["lat"])
        if not math.isfinite(longitude) or not math.isfinite(latitude) or not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError("invalid_geometry")
        result.append((longitude, latitude))
    if result[0] != result[-1]:
        raise ValueError("unclosed_geometry")
    return result


def _polygons(element: dict) -> list[Polygon]:
    parts = element.get("polygons")
    if parts is None:
        parts = [{"outer": element.get("geometry"), "holes": []}]
    if not isinstance(parts, list) or not parts:
        raise ValueError("missing_geometry")
    result = []
    for part in parts:
        polygon = Polygon(_ring(part["outer"]), [_ring(hole) for hole in part.get("holes", [])])
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
            raise ValueError("invalid_geometry")
        result.append(polygon)
    return result


def _boundary(snapshot: dict) -> tuple[Any, str]:
    explicit = snapshot.get("boundary_geojson")
    if explicit is not None:
        geometry = shape(explicit.get("geometry", explicit))
        method = "provided_boundary"
    else:
        boundary_id = snapshot.get("boundary_relation_id")
        found = [entry for entry in snapshot["elements"] if entry.get("type") == "relation" and entry.get("id") == boundary_id]
        if len(found) > 1:
            raise ValueError("Duplicate city boundary relation.")
        if found:
            tags = found[0].get("tags", {})
            if tags.get("boundary") != "administrative":
                raise ValueError("The supplied city boundary is not administrative.")
            parts = _polygons(found[0])
            geometry = parts[0] if len(parts) == 1 else MultiPolygon(parts)
            method = f"osm_administrative_relation_{boundary_id}"
        else:
            return box(ASTANA_BBOX["longitude_min"], ASTANA_BBOX["latitude_min"],
                       ASTANA_BBOX["longitude_max"], ASTANA_BBOX["latitude_max"]), "astana_bbox_only"
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.is_empty or not geometry.is_valid:
        raise ValueError("Invalid city boundary geometry.")
    return geometry, method


def _true(value: Any) -> bool:
    return value is True or isinstance(value, (int, float)) and value == 1 or isinstance(value, str) and value.casefold() == "true"


def _conflicts(row: dict) -> list[str]:
    value = row.get("catalog_conflict_reasons_json", "[]")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(item) for item in parsed] if isinstance(parsed, list) else ["unparsed_catalog_conflicts"]
        except ValueError:
            return ["unparsed_catalog_conflicts"]
    return []


def _building_rejection(tags: dict) -> str | None:
    lifecycle = ("abandoned", "demolished", "disused", "razed", "ruins", "construction", "proposed")
    if (any(tags.get(name) not in {None, "no", "false", "0"} for name in lifecycle)
            or any(tags.get(f"{name}:building") not in {None, "no", "false", "0"} for name in lifecycle)
            or tags.get("building") in {"abandoned", "demolished", "ruins", "construction", "proposed"}):
        return "inactive_or_unbuilt_building"
    if tags.get("building") not in _RESIDENTIAL_BUILDINGS or tags.get("building:part", "no") != "no" or tags.get("type") == "site":
        return "not_residential_building_footprint"
    if tags.get("amenity") in {"school", "kindergarten", "college", "university", "hospital", "clinic", "prison",
                               "police", "fire_station", "courthouse", "townhall", "place_of_worship", "parking", "fuel"}:
        return "explicit_nonresidential_amenity"
    if tags.get("building:use") in {"garage", "garages", "kiosk", "school", "industrial", "warehouse", "commercial"}:
        return "explicit_nonresidential_use"
    return None


def match_addresses(candidates: pd.DataFrame, osm_snapshot: dict) -> dict[str, Any]:
    """Return exact matches and one audit row per house_key; never use prices.

    Snapshot follows the local OSM extraction schema (elements + snapshot source
    metadata). An exact address match is kept for audit even when unresolved
    catalog identity conflicts make it ineligible for automatic enrichment.
    """
    required = {"house_key", "address_normalized", "district"}
    if not isinstance(candidates, pd.DataFrame) or not candidates.columns.is_unique or not required.issubset(candidates):
        raise ValueError("Candidates require house_key, address_normalized and district.")
    if not candidates.house_key.map(lambda value: isinstance(value, str) and bool(value.strip())).all() or not candidates.house_key.is_unique:
        raise ValueError("Candidate house keys must be unique, nonempty strings.")
    if not isinstance(osm_snapshot, dict) or not isinstance(osm_snapshot.get("elements"), list):
        raise ValueError("OSM snapshot requires an elements list.")
    source = osm_snapshot.get("snapshot", {})
    if not isinstance(source, dict) or not all(source.get(key) for key in ("source_url", "source_sha256", "fetched_at", "license")):
        raise ValueError("OSM source URL, checksum, access date and license are required.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(source["source_sha256"])):
        raise ValueError("Invalid snapshot source SHA256.")
    boundary, boundary_method = _boundary(osm_snapshot)
    # Direct aliases on ONE road feature only: no transitive name union and no old_name.
    aliases: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for entry in osm_snapshot["elements"]:
        tags = entry.get("tags", {})
        if entry.get("type") not in {"way", "relation"} or not tags.get("highway"):
            continue
        names = [tags.get(key) for key in ("name", "name:ru", "name:kk")]
        names += str(tags.get("alt_name", "")).split(";")
        normalized = {normalize_street(name) for name in names} - {None}
        identifier = f"{entry.get('type')}/{entry.get('id')}"
        for left in normalized:
            for right in normalized - {left}:
                aliases[left][right].add(identifier)

    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rejected_address_objects: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rejections: Counter = Counter()
    source_object_seen: set[tuple] = set()
    for entry in osm_snapshot["elements"]:
        tags = entry.get("tags", {})
        if not tags.get("addr:street") or not tags.get("addr:housenumber"):
            continue
        if entry.get("type") not in {"way", "relation"}:
            rejections["address_node_not_building_footprint"] += 1
            continue
        street, house = normalize_street(tags["addr:street"]), normalize_house_number(tags["addr:housenumber"])
        if not street or not house:
            rejections["ambiguous_osm_address"] += 1
            continue
        rejection = _building_rejection(tags)
        polygons = None
        try:
            polygons = _polygons(entry)
            if len(polygons) != 1:
                raise ValueError("multiple_building_footprints")
            polygon = polygons[0]
            if not boundary.covers(polygon):
                raise ValueError("outside_astana_boundary")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            rejection = rejection or (str(error) if isinstance(error, ValueError) else "invalid_geometry")
        osm_id, version = entry.get("id"), entry.get("version")
        if isinstance(osm_id, bool) or not isinstance(osm_id, int) or osm_id <= 0 or not isinstance(version, int) or version <= 0:
            rejection = rejection or "missing_osm_identity_or_version"
        if rejection:
            rejections[rejection] += 1
            # A rejected second building must not make the first address look
            # unique. Only geometries proven entirely outside the city are
            # irrelevant to ambiguity within Astana. Missing/invalid geometry
            # cannot establish uniqueness and remains blocking evidence.
            if not polygons or not all(boundary.disjoint(part) for part in polygons):
                rejected_address_objects[(street, house)].append({
                    "object": f"{entry.get('type')}/{entry.get('id')}", "reason": rejection})
            continue
        identity = (entry["type"], osm_id)
        if identity in source_object_seen:
            raise ValueError(f"Duplicate OSM source object: {identity}")
        source_object_seen.add(identity)
        index[(street, house)].append({"entry": entry, "polygon": polygon, "street": street, "house": house,
                                       "source_url": f"https://www.openstreetmap.org/{entry['type']}/{osm_id}"})

    records = candidates.to_dict("records")
    candidate_districts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in records:
        address = _address(row["address_normalized"])
        if address and row["house_key"] == canonical_house_key(row["address_normalized"], row["district"]):
            candidate_districts[address].add(_text(row["district"]))
    matches, audit = [], []
    matched_polygons: dict[str, Polygon] = {}
    for row in records:
        address = _address(row["address_normalized"])
        existing = _true(row.get("model_house_mapped", False))
        conflicts = _conflicts(row)
        needs_review = bool(conflicts)
        item = {"house_key": row["house_key"], "address_normalized": row["address_normalized"], "district": row["district"],
                "decision": "rejected", "reason": "invalid_candidate_address", "source_object_count": 0,
                "distinct_footprint_count": 0, "osm_objects_json": "[]", "existing_model_coordinate": existing,
                "identity_review_required": needs_review}
        if not address or not _text(row["district"]):
            audit.append(item)
            continue
        if row["house_key"] != canonical_house_key(row["address_normalized"], row["district"]):
            item["reason"] = "candidate_house_key_mismatch"
            audit.append(item)
            continue
        street, house = address
        street_options = {street, *aliases.get(street, {})}
        sources = [entry for option in sorted(street_options) for entry in index.get((option, house), [])]
        item["source_object_count"] = len(sources)
        item["osm_objects_json"] = json.dumps(sorted(entry["source_url"] for entry in sources))
        if not sources:
            item.update(decision="unmatched", reason="no_exact_building_address")
            audit.append(item)
            continue
        blocked = [entry for option in sorted(street_options) for entry in rejected_address_objects.get((option, house), [])]
        if blocked:
            item.update(reason="conflicting_or_unresolved_source_address_objects",
                        source_object_count=len(sources) + len(blocked),
                        osm_objects_json=json.dumps(sorted([entry["source_url"] for entry in sources]
                                                          + [entry["object"] for entry in blocked])))
            audit.append(item)
            continue
        # Never let a district filter hide a second physical building at this address.
        footprints: list[list[dict]] = []
        for entry in sources:
            equivalent = next((group for group in footprints if entry["polygon"].equals(group[0]["polygon"])), None)
            if equivalent is None:
                footprints.append([entry])
            else:
                equivalent.append(entry)
        item["distinct_footprint_count"] = len(footprints)
        if len(footprints) != 1:
            item["reason"] = "multiple_distinct_building_footprints"
            audit.append(item)
            continue
        source_districts = {_text(entry["entry"]["tags"].get("addr:district")) for entry in sources} - {""}
        if source_districts and source_districts != {_text(row["district"])}:
            item["reason"] = "osm_district_conflict"
            audit.append(item)
            continue
        districts = {district for option in street_options for district in candidate_districts.get((option, house), set())}
        if len(districts) > 1 and not source_districts:
            item["reason"] = "candidate_district_ambiguous"
            audit.append(item)
            continue
        chosen = sorted(sources, key=lambda entry: (entry["street"] != street, entry["entry"]["type"], entry["entry"]["id"]))[0]
        point = chosen["polygon"].representative_point()
        alias_evidence = sorted({evidence for entry in sources if entry["street"] != street
                                 for evidence in aliases[street][entry["street"]]})
        matched = {key: row[key] for key in ("house_key", "address_normalized", "district")}
        review_reasons = list(conflicts)
        boundary_confirmed = boundary_method != "astana_bbox_only"
        if not boundary_confirmed:
            review_reasons.append("administrative_boundary_missing")
        matched.update(latitude=float(point.y), longitude=float(point.x), coordinate_status="osm_exact_address", source_status="osm_exact_address",
                       match_method="exact_address_and_building_footprint", matched_osm_street=chosen["entry"]["tags"]["addr:street"],
                       matched_osm_housenumber=chosen["entry"]["tags"]["addr:housenumber"], osm_type=chosen["entry"]["type"],
                       osm_id=chosen["entry"]["id"], osm_version=chosen["entry"]["version"], osm_timestamp=chosen["entry"].get("timestamp"),
                       source_url=chosen["source_url"], osm_url=chosen["source_url"], equivalent_osm_objects_json=item["osm_objects_json"],
                       street_alias_evidence_json=json.dumps(alias_evidence), boundary_check=boundary_method,
                       snapshot_source_url=source["source_url"], snapshot_source_sha256=source["source_sha256"],
                       source_accessed_at=source["fetched_at"], source_snapshot_at=source.get("pbf_header", {}).get("osmosis_replication_timestamp"),
                       source_license=source["license"], source_attribution=source.get("attribution", "OpenStreetMap contributors"),
                       existing_model_coordinate=existing, identity_review_required=needs_review,
                       eligible_for_enrichment=not existing and not needs_review and boundary_confirmed,
                       review_reasons_json=json.dumps(review_reasons, ensure_ascii=False))
        matches.append(matched)
        matched_polygons[row["house_key"]] = chosen["polygon"]
        item.update(decision="matched", reason="existing_model_coordinate_preserved" if existing else
                    ("catalog_identity_review_required" if needs_review else
                     ("unique_exact_osm_building" if boundary_confirmed else "administrative_boundary_missing")))
        audit.append(item)
    # Direct aliases can converge on a third name without being aliases of each
    # other. Check the actual matched footprints across all candidates as well;
    # this is ambiguity detection, not transitive street-name inference.
    footprint_groups: dict[tuple, list[list[dict]]] = defaultdict(list)
    for matched in matches:
        polygon = matched_polygons[matched["house_key"]]
        groups = footprint_groups[polygon.bounds]
        equivalent = next((group for group in groups if polygon.equals(matched_polygons[group[0]["house_key"]])), None)
        if equivalent is None:
            groups.append([matched])
        else:
            equivalent.append(matched)
    ambiguous_keys = {matched["house_key"] for groups in footprint_groups.values() for group in groups
                      if len({_text(item["district"]) for item in group}) > 1 for matched in group}
    if ambiguous_keys:
        matches = [item for item in matches if item["house_key"] not in ambiguous_keys]
        for item in audit:
            if item["house_key"] in ambiguous_keys:
                item.update(decision="rejected", reason="matched_footprint_multiple_candidate_districts")
    matched_frame = pd.DataFrame(matches, columns=MATCH_COLUMNS)
    audit_frame = pd.DataFrame(audit, columns=AUDIT_COLUMNS)
    for column in ("existing_model_coordinate", "identity_review_required", "eligible_for_enrichment"):
        matched_frame[column] = matched_frame[column].astype(bool)
    surviving_groups = [group for groups in footprint_groups.values() for group in groups
                        if any(item["house_key"] not in ambiguous_keys for item in group)]
    eligible_groups = [group for group in surviving_groups if any(item["eligible_for_enrichment"] for item in group)]
    summary = {"schema_version": 1, "candidate_house_keys": len(candidates), "exact_matches": len(matches),
               "eligible_new_matches": int(matched_frame.eligible_for_enrichment.sum()),
               "unique_matched_buildings": len(surviving_groups),
               "unique_eligible_new_buildings": len(eligible_groups),
               "eligible_buildings_also_matching_existing_model": sum(any(item["existing_model_coordinate"] for item in group)
                                                                       for group in eligible_groups),
               "existing_model_matches_preserved": int(matched_frame.existing_model_coordinate.sum()),
               "identity_review_matches": int(matched_frame.identity_review_required.sum()),
               "audit_reason_counts": dict(Counter(item["reason"] for item in audit)),
               "source_rejection_counts": dict(rejections), "accepted_osm_objects": len(source_object_seen),
               "boundary_check": boundary_method, "snapshot": source,
               "limitations": ["OSM exact address is not an official verification or a guarantee of entrance accuracy.",
                               "Point lies on the mapped building surface; apartment/entrance location is unknown.",
                               "No fuzzy matching, guessed translations, old street names, or complex centroids were used.",
                               "Catalog trait conflicts remain unresolved and existing model coordinates are not overridden."]}
    return {"matches": matched_frame, "audit": audit_frame, "summary": summary}

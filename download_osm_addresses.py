"""Cache one public Geofabrik snapshot and extract Astana OSM features locally.

No geocoding, scraping, model fitting, or catalogue changes are performed.
The output is ODbL-1.0 data, attributed to OpenStreetMap contributors.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import osmium
import requests


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "osm_2026_10_13"
SOURCE_NAME = "kazakhstan-261012.osm.pbf"
SOURCE_URL = f"https://download.geofabrik.de/asia/{SOURCE_NAME}"
SOURCE_PAGE = "https://download.geofabrik.de/asia/kazakhstan.html"
ASTANA_BBOX = (50.8, 70.8, 51.4, 71.9)  # south, west, north, east
# Verified in the pinned PBF: boundary=administrative, admin_level=4,
# ISO3166-2=KZ-71. Wikidata's P402 points to place polygon 3087156 instead.
ASTANA_RELATION_ID = 3087155
LICENSE_URL = "https://opendatacommons.org/licenses/odbl/1-0/"
COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"
USER_AGENT = "AstanaApartmentResearch/1.0 (single cached regional OSM extract)"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path: Path, document: dict) -> None:
    """Never replace an existing immutable snapshot or extraction artifact."""
    payload = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace existing artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(payload)


def fetch_snapshot(directory: Path, *, extract_only: bool = False) -> tuple[Path, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SOURCE_NAME
    manifest_path = directory / "kazakhstan-261012.source.json"
    if path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != sha256_file(path) or manifest.get("bytes") != path.stat().st_size:
            raise ValueError("Cached PBF does not match its provenance manifest")
        if manifest.get("url") != SOURCE_URL:
            raise ValueError("Cached PBF has an unexpected source URL")
        return path, manifest
    if extract_only:
        raise FileNotFoundError(path)
    if manifest_path.exists():
        raise FileExistsError("Source manifest exists without its PBF; inspect before downloading")
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"Partial download requires inspection: {partial}")
    digest = hashlib.sha256()
    size = 0
    started_at = utc_now()
    last_progress = time.monotonic()
    with requests.get(SOURCE_URL, stream=True, headers={"User-Agent": USER_AGENT}, timeout=(30, 90)) as response:
        response.raise_for_status()
        expected_size = int(response.headers.get("Content-Length", "0"))
        with partial.open("xb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if time.monotonic() - last_progress >= 30:
                    print(json.dumps({"downloaded_bytes": size, "expected_bytes": expected_size}), flush=True)
                    last_progress = time.monotonic()
        if expected_size and size != expected_size:
            raise ValueError("Incomplete PBF download; partial file retained for inspection")
        metadata = {
            "schema_version": 1, "source": "Geofabrik Kazakhstan extract of OpenStreetMap",
            "url": SOURCE_URL, "final_url": response.url, "source_page": SOURCE_PAGE,
            "snapshot_filename": SOURCE_NAME, "download_started_at": started_at, "fetched_at": utc_now(),
            "http_status": response.status_code,
            "response_headers": {key: response.headers.get(key) for key in ("Content-Length", "Content-Type", "Last-Modified", "ETag")},
            "bytes": size, "sha256": digest.hexdigest(), "license": "ODbL-1.0", "license_url": LICENSE_URL,
            "attribution": "© OpenStreetMap contributors", "attribution_url": COPYRIGHT_URL,
            "notes": ["One cached public regional download; no geocoding requests.",
                      "OSM timestamps are edit/database dates, not physical survey dates."],
        }
    # Destination was checked above; a concurrent creator must never be replaced.
    partial.rename(path)
    write_new_json(manifest_path, metadata)
    return path, metadata


def _coordinates(nodes) -> list[dict[str, float]] | None:
    result = []
    for node in nodes:
        if not node.location.valid():
            return None
        result.append({"lat": node.lat, "lon": node.lon})
    return result


def _bbox_intersects(geometry: list[dict[str, float]]) -> bool:
    if not geometry:
        return False
    south, west, north, east = ASTANA_BBOX
    lats = [point["lat"] for point in geometry]
    lons = [point["lon"] for point in geometry]
    return min(lats) <= north and max(lats) >= south and min(lons) <= east and max(lons) >= west


def _nodes_bbox_intersects(nodes) -> bool:
    """Avoid allocating geometry dictionaries for every building in Kazakhstan."""
    south, west, north, east = ASTANA_BBOX
    minlat = minlon = float("inf")
    maxlat = maxlon = -float("inf")
    for node in nodes:
        if not node.location.valid():
            return False
        lat, lon = node.lat, node.lon
        minlat, maxlat = min(minlat, lat), max(maxlat, lat)
        minlon, maxlon = min(minlon, lon), max(maxlon, lon)
    return minlat <= north and maxlat >= south and minlon <= east and maxlon >= west


def _has_address(tags: dict) -> bool:
    return bool(tags.get("addr:housenumber"))


def _named_road(tags: dict) -> bool:
    return bool(tags.get("highway")) and any(
        key == "name" or key.startswith("name:") or key in ("alt_name", "old_name", "official_name", "short_name")
        for key in tags
    )


def _identity(obj, kind: str, object_id: int, tags: dict) -> dict:
    # Never copy contributor user/uid/changeset attributes into the extraction.
    return {"type": kind, "id": object_id, "tags": tags, "version": obj.version,
            "timestamp": obj.timestamp.isoformat(), "osm_url": f"https://www.openstreetmap.org/{kind}/{object_id}"}


def extract_features(path: Path) -> tuple[list[dict], dict]:
    """Read full topology before applying bbox filters; preserve polygon holes."""
    elements: dict[tuple[str, int], dict] = {}
    relation_candidates: dict[int, dict] = {}
    assembled_relations: set[int] = set()
    counts: Counter = Counter()
    south, west, north, east = ASTANA_BBOX
    last_progress = time.monotonic()
    wanted = ("addr:housenumber", "building", "highway", "boundary")
    processor = (osmium.FileProcessor(path)
                 .with_areas(osmium.filter.KeyFilter("addr:housenumber", "building", "boundary"))
                 .with_filter(osmium.filter.KeyFilter(*wanted)))
    for obj in processor:
        tags = dict(obj.tags)
        counts["filtered_objects_seen"] += 1
        if time.monotonic() - last_progress >= 30:
            print(json.dumps({"stage": "local_extraction", "filtered_objects_seen": counts["filtered_objects_seen"],
                              "retained_features": len(elements)}), flush=True)
            last_progress = time.monotonic()
        if obj.is_node():
            if not (_has_address(tags) or tags.get("building")) or not obj.location.valid():
                continue
            if south <= obj.lat <= north and west <= obj.lon <= east:
                element = _identity(obj, "node", obj.id, tags)
                element.update(lat=obj.lat, lon=obj.lon, geometry_kind="point")
                elements[("node", obj.id)] = element
        elif obj.is_way():
            if not (_has_address(tags) or tags.get("building") or _named_road(tags)):
                continue
            if not _nodes_bbox_intersects(obj.nodes):
                continue
            geometry = _coordinates(obj.nodes)
            if geometry is None or len(geometry) < 2:
                counts["unusable_way_geometry_countrywide"] += 1
                continue
            if not _bbox_intersects(geometry):
                continue
            element = _identity(obj, "way", obj.id, tags)
            element.update(geometry=geometry, geometry_kind="closed_way" if obj.is_closed() else "linestring")
            elements[("way", obj.id)] = element
        elif obj.is_relation():
            if _has_address(tags) or tags.get("building") or obj.id == ASTANA_RELATION_ID:
                relation_candidates[obj.id] = _identity(obj, "relation", obj.id, tags)
                relation_candidates[obj.id]["member_ways"] = [
                    {"id": member.ref, "role": member.role}
                    for member in obj.members if member.type == "w"
                ]
        elif obj.is_area():
            kind = "way" if obj.from_way() else "relation"
            object_id = obj.orig_id()
            if not (_has_address(tags) or tags.get("building") or (kind == "relation" and object_id == ASTANA_RELATION_ID)):
                continue
            if kind == "relation":
                assembled_relations.add(object_id)
            if not (kind == "relation" and object_id == ASTANA_RELATION_ID):
                if not any(_nodes_bbox_intersects(outer) for outer in obj.outer_rings()):
                    continue
            polygons = []
            valid = True
            for outer in obj.outer_rings():
                exterior = _coordinates(outer)
                holes = [_coordinates(inner) for inner in obj.inner_rings(outer)]
                if exterior is None or any(hole is None for hole in holes):
                    valid = False
                    break
                polygons.append({"outer": exterior, "holes": holes})
            if not valid or not polygons:
                counts["unusable_area_geometry_countrywide"] += 1
                continue
            if object_id != ASTANA_RELATION_ID or kind != "relation":
                if not any(_bbox_intersects(polygon["outer"]) for polygon in polygons):
                    continue
            element = _identity(obj, kind, object_id, tags)
            element.update(polygons=polygons, geometry_kind="polygon" if len(polygons) == 1 else "multipolygon")
            # A flattened multi-ring outline is unsafe: leave geometry absent for
            # multi-outer relations, requiring downstream explicit handling.
            if len(polygons) == 1:
                element["geometry"] = polygons[0]["outer"]
            elements[(kind, object_id)] = element
    counts["unassembled_address_or_building_relations_countrywide"] = len(set(relation_candidates) - assembled_relations)
    boundary = elements.get(("relation", ASTANA_RELATION_ID))
    counts["astana_boundary_assembled"] = int(boundary is not None and "polygons" in boundary)
    counts.update({f"retained_{kind}": sum(element["type"] == kind for element in elements.values())
                   for kind in ("node", "way", "relation")})
    counts["retained_addressed_features"] = sum(_has_address(element["tags"]) for element in elements.values())
    counts["retained_building_features"] = sum(bool(element["tags"].get("building")) for element in elements.values())
    counts["retained_named_roads"] = sum(_named_road(element["tags"]) for element in elements.values())
    for element in elements.values():
        if element["type"] == "relation":
            candidate = relation_candidates.get(element["id"], {})
            element["member_ways"] = candidate.get("member_ways", [])
    # Snapshot topology may duplicate a relation's tagged member way. Keeping
    # identities is intentional; downstream matching must resolve duplicates.
    result = sorted(elements.values(), key=lambda item: (item["type"], item["id"]))
    return result, dict(counts)


def run(directory: Path, *, extract_only: bool = False) -> dict:
    path, source = fetch_snapshot(directory, extract_only=extract_only)
    source_sha = source["sha256"]
    code_sha = sha256_file(Path(__file__))
    output_path = directory / "astana_addresses.json"
    metadata_path = directory / "astana_addresses.source.json"
    if output_path.exists() or metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (saved.get("input_sha256") != source_sha or saved.get("extractor_sha256") != code_sha
                or saved.get("output_sha256") != sha256_file(output_path)):
            raise ValueError("Existing extraction is from different code/source or has changed; use a separate directory")
        print(json.dumps({"status": "reused", "path": str(output_path), "counts": saved["counts"]}), flush=True)
        return saved
    with osmium.io.Reader(path) as reader:
        header = reader.header()
        pbf_header = {key: header.get(key, "") for key in
                      ("generator", "osmosis_replication_timestamp", "osmosis_replication_sequence_number", "osmosis_replication_base_url")}
    elements, counts = extract_features(path)
    boundary = next((element for element in elements
                     if element["type"] == "relation" and element["id"] == ASTANA_RELATION_ID), None)
    if (not boundary or not boundary.get("polygons")
            or boundary["tags"].get("boundary") != "administrative"
            or boundary["tags"].get("admin_level") != "4"
            or boundary["tags"].get("ISO3166-2") != "KZ-71"):
        raise ValueError("Pinned Astana administrative boundary was not assembled and validated")
    if sha256_file(path) != source_sha or sha256_file(Path(__file__)) != code_sha:
        raise ValueError("Source PBF or extractor changed while extraction was running")
    extracted_at = utc_now()
    snapshot = {"source_url": SOURCE_URL, "source_sha256": source_sha, "pbf_header": pbf_header,
                "fetched_at": source["fetched_at"], "extracted_at": extracted_at,
                "license": "ODbL-1.0", "license_url": LICENSE_URL,
                "attribution": "© OpenStreetMap contributors", "attribution_url": COPYRIGHT_URL}
    output = {"schema_version": 1, "snapshot": snapshot, "bbox_south_west_north_east": list(ASTANA_BBOX),
              "boundary_relation_id": ASTANA_RELATION_ID, "elements": elements}
    write_new_json(output_path, output)
    metadata = {"schema_version": 1, "input_path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
                "input_sha256": source_sha, "extractor_sha256": code_sha, "output_path": output_path.name,
                "output_sha256": sha256_file(output_path), "output_bytes": output_path.stat().st_size,
                "element_count": len(elements), "counts": counts, "snapshot": snapshot,
                "versions": {"osmium": version("osmium"), "requests": version("requests")},
                "bbox_south_west_north_east": list(ASTANA_BBOX), "boundary_relation_id": ASTANA_RELATION_ID,
                "selection": "Addressed nodes/ways/area relations, building features, named road ways, and Astana boundary. Bbox intersection is a coarse extraction filter, not proof of city membership.",
                "limitations": ["Coordinates and addresses are crowdsourced OSM evidence, not cadastral verification.",
                    "Object timestamps record edits, not survey dates. No contributor IDs/user names/changesets copied.",
                    "Single-ring geometry is an outline, not a centroid. Polygon holes and multiple outer parts remain in polygons.",
                    "Building relations and member ways may describe the same physical building; address matching must resolve duplicate identities.",
                    "Unassembled relation counts refer to the country: absent geometry cannot safely be assigned to Astana.",
                    "No interpolation, road snapping, fuzzy geocoding, or complex-centroid substitution was performed."]}
    write_new_json(metadata_path, metadata)
    print(json.dumps({"status": "extracted", "path": str(output_path), "counts": counts, "sha256": metadata["output_sha256"]}), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--extract-only", action="store_true", help="Require an existing, SHA-validated PBF; make no network calls")
    args = parser.parse_args()
    run(args.directory.resolve(), extract_only=args.extract_only)


if __name__ == "__main__":
    main()

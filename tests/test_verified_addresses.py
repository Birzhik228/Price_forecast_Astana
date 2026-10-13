"""Synthetic building geometry tests; no downloads or real-model changes."""

from __future__ import annotations

from copy import deepcopy
import unittest

import pandas as pd
from shapely.geometry import Point, Polygon, mapping

from building_catalog import canonical_house_key
from verified_addresses import match_addresses, normalize_house_number, normalize_street


def ring(x=71.4, y=51.1, width=.001):
    return [{"lon": x, "lat": y}, {"lon": x + width, "lat": y},
            {"lon": x + width, "lat": y + width}, {"lon": x, "lat": y + width}, {"lon": x, "lat": y}]


def building(number=1, street="улица Тестовая", house="44а", **changes):
    value = {"type": "way", "id": number, "version": 3, "timestamp": "2026-10-13T00:00:00Z",
             "tags": {"building": "apartments", "addr:street": street, "addr:housenumber": house},
             "geometry": ring(), "geometry_kind": "closed_way"}
    value.update(changes)
    return value


def snapshot(*elements):
    city = Polygon([(70.8, 50.8), (71.9, 50.8), (71.9, 51.4), (70.8, 51.4), (70.8, 50.8)])
    return {"schema_version": 1, "elements": list(elements), "boundary_geojson": mapping(city), "snapshot": {
        "source_url": "https://example.test/snapshot.osm.pbf", "source_sha256": "a" * 64,
        "fetched_at": "2026-10-13T00:00:00Z", "license": "ODbL", "attribution": "OpenStreetMap contributors",
        "pbf_header": {"osmosis_replication_timestamp": "2026-10-12T20:00:00Z"},
    }}


def candidates(address="тестовая 44a", district="Есиль", **changes):
    value = {"house_key": canonical_house_key(address, district), "address_normalized": address, "district": district}
    value.update(changes)
    return pd.DataFrame([value])


class AddressMatchingTests(unittest.TestCase):
    def test_exact_footprint_suffix_and_source_provenance(self):
        source = snapshot(building())
        original = deepcopy(source)
        result = match_addresses(candidates(), source)
        row = result["matches"].iloc[0]
        self.assertEqual(row.coordinate_status, "osm_exact_address")
        self.assertEqual(row.source_status, row.coordinate_status)
        self.assertEqual(row.osm_url, row.source_url)
        self.assertTrue(row.eligible_for_enrichment)
        self.assertEqual(row.osm_version, 3)
        self.assertEqual(row.source_url, "https://www.openstreetmap.org/way/1")
        self.assertEqual(row.snapshot_source_sha256, "a" * 64)
        polygon = Polygon([(point["lon"], point["lat"]) for point in ring()])
        self.assertTrue(polygon.contains(Point(row.longitude, row.latitude)))
        self.assertEqual(source, original)

    def test_house_suffix_only_and_no_street_transliteration(self):
        self.assertEqual(normalize_house_number("44A"), "44а")
        self.assertEqual(normalize_house_number("44/1 корпус C"), "44/1 корпус с")
        self.assertNotEqual(normalize_street("A-52 көшесі"), normalize_street("А-52 улица"))
        self.assertTrue(match_addresses(candidates("A-52 44а"), snapshot(building(street="А-52")))["matches"].empty)

    def test_slash_and_corpus_are_not_dropped(self):
        for address in ("тестовая 44/1", "тестовая 44 корпус а", "тестовая 441"):
            self.assertTrue(match_addresses(candidates(address), snapshot(building(house="44")))["matches"].empty)

    def test_alias_requires_same_road_feature_and_old_name_is_ignored(self):
        road = {"type": "way", "id": 99, "tags": {"highway": "residential", "name:ru": "улица Тестовая",
                                                          "name:kk": "Сынақ көшесі", "old_name": "Старая"}}
        result = match_addresses(candidates(), snapshot(building(street="Сынақ көшесі"), road))
        self.assertEqual(len(result["matches"]), 1)
        self.assertIn("way/99", result["matches"].iloc[0].street_alias_evidence_json)
        self.assertTrue(match_addresses(candidates("старая 44а"), snapshot(building(), road))["matches"].empty)
        self.assertTrue(match_addresses(candidates(), snapshot(building(street="Сынақ көшесі")))["matches"].empty)

    def test_aliases_are_not_transitively_inferred(self):
        roads = [{"type": "way", "id": 90, "tags": {"highway": "residential", "name": "Тестовая", "alt_name": "Средняя"}},
                 {"type": "way", "id": 91, "tags": {"highway": "residential", "name": "Средняя", "alt_name": "Дальняя"}}]
        self.assertTrue(match_addresses(candidates(), snapshot(building(street="Дальняя"), *roads))["matches"].empty)

    def test_word_order_and_genitive_remain_significant(self):
        for street in ("Тестовой", "Тестовая Новая"):
            self.assertTrue(match_addresses(candidates("новая тестовая 44а"), snapshot(building(street=street)))["matches"].empty)

    def test_different_footprints_at_same_address_are_ambiguous_even_nearby(self):
        result = match_addresses(candidates(), snapshot(building(), building(2, geometry=ring(x=71.4001))))
        self.assertTrue(result["matches"].empty)
        self.assertEqual(result["audit"].iloc[0].reason, "multiple_distinct_building_footprints")

    def test_rejected_building_cannot_hide_same_address_ambiguity(self):
        multi = building(2, type="relation", polygons=[{"outer": ring(x=71.41)}, {"outer": ring(x=71.42)}])
        invalid = building(2, geometry=ring(x=71.41)[:-1])
        garage = building(2, geometry=ring(x=71.41))
        garage["tags"]["building"] = "garage"
        for other in (multi, invalid, garage):
            with self.subTest(other=other):
                result = match_addresses(candidates(), snapshot(building(), other))
                self.assertTrue(result["matches"].empty)
                self.assertEqual(result["audit"].iloc[0].reason, "conflicting_or_unresolved_source_address_objects")

    def test_rejected_building_proven_outside_city_does_not_block_address(self):
        result = match_addresses(candidates(), snapshot(building(), building(2, geometry=ring(x=30))))
        self.assertEqual(result["summary"]["eligible_new_matches"], 1)

    def test_aliases_converging_on_one_building_cannot_bypass_district_check(self):
        roads = [{"type": "way", "id": 90, "tags": {"highway": "residential", "name": "Первая", "alt_name": "Средняя"}},
                 {"type": "way", "id": 91, "tags": {"highway": "residential", "name": "Средняя", "alt_name": "Третья"}}]
        frame = pd.concat([candidates("первая 44а", "Есиль"), candidates("третья 44а", "Нура")], ignore_index=True)
        result = match_addresses(frame, snapshot(building(street="Средняя"), *roads))
        self.assertTrue(result["matches"].empty)
        self.assertEqual(set(result["audit"].reason), {"matched_footprint_multiple_candidate_districts"})

    def test_same_district_alias_keys_match_but_count_one_physical_building(self):
        road = {"type": "way", "id": 90, "tags": {"highway": "residential", "name": "Первая", "alt_name": "Средняя"}}
        frame = pd.concat([candidates("первая 44а", "Есиль"), candidates("средняя 44а", "Есиль")], ignore_index=True)
        result = match_addresses(frame, snapshot(building(street="Средняя"), road))
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(result["summary"]["eligible_new_matches"], 2)
        self.assertEqual(result["summary"]["unique_matched_buildings"], 1)
        self.assertEqual(result["summary"]["unique_eligible_new_buildings"], 1)
        frame["model_house_mapped"] = [True, False]
        result = match_addresses(frame, snapshot(building(street="Средняя"), road))
        self.assertEqual(result["summary"]["eligible_buildings_also_matching_existing_model"], 1)

    def test_house_key_cannot_name_another_address(self):
        result = match_addresses(candidates(house_key="astana|есиль|другая 999"), snapshot(building()))
        self.assertTrue(result["matches"].empty)
        self.assertEqual(result["audit"].iloc[0].reason, "candidate_house_key_mismatch")

    def test_equivalent_way_relation_rotation_and_reversal_are_deduplicated(self):
        vertices = ring()[:-1]
        changed = list(reversed(vertices[2:] + vertices[:2]))
        changed.append(changed[0])
        relation = building(2, type="relation", polygons=[{"outer": changed, "holes": []}])
        result = match_addresses(candidates(), snapshot(building(), relation))
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["audit"].iloc[0].distinct_footprint_count, 1)
        self.assertIn("way/1", result["matches"].iloc[0].equivalent_osm_objects_json)
        self.assertIn("relation/2", result["matches"].iloc[0].equivalent_osm_objects_json)

    def test_nodes_poi_parts_and_multiple_outer_relations_rejected(self):
        node = building(type="node", lat=51.1, lon=71.4)
        part = building(2)
        part["tags"]["building:part"] = "yes"
        poi = building(3)
        poi["tags"].pop("building")
        poi["tags"]["amenity"] = "school"
        relation = building(4, type="relation", polygons=[{"outer": ring()}, {"outer": ring(x=71.41)}])
        result = match_addresses(candidates(), snapshot(node, part, poi, relation))
        self.assertTrue(result["matches"].empty)
        self.assertEqual(result["summary"]["source_rejection_counts"]["multiple_building_footprints"], 1)

    def test_unclosed_self_intersecting_and_outside_geometry_rejected(self):
        invalid = ring()
        invalid[1], invalid[2] = invalid[2], invalid[1]
        for points in (ring()[:-1], invalid, ring(x=30)):
            self.assertTrue(match_addresses(candidates(), snapshot(building(geometry=points)))["matches"].empty)

    def test_polygon_hole_representative_point_not_inside_hole(self):
        element = building(polygons=[{"outer": ring(width=.01), "holes": [ring(x=71.402, y=51.102, width=.006)]}])
        result = match_addresses(candidates(), snapshot(element))
        row = result["matches"].iloc[0]
        hole = Polygon([(p["lon"], p["lat"]) for p in element["polygons"][0]["holes"][0]])
        self.assertFalse(hole.covers(Point(row.longitude, row.latitude)))

    def test_multicomponent_administrative_boundary_and_holes(self):
        boundary = {"type": "relation", "id": 777, "tags": {"boundary": "administrative"},
                    "polygons": [{"outer": ring(x=71.39, y=51.09, width=.03), "holes": [ring(x=71.399, y=51.099, width=.004)]},
                                 {"outer": ring(x=71.5, y=51.2, width=.01), "holes": []}]}
        source = snapshot(building(), boundary)
        source.pop("boundary_geojson")
        source["boundary_relation_id"] = 777
        self.assertTrue(match_addresses(candidates(), source)["matches"].empty)
        source["elements"][0]["geometry"] = ring(x=71.502, y=51.202)
        result = match_addresses(candidates(), source)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["summary"]["boundary_check"], "osm_administrative_relation_777")

    def test_place_polygon_is_not_mislabeled_administrative_boundary(self):
        source = snapshot(building(), {"type": "relation", "id": 777, "tags": {"place": "city"}, "geometry": ring(width=.02)})
        source.pop("boundary_geojson")
        source["boundary_relation_id"] = 777
        with self.assertRaisesRegex(ValueError, "not administrative"):
            match_addresses(candidates(), source)

    def test_explicit_boundary_is_used_instead_of_bbox(self):
        source = snapshot(building())
        source["boundary_geojson"] = mapping(Polygon([(p["lon"], p["lat"]) for p in ring(x=71.5)]))
        self.assertTrue(match_addresses(candidates(), source)["matches"].empty)

    def test_bbox_only_is_audit_match_but_not_eligible_enrichment(self):
        source = snapshot(building())
        source.pop("boundary_geojson")
        result = match_addresses(candidates(), source)
        self.assertEqual(len(result["matches"]), 1)
        self.assertFalse(result["matches"].iloc[0].eligible_for_enrichment)
        self.assertEqual(result["audit"].iloc[0].reason, "administrative_boundary_missing")

    def test_nonresidential_and_inactive_buildings_rejected_with_reason(self):
        scenarios = [({"building": "garage"}, "not_residential_building_footprint"),
                     ({"building": "yes", "amenity": "school"}, "explicit_nonresidential_amenity"),
                     ({"building": "yes", "building:use": "kiosk"}, "explicit_nonresidential_use"),
                     ({"abandoned:building": "apartments"}, "inactive_or_unbuilt_building"),
                     ({"demolished": "yes"}, "inactive_or_unbuilt_building"),
                     ({"building": "construction"}, "inactive_or_unbuilt_building")]
        for tags, reason in scenarios:
            with self.subTest(tags=tags):
                element = building()
                element["tags"].update(tags)
                result = match_addresses(candidates(), snapshot(element))
                self.assertTrue(result["matches"].empty)
                self.assertEqual(result["summary"]["source_rejection_counts"][reason], 1)

    def test_crossdistrict_candidates_require_explicit_osm_district(self):
        frame = pd.concat([candidates(), candidates(district="Нура", house_key="astana|нура|тестовая 44а")], ignore_index=True)
        result = match_addresses(frame, snapshot(building()))
        self.assertTrue(result["matches"].empty)
        self.assertEqual(set(result["audit"].reason), {"candidate_district_ambiguous"})
        element = building()
        element["tags"]["addr:district"] = "Есиль"
        result = match_addresses(frame, snapshot(element))
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"].iloc[0].district, "Есиль")

    def test_existing_map_and_conflict_matches_are_separate_from_enrichment(self):
        for changes in ({"model_house_mapped": True}, {"catalog_conflict_reasons_json": '["year_built_conflict"]'}):
            result = match_addresses(candidates(**changes), snapshot(building()))
            self.assertEqual(len(result["matches"]), 1)
            self.assertEqual(result["summary"]["eligible_new_matches"], 0)
            self.assertFalse(result["matches"].iloc[0].eligible_for_enrichment)

    def test_multiaddress_and_unsafe_source_metadata_fail_closed(self):
        self.assertTrue(match_addresses(candidates(), snapshot(building(house="44а;46")))["matches"].empty)
        with self.assertRaisesRegex(ValueError, "source URL"):
            match_addresses(candidates(), {"elements": []})

    def test_empty_flow_keeps_schema_and_boolean_filter(self):
        result = match_addresses(candidates().iloc[:0], snapshot())
        frame = result["matches"]
        self.assertEqual(str(frame.eligible_for_enrichment.dtype), "bool")
        self.assertIn("latitude", frame.loc[frame.eligible_for_enrichment].columns)
        self.assertEqual(len(result["audit"]), 0)


if __name__ == "__main__":
    unittest.main()

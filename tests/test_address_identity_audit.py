import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from address_identity_audit import audit_numbered_street_addresses, load_numbered_street_evidence


def catalog(*addresses):
    return pd.DataFrame([{"record_id": f"row-{i}", "house_key": f"astana|нура|{value}",
                          "address_normalized": value} for i, value in enumerate(addresses)])


def road(identifier=1, russian_name="улица Е 535", **tags):
    return {"type": "way", "id": identifier, "tags": {"highway": "residential", "name:ru": russian_name, **tags}}


class NumberedStreetAuditTests(unittest.TestCase):
    def test_exact_street_without_house_is_evidence_but_complete_address_is_not(self):
        frame = catalog("е 535", "е 535 2", "е 535 2а", "е 535 2 корпус 1")
        original = frame.copy(deep=True)
        got = audit_numbered_street_addresses(frame, [road()], source_url="source")
        self.assertEqual(got.record_id.tolist(), ["row-0"])
        self.assertEqual(got.iloc[0].match_type, "exact_name")
        self.assertEqual(got.iloc[0].source_url, "source")
        pd.testing.assert_frame_equal(frame, original)

    def test_separator_equivalence_keeps_script_and_digits(self):
        got = audit_numbered_street_addresses(catalog("е 915", "e 915", "е 0915", "е 916"),
                                              [road(russian_name="улица Е-915")])
        self.assertEqual(got.record_id.tolist(), ["row-0"])
        self.assertEqual(got.iloc[0].match_type, "number_separator_only")

    def test_names_from_buildings_or_pois_do_not_supply_road_evidence(self):
        got = audit_numbered_street_addresses(catalog("е 535"), [
            {"type": "node", "id": 2, "tags": {"name": "Е 535", "highway": "bus_stop"}},
            {"type": "way", "id": 3, "tags": {"name": "Е 535", "building": "apartments"}},
        ])
        self.assertTrue(got.empty)
        self.assertIn("old_house_key", got.columns)

    def test_multilingual_segments_have_one_deterministic_record_with_all_evidence(self):
        roads = [road(20, "улица Е-535", name="Е-535 көшесі"), road(10)]
        a = audit_numbered_street_addresses(catalog("е 535"), roads)
        b = audit_numbered_street_addresses(catalog("е 535"), reversed(roads))
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 1)
        self.assertEqual(a.iloc[0].road_id, 10)
        self.assertEqual(len(json.loads(a.iloc[0].road_evidence_json)), 3)

    def test_missing_house_identity_is_not_a_correction(self):
        frame = catalog("е 535")
        frame.loc[0, "house_key"] = pd.NA
        self.assertTrue(audit_numbered_street_addresses(frame, [road()]).empty)

    def test_duplicate_record_identity_is_rejected(self):
        frame = catalog("е 535", "е 535")
        frame["record_id"] = "same"
        with self.assertRaisesRegex(ValueError, "unique"):
            audit_numbered_street_addresses(frame, [road()])

    def test_local_manifest_hash_is_enforced_and_no_files_are_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = json.dumps({"elements": [road()]}).encode("utf-8")
            (directory / "osm.json").write_bytes(raw)
            manifest = {"output_path": "osm.json", "output_sha256": hashlib.sha256(raw).hexdigest(),
                        "output_bytes": len(raw), "element_count": 1,
                        "snapshot": {"source_url": "public-source", "source_sha256": "source-hash"}}
            manifest_path = directory / "source.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            got = load_numbered_street_evidence(catalog("е 535"), manifest_path)
            self.assertEqual(got.iloc[0].extract_sha256, manifest["output_sha256"])
            self.assertEqual((directory / "osm.json").read_bytes(), raw)
            (directory / "osm.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_numbered_street_evidence(catalog("е 535"), manifest_path)


if __name__ == "__main__":
    unittest.main()

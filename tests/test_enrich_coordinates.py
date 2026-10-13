"""Registry import must preserve the address review decision, without prices."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from enrich_coordinates import make_registry, write_new


class RegistryImportTests(unittest.TestCase):
    def test_only_explicit_eligible_matches_are_imported_with_provenance(self):
        rows = [{"house_key": f"astana|есиль|тестовая {i}", "eligible_for_enrichment": eligible,
                 "source_status": "osm_exact_address", "latitude": 51.1, "longitude": 71.4,
                 "osm_url": "https://www.openstreetmap.org/way/123", "osm_version": 7}
                for i, eligible in enumerate((True, False, None, "true"), 1)]
        registry = make_registry(pd.DataFrame(rows), {"created_at_utc": "2026-10-13T00:00:00Z"})
        self.assertEqual(list(registry["entries"]), ["astana|есиль|тестовая 1"])
        self.assertEqual(registry["entries"]["astana|есиль|тестовая 1"]["osm_version"], 7)
        self.assertEqual(registry["license"], "ODbL-1.0")

    def test_duplicate_or_unproven_match_cannot_enter_registry(self):
        row = {"house_key": "astana|есиль|тестовая 1", "eligible_for_enrichment": True,
               "source_status": "osm_exact_address"}
        for rows in ([row, row], [{**row, "source_status": "fuzzy"}]):
            with self.assertRaises(ValueError):
                make_registry(pd.DataFrame(rows), {"created_at_utc": "2026-10-13"})

    def test_artifact_writer_refuses_different_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            write_new(path, b"first")
            write_new(path, b"first")
            with self.assertRaises(FileExistsError):
                write_new(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()

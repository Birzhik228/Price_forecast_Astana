"""Диагностика поиска и безопасное отображение значений в HTML-отчёте."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from property_report import check_retrieval, render_report


def listings_fixture() -> pd.DataFrame:
    return pd.DataFrame([{
        "record_id": f"2026-01:{number}", "group": f"g{number}",
        "url": f"https://krisha.kz/a/show/{number}", "dataset_period": "2026-01",
        "area": 60.0, "rooms": 2, "district": "Есиль", "price": 30_000_000,
    } for number in range(7)])


def fixture_finder(listings: pd.DataFrame, query: dict, limit: int) -> pd.DataFrame:
    number = int(query["exclude_record_id"].split(":")[-1])
    n = number if number < 3 else limit
    result = listings.loc[listings.record_id.ne(query["exclude_record_id"])].sort_values("record_id").head(n).copy()
    result["location_match"] = "building" if number % 2 else "district"
    return result


class RetrievalDiagnosticsTests(unittest.TestCase):
    def test_random_sample_is_reproducible_and_counts_queries_separately_from_rows(self) -> None:
        frame = listings_fixture()
        first = check_retrieval(frame, fixture_finder, sample_size=200)
        second = check_retrieval(frame.iloc[::-1], fixture_finder, sample_size=200)
        self.assertEqual(first, second)
        self.assertEqual(first["queries"], 7)
        self.assertEqual(first["result_count_bins"], {"0": 1, "1–2": 2, "3–5": 4})
        self.assertEqual(first["query_location_counts"], {"with_local": 3, "district_only": 3, "empty": 1})
        self.assertEqual(first["queries_with_results"], 6)
        self.assertEqual(first["returned_rows"], 23)
        self.assertEqual(first["result_location_counts"], {"building": 11, "district": 12})

    def test_check_fails_if_a_query_is_returned_as_its_own_analog(self) -> None:
        def self_finder(listings: pd.DataFrame, query: dict, limit: int) -> pd.DataFrame:
            result = listings.loc[listings.record_id.eq(query["exclude_record_id"])].copy()
            result["location_match"] = "building"
            return result

        with self.assertRaisesRegex(AssertionError, "исходную квартиру"):
            check_retrieval(listings_fixture(), self_finder, sample_size=1)

    def test_report_escapes_source_text_and_does_not_claim_unchanged_missing_models(self) -> None:
        retrieval = check_retrieval(listings_fixture(), fixture_finder, sample_size=200)
        summary = {
            "created_at_utc": "2026-10-13T00:00:00+00:00", "listings": 7, "buildings": 3,
            "issues": 1, "coverage": {"Координаты": 4}, "source_counts": {"2026-01": 7},
            "protected_artifacts": {"models/missing.joblib": {"kind": "Модель", "unchanged": False, "actual_sha256": None}},
        }
        payload = '<script>alert("source")</script>'
        issues = pd.DataFrame([{"record_id": "2026-01:1", "detail": payload}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            render_report(summary, retrieval, {"warning": payload}, issues, path)
            html = path.read_text(encoding="utf-8")
        self.assertIn("6 / 7", html)
        self.assertIn("Неизменность моделей не подтверждена", html)
        self.assertNotIn("совпали с зафиксированными значениями", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("{retrieval[", html)
        self.assertNotIn("{summary[", html)


if __name__ == "__main__":
    unittest.main()

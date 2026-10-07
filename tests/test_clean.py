"""Автономные проверки очистки на искусственных данных только для тестов."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from clean import CleaningConfig, clean_dataframe, normalize_district, parse_numeric, run_cleaning


def listing(**changes: Any) -> dict[str, Any]:
    """Полная вымышленная строка, не попадающая в рабочий датасет."""
    row: dict[str, Any] = {
        "price": 35_000_000,
        "area": 65.5,
        "rooms": 2,
        "floor": 4,
        "total_floors": 9,
        "year_built": 2015,
        "building_type": "кирпичный",
        "condition": "хорошее",
        "district": "Есиль",
        "residential_complex": "Тестовый ЖК",
        "address": "Тестовая улица, 1",
        "url": "https://krisha.kz/a/show/123456789",
        "collected_at": "2026-01-05",
    }
    row.update(changes)
    return row


class NumericParsingTests(unittest.TestCase):
    def test_price_with_nonbreaking_spaces_and_currency(self) -> None:
        self.assertEqual(parse_numeric("35\u00a0000\u00a0000 ₸", kind="price"), 35_000_000)
        self.assertEqual(parse_numeric("35 000 000 KZT", kind="price"), 35_000_000)

    def test_price_in_millions_with_decimal_comma(self) -> None:
        self.assertEqual(parse_numeric("35,5 млн ₸", kind="price"), 35_500_000)

    def test_area_with_units_and_decimal_comma(self) -> None:
        self.assertEqual(parse_numeric("65,5 м²", kind="area"), 65.5)

    def test_valid_thousands_grouping_is_preserved(self) -> None:
        self.assertEqual(parse_numeric("1 234,5 м²", kind="area"), 1234.5)
        self.assertEqual(parse_numeric("25 000 000 ₸", kind="price"), 25_000_000)

    def test_malformed_spacing_is_not_reinterpreted_as_a_number(self) -> None:
        for value, kind in (("3 6 млн", "price"), ("1\n2 млн", "price"), ("3 6 м²", "area")):
            with self.subTest(value=value, kind=kind):
                self.assertTrue(math.isnan(parse_numeric(value, kind=kind)))

    def test_plain_numeric_values(self) -> None:
        self.assertEqual(parse_numeric(7), 7.0)
        self.assertEqual(parse_numeric(65.5, kind="area"), 65.5)

    def test_missing_and_nonnumeric_values(self) -> None:
        for value in (None, pd.NA, float("nan"), "", "не указано", "договорная"):
            with self.subTest(value=value):
                self.assertTrue(math.isnan(parse_numeric(value)))


class DistrictTests(unittest.TestCase):
    def test_six_districts_remain_distinct(self) -> None:
        districts = ("Нура", "Есиль", "Сарыарка", "Сарайшык", "Алматы", "Байконур")
        normalized = [normalize_district(value) for value in districts]
        self.assertEqual(normalized, list(districts))
        self.assertEqual(len(set(normalized)), 6)

    def test_district_prefix_suffix_and_spacing(self) -> None:
        self.assertEqual(normalize_district("  Есильский район  "), "Есиль")
        self.assertEqual(normalize_district("р-н Алматы"), "Алматы")

    def test_missing_district_stays_missing(self) -> None:
        for value in (None, pd.NA, ""):
            with self.subTest(value=value):
                self.assertTrue(pd.isna(normalize_district(value)))


class CleaningTests(unittest.TestCase):
    def test_kaggle_aliases_and_unused_columns(self) -> None:
        row = listing()
        for old, new in (
            ("price", "price_local"),
            ("building_type", "house_type"),
            ("residential_complex", "residential_complex_name"),
            ("collected_at", "parsed_at"),
        ):
            row[new] = row.pop(old)
        row.update(price_usd=70_000, price_per_m2=534_351, description="Тест")
        cleaned, rejected, _ = clean_dataframe(pd.DataFrame([row]))
        self.assertEqual(len(cleaned), 1)
        self.assertTrue(rejected.empty)
        self.assertEqual(cleaned.loc[0, "price"], 35_000_000)
        self.assertEqual(cleaned.loc[0, "building_type"], "кирпичный")
        self.assertEqual(cleaned.loc[0, "residential_complex"].casefold(), "тестовый жк")
        for extra in ("price_usd", "price_per_m2", "description", "price_local"):
            self.assertNotIn(extra, cleaned.columns)

    def test_minimal_schema_keeps_optional_missing(self) -> None:
        cleaned, rejected, report = clean_dataframe(
            pd.DataFrame([{"price": 20_000_000, "area": 40, "rooms": 1}])
        )
        self.assertEqual(len(cleaned), 1)
        self.assertTrue(rejected.empty)
        for column in ("floor", "total_floors", "year_built", "condition", "district"):
            self.assertTrue(pd.isna(cleaned.loc[0, column]))
            self.assertEqual(report["missing_after"][column], 1)

    def test_missing_required_column_is_clear_error(self) -> None:
        for column in ("price", "area", "rooms"):
            with self.subTest(column=column), self.assertRaises(ValueError):
                clean_dataframe(pd.DataFrame([listing()]).drop(columns=column))

    def test_invalid_critical_values_are_rejected(self) -> None:
        raw = pd.DataFrame([
            listing(price=None), listing(area="неизвестно"),
            listing(price=999_999), listing(price=2_000_000_001),
            listing(area=9.99), listing(area=1000.01),
        ])
        cleaned, rejected, report = clean_dataframe(raw)
        self.assertTrue(cleaned.empty)
        self.assertEqual(len(rejected), 6)
        self.assertEqual(report["removed"]["missing_price_or_area"], 2)
        self.assertEqual(report["removed"]["price_out_of_bounds"], 2)
        self.assertEqual(report["removed"]["area_out_of_bounds"], 2)
        self.assertEqual(set(rejected["source_row"]), set(range(1, 7)))
        self.assertIn("reason", rejected.columns)

    def test_boundary_prices_and_areas_are_included(self) -> None:
        rows = [listing(price=1_000_000, area=10, url=None),
                listing(price=2_000_000_000, area=1000, url=None)]
        cleaned, rejected, _ = clean_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(rejected.empty)

    def test_optional_invalid_numbers_become_missing(self) -> None:
        raw = pd.DataFrame([listing(rooms=0, floor=101, total_floors=-1, year_built=1800)])
        cleaned, rejected, _ = clean_dataframe(raw)
        self.assertEqual(len(cleaned), 1)
        self.assertTrue(rejected.empty)
        for column in ("rooms", "floor", "total_floors", "year_built"):
            self.assertTrue(pd.isna(cleaned.loc[0, column]))

    def test_fractional_integer_features_are_not_rounded(self) -> None:
        raw = pd.DataFrame([listing(rooms=2.5, floor=4.5, total_floors=9.5, year_built=2015.5)])
        cleaned, _, _ = clean_dataframe(raw)
        for column in ("rooms", "floor", "total_floors", "year_built"):
            self.assertTrue(pd.isna(cleaned.loc[0, column]))

    def test_inconsistent_floor_pair_is_missing(self) -> None:
        cleaned, _, _ = clean_dataframe(pd.DataFrame([listing(floor=10, total_floors=9)]))
        self.assertTrue(pd.isna(cleaned.loc[0, "floor"]))
        self.assertTrue(pd.isna(cleaned.loc[0, "total_floors"]))

    def test_planned_construction_year_is_preserved(self) -> None:
        rows = [listing(year_built=2028), listing(year_built=2032, price=36_000_000, url=None)]
        cleaned, _, _ = clean_dataframe(pd.DataFrame(rows), CleaningConfig(reference_year=2026))
        self.assertEqual(cleaned.iloc[0]["year_built"], 2028)
        self.assertTrue(pd.isna(cleaned.iloc[1]["year_built"]))

    def test_url_duplicates_ignore_tracking(self) -> None:
        rows = [listing(), listing(url="https://krisha.kz/a/show/123456789/?utm_source=test", price=36_000_000)]
        cleaned, rejected, report = clean_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["price"], 35_000_000)
        self.assertEqual(report["removed"]["duplicate_url"], 1)
        self.assertEqual(rejected.iloc[0]["source_row"], 2)

    def test_missing_urls_do_not_create_url_duplicates(self) -> None:
        rows = [listing(url=None, price=31_000_000), listing(url=None, price=32_000_000)]
        cleaned, _, report = clean_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["removed"]["duplicate_url"], 0)

    def test_invalid_urls_become_missing_without_losing_rows(self) -> None:
        rows = [listing(url="не ссылка"), listing(url="https://example.com/a/show/123", price=36_000_000)]
        cleaned, _, _ = clean_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(cleaned["url"].isna().all())

    def test_invalid_row_does_not_displace_valid_url_duplicate(self) -> None:
        rows = [listing(price=None), listing()]
        cleaned, rejected, report = clean_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["source_row"], 2)
        self.assertEqual(report["removed"]["duplicate_url"], 0)
        self.assertEqual(rejected.iloc[0]["reason"], "missing_price_or_area")

    def test_complete_feature_duplicates_with_different_urls(self) -> None:
        rows = [listing(), listing(url="https://krisha.kz/a/show/987654321")]
        cleaned, rejected, report = clean_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report["removed"]["duplicate_features"], 1)
        self.assertEqual(rejected.iloc[0]["reason"], "duplicate_features")

    def test_distinct_addresses_are_not_merged(self) -> None:
        rows = [listing(), listing(address="Тестовая улица, 2", url=None)]
        cleaned, _, _ = clean_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(cleaned), 2)

    def test_sparse_rows_are_not_merged_by_features(self) -> None:
        sparse = {"price": 20_000_000, "area": 40, "rooms": 1}
        cleaned, _, report = clean_dataframe(pd.DataFrame([sparse, sparse.copy()]))
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["removed"]["duplicate_features"], 0)

    def test_unknown_floors_prevent_feature_deduplication(self) -> None:
        rows = [listing(floor=None, url=None), listing(floor=None, url=None)]
        cleaned, _, report = clean_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["removed"]["duplicate_features"], 0)

    def test_report_reconciles_rows_and_preserves_input(self) -> None:
        raw = pd.DataFrame([listing(), listing(), listing(price=None), listing(price=42_000_000, url=None)])
        original = raw.copy(deep=True)
        cleaned, rejected, report = clean_dataframe(raw)
        pd.testing.assert_frame_equal(raw, original)
        self.assertEqual(report["rows_before"], 4)
        self.assertEqual(report["rows_after"], len(cleaned))
        self.assertEqual(len(raw), len(cleaned) + len(rejected))
        self.assertEqual(sum(report["removed"].values()), len(rejected))
        self.assertEqual(report["missing_before"]["price"], 1)
        self.assertEqual(report["missing_after"]["price"], 0)

    def test_source_rows_refer_to_record_positions_not_dataframe_index(self) -> None:
        raw = pd.DataFrame([listing(), listing(price=None)], index=[100, 7])
        cleaned, rejected, _ = clean_dataframe(raw)
        self.assertEqual(cleaned.iloc[0]["source_row"], 1)
        self.assertEqual(rejected.iloc[0]["source_row"], 2)

    def test_invalid_config_is_rejected(self) -> None:
        for options in ({"price_min": 2_000_000, "price_max": 1_000_000},
                        {"area_min": -1}, {"area_min": 100, "area_max": 10}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                clean_dataframe(pd.DataFrame([listing()]), CleaningConfig(**options))


class CleaningFileTests(unittest.TestCase):
    def test_duplicate_csv_header_preserves_source_and_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("price,price,area,rooms\n35000000,36000000,65.5,2\n", encoding="utf-8")
            outputs = [root / "clean.csv", root / "report.json", root / "rejected.csv"]
            for index, output in enumerate(outputs):
                output.write_text(f"existing output {index}\n", encoding="utf-8")
            original_bytes = {path: path.read_bytes() for path in [source, *outputs]}
            with self.assertRaises(ValueError):
                run_cleaning(source, *outputs)
            for path, original in original_bytes.items():
                with self.subTest(path=path.name):
                    self.assertEqual(path.read_bytes(), original)

    def test_run_writes_all_artifacts_and_keeps_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw.csv", root / "processed" / "clean.csv"
            report_path, rejected_path = root / "reports" / "report.json", root / "reports" / "rejected.csv"
            pd.DataFrame([listing(), listing(area=None)]).to_csv(source, index=False)
            original = source.read_bytes()
            report = run_cleaning(source, output, report_path, rejected_path, CleaningConfig(reference_year=2026))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(len(pd.read_csv(output)), 1)
            self.assertEqual(len(pd.read_csv(rejected_path)), 1)
            saved_report = json.loads(report_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(saved_report["rows_after"], 1)
            self.assertEqual(report["rows_after"], 1)

    def test_input_cannot_be_overwritten_by_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame([listing()]).to_csv(source, index=False)
            original = source.read_bytes()
            ordinary = [root / "clean.csv", root / "report.json", root / "rejected.csv"]
            for index in range(3):
                destinations = ordinary.copy()
                destinations[index] = source
                with self.subTest(index=index), self.assertRaises(ValueError):
                    run_cleaning(source, *destinations)
                self.assertEqual(source.read_bytes(), original)

    def test_outputs_cannot_share_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, collision = root / "source.csv", root / "collision.csv"
            pd.DataFrame([listing()]).to_csv(source, index=False)
            with self.assertRaises(ValueError):
                run_cleaning(source, collision, root / "report.json", collision)


if __name__ == "__main__":
    unittest.main()

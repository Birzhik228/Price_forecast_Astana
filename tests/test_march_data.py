"""Проверки адаптера Turarr на синтетических строках и временных файлах."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from clean import COLUMNS
from config import PROJECT_ROOT
from march_data import adapt_march_schema, prepare_march_data


def raw_listing(**changes: object) -> dict[str, object]:
    row = {
        "url": "https://krisha.kz/a/show/123456789",
        "title": "2-комнатная квартира · 64.5 м² · 9/9 этаж, Тестовая улица 25 — парк",
        "price": "45500000〒", "Площадь": "64.5 м², жилая — 40 м², Площадь кухни — 10.8 м²",
        "Этаж": "9 из 9", "Год постройки": "2022", "Город": "Астана, Сарайшык р-н",
        "Тип дома": "Кирпичный", "Состояние квартиры": "Свежий ремонт", "Жилой комплекс": "Тестовый ЖК",
    }
    row.update(changes)
    return row


class MarchSchemaTests(unittest.TestCase):
    def test_explicit_fields_unicode_and_address(self) -> None:
        original = pd.DataFrame([raw_listing()], index=[999])
        before = original.copy(deep=True)
        result = adapt_march_schema(original)
        self.assertEqual(list(result.columns), list(COLUMNS))
        row = result.iloc[0]
        self.assertEqual(row["source_row"], 1)
        self.assertEqual(row["price"], "45500000₸")
        self.assertEqual(row["area"], 64.5)
        self.assertEqual(row["rooms"], 2)
        self.assertEqual((row["floor"], row["total_floors"]), (9, 9))
        self.assertEqual(row["address"], "тестовая улица 25 — парк")
        self.assertEqual(row["district"], "Сарайшык")
        self.assertEqual(row["building_type"], "кирпичный")
        self.assertEqual(row["condition"], "свежий ремонт")
        self.assertTrue(pd.isna(row["collected_at"]))
        pd.testing.assert_frame_equal(original, before)

    def test_districts_remain_distinct_and_city_only_is_missing(self) -> None:
        cities = ["Астана, Сарайшык р-н", "Астана, Сарыарка р-н", "Астана, р-н Байконур", "Астана, Есильский р-н", "Астана"]
        result = adapt_march_schema(pd.DataFrame([raw_listing(**{"Город": city}) for city in cities]))
        self.assertEqual(result["district"].iloc[:4].tolist(), ["Сарайшык", "Сарыарка", "Байконур", "Есиль"])
        self.assertTrue(pd.isna(result.iloc[4]["district"]))

    def test_optional_missing_not_filled_from_title(self) -> None:
        rows = [raw_listing(**{"Этаж": "", "Тип дома": "", "Состояние квартиры": "", "Жилой комплекс": ""}),
                raw_listing(**{"Этаж": "6"})]
        result = adapt_march_schema(pd.DataFrame(rows))
        for field in ("floor", "total_floors", "building_type", "condition", "residential_complex"):
            self.assertTrue(pd.isna(result.iloc[0][field]), field)
        self.assertEqual(result.iloc[1]["floor"], 6)
        self.assertTrue(pd.isna(result.iloc[1]["total_floors"]))

    def test_ambiguous_area_and_floor_not_extracted_as_arbitrary_numbers(self) -> None:
        row = raw_listing(**{"Площадь": "64.5 или 80 м²", "Этаж": "4 или 5 из 9"})
        result = adapt_march_schema(pd.DataFrame([row]))
        for field in ("area", "floor", "total_floors"):
            self.assertTrue(pd.isna(result.iloc[0][field]))
        reasons = {item["reason"] for item in result.attrs["parsing_anomalies"]}
        self.assertEqual(reasons, {"unrecognized_total_area", "unrecognized_floor_pair"})

    def test_area_column_is_authoritative_and_disagreement_reported(self) -> None:
        result = adapt_march_schema(pd.DataFrame([raw_listing(**{"Площадь": "65 м²"})]))
        self.assertEqual(result.iloc[0]["area"], 65)
        self.assertIn("title_area_differs_from_area_column", {item["reason"] for item in result.attrs["parsing_anomalies"]})

    def test_missing_required_rooms_and_non_exact_prices_are_flagged(self) -> None:
        rows = [raw_listing(title="Квартира без числа комнат"), raw_listing(price="от 45500000〒"),
                raw_listing(title="21-комнатная квартира · 64.5 м², Тестовая улица 25")]
        result = adapt_march_schema(pd.DataFrame(rows))
        self.assertEqual(result.attrs["adapter_rejections"], {1: "invalid_required_rooms", 2: "non_exact_price", 3: "invalid_required_rooms"})

    def test_schema_requires_unambiguous_source_columns(self) -> None:
        with self.assertRaises(ValueError):
            adapt_march_schema(pd.DataFrame([raw_listing()]).drop(columns="title"))
        duplicated = pd.DataFrame([raw_listing()])
        duplicated.columns = ["title"] * len(duplicated.columns)
        with self.assertRaises(ValueError):
            adapt_march_schema(duplicated)


class MarchFileTests(unittest.TestCase):
    def setUp(self) -> None:
        # Указанный workspace доступен также в ограниченном Windows-окружении.
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT, prefix=".march-test-")
        self.root = Path(self.temporary.name).resolve()
        if not self.root.is_relative_to(PROJECT_ROOT.resolve()) or self.root == PROJECT_ROOT.resolve():
            raise AssertionError("Временная директория должна быть внутри workspace.")
        self.addCleanup(self.temporary.cleanup)
        self.source = self.root / "raw.csv"
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "clean.csv"
        self.rejected = self.root / "rejected.csv"
        self.report = self.root / "report.json"
        self.addCleanup(patch.stopall)
        patch("march_data.SOURCE_MANIFEST_PATH", self.manifest).start()

    def write_source(self, rows: list[dict[str, object]]) -> None:
        frame = pd.DataFrame(rows)
        frame.to_csv(self.source, index=False, encoding="utf-8-sig")
        payload = self.source.read_bytes()
        self.manifest.write_text(json.dumps({
            "sha256": hashlib.sha256(payload).hexdigest(), "file_bytes": len(payload),
            "rows": len(frame), "columns": frame.columns.tolist(), "declared_data_period": "March 2026",
        }), encoding="utf-8")

    def run_adapter(self) -> dict:
        return prepare_march_data(self.source, self.output, self.rejected, self.report)

    def test_invalid_first_row_does_not_displace_valid_duplicate_and_positions_survive(self) -> None:
        self.write_source([
            raw_listing(title="Квартира без числа комнат"), raw_listing(),
            raw_listing(price="от 45500000〒", url="https://krisha.kz/a/show/987654321"),
            raw_listing(url="https://krisha.kz/a/show/111111111", price="46000000〒"),
        ])
        source_bytes = self.source.read_bytes()
        report = self.run_adapter()
        result = pd.read_csv(self.output)
        rejected = pd.read_csv(self.rejected)
        self.assertEqual(result["source_row"].tolist(), [2, 4])
        self.assertEqual(rejected["source_row"].tolist(), [1, 3])
        self.assertTrue(pd.api.types.is_integer_dtype(result["source_row"]))
        self.assertTrue(pd.api.types.is_integer_dtype(rejected["source_row"]))
        self.assertEqual(rejected["reason"].tolist(), ["invalid_required_rooms", "non_exact_price"])
        self.assertEqual(rejected.iloc[1]["price"], "от 45500000₸")
        self.assertEqual(report["rows_before"], 4)
        self.assertEqual(report["rows_after"], 2)
        self.assertEqual(sum(report["removed"].values()), 2)
        self.assertEqual(report["input_sha256"], hashlib.sha256(source_bytes).hexdigest())
        self.assertEqual(report["output_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertTrue(result["collected_at"].isna().all())

    def test_precise_price_and_missing_values_are_retained(self) -> None:
        self.write_source([raw_listing(**{"Этаж": "", "Состояние квартиры": ""})])
        report = self.run_adapter()
        result = pd.read_csv(self.output)
        self.assertEqual(result.iloc[0]["price"], 45_500_000)
        self.assertTrue(result[["floor", "total_floors", "condition", "collected_at"]].isna().all().all())
        self.assertEqual(report["rows_removed"], 0)
        self.assertEqual(report["missing_after"]["condition"], 1)

    def test_checksum_failure_preserves_existing_outputs(self) -> None:
        self.write_source([raw_listing()])
        for destination in (self.output, self.rejected, self.report):
            destination.write_bytes(b"unchanged")
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.run_adapter()
        for destination in (self.output, self.rejected, self.report):
            self.assertEqual(destination.read_bytes(), b"unchanged")

    def test_malformed_record_rejected_even_with_valid_hash(self) -> None:
        self.write_source([raw_listing()])
        payload = self.source.read_bytes().rstrip(b"\r\n") + b",unexpected\n"
        self.source.write_bytes(payload)
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest.update(sha256=hashlib.sha256(payload).hexdigest(), file_bytes=len(payload))
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ожидалось"):
            self.run_adapter()
        self.assertFalse(self.output.exists())

    def test_source_and_result_paths_must_differ(self) -> None:
        self.write_source([raw_listing()])
        source_bytes = self.source.read_bytes()
        with self.assertRaises(ValueError):
            prepare_march_data(self.source, self.source, self.rejected, self.report)
        self.assertEqual(self.source.read_bytes(), source_bytes)

    def test_all_invalid_leaves_outputs_unchanged(self) -> None:
        self.write_source([raw_listing(price="от 45500000〒")])
        self.output.write_bytes(b"old result")
        with self.assertRaisesRegex(ValueError, "не осталось строк"):
            self.run_adapter()
        self.assertEqual(self.output.read_bytes(), b"old result")


if __name__ == "__main__":
    unittest.main()

"""Строгий адаптер исходного CSV Turarr без импутации и изменения января."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from clean import (
    COLUMNS, DISTRICTS, CleaningConfig, atomic_write, clean_dataframe,
    missing_counts, normalize_district, normalize_text, parse_numeric,
)
from config import PROJECT_ROOT, RAW_DATA_DIR


SOURCE_MANIFEST_PATH = PROJECT_ROOT / "data" / "turarr_march_2026.source.json"
MARCH_RAW_PATH = RAW_DATA_DIR / "turarr_march_2026" / "astana_apartments_raw.csv"
MARCH_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "march_2026"
MARCH_REPORT_PATH = PROJECT_ROOT / "reports" / "march_update" / "cleaning_report.json"
REQUIRED_SOURCE_COLUMNS = {
    "url", "title", "price", "Площадь", "Этаж", "Год постройки",
    "Город", "Тип дома", "Состояние квартиры", "Жилой комплекс",
}
AREA_VALUE = r"\d+(?:[.,]\d+)?\s*м2"
AREA_PATTERN = re.compile(
    rf"(?P<total>{AREA_VALUE})(?:,\s*(?:жилая|жилая площадь|площадь кухни)\s*[—-]\s*{AREA_VALUE})*"
)
TITLE_PATTERN = re.compile(
    rf"(?P<rooms>\d+)-комнатная квартира\s*·\s*(?P<area>{AREA_VALUE})"
    r"(?:\s*·\s*(?P<floor>\d+)(?:/(?P<total_floors>\d+))?\s*этаж)?"
    r",\s*(?P<address>.+)"
)
ROOMS_PATTERN = re.compile(r"^(?P<rooms>\d+)-комнатная квартира(?:\s*·|$)")
FLOOR_PATTERN = re.compile(r"(?P<floor>\d+)(?:\s+из\s+(?P<total_floors>\d+))?")


def _normalized(value: object, *, lowercase: bool = False) -> str | None:
    result = normalize_text(value, lowercase=lowercase)
    return None if pd.isna(result) else str(result)


def _adapt_with_diagnostics(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[int, str]]:
    if not raw.columns.is_unique:
        raise ValueError("В исходной таблице Turarr повторяются колонки.")
    if missing := REQUIRED_SOURCE_COLUMNS - set(raw.columns):
        raise ValueError(f"В схеме Turarr отсутствуют колонки: {', '.join(sorted(missing))}.")
    records: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    rejected: dict[int, str] = {}

    def anomaly(source_row: int, field: str, value: object, reason: str, action: str) -> None:
        anomalies.append({
            "source_row": source_row, "field": field, "source_value": _normalized(value),
            "reason": reason, "action": action,
        })

    for source_row, original in enumerate(raw.to_dict(orient="records"), start=1):
        row: dict[str, Any] = {name: pd.NA for name in COLUMNS}
        row["source_row"] = source_row
        for target, source in (
            ("url", "url"), ("year_built", "Год постройки"),
            ("building_type", "Тип дома"), ("condition", "Состояние квартиры"),
            ("residential_complex", "Жилой комплекс"),
        ):
            row[target] = normalize_text(
                original[source], lowercase=target in {"building_type", "condition", "residential_complex"},
            )

        price_text = _normalized(original["price"], lowercase=True)
        if price_text is not None:
            # U+3012 — буквальный маркер валюты этого закреплённого источника.
            normalized_price = price_text.replace("〒", "₸")
            if re.match(r"^(?:от|до)\s+", normalized_price):
                rejected[source_row] = "non_exact_price"
                row["price"] = normalized_price
                anomaly(source_row, "price", original["price"], "non_exact_price", "reject")
            else:
                row["price"] = normalized_price
                if not math.isfinite(parse_numeric(normalized_price, "price")):
                    anomaly(source_row, "price", original["price"], "unrecognized_price", "reject_in_general_cleaning")

        area_text = _normalized(original["Площадь"], lowercase=True)
        area_match = AREA_PATTERN.fullmatch(area_text) if area_text else None
        if area_match is not None:
            row["area"] = parse_numeric(area_match.group("total"), "area")
        elif area_text is not None:
            anomaly(source_row, "Площадь", original["Площадь"], "unrecognized_total_area", "reject_in_general_cleaning")

        title = _normalized(original["title"], lowercase=True)
        rooms_match = ROOMS_PATTERN.match(title) if title else None
        if rooms_match is not None and 1 <= int(rooms_match.group("rooms")) <= 20:
            row["rooms"] = int(rooms_match.group("rooms"))
        else:
            rejected.setdefault(source_row, "invalid_required_rooms")
            anomaly(source_row, "title", original["title"], "invalid_required_rooms", "reject")
        title_match = TITLE_PATTERN.fullmatch(title) if title else None
        if title_match is not None:
            row["address"] = normalize_text(title_match.group("address"), lowercase=True)
            title_area = parse_numeric(title_match.group("area"), "area")
            if not pd.isna(row["area"]) and not math.isclose(row["area"], title_area, rel_tol=0, abs_tol=1e-6):
                anomaly(source_row, "title", original["title"], "title_area_differs_from_area_column", "retain_total_from_area_column")
        else:
            anomaly(source_row, "title", original["title"], "unrecognized_title_address", "address_to_missing")

        floor_text = _normalized(original["Этаж"], lowercase=True)
        floor_match = FLOOR_PATTERN.fullmatch(floor_text) if floor_text else None
        if floor_match is not None:
            row["floor"] = int(floor_match.group("floor"))
            if floor_match.group("total_floors") is not None:
                row["total_floors"] = int(floor_match.group("total_floors"))
            # Заголовок не используется для заполнения отсутствующего поля Этаж.
            if title_match is not None:
                for field in ("floor", "total_floors"):
                    title_value = title_match.group(field)
                    if title_value is not None and not pd.isna(row[field]) and int(title_value) != row[field]:
                        anomaly(source_row, "title", original["title"], f"title_{field}_differs_from_floor_column", "retain_floor_column")
        elif floor_text is not None:
            anomaly(source_row, "Этаж", original["Этаж"], "unrecognized_floor_pair", "floor_and_total_floors_to_missing")

        city = _normalized(original["Город"], lowercase=True)
        city_match = re.fullmatch(r"астана(?:\s*,\s*(?P<district>.+))?", city) if city else None
        if city_match is not None:
            row["district"] = normalize_district(city_match.group("district"))
            if not pd.isna(row["district"]) and row["district"] not in set(DISTRICTS.values()):
                anomaly(source_row, "Город", original["Город"], "unknown_astana_district", "retain_normalized_category")
        elif city is not None:
            rejected.setdefault(source_row, "unsupported_city")
            anomaly(source_row, "Город", original["Город"], "unsupported_city", "reject")
        # collected_at намеренно отсутствует: дата скачивания не является датой наблюдения.
        records.append(row)
    frame = pd.DataFrame(records, columns=list(COLUMNS))
    return frame, anomalies, rejected


def adapt_march_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Перевести схему Turarr в COLUMNS; диагностика доступна в attrs.

    Это адаптация, а не окончательная выборка: строки с нестрогой ценой или
    отсутствующими обязательными комнатами исключает prepare_march_data.
    Позиции source_row относятся к raw, независимо от индекса DataFrame.
    """
    frame, anomalies, rejected = _adapt_with_diagnostics(raw)
    frame.attrs["parsing_anomalies"] = anomalies
    frame.attrs["adapter_rejections"] = rejected
    return frame


def _read_verified_source(path: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != str(manifest.get("sha256", "")).lower():
        raise ValueError("SHA-256 мартовского исходника не совпадает с закреплённым манифестом.")
    if len(payload) != manifest.get("file_bytes"):
        raise ValueError("Размер мартовского исходника не совпадает с манифестом.")
    reader = csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    header = next(reader, [])
    if not header or len(header) != len(set(header)):
        raise ValueError("Отсутствующий или неоднозначный заголовок CSV Turarr.")
    if header != manifest.get("columns"):
        raise ValueError("Колонки CSV Turarr не совпадают с закреплённым манифестом.")
    records = []
    for source_row, record in enumerate(reader, start=1):
        if len(record) != len(header):
            raise ValueError(f"Строка {source_row}: ожидалось {len(header)} полей, получено {len(record)}.")
        records.append(record)
    if len(records) != manifest.get("rows"):
        raise ValueError("Число записей CSV Turarr не совпадает с закреплённым манифестом.")
    return pd.DataFrame(records, columns=header, dtype="string"), manifest, digest


def _clean_preserving_positions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_rows = frame["source_row"].to_numpy()
    # prepare_schema внутри общего очистителя нумерует переданную часть с 1.
    cleaned, rejected, report = clean_dataframe(frame, CleaningConfig())
    for part in (cleaned, rejected):
        part["source_row"] = pd.Series(
            [int(source_rows[int(position) - 1]) for position in part["source_row"]],
            index=part.index, dtype="int64",
        )
    for candidate in report["review_candidates"]:
        candidate["source_row"] = int(source_rows[candidate["source_row"] - 1])
    return cleaned, rejected, report


def prepare_march_data(
    raw_path: Path, output_path: Path, rejected_path: Path, report_path: Path,
) -> dict[str, Any]:
    """Проверить исходные байты, адаптировать, очистить и сохранить отдельно."""
    raw_path, output_path, rejected_path, report_path = (
        Path(path).resolve() for path in (raw_path, output_path, rejected_path, report_path)
    )
    all_paths = (raw_path, output_path, rejected_path, report_path, SOURCE_MANIFEST_PATH.resolve())
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Исходник, манифест, очищенный CSV, исключённые строки и отчёт должны иметь разные пути.")
    for path in (output_path, rejected_path, report_path):
        if path.is_relative_to(RAW_DATA_DIR.resolve()):
            raise ValueError("Результаты адаптации нельзя записывать в data/raw.")
    raw, manifest, input_digest = _read_verified_source(raw_path)
    adapted, anomalies, adapter_reasons = _adapt_with_diagnostics(raw)
    adapter_invalid = adapted["source_row"].isin(adapter_reasons)
    cleaned, general_rejected, general_report = _clean_preserving_positions(adapted.loc[~adapter_invalid])
    invalid_cleaned, invalid_rejected, invalid_report = _clean_preserving_positions(adapted.loc[adapter_invalid])
    adapter_rejected = pd.concat(
        [invalid_cleaned, invalid_rejected.drop(columns="reason")], ignore_index=True,
    )
    adapter_rejected["reason"] = adapter_rejected["source_row"].map(adapter_reasons)
    # Для неопределённой цены сохраняем исходное выражение, а не ошибочную сумму.
    if len(adapter_rejected):
        original_prices = adapted.set_index("source_row")["price"]
        non_exact = adapter_rejected["reason"].eq("non_exact_price")
        adapter_rejected["price"] = adapter_rejected["price"].astype(object)
        adapter_rejected.loc[non_exact, "price"] = adapter_rejected.loc[non_exact, "source_row"].map(original_prices)
    rejected = pd.concat([general_rejected, adapter_rejected], ignore_index=True).sort_values("source_row", kind="stable").reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("После адаптации и очистки не осталось строк; выходные файлы не изменены.")
    reason_counts = dict(general_report["removed"])
    for reason in sorted(set(adapter_reasons.values())):
        reason_counts[reason] = sum(value == reason for value in adapter_reasons.values())
    report = dict(general_report)
    report.update({
        "adapter": "turarr_march_2026_v1", "rows_before": len(raw),
        "rows_after": len(cleaned), "rows_removed": len(rejected), "removed": reason_counts,
        "rows_entering_general_cleaning": int((~adapter_invalid).sum()),
        "rows_rejected_by_adapter": int(adapter_invalid.sum()),
        "general_cleaning_report": general_report,
        "missing_before": {
            name: general_report["missing_before"][name] + invalid_report["missing_before"][name]
            for name in general_report["missing_before"]
        },
        "missing_after": missing_counts(cleaned),
        "input_file": str(raw_path), "output_file": str(output_path),
        "rejected_file": str(rejected_path), "source_manifest": str(SOURCE_MANIFEST_PATH.resolve()),
        "input_sha256": input_digest, "original_raw_sha256": input_digest,
        "declared_data_period": manifest.get("declared_data_period"),
        "row_collection_dates_available": False,
        "parsing_anomalies": anomalies,
        "parsing_anomaly_counts": dict(pd.Series([item["reason"] for item in anomalies], dtype="string").value_counts().astype(int).items()),
        "adapter_normalizations": {
            "price_currency_marker_u3012_to_tenge": int(raw["price"].str.contains("〒", regex=False).sum()),
            "standalone_floor_without_total": int(raw["Этаж"].map(lambda value: bool(re.fullmatch(r"\d+", _normalized(value) or ""))).sum()),
        },
        "source_missing_counts": {
            name: int(raw[name].map(normalize_text).isna().sum()) for name in raw.columns
        },
        "category_values": {
            name: sorted(cleaned[name].dropna().unique().tolist())
            for name in ("district", "building_type", "condition")
        },
        "ignored_source_columns": sorted(set(raw.columns) - REQUIRED_SOURCE_COLUMNS),
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
    })
    for key in ("numeric_parse_failures", "invalid_optional_to_missing"):
        report[key] = {
            name: general_report[key][name] + invalid_report[key][name] for name in general_report[key]
        }
    for key in ("inconsistent_floor_pairs_to_missing", "invalid_urls_to_missing"):
        report[key] = general_report[key] + invalid_report[key]
    report["notes"] = [
        *general_report["notes"],
        "The period is publisher-declared March 2026; no row-level collection timestamp is available.",
        "collected_at remains missing; neither publication nor download time is substituted.",
        "Prices introduced by от/до are not exact asking prices and are excluded before URL deduplication.",
        "Missing, unrecognized or out-of-range required rooms are rejected before URL deduplication.",
        "Missing optional fields are preserved; no imputation or target-quantile trimming is performed.",
        "Standalone floor values do not imply total_floors; source Этаж is authoritative over the title.",
        "source_row refers to the original raw CSV record, excluding its header, including adapter rejections.",
        "missing_before describes adapted fields before general cleaning; source_missing_counts describes raw fields.",
        "Counts of normalization failures cover all raw records; removal reasons use adapter checks first.",
    ]
    if len(cleaned) + len(rejected) != len(raw) or sum(reason_counts.values()) != len(rejected):
        raise AssertionError("Нарушен баланс строк при адаптации Turarr.")
    output_bytes = cleaned.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    rejected_bytes = rejected.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    report["output_sha256"] = hashlib.sha256(output_bytes).hexdigest()
    report["rejected_sha256"] = hashlib.sha256(rejected_bytes).hexdigest()
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(output_path, output_bytes)
    atomic_write(rejected_path, rejected_bytes)
    atomic_write(report_path, report_bytes)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=MARCH_RAW_PATH)
    parser.add_argument("--output", type=Path, default=MARCH_OUTPUT_DIR / "listings_clean.csv")
    parser.add_argument("--rejected-output", type=Path, default=MARCH_OUTPUT_DIR / "listings_rejected.csv")
    parser.add_argument("--report", type=Path, default=MARCH_REPORT_PATH)
    args = parser.parse_args()
    try:
        report = prepare_march_data(
            PROJECT_ROOT / args.input, PROJECT_ROOT / args.output,
            PROJECT_ROOT / args.rejected_output, PROJECT_ROOT / args.report,
        )
    except (ValueError, OSError, csv.Error) as error:
        print(f"Ошибка адаптации: {error}")
        return 1
    print(json.dumps({key: report[key] for key in ("rows_before", "rows_after", "removed", "parsing_anomaly_counts", "output_file")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

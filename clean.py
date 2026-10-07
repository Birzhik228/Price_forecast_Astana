"""Очистка CSV Kaggle или нашего парсера без изменения исходных данных."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from config import (
    CLEANING_REPORT_PATH, CLEAN_LISTINGS_PATH, KAGGLE_LISTINGS_PATH,
    PROJECT_ROOT, RAW_DATA_DIR, REFERENCE_YEAR, REJECTED_LISTINGS_PATH,
)


FEATURE_COLUMNS = (
    "district", "area", "rooms", "floor", "total_floors",
    "year_built", "building_type", "condition",
)
NUMERIC_COLUMNS = ("price", "area", "rooms", "floor", "total_floors", "year_built")
COLUMNS = (
    "source_row", "price", "area", "rooms", "floor", "total_floors",
    "year_built", "building_type", "condition", "district",
    "residential_complex", "address", "url", "collected_at",
)
ALIASES = {
    "price_local": "price",
    "house_type": "building_type",
    "residential_complex_name": "residential_complex",
    "parsed_at": "collected_at",
}
MISSING_MARKERS = {"", "nan", "none", "null", "na", "n/a", "не указано", "не указан", "-", "—"}
DISTRICTS = {
    "нура": "Нура", "есиль": "Есиль", "есильский": "Есиль",
    "сарыарка": "Сарыарка", "сарыаркинский": "Сарыарка",
    "сарайшык": "Сарайшык", "алматы": "Алматы",
    "байконур": "Байконур", "байконыр": "Байконур", "байқоңыр": "Байконур",
}
DEDUP_COLUMNS = ("price", *FEATURE_COLUMNS, "address", "residential_complex")


@dataclass(frozen=True)
class CleaningConfig:
    """Широкие фиксированные границы, а не квантили всего датасета."""

    price_min: float = 1_000_000
    price_max: float = 2_000_000_000
    area_min: float = 10.0
    area_max: float = 1_000.0
    reference_year: int = REFERENCE_YEAR

    def __post_init__(self) -> None:
        for low, high in ((self.price_min, self.price_max), (self.area_min, self.area_max)):
            if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
                raise ValueError("Границы должны быть конечными: 0 < минимум < максимум.")
        if not isinstance(self.reference_year, int) or not 1900 <= self.reference_year <= 2100:
            raise ValueError("Опорный год должен быть целым числом от 1900 до 2100.")


def normalize_text(value: object, *, lowercase: bool = False) -> object:
    """Сохраняет пропуски пропусками, унифицирует Unicode и пробелы."""
    if pd.isna(value):
        return pd.NA
    text = " ".join(unicodedata.normalize("NFKC", str(value)).split())
    if text.casefold() in MISSING_MARKERS:
        return pd.NA
    return text.casefold() if lowercase else text


def parse_numeric(value: object, kind: str = "number") -> float:
    """Разбирает целое значение, не выдёргивая цифры из произвольного текста."""
    normalized = normalize_text(value, lowercase=True)
    if pd.isna(normalized):
        return np.nan
    text = str(normalized)
    # Пробелы внутри числа допустимы только между группами из трёх цифр.
    number = r"(?P<number>[+-]?(?:\d{1,3}(?: \d{3})+|\d+)(?:[.,]\d+)?(?:e[+-]?\d+)?)"
    suffix = ""
    if kind == "price":
        suffix = r"(?:\s*(?P<scale>тыс|млн|млрд)\.?)?(?:\s*(?:₸|тг\.?|тенге|kzt))?"
    elif kind == "area":
        suffix = r"(?:\s*(?:м2|кв\.?\s*м\.?))?"
    elif kind != "number":
        raise ValueError(f"Неизвестный вид числа: {kind}")
    match = re.fullmatch(number + suffix, text)
    if match is None:
        return np.nan
    result = float(match.group("number").replace(" ", "").replace(",", "."))
    if kind == "price":
        result *= {None: 1, "тыс": 1_000, "млн": 1_000_000, "млрд": 1_000_000_000}[match.group("scale")]
    return result if math.isfinite(result) else np.nan


def normalize_district(value: object) -> object:
    text = normalize_text(value, lowercase=True)
    if pd.isna(text):
        return pd.NA
    text = re.sub(r"^(?:р-н|район)\s+|\s+(?:р-н|район)$", "", str(text)).strip()
    return DISTRICTS.get(text, text)


def normalize_url(value: object) -> object:
    text = normalize_text(value)
    if pd.isna(text):
        return pd.NA
    try:
        parsed = urlsplit(str(text))
    except ValueError:
        return pd.NA
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {"krisha.kz", "www.krisha.kz"}:
        return pd.NA
    match = re.fullmatch(r"/a/show/(\d+)/?", parsed.path)
    return f"https://krisha.kz/a/show/{match.group(1)}" if match else pd.NA


def missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(frame[column].isna().sum()) for column in COLUMNS if column != "source_row"}


def prepare_schema(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.columns.duplicated().any():
        raise ValueError("В исходной таблице повторяются имена колонок.")
    for old, new in ALIASES.items():
        if old in raw.columns and new in raw.columns:
            raise ValueError(f"Неоднозначная схема: одновременно присутствуют {old} и {new}.")
    data = raw.rename(columns=ALIASES).copy().reset_index(drop=True)
    required = {"price", "area", "rooms"}
    if missing := required - set(data.columns):
        raise ValueError(f"Не хватает обязательных колонок: {', '.join(sorted(missing))}.")
    # Явный список исключает price_usd и price_per_m2 — источники утечки цены.
    for column in COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    data = data.loc[:, list(COLUMNS)]
    data["source_row"] = np.arange(1, len(data) + 1)
    for column in COLUMNS:
        if column != "source_row":
            data[column] = data[column].map(normalize_text).astype("string")
    return data


def clean_dataframe(raw: pd.DataFrame, config: CleaningConfig | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Возвращает очищенные строки, исключённые строки и воспроизводимый отчёт."""
    settings = config or CleaningConfig()
    data = prepare_schema(raw)
    before = missing_counts(data)
    numeric_failures: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        kind = column if column in {"price", "area"} else "number"
        original = data[column].notna()
        data[column] = data[column].map(lambda value: parse_numeric(value, kind)).astype("Float64")
        numeric_failures[column] = int((original & data[column].isna()).sum())
    for column in ("building_type", "condition", "residential_complex", "address"):
        data[column] = data[column].map(lambda value: normalize_text(value, lowercase=True)).astype("string")
    data["district"] = data["district"].map(normalize_district).astype("string")
    original_urls = data["url"].notna()
    data["url"] = data["url"].map(normalize_url).astype("string")
    invalid_urls = int((original_urls & data["url"].isna()).sum())

    # Ошибка в необязательном признаке не уничтожает корректную цену квартиры.
    invalid_optional: dict[str, int] = {}
    optional_limits = {
        "rooms": (1, 20), "floor": (1, 100), "total_floors": (1, 100),
        "year_built": (1850, settings.reference_year + 5),
    }
    for column, (low, high) in optional_limits.items():
        invalid = (data[column].notna() & (
            ~data[column].between(low, high) | data[column].mod(1).ne(0)
        )).fillna(False)
        invalid_optional[column] = int(invalid.sum())
        data.loc[invalid, column] = pd.NA
        data[column] = data[column].astype("Int64")
    inconsistent_floors = (data["floor"] > data["total_floors"]).fillna(False)
    inconsistent_count = int(inconsistent_floors.sum())
    # По двум противоречащим числам нельзя определить, какое из них верное.
    data.loc[inconsistent_floors, ["floor", "total_floors"]] = pd.NA

    removed: dict[str, int] = {}
    rejected_parts: list[pd.DataFrame] = []

    def reject(mask: pd.Series, reason: str) -> None:
        nonlocal data
        mask = mask.fillna(False)
        removed[reason] = int(mask.sum())
        rejected_parts.append(data.loc[mask].assign(reason=reason))
        data = data.loc[~mask].copy()

    reject(data[["price", "area"]].isna().any(axis=1), "missing_price_or_area")
    reject(~data["price"].between(settings.price_min, settings.price_max), "price_out_of_bounds")
    reject(~data["area"].between(settings.area_min, settings.area_max), "area_out_of_bounds")
    reject(data["url"].notna() & data["url"].duplicated(keep="first"), "duplicate_url")

    # Для неполных строк совпадения нескольких признаков недостаточно.
    core_present = data[["rooms", "floor", "total_floors"]].notna().all(axis=1)
    location_or_complete = data["address"].notna() | data[list(FEATURE_COLUMNS)].notna().all(axis=1)
    eligible = core_present & location_or_complete
    feature_duplicates = pd.Series(False, index=data.index)
    feature_duplicates.loc[eligible] = data.loc[eligible].duplicated(subset=list(DEDUP_COLUMNS), keep="first")
    reject(feature_duplicates, "duplicate_features")

    # Только диагностика: сомнительно дёшево не означает доказанную ошибку.
    ratio = data["price"] / data["area"]
    review_mask = ratio.lt(100_000)
    review_candidates = [
        {"source_row": int(row["source_row"]),
         "url": None if pd.isna(row["url"]) else str(row["url"]),
         "reason": "price_per_m2_below_100000; retained_for_review"}
        for _, row in data.loc[review_mask, ["source_row", "url"]].iterrows()
    ]
    rejected = pd.concat(rejected_parts, ignore_index=True).sort_values("source_row", kind="stable")
    cleaned = data.reset_index(drop=True)
    report: dict[str, object] = {
        "rows_before": len(raw), "rows_after": len(cleaned), "removed": removed,
        "rows_removed": len(rejected),
        "missing_before": before, "missing_after": missing_counts(cleaned),
        "numeric_parse_failures": numeric_failures,
        "invalid_optional_to_missing": invalid_optional,
        "inconsistent_floor_pairs_to_missing": inconsistent_count,
        "invalid_urls_to_missing": invalid_urls,
        "thresholds": asdict(settings),
        "optional_limits": {key: list(bounds) for key, bounds in optional_limits.items()},
        "dedup_columns": list(DEDUP_COLUMNS),
        "dedup_keep": "first_valid_row_in_source_order",
        "review_candidates": review_candidates,
        "ignored_source_columns": [name for name in raw.columns if ALIASES.get(name, name) not in COLUMNS],
        "notes": [
            "Optional missing values are not imputed before train/test split.",
            "Feature duplicates are candidates, not proven apartment identities; rejected rows are preserved.",
            "collected_at retains the source text and its original timezone information (if any).",
            "source_row is a 1-based record position, excluding the header, in the original CSV.",
        ],
    }
    return cleaned, rejected.reset_index(drop=True), report


def atomic_write(path: Path, payload: bytes) -> None:
    """Подменяет готовый файл только после успешной записи во временный."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_cleaning(input_path: Path, output_path: Path, report_path: Path,
                 rejected_path: Path, config: CleaningConfig | None = None) -> dict[str, object]:
    paths = [path.resolve() for path in (input_path, output_path, report_path, rejected_path)]
    if len(set(paths)) != len(paths):
        raise ValueError("Исходный CSV, результат, исключённые строки и отчёт должны иметь разные пути.")
    source, output, report_file, rejected_file = paths
    for path in (output, report_file, rejected_file):
        if path.is_relative_to(RAW_DATA_DIR.resolve()):
            raise ValueError("Результаты нельзя записывать в data/raw: исходные файлы сохраняются неизменными.")
    # pandas сам переименовывает повторные заголовки; проверяем их заранее.
    with source.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, strict=True), [])
    if len(header) != len(set(header)):
        raise ValueError("В исходном CSV повторяются имена колонок.")
    raw = pd.read_csv(source, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    cleaned, rejected, report = clean_dataframe(raw, config)
    if cleaned.empty:
        raise ValueError("После очистки не осталось строк. Выходные файлы не изменены; проверьте вход и пороги.")
    output_bytes = cleaned.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    rejected_bytes = rejected.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    report.update({
        "input_file": str(source), "output_file": str(output), "rejected_file": str(rejected_file),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
    })
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(output, output_bytes)
    atomic_write(rejected_file, rejected_bytes)
    atomic_write(report_file, report_bytes)
    return report


def format_report(report: dict[str, object]) -> str:
    lines = [f"Строк до очистки: {report['rows_before']}",
             f"Строк после очистки: {report['rows_after']}",
             f"Исключено всего: {report['rows_removed']}", "", "Причины исключения (без двойного счёта):"]
    labels = {
        "missing_price_or_area": "Нет корректной цены или площади",
        "price_out_of_bounds": "Цена вне заданных границ",
        "area_out_of_bounds": "Площадь вне заданных границ",
        "duplicate_url": "Повторная ссылка",
        "duplicate_features": "Совпадение признаков и сведений о расположении",
    }
    for key, count in report["removed"].items():
        lines.append(f"  {labels[key]}: {count}")
    lines.extend(["", "Пропуски по колонкам: до -> после"])
    for column, count in report["missing_before"].items():
        lines.append(f"  {column}: {count} -> {report['missing_after'][column]}")
    lines.extend(["", f"Сомнительно низкая цена за м², оставлены для проверки: {len(report['review_candidates'])}",
                  "Пропуски признаков сохранены для обработки внутри train-пайплайна."])
    return "\n".join(lines)


def main() -> int:
    defaults = CleaningConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=KAGGLE_LISTINGS_PATH, help="CSV Kaggle или нашего парсера.")
    parser.add_argument("--output", type=Path, default=CLEAN_LISTINGS_PATH, help="Очищенный CSV.")
    parser.add_argument("--report", type=Path, default=CLEANING_REPORT_PATH, help="JSON-отчёт.")
    parser.add_argument("--rejected-output", type=Path, default=REJECTED_LISTINGS_PATH, help="Исключённые строки с причинами.")
    parser.add_argument("--min-price", type=float, default=defaults.price_min)
    parser.add_argument("--max-price", type=float, default=defaults.price_max)
    parser.add_argument("--min-area", type=float, default=defaults.area_min)
    parser.add_argument("--max-area", type=float, default=defaults.area_max)
    parser.add_argument("--reference-year", type=int, default=defaults.reference_year)
    args = parser.parse_args()
    try:
        settings = CleaningConfig(args.min_price, args.max_price, args.min_area, args.max_area, args.reference_year)
        report = run_cleaning(
            PROJECT_ROOT / args.input, PROJECT_ROOT / args.output,
            PROJECT_ROOT / args.report, PROJECT_ROOT / args.rejected_output, settings,
        )
    except (ValueError, OSError, pd.errors.ParserError, csv.Error) as error:
        print(f"Ошибка очистки: {error}")
        return 1
    print(format_report(report))
    print(f"\nОчищенные данные: {report['output_file']}")
    print(f"Исключённые строки: {report['rejected_file']}")
    print(f"Подробный отчёт: {(PROJECT_ROOT / args.report).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

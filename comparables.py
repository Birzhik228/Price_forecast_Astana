"""Исторические аналоги по характеристикам; цены не участвуют в ранжировании.

Каталог не является оценкой качества модели. Для исследовательской проверки
передавайте только разрешённые обучающие строки и исключайте объект запроса.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd


RESULT_COLUMNS = ("similarity_reason", "distance_m", "location_match", "price_per_m2")
LISTING_COLUMNS = frozenset({
    "group", "record_id", "price", "area", "rooms", "district", "url",
    "building_id", "complex_id", "complex_name", "address_normalized",
    "latitude", "longitude", "coordinate_status", "parking_type", "date_label",
})
BUILDING_COLUMNS = frozenset({
    "building_id", "complex_id", "complex_name", "address_normalized", "district",
    "latitude", "longitude", "coordinate_status", "parking_type", "listing_count",
    "display_label",
})
# Та же область допустимых координат, что у локального сборщика каталога.
LATITUDE_BOUNDS = (50.8, 51.4)
LONGITUDE_BOUNDS = (70.8, 71.9)
MAX_DISTANCE_M = 2000.0


@dataclass(frozen=True)
class PropertyCatalog:
    listings: pd.DataFrame
    buildings: pd.DataFrame
    manifest: dict[str, Any]


def _text(value: Any) -> str | None:
    """Не превращать None/NaN/pd.NA в совпадающие строковые идентификаторы."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value else None


def _normalized(value: Any) -> str | None:
    text = _text(value)
    return " ".join(text.casefold().replace("ё", "е").split()) if text else None


def _number(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def safe_listing_url(value: Any) -> str | None:
    """Только ссылки на карточки Krisha; удалить трекинг и использовать HTTPS."""
    if isinstance(value, str) and any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    text = _text(value)
    if not text:
        return None
    try:
        parsed = urlsplit(text)
    except ValueError:
        return None
    if (parsed.scheme.lower() not in {"http", "https"}
            or parsed.netloc.lower() not in {"krisha.kz", "www.krisha.kz"}):
        return None
    match = re.fullmatch(r"/a/show/([0-9]+)/?", parsed.path)
    return f"https://krisha.kz/a/show/{match.group(1)}" if match else None


def _read_catalog_table(root: Path, specification: Any, required: frozenset[str]) -> pd.DataFrame:
    if not isinstance(specification, dict):
        raise ValueError("Манифест каталога не содержит описания CSV.")
    filename, checksum = specification.get("path"), specification.get("sha256")
    if (not isinstance(filename, str) or not filename
            or not isinstance(checksum, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checksum) is None):
        raise ValueError("В манифесте каталога некорректны путь или SHA-256.")
    relative = Path(filename)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("CSV каталога должен находиться внутри его папки.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".csv":
        raise ValueError("CSV каталога должен находиться внутри его папки.")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != checksum.lower():
        raise ValueError(f"Контрольная сумма каталога не совпадает: {filename}.")
    # Проверка и разбор одних байтов исключают смешивание двух версий CSV.
    frame = pd.read_csv(io.BytesIO(content), dtype={
        column: "string" for column in (
            "group", "record_id", "building_id", "complex_id", "url", "district",
        )
    })
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Несовместимая схема каталога: отсутствуют " + ", ".join(sorted(missing)))
    if "rows" in specification and (
        type(specification["rows"]) is not int or specification["rows"] != len(frame)
    ):
        raise ValueError("Количество строк каталога не совпадает с манифестом.")
    if "columns" in specification and specification["columns"] != frame.columns.tolist():
        raise ValueError("Колонки каталога не совпадают с манифестом.")
    return frame


def load_catalog(path: Path) -> PropertyCatalog:
    """Прочитать папку каталога или manifest.json, проверить схему и оба SHA-256."""
    path = Path(path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or not isinstance(manifest.get("files"), dict)):
        raise ValueError("Несовместимая версия или схема манифеста каталога.")
    root = manifest_path.parent.resolve()
    listings = _read_catalog_table(root, manifest["files"].get("listings"), LISTING_COLUMNS)
    buildings = _read_catalog_table(root, manifest["files"].get("buildings"), BUILDING_COLUMNS)
    for frame, column in ((listings, "record_id"), (buildings, "building_id")):
        values = frame[column].map(_text)
        if values.isna().any() or values.duplicated().any():
            raise ValueError(f"Каталог содержит пустые или повторные {column}.")
    known_buildings = set(buildings["building_id"])
    if not set(listings["building_id"].dropna()).issubset(known_buildings):
        raise ValueError("Объявление ссылается на отсутствующий дом в каталоге.")
    return PropertyCatalog(listings=listings, buildings=buildings, manifest=manifest)


def _coordinates(row: dict[str, Any]) -> tuple[float, float] | None:
    # В текущем каталоге нет координат, проверенных по внешнему источнику.
    if _text(row.get("coordinate_status")) != "plausible":
        return None
    latitude, longitude = _number(row.get("latitude")), _number(row.get("longitude"))
    if (latitude is None or longitude is None
            or not LATITUDE_BOUNDS[0] <= latitude <= LATITUDE_BOUNDS[1]
            or not LONGITUDE_BOUNDS[0] <= longitude <= LONGITUDE_BOUNDS[1]):
        return None
    return latitude, longitude


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*first, *second))
    haversine = math.sin((lat2 - lat1) / 2) ** 2 + (
        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_008.8 * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))


def _identity_components(rows: list[dict[str, Any]]) -> list[int]:
    """Связные компоненты по group, URL и record_id, включая цепочки псевдонимов."""
    parent = list(range(len(rows)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for kind, value in (
            ("group", _text(row.get("group"))),
            ("record_id", _text(row.get("record_id"))),
            ("url", safe_listing_url(row.get("url"))),
        ):
            if value is None:
                continue
            key = kind, value
            if key in owners:
                parent[root(index)] = root(owners[key])
            else:
                owners[key] = index
    return [root(index) for index in range(len(rows))]


def _empty(listings: pd.DataFrame) -> pd.DataFrame:
    result = listings.iloc[:0].copy()
    for column in RESULT_COLUMNS:
        result[column] = pd.Series(dtype=float if column in {"distance_m", "price_per_m2"} else str)
    return result.reset_index(drop=True)


def _structural_difference(row: dict[str, Any], query: dict[str, Any]) -> tuple[float, ...]:
    category_penalty = 0.0
    for column in ("building_type", "condition"):
        expected, value = _normalized(query.get(column)), _normalized(row.get(column))
        if expected is not None:
            category_penalty += 0.5 if value is None else float(value != expected)
    differences = []
    for column, missing_penalty in (("floor", 100), ("total_floors", 100), ("year_built", 200)):
        expected, value = _number(query.get(column)), _number(row.get(column))
        differences.append(0.0 if expected is None else missing_penalty if value is None else abs(value - expected))
    return (category_penalty, *differences)


def find_comparables(listings: pd.DataFrame, query: dict[str, Any], limit: int = 5) -> pd.DataFrame:
    """Найти до limit исторических объявлений с теми же комнатами и площадью ±20%.

    Порядок местоположений: дом, ЖК, до 2 км по приблизительным координатам,
    район. Внутри nearby сначала расстояние; дальше близость площади и прочих
    характеристик. Цена используется только для проверки пригодности и вывода.
    Небезопасные/отсутствующие URL не выдаются; разрешённые URL канонизируются.
    exclude_record_id/exclude_group/exclude_url исключают всю связанную группу,
    даже если исходная строка не проходит фильтр цены, площади или числа комнат.
    Без пригодного местоположения возвращается пустая таблица.
    """
    if not isinstance(listings, pd.DataFrame) or not isinstance(query, dict):
        raise ValueError("Нужны таблица объявлений и словарь параметров квартиры.")
    if not {"price", "area", "rooms"}.issubset(listings.columns):
        raise ValueError("Каталог должен содержать price, area и rooms.")
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit < 0:
        raise ValueError("Количество аналогов должно быть неотрицательным целым числом.")
    area, rooms = _number(query.get("area")), _number(query.get("rooms"))
    if area is None or area <= 0 or rooms is None or rooms <= 0 or not rooms.is_integer():
        raise ValueError("Для поиска аналогов укажите положительную площадь и целое число комнат.")
    building, complex_id = _text(query.get("building_id")), _text(query.get("complex_id"))
    district, coordinates = _normalized(query.get("district")), _coordinates(query)
    if limit == 0 or not (building or complex_id or district or coordinates):
        return _empty(listings)

    rows = listings.to_dict("records")
    components = _identity_components(rows)
    excluded: set[int] = set()
    exclusions = {
        "record_id": _text(query.get("exclude_record_id")),
        "group": _text(query.get("exclude_group")),
        "url": safe_listing_url(query.get("exclude_url")),
    }
    for row, component in zip(rows, components):
        if any(value is not None and value == (
            safe_listing_url(row.get(key)) if key == "url" else _text(row.get(key))
        ) for key, value in exclusions.items()):
            excluded.add(component)

    candidates: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
    for row, component in zip(rows, components):
        if component in excluded:
            continue
        price, candidate_area = _number(row.get("price")), _number(row.get("area"))
        url = safe_listing_url(row.get("url"))
        if (price is None or price <= 0 or candidate_area is None or candidate_area <= 0
                or _number(row.get("rooms")) != rooms or url is None):
            continue
        lower, upper = 0.8 * area, 1.2 * area
        # Например, 0.8 * 68.5 даёт 54.800000000000004 в float.
        if ((candidate_area < lower and not math.isclose(candidate_area, lower, rel_tol=1e-12))
                or (candidate_area > upper and not math.isclose(candidate_area, upper, rel_tol=1e-12))):
            continue
        candidate_coordinates = _coordinates(row)
        distance = _distance(coordinates, candidate_coordinates) if coordinates and candidate_coordinates else math.nan
        if building and building == _text(row.get("building_id")):
            priority, location, reason = 0, "building", "Тот же дом"
        elif complex_id and complex_id == _text(row.get("complex_id")):
            priority, location, reason = 1, "complex", "Тот же ЖК"
        elif math.isfinite(distance) and distance <= MAX_DISTANCE_M:
            priority, location = 2, "nearby"
            reason = f"Примерно {distance:.0f} м (координаты не проверены)"
        elif district and district == _normalized(row.get("district")):
            priority, location, reason = 3, "district", "Тот же район; близость домов не установлена"
        else:
            continue
        area_difference = abs(candidate_area - area) / area
        # Не включать price, price_per_m2 или прогноз в ключ ранжирования.
        tie = (_text(row.get("record_id")) or "", _text(row.get("group")) or "", url)
        rank = (priority, distance if location == "nearby" else 0.0, area_difference,
                *_structural_difference(row, query), *tie)
        result = {**row, "url": url, "similarity_reason": reason + "; те же комнаты, площадь ±20%",
                  "distance_m": distance, "location_match": location, "price_per_m2": price / candidate_area}
        candidates.append((rank, component, result))

    selected, seen = [], set()
    for _, component, row in sorted(candidates, key=lambda candidate: candidate[0]):
        if component in seen:
            continue
        selected.append(row)
        seen.add(component)
        if len(selected) == limit:
            break
    if not selected:
        return _empty(listings)
    return pd.DataFrame(selected, columns=list(dict.fromkeys([*listings.columns, *RESULT_COLUMNS])))

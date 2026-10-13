"""Консервативный локальный каталог архивных объявлений и адресов домов.

Идентификатор дома означает совпадение явно указанного адреса внутри района,
а не проверку объекта в государственном реестре. Цена в сопоставлении не участвует.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from clean import normalize_text, normalize_url


ASTANA_BBOX = {"latitude_min": 50.8, "latitude_max": 51.4,
               "longitude_min": 70.8, "longitude_max": 71.9}
MAX_COORDINATE_RADIUS_M = 125.0
ISSUE_COLUMNS = ["record_id", "building_id", "issue", "field", "value", "detail"]
BUILDING_COLUMNS = [
    "building_id", "complex_id", "complex_name", "address_normalized", "district",
    "latitude", "longitude", "coordinate_status", "coordinate_source",
    "coordinate_spread_m", "coordinate_observations", "parking_type", "parking_status",
    "listing_count", "display_label", "year_built", "total_floors", "building_type",
    "year_built_status", "total_floors_status", "building_type_status",
    "complex_status", "developer", "elevator",
]
EXTRA_COLUMNS = [
    "building_id", "complex_id", "complex_name", "address_normalized", "address_status",
    "latitude", "longitude", "coordinate_status", "coordinate_source",
    "coordinate_spread_m", "parking_type", "parking_status", "date_label",
    "developer", "elevator",
]
_HOUSE = r"\d+[а-яёa-z]?(?:[/-]\d+[а-яёa-z]?)*"
_HOUSE_ADDRESS = re.compile(
    rf"^(?P<street>.+?)\s+(?:(?:д\.?|дом)\s+)?(?P<house>{_HOUSE})"
    rf"(?P<unit>\s+(?:корпус|корп\.?|к\.?|блок|строение|стр\.?)\s+[\dа-яёa-z]+)?$"
)


def _text(value: object, *, lower: bool = False) -> str | None:
    value = normalize_text(value, lowercase=lower)
    return None if pd.isna(value) else str(value)


def _key(value: object) -> str | None:
    text = _text(value, lower=True)
    if text is None:
        return None
    # Format characters never distinguish houses; letters, scripts and numbers do.
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    return " ".join(text.split()) or None


def _identifier(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def normalize_address(value: object, complex_name: object = None) -> tuple[str | None, str]:
    """Возвращает нормализованный полный адрес и осторожную оценку его полноты.

    Никаких исправлений опечаток, транслитерации, отбрасывания корпуса или
    пересечений улиц. Адрес с несколькими домами либо пояснениями требует разбора.
    """
    text = _key(value)
    if text is None:
        return None, "missing"
    text = text.replace("‑", "-").replace("–", "-")
    if re.search(r"[,;|!?():—]|\s-\s", text):
        return text, "ambiguous"
    match = _HOUSE_ADDRESS.fullmatch(text)
    if not match:
        return text, "incomplete"
    street = match["street"]
    if (not re.search(r"[a-zа-яёәғқңөұүһі]", street)
            or street == _key(complex_name)
            or re.match(r"^(?:жк\b|жилой комплекс\b|мкр\b|микрорайон\b)", street)
            or re.search(r"\b(?:корпус|корп|блок|строение|дом)\b", street)):
        return text, "ambiguous"
    unit = match["unit"] or ""
    # Explicit optional 'дом/д.' is syntax, but corpus identifiers remain in key.
    return f"{street} {match['house']}{unit}", "house"


def _distance_m(latitudes: np.ndarray, longitudes: np.ndarray, lat: float, lon: float) -> np.ndarray:
    a, b = np.radians(latitudes), np.radians(longitudes)
    c, d = math.radians(lat), math.radians(lon)
    h = np.sin((a - c) / 2) ** 2 + np.cos(a) * math.cos(c) * np.sin((b - d) / 2) ** 2
    return 6_371_000 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _coordinate(lat: object, lon: object) -> tuple[float | None, float | None, str]:
    lat_text, lon_text = _text(lat), _text(lon)
    if lat_text is None and lon_text is None:
        return None, None, "missing"
    try:
        a, b = float(lat_text), float(lon_text)
    except (TypeError, ValueError):
        return None, None, "invalid"
    if not math.isfinite(a) or not math.isfinite(b):
        return None, None, "invalid"
    if not (ASTANA_BBOX["latitude_min"] <= a <= ASTANA_BBOX["latitude_max"]
            and ASTANA_BBOX["longitude_min"] <= b <= ASTANA_BBOX["longitude_max"]):
        return a, b, "invalid"
    return a, b, "unverified"


def _consensus(values: pd.Series) -> tuple[object, str]:
    unique = values.dropna().unique()
    if len(unique) == 0:
        return None, "unknown"
    if len(unique) == 1:
        return unique[0], "observed"
    return None, "conflict"


def build_property_catalog(
    combined: pd.DataFrame, january_raw: pd.DataFrame, march_raw: pd.DataFrame,
) -> dict[str, Any]:
    """Вернуть listings/buildings/issues/summary, не изменяя входы и не пиша файлы.

    Raw positional lookup is guarded by period, original_source_row, record_id,
    and matching canonical URL. A changed ordering of a raw file fails closed.
    """
    required = {"dataset_period", "original_source_row", "record_id", "group", "url",
                "address", "residential_complex", "district", "collected_at",
                "year_built", "total_floors", "building_type"}
    if not combined.columns.is_unique or not required.issubset(combined.columns):
        raise ValueError(f"Нужны уникальные колонки combined: {sorted(required)}")
    if set(EXTRA_COLUMNS) & set(combined.columns):
        raise ValueError("В combined уже есть зарезервированные колонки каталога.")
    if combined["record_id"].isna().any() or not combined["record_id"].is_unique:
        raise ValueError("record_id должны быть непустыми и уникальными.")
    for raw, required_raw in [(january_raw, {"url", "latitude", "longitude", "parking"}),
                              (march_raw, {"url", "Парковка"})]:
        if not raw.columns.is_unique or not required_raw.issubset(raw.columns):
            raise ValueError(f"Нет ожидаемых raw колонок: {sorted(required_raw)}")
    listings = combined.copy(deep=True).reset_index(drop=True)
    issues: list[dict[str, object]] = []

    def issue(row: dict | pd.Series, code: str, field: str, value: object, detail: str) -> None:
        issues.append({"record_id": row.get("record_id"), "building_id": row.get("building_id"),
                       "issue": code, "field": field, "value": _text(value), "detail": detail})

    additions = []
    for row in listings.to_dict(orient="records"):
        period = str(row["dataset_period"])
        if period not in {"2026-01", "2026-03"}:
            raise ValueError(f"Неизвестный dataset_period: {period}")
        raw = january_raw if period == "2026-01" else march_raw
        try:
            number = float(row["original_source_row"])
        except (ValueError, TypeError):
            raise ValueError("original_source_row должен быть целой позицией raw.") from None
        if not math.isfinite(number) or not number.is_integer() or not 1 <= number <= len(raw):
            raise ValueError("original_source_row выходит за границы исходного raw.")
        if row["record_id"] != f"{period}:{int(number)}":
            raise ValueError("record_id не соответствует периоду и original_source_row.")
        original = raw.iloc[int(number) - 1]
        left_url, right_url = _text(normalize_url(row["url"])), _text(normalize_url(original["url"]))
        if not left_url or left_url != right_url:
            raise ValueError(f"URL provenance mismatch для {row['record_id']}; raw может быть переупорядочен.")
        complex_name = _text(row["residential_complex"])
        source_name = original.get("residential_complex_name" if period == "2026-01" else "Жилой комплекс")
        if _key(source_name) == _key(complex_name) and _text(source_name):
            complex_name = _text(source_name)
        complex_key = _key(complex_name)
        complex_id = _identifier("complex_", "astana|" + complex_key) if complex_key else None
        address, address_status = normalize_address(row["address"], complex_name)
        district = _key(row["district"])
        if address_status == "house" and not district:
            address_status = "district_missing"
        building_id = (_identifier("building_", f"astana|{district}|{address}")
                       if address_status == "house" else None)
        parking = _key(original.get("parking" if period == "2026-01" else "Парковка"))
        parking_status = "observed" if parking else "unknown"
        if period == "2026-01":
            latitude, longitude, coordinate_status = _coordinate(original["latitude"], original["longitude"])
            collected = _text(row["collected_at"])
            date_label = (f"Сбор: {collected[:10]}; часовой пояс не указан" if collected else
                          "Январь 2026; точная дата сбора отсутствует")
        else:
            latitude, longitude, coordinate_status = None, None, "missing"
            date_label = "Март 2026 — период по заявлению источника; дата объявления неизвестна"
        extra = dict(zip(EXTRA_COLUMNS, [
            building_id, complex_id, complex_name, address, address_status,
            latitude, longitude, coordinate_status, "january_raw" if coordinate_status != "missing" else None,
            None, parking, parking_status, date_label, None, None,
        ]))
        additions.append(extra)
        tagged = {**row, **extra}
        if address_status != "house":
            issue(tagged, "address_" + address_status, "address", address,
                  "Дом не сопоставлен: нужен один однозначный адрес и район; название ЖК дом не заменяет.")
        if coordinate_status == "invalid":
            issue(tagged, "coordinate_invalid", "latitude,longitude",
                  f"{original.get('latitude')},{original.get('longitude')}", "Вне bbox Астаны либо нечисловая/неполная пара.")
    listings = pd.concat([listings, pd.DataFrame(additions, columns=EXTRA_COLUMNS)], axis=1)

    # A reused exact marker across different explicit houses may be a complex centroid.
    # It cannot validate a house-level location even when many ads repeat that marker.
    markers: dict[tuple[float, float], set[str]] = {}
    valid = listings["coordinate_status"].eq("unverified") & listings["building_id"].notna()
    for row in listings.loc[valid].to_dict(orient="records"):
        marker = (round(row["latitude"], 6), round(row["longitude"], 6))
        markers.setdefault(marker, set()).add(row["building_id"])
    reused = {marker for marker, ids in markers.items() if len(ids) > 1}
    buildings = []
    for building_id, part in listings.dropna(subset=["building_id"]).groupby("building_id", sort=True):
        first = part.iloc[0]
        values: dict[str, object] = {
            "building_id": building_id, "address_normalized": first["address_normalized"],
            "district": first["district"], "listing_count": len(part), "developer": None, "elevator": None,
        }
        complex_id, complex_status = _consensus(part["complex_id"])
        values.update(complex_id=complex_id, complex_status=complex_status)
        names = part["complex_name"].dropna().tolist()
        values["complex_name"] = sorted(names, key=lambda x: (x.casefold(), x))[0] if names and complex_id else None
        for field in ("parking_type", "year_built", "total_floors", "building_type"):
            value, status = _consensus(part[field])
            values[field] = value
            values["parking_status" if field == "parking_type" else field + "_status"] = status
            if status == "conflict":
                issue({"building_id": building_id}, "building_field_conflict", field,
                      " | ".join(map(str, sorted(part[field].dropna().unique(), key=str))),
                      "Непустые архивные значения расходятся; агрегат оставлен пустым.")
        if complex_status == "conflict":
            issue({"building_id": building_id}, "building_complex_conflict", "complex_name",
                  " | ".join(sorted(set(names))), "Один адрес связан с разными названиями ЖК; ЖК дома не выбран.")
        good = part.loc[part["coordinate_status"].eq("unverified")]
        has_invalid = part["coordinate_status"].eq("invalid").any()
        lat, lon, radius = None, None, None
        status = "invalid" if has_invalid else "missing"
        has_reused = False
        if len(good):
            lat, lon = float(good["latitude"].median()), float(good["longitude"].median())
            radius = float(_distance_m(good["latitude"].to_numpy(float), good["longitude"].to_numpy(float), lat, lon).max())
            has_reused = any((round(a, 6), round(b, 6)) in reused
                             for a, b in zip(good["latitude"], good["longitude"]))
            status = ("plausible" if len(good) >= 2 and radius <= MAX_COORDINATE_RADIUS_M
                      and not has_invalid and not has_reused else "unverified")
        values.update(latitude=lat, longitude=lon, coordinate_status=status,
                      coordinate_source="january_raw_consensus" if len(good) else None,
                      coordinate_spread_m=radius, coordinate_observations=len(good))
        if radius is not None and radius > MAX_COORDINATE_RADIUS_M:
            issue({"building_id": building_id}, "coordinate_spread", "coordinate_spread_m", radius,
                  "Радиус от медианы превышает 125 м; расстояния и перенос координат запрещены.")
        if has_reused:
            issue({"building_id": building_id}, "coordinate_reused_across_houses", "latitude,longitude", None,
                  "Одинаковая точка встречается у нескольких адресов домов; возможно, центр ЖК.")
        if has_invalid and len(good):
            issue({"building_id": building_id}, "coordinate_mixed_validity", "latitude,longitude", None,
                  "В доме присутствуют пригодные по bbox и невалидные точки; большинство не разрешает конфликт.")
        listings.loc[part.index, "coordinate_spread_m"] = radius
        if status == "plausible":
            listings.loc[good.index, "coordinate_status"] = "plausible"
            missing = part.index[part["coordinate_status"].eq("missing")]
            listings.loc[missing, ["latitude", "longitude", "coordinate_status", "coordinate_source"]] = [
                lat, lon, "plausible", "building_consensus",
            ]
        title = values["complex_name"]
        values["display_label"] = (f"{title} · " if title else "") + f"{values['address_normalized']} · {values['district']}"
        buildings.append(values)
    building_frame = pd.DataFrame(buildings, columns=BUILDING_COLUMNS)
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    summary = {
        "listing_count": len(listings), "building_count": len(building_frame),
        "complex_count": int(listings["complex_id"].nunique()),
        "listings_with_building_id": int(listings["building_id"].notna().sum()),
        "listings_with_complex_id": int(listings["complex_id"].notna().sum()),
        "listings_with_parking": int(listings["parking_type"].notna().sum()),
        "coordinate_status_counts": dict(sorted(Counter(listings["coordinate_status"]).items())),
        "building_coordinate_status_counts": dict(sorted(Counter(building_frame["coordinate_status"]).items())),
        "address_status_counts": dict(sorted(Counter(listings["address_status"]).items())),
        "dataset_period_counts": dict(sorted(Counter(listings["dataset_period"]).items())),
        "coordinate_inherited_count": int(listings["coordinate_source"].eq("building_consensus").sum()),
        "issue_counts": dict(sorted(Counter(issue_frame["issue"]).items())),
        "building_conflicts": {field: int(building_frame[status].eq("conflict").sum()) for field, status in [
            ("complex", "complex_status"), ("parking", "parking_status"),
            ("year_built", "year_built_status"), ("total_floors", "total_floors_status"),
            ("building_type", "building_type_status"),
        ]},
        "policy": {"bbox": ASTANA_BBOX, "maximum_radius_from_median_m": MAX_COORDINATE_RADIUS_M,
                   "minimum_coordinate_observations": 2, "distance_allowed_statuses": ["plausible"],
                   "building_key": "astana + exact normalized district + single explicit house address",
                   "complex_key": "astana + exact normalized complex name"},
        "limitations": [
            "Адреса и ЖК сопоставлены консервативно без справочника; разные написания не объединены.",
            "building_id не является реестровым ID; корпуса и дроби сохранены, неоднозначные адреса исключены.",
            "plausible означает только согласованность архивных точек: точность дома внешне не подтверждена.",
            "Одинаковые координаты у разных домов и внутридомовые конфликты запрещают расчёт расстояний.",
            "Парковка — текст объявления, не наличие места в собственности и не включение в цену; пропуск неизвестен.",
            "Застройщик и лифт неизвестны; большинством конфликтующие характеристики дома не выбираются.",
            "Январские даты — сбор без часового пояса; март — период по заявлению автора, без дат строк.",
            "Цены являются архивными ценами предложения; ссылки и актуальная доступность не проверялись.",
        ],
    }
    return {"listings": listings, "buildings": building_frame, "issues": issue_frame, "summary": summary}

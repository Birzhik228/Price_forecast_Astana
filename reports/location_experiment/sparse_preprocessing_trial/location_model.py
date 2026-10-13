"""Отдельная схема 2 для эксперимента с ЖК и координатами, изученными на train.

Геокоординаты не проверены внешним источником. Согласованность точек из исходных
объявлений проверяется только внутри fit; transform не использует сырые точки
проверочных строк и не пересчитывает справочник домов. Старые модели не меняются.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from building_catalog import (
    ASTANA_BBOX, MAX_COORDINATE_RADIUS_M, normalize_complex_name,
)
from features import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, MISSING_CATEGORY, NUMERIC_FEATURES,
    build_preprocessor,
)


LOCATION_SCHEMA_VERSION = 2
LOCATION_VARIANTS = ("baseline8", "complex", "coords", "complex_coords")
LOCATION_INPUT_COLUMNS = (*FEATURE_COLUMNS, "residential_complex", "house_key",
                          "raw_latitude", "raw_longitude", "raw_coordinate_status")
LOCATION_GEOGRAPHY_COLUMNS = ("latitude", "longitude")
MIN_COORDINATE_OBSERVATIONS = 2
COMPLEX_MIN_FREQUENCY = 10
COMPLEX_MAX_CATEGORIES = 256
_TRAITS = ("residential_complex", "year_built", "total_floors")
_AUDIT_COLUMNS = ("house_key", "coordinate_status", "coordinate_observations",
                  "coordinate_spread_m", "has_invalid", "has_reused_marker",
                  "conflicting_traits", "latitude", "longitude")


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _finite(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def prepare_location_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Отобрать 13 разрешённых входов, не подмешивая цены и каталоговые координаты.

    house_key должен быть row-local результатом canonical_house_key. raw_* —
    собственные исходные координаты, привязанные по проверенному происхождению.
    Пропущенное значение допустимо; отсутствие колонки означает ошибку схемы.
    Наличие полного каталога в frame безопасно: все остальные поля отбрасываются.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Нужен DataFrame с уникальными колонками схемы location.")
    missing = set(LOCATION_INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Нет исходных колонок location: " + ", ".join(sorted(missing)))
    result = frame.loc[:, list(LOCATION_INPUT_COLUMNS)].copy(deep=True)
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    result["residential_complex"] = result["residential_complex"].map(
        lambda value: normalize_complex_name(value) or np.nan
    ).astype(object)
    result["house_key"] = result["house_key"].map(_optional_text)
    # Не превращать нечисловую исходную точку в пропуск: invalid блокирует дом.
    # Raw значения сохраняются до fit; в transform они не являются признаками.
    statuses = result["raw_coordinate_status"].map(_optional_text).fillna("missing")
    if not statuses.isin(("missing", "invalid", "unverified")).all():
        raise ValueError("raw_coordinate_status допускает только missing/invalid/unverified; используйте исходные точки.")
    result["raw_coordinate_status"] = statuses
    return result


def _raw_point(row: dict[str, Any]) -> tuple[tuple[float, float] | None, bool]:
    status = row["raw_coordinate_status"]
    a, b = _finite(row["raw_latitude"]), _finite(row["raw_longitude"])
    if status == "invalid":
        return None, True
    if status == "missing":
        # Непустые значения, помеченные missing, не служат источником координат.
        return None, a is not None or b is not None
    if (a is None or b is None or not ASTANA_BBOX["latitude_min"] <= a <= ASTANA_BBOX["latitude_max"]
            or not ASTANA_BBOX["longitude_min"] <= b <= ASTANA_BBOX["longitude_max"]):
        return None, True
    return (a, b), False


def _radius(points: list[tuple[float, float]], latitude: float, longitude: float) -> float:
    coordinates = np.radians(np.asarray(points, dtype=float))
    lat, lon = math.radians(latitude), math.radians(longitude)
    haversine = np.sin((coordinates[:, 0] - lat) / 2) ** 2 + (
        np.cos(coordinates[:, 0]) * math.cos(lat) * np.sin((coordinates[:, 1] - lon) / 2) ** 2
    )
    distances = 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    return float(distances.max())


def _trait(row: dict[str, Any], column: str) -> str | float | None:
    return _optional_text(row[column]) if column == "residential_complex" else _finite(row[column])


class LocationFeatureTransformer(TransformerMixin, BaseEstimator):
    """Clone-friendly признаки с train-only справочником пригодных координат домов."""

    def __init__(self, variant: str = "complex_coords") -> None:
        self.variant = variant

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LocationFeatureTransformer":
        if self.variant not in LOCATION_VARIANTS:
            raise ValueError(f"Неизвестный вариант location: {self.variant}")
        frame = prepare_location_input(X)
        if frame.empty:
            raise ValueError("Для fit location нужны обучающие строки.")
        self.feature_names_in_ = np.asarray(LOCATION_INPUT_COLUMNS, dtype=object)
        self.n_features_in_ = len(LOCATION_INPUT_COLUMNS)
        self.house_coordinate_map_: dict[str, tuple[float, float]] = {}
        self.house_traits_: dict[str, dict[str, str | float | None]] = {}
        audit: list[dict[str, Any]] = []
        if self.variant in {"coords", "complex_coords"}:
            houses: dict[str, list[dict[str, Any]]] = {}
            markers: dict[tuple[float, float], set[str]] = {}
            for row in frame.to_dict("records"):
                key = row["house_key"]
                if key is None:
                    continue
                point, invalid = _raw_point(row)
                row["_point"], row["_invalid"] = point, invalid
                houses.setdefault(key, []).append(row)
                if point is not None:
                    markers.setdefault(tuple(round(value, 6) for value in point), set()).add(key)
            reused = {point for point, keys in markers.items() if len(keys) > 1}
            for key, rows in sorted(houses.items()):
                traits, conflicting = {}, []
                for column in _TRAITS:
                    values = {_trait(row, column) for row in rows} - {None}
                    traits[column] = next(iter(values)) if len(values) == 1 else None
                    if len(values) > 1:
                        conflicting.append(column)
                self.house_traits_[key] = traits
                points = [row["_point"] for row in rows if row["_point"] is not None]
                invalid = any(row["_invalid"] for row in rows)
                has_reused = any(tuple(round(value, 6) for value in point) in reused for point in points)
                latitude = longitude = radius = math.nan
                if points:
                    latitude, longitude = map(float, np.median(np.asarray(points), axis=0))
                    radius = _radius(points, latitude, longitude)
                if conflicting:
                    status = "trait_conflict"
                elif invalid:
                    status = "invalid"
                elif has_reused:
                    status = "reused_marker"
                elif not points:
                    status = "missing"
                elif len(points) < MIN_COORDINATE_OBSERVATIONS:
                    status = "insufficient_observations"
                elif radius > MAX_COORDINATE_RADIUS_M:
                    status = "coordinate_spread"
                else:
                    status = "plausible"
                    self.house_coordinate_map_[key] = (latitude, longitude)
                audit.append(dict(house_key=key, coordinate_status=status, coordinate_observations=len(points),
                                  coordinate_spread_m=radius, has_invalid=invalid, has_reused_marker=has_reused,
                                  conflicting_traits=", ".join(conflicting), latitude=latitude, longitude=longitude))
        self.coordinate_audit_ = pd.DataFrame(audit, columns=list(_AUDIT_COLUMNS))
        self.training_geo_summary_ = {"fit_rows": len(frame), "houses_seen": len(audit),
                                      "plausible_houses": len(self.house_coordinate_map_),
                                      "externally_verified": False}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "house_coordinate_map_")
        frame = prepare_location_input(X)
        result = frame.loc[:, list(FEATURE_COLUMNS)].copy()
        if self.variant in {"complex", "complex_coords"}:
            result["residential_complex"] = frame["residential_complex"]
        if self.variant in {"coords", "complex_coords"}:
            coordinates = []
            for row in frame.to_dict("records"):
                key = row["house_key"]
                point = self.house_coordinate_map_.get(key)
                known = self.house_traits_.get(key, {})
                # Только собственные характеристики запроса, не соседние test-строки.
                if point is not None and any(
                    known.get(column) is not None and _trait(row, column) is not None
                    and known[column] != _trait(row, column) for column in _TRAITS
                ):
                    point = None
                coordinates.append(point if point is not None else (math.nan, math.nan))
            result[list(LOCATION_GEOGRAPHY_COLUMNS)] = np.asarray(coordinates, dtype=float).reshape(-1, 2)
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "house_coordinate_map_")
        columns = list(FEATURE_COLUMNS)
        if self.variant in {"complex", "complex_coords"}:
            columns.append("residential_complex")
        if self.variant in {"coords", "complex_coords"}:
            columns.extend(LOCATION_GEOGRAPHY_COLUMNS)
        return np.asarray(columns, dtype=object)


def _location_preprocessor(variant: str) -> ColumnTransformer:
    if variant == "baseline8":
        return build_preprocessor()
    numeric_columns = list(NUMERIC_FEATURES)
    if variant in {"coords", "complex_coords"}:
        numeric_columns.extend(LOCATION_GEOGRAPHY_COLUMNS)
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY, keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    transformers = [("numeric", numeric, numeric_columns),
                    ("categorical", categorical, list(CATEGORICAL_FEATURES))]
    if variant in {"complex", "complex_coords"}:
        complex_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY, keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=True,
                                      min_frequency=COMPLEX_MIN_FREQUENCY, max_categories=COMPLEX_MAX_CATEGORIES)),
        ])
        transformers.append(("complex", complex_pipeline, ["residential_complex"]))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def make_location_pipeline(base_regressor: BaseEstimator, variant: str) -> Pipeline:
    """Четыре варианта одного регрессора: все параметры, кроме n_jobs=2, сохранены."""
    if variant not in LOCATION_VARIANTS:
        raise ValueError(f"Неизвестный вариант location: {variant}")
    model = clone(base_regressor)
    if "n_jobs" in model.get_params(deep=False):
        model.set_params(n_jobs=2)
    return Pipeline([
        ("location_features", LocationFeatureTransformer(variant=variant)),
        ("preprocess", _location_preprocessor(variant)),
        ("model", model),
    ])

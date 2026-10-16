"""Адаптер приложения для схем 1–3 без изменения сохранённых моделей."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from building_catalog import canonical_house_key, normalize_complex_name
from clean import normalize_district
from features import FEATURE_COLUMNS
from location_model import LOCATION_INPUT_COLUMNS, prepare_location_input
from prediction import (
    PredictionResult, load_artifacts, predict_apartment, prepare_input, validate_metadata,
)


@dataclass(frozen=True)
class LocationPredictionResult(PredictionResult):
    """Обычный результат плюс пояснения об использованных необязательных сведениях."""

    location_notes: tuple[str, ...] = ()


def _validate_location_metadata(metadata: dict[str, Any]) -> None:
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 2
            or metadata.get("variant") != "complex_coords"
            or metadata.get("feature_columns") != list(LOCATION_INPUT_COLUMNS)):
        raise ValueError("Описание модели не соответствует схеме ЖК и координат.")
    # Старая проверка обслуживает именно восемь общих полей формы и диапазон.
    projected = {**metadata, "schema_version": 1, "feature_columns": list(FEATURE_COLUMNS)}
    validate_metadata(projected)


def _known_complexes(pipeline: Pipeline) -> set[str]:
    try:
        categories = pipeline.named_steps["preprocess"].named_transformers_["complex"].named_steps["encoder"].categories_[0]
    except (AttributeError, KeyError, IndexError, TypeError) as error:
        raise ValueError("В модели отсутствует обученный словарь ЖК.") from error
    return {name for value in categories if (name := normalize_complex_name(value)) is not None}


def load_app_artifacts(model_path: Path, metadata_path: Path) -> tuple[Any, dict[str, Any]]:
    """Загрузить собственный локальный артефакт приложения с проверкой SHA и схемы."""
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(metadata, dict) and metadata.get("schema_version") == 1:
        return load_artifacts(model_path, metadata_path)
    if isinstance(metadata, dict) and metadata.get("schema_version") == 3:
        from catboost_prediction import load_catboost_artifacts
        return load_catboost_artifacts(model_path, metadata_path)
    _validate_location_metadata(metadata)
    with model_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != metadata["model_sha256"]:
            raise ValueError("Модель и её описание не совпадают. Восстановите оба файла одного запуска.")
        stream.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = joblib.load(stream)
    if (not isinstance(pipeline, Pipeline) or list(getattr(pipeline, "feature_names_in_", [])) != list(LOCATION_INPUT_COLUMNS)
            or getattr(pipeline.named_steps.get("location_features"), "variant", None) != "complex_coords"):
        raise ValueError("Пайплайн не соответствует входам модели ЖК и координат.")
    if not hasattr(pipeline.named_steps["location_features"], "house_coordinate_map_"):
        raise ValueError("В модели отсутствует обученный справочник координат домов.")
    _known_complexes(pipeline)
    return pipeline, metadata


def _text(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    raise ValueError("Название ЖК, адрес и район должны быть строками или оставаться пустыми.")


def _first_text(location: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _text(location.get(key))
        if value is not None:
            return value
    return None


def _district(value: Any) -> str | None:
    value = _text(value)
    normalized = normalize_district(value)
    return None if pd.isna(normalized) else str(normalized)


def predict_app_apartment(
    pipeline: Any, metadata: dict[str, Any], values: dict[str, Any],
    location: dict[str, Any] | None = None,
) -> PredictionResult:
    """Использовать явный адрес и версионированную географию pipeline.

    Никакие переданные latitude/longitude/raw_*/house_key/building_id не являются
    входами модели. Ключ дома вычисляется заново из адреса и согласованного района.
    Схема 1 полностью сохраняет прежний путь прогнозирования.
    """
    if isinstance(metadata, dict) and metadata.get("schema_version") == 1:
        return predict_apartment(pipeline, metadata, values)
    if isinstance(metadata, dict) and metadata.get("schema_version") == 3:
        from catboost_prediction import predict_catboost_apartment
        return predict_catboost_apartment(pipeline, metadata, values, location)
    _validate_location_metadata(metadata)
    if location is not None and not isinstance(location, dict):
        raise ValueError("Местоположение должно быть словарём параметров.")
    location = location or {}
    basic, notices = prepare_input(values, metadata)
    form_district = _district(basic.iloc[0]["district"])
    location_district = _district(location.get("location_district"))
    alternate_district = _district(location.get("district"))
    if location_district and alternate_district and location_district != alternate_district:
        raise ValueError("Район выбранного дома указан противоречиво.")
    location_district = location_district or alternate_district
    if form_district and location_district and form_district != location_district:
        raise ValueError("Район квартиры не совпадает с районом выбранного дома. Уточните район или выбор дома.")
    complex_display = _first_text(location, "complex_name", "residential_complex")
    complex_name = normalize_complex_name(complex_display)
    address = _first_text(location, "address", "address_normalized")
    house_key = canonical_house_key(address, location_district or form_district, complex_name)
    frame = basic.assign(residential_complex=complex_name, house_key=house_key,
                         raw_latitude=np.nan, raw_longitude=np.nan, raw_coordinate_status="missing")
    frame = prepare_location_input(frame)
    notes = []
    if complex_name is None:
        notes.append("ЖК не указан. Название комплекса не учтено в оценке.")
    elif complex_name not in _known_complexes(pipeline):
        notices.append(f"ЖК «{complex_display}» не встречался в обучающем наборе; оценка для него менее надёжна.")
    else:
        notes.append(f"Учтён ЖК: {complex_display}.")
    transformer = pipeline.named_steps["location_features"]
    transformed = transformer.transform(frame)
    coordinates = transformed.loc[:, ["latitude", "longitude"]].to_numpy(dtype=float)
    if coordinates.shape != (1, 2):
        raise ValueError("Модель вернула некорректные признаки местоположения.")
    if np.isfinite(coordinates).all():
        provenance = transformer.provenance(house_key) if hasattr(transformer, "provenance") else {}
        if provenance.get("source_status") == "osm_exact_address":
            notes.append("Координаты учтены по точному совпадению адреса с контуром здания в OpenStreetMap. "
                         f"[Объект на карте]({provenance['osm_url']}). Это не кадастровая проверка или положение подъезда.")
            notes.append("© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) · "
                         "[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).")
        else:
            notes.append("Координаты выбранного дома учтены из обучающего справочника; точность до дома внешне не подтверждена.")
    else:
        notes.append("Пригодные координаты дома недоступны для этого запроса. Оценка выполнена с неполными сведениями о местоположении.")
        if house_key in transformer.house_coordinate_map_:
            notices.append("Параметры дома отличаются от сведений обучающего набора. Его координаты не использованы.")
    predicted = np.asarray(pipeline.predict(frame), dtype=float)
    if predicted.shape != (1,) or not np.isfinite(predicted).all() or predicted[0] <= 0:
        raise ValueError("Модель не смогла получить положительную конечную цену для этих параметров.")
    price, mae = float(predicted[0]), float(metadata["test_metrics"]["mae_tenge"])
    if not math.isfinite(mae) or mae < 0:
        raise ValueError("В описании модели отсутствует корректная MAE.")
    return LocationPredictionResult(
        price_tenge=price, price_per_m2=price / float(basic.iloc[0]["area"]),
        lower_tenge=max(0.0, price - mae), upper_tenge=price + mae,
        warnings=tuple(notices), location_notes=tuple(notes),
    )

"""Загрузка локального пайплайна, проверка ввода и расчёт цены."""

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

from config import REFERENCE_YEAR
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


# Те же широкие границы, что на этапе очистки; это не диапазоны рынка.
INPUT_LIMITS: dict[str, tuple[float, float]] = {
    "area": (10, 1000), "rooms": (1, 20), "floor": (1, 100),
    "total_floors": (1, 100), "year_built": (1850, REFERENCE_YEAR + 5),
}
FIELD_LABELS = {
    "area": "Площадь", "rooms": "Количество комнат", "floor": "Этаж",
    "total_floors": "Этажность дома", "year_built": "Год постройки",
    "district": "Район", "building_type": "Тип дома", "condition": "Состояние",
}


@dataclass(frozen=True)
class PredictionResult:
    """Цены в тенге; диапазон описательный, без гарантии покрытия."""

    price_tenge: float
    price_per_m2: float
    lower_tenge: float
    upper_tenge: float
    warnings: tuple[str, ...]


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Не смешивать модель с метриками или схемой другого проекта."""
    if not isinstance(metadata, dict):
        raise ValueError("Описание модели должно быть JSON-объектом.")
    if (metadata.get("schema_version") != 1 or metadata.get("currency") != "KZT"
            or metadata.get("target") != "asking_price"
            or metadata.get("feature_columns") != list(FEATURE_COLUMNS)):
        raise ValueError("Описание модели содержит несовместимую схему признаков или цены.")
    try:
        mae = float(metadata["test_metrics"]["mae_tenge"])
        half_width = float(metadata["approximate_range"]["half_width_tenge"])
        categories = metadata["categories_from_train"]
        ranges = metadata["numeric_ranges_from_train"]
        checksum = metadata["model_sha256"]
        if not math.isfinite(mae) or mae < 0 or not math.isclose(mae, half_width):
            raise ValueError("MAE и ширина диапазона должны совпадать и быть неотрицательными.")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("Отсутствует контрольная сумма модели.")
        int(checksum, 16)
        for column in CATEGORICAL_FEATURES:
            values = categories[column]
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"Некорректный список категорий: {column}.")
            if len(values) != len(set(values)):
                raise ValueError(f"Повторные категории: {column}.")
        for column in NUMERIC_FEATURES:
            if column not in ranges:
                continue  # Полностью пустой столбец не имеет диапазона train.
            low, high = float(ranges[column]["min"]), float(ranges[column]["max"])
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError(f"Некорректный диапазон train: {column}.")
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Описание модели неполное или повреждено.") from error


def load_artifacts(model_path: Path, metadata_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Загружать только собственные локальные joblib-файлы проекта."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata)
    with model_path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != metadata["model_sha256"]:
            raise ValueError("Модель и её описание не совпадают. Восстановите оба файла одного запуска.")
        stream.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = joblib.load(stream)
    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict"):
        raise ValueError("Файл не содержит обученный Pipeline.")
    if list(getattr(pipeline, "feature_names_in_", [])) != list(FEATURE_COLUMNS):
        raise ValueError("Входные признаки пайплайна не совпадают с формой.")
    return pipeline, metadata


def prepare_input(
    values: dict[str, Any], metadata: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Проверить поля формы и подготовить одну строку без ручной импутации."""
    row: dict[str, Any] = {}
    errors: list[str] = []
    notices: list[str] = []
    missing_labels: list[str] = []
    for column in NUMERIC_FEATURES:
        value = values.get(column)
        label = FIELD_LABELS[column]
        if value is None:
            if column in ("area", "rooms"):
                errors.append(f"{label}: заполните поле.")
            else:
                row[column] = np.nan
                missing_labels.append(label.lower())
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{label}: введите число.")
            continue
        low, high = INPUT_LIMITS[column]
        if isinstance(value, (bool, np.bool_)) or not math.isfinite(number):
            errors.append(f"{label}: требуется конечное число.")
        elif column != "area" and not number.is_integer():
            errors.append(f"{label}: требуется целое число.")
        elif not low <= number <= high:
            errors.append(f"{label}: допустимы значения от {low:g} до {high:g}.")
        else:
            row[column] = number
            bounds = metadata["numeric_ranges_from_train"].get(column)
            if bounds and not bounds["min"] <= number <= bounds["max"]:
                notices.append(
                    f"{label}: в обучающих данных были значения от {bounds['min']:g} "
                    f"до {bounds['max']:g}. Для вашего значения прогноз менее надёжен."
                )
    for column in CATEGORICAL_FEATURES:
        value = values.get(column)
        if value is None:
            row[column] = np.nan
            missing_labels.append(FIELD_LABELS[column].lower())
        elif not isinstance(value, str) or value not in metadata["categories_from_train"][column]:
            errors.append(f"{FIELD_LABELS[column]}: выберите значение из списка или «Не указано».")
        else:
            row[column] = value
    if row.get("floor", np.nan) > row.get("total_floors", np.nan):
        errors.append("Этаж квартиры не может превышать этажность дома.")
    if errors:
        raise ValueError("\n".join(errors))
    if missing_labels:
        notices.append("Не указаны: " + ", ".join(missing_labels) + ". Прогноз учитывает неполные сведения.")
    return pd.DataFrame([row], columns=list(FEATURE_COLUMNS)), notices


def predict_apartment(
    pipeline: Pipeline, metadata: dict[str, Any], values: dict[str, Any],
) -> PredictionResult:
    """Рассчитать полную цену, затем цену за м² и диапазон на основе test MAE."""
    frame, notices = prepare_input(values, metadata)
    predicted = np.asarray(pipeline.predict(frame), dtype=float)
    if predicted.shape != (1,) or not np.isfinite(predicted).all() or predicted[0] <= 0:
        raise ValueError("Модель не смогла получить положительную конечную цену для этих параметров.")
    price = float(predicted[0])
    mae = float(metadata["test_metrics"]["mae_tenge"])
    if not math.isfinite(mae) or mae < 0:
        raise ValueError("В описании модели отсутствует корректная MAE.")
    return PredictionResult(
        price_tenge=price, price_per_m2=price / float(frame.iloc[0]["area"]),
        lower_tenge=max(0.0, price - mae), upper_tenge=price + mae,
        warnings=tuple(notices),
    )

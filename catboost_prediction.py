"""Application adapter for the saved MAE CatBoost; never fits or mutates it."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

from building_catalog import normalize_complex_name
from catboost_model import CATBOOST_MODEL_COLUMNS, CatBoostPriceRegressor
from features import FEATURE_COLUMNS
from location_model import LOCATION_INPUT_COLUMNS, prepare_location_input
from prediction import prepare_input, validate_metadata


def validate_catboost_metadata(metadata: dict[str, Any]) -> None:
    """Keep the model objective and effective feature contract explicit."""
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 3
            or metadata.get("variant") != "catboost_mae"
            or metadata.get("target_mode") != "price_mae"
            or metadata.get("uses_coordinates") is not False
            or metadata.get("feature_columns") != list(LOCATION_INPUT_COLUMNS)
            or metadata.get("actual_model_features") != list(CATBOOST_MODEL_COLUMNS)):
        raise ValueError("Описание модели не соответствует CatBoost с MAE и ЖК.")
    validate_metadata({**metadata, "schema_version": 1, "feature_columns": list(FEATURE_COLUMNS)})
    complexes = metadata.get("known_complexes_from_train")
    if (not isinstance(complexes, list)
            or any(not isinstance(value, str) or not value or normalize_complex_name(value) != value
                   for value in complexes)
            or len(set(complexes)) != len(complexes)):
        raise ValueError("В описании CatBoost отсутствует корректный словарь ЖК из train.")


def load_catboost_artifacts(model_path: Path, metadata_path: Path) -> tuple[CatBoostPriceRegressor, dict[str, Any]]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    validate_catboost_metadata(metadata)
    with Path(model_path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != metadata["model_sha256"]:
            raise ValueError("Модель и её описание не совпадают. Восстановите оба файла одного запуска.")
        stream.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(stream)
    if (not isinstance(model, CatBoostPriceRegressor)
            or getattr(model, "target_mode_", None) != "price_mae"
            or getattr(model, "training_loss_", None) != "MAE"
            or list(getattr(model, "feature_names_in_", [])) != list(LOCATION_INPUT_COLUMNS)
            or list(getattr(model, "model_feature_names_", [])) != list(CATBOOST_MODEL_COLUMNS)
            or getattr(getattr(model, "location_transformer_", None), "variant", None) != "complex"):
        raise ValueError("Файл не содержит обученный CatBoost с MAE и ожидаемыми признаками.")
    native = getattr(model, "model_", None)
    if (native is None or not native.is_fitted()
            or native.get_all_params().get("loss_function") != "MAE"
            or list(native.feature_names_) != list(CATBOOST_MODEL_COLUMNS)):
        raise ValueError("Обученный регрессор CatBoost не соответствует MAE и признакам.")
    return model, metadata


def predict_catboost_apartment(model, metadata, values, location=None):
    # Imported at call time: the public router delegates here after its own
    # module has loaded; legacy helpers retain exactly the same text rules.
    from location_prediction import LocationPredictionResult, _district, _first_text

    validate_catboost_metadata(metadata)
    if location is not None and not isinstance(location, dict):
        raise ValueError("Местоположение должно быть словарём параметров.")
    location = location or {}
    basic, notices = prepare_input(values, metadata)
    form_district = _district(basic.iloc[0]["district"])
    house_district = _district(location.get("location_district"))
    alternate_district = _district(location.get("district"))
    if house_district and alternate_district and house_district != alternate_district:
        raise ValueError("Район выбранного дома указан противоречиво.")
    house_district = house_district or alternate_district
    if form_district and house_district and form_district != house_district:
        raise ValueError("Район квартиры не совпадает с районом выбранного дома. Уточните район или выбор дома.")
    display_name = _first_text(location, "complex_name", "residential_complex")
    complex_name = normalize_complex_name(display_name)
    frame = prepare_location_input(basic.assign(
        residential_complex=complex_name, house_key=None,
        raw_latitude=np.nan, raw_longitude=np.nan, raw_coordinate_status="missing",
    ))
    notes = []
    if complex_name is None:
        notes.append("ЖК не указан. Название комплекса не учтено в оценке.")
    elif complex_name not in metadata["known_complexes_from_train"]:
        notices.append(f"ЖК «{display_name}» не встречался в обучающем наборе; оценка для него менее надёжна.")
    else:
        notes.append(f"Учтён ЖК: {display_name}.")
    notes.append("CatBoost оценивает параметры квартиры и ЖК. Адрес и координаты используются только для подбора архивных аналогов.")
    predicted = np.asarray(model.predict(frame), dtype=float)
    if predicted.shape != (1,) or not np.isfinite(predicted).all() or predicted[0] <= 0:
        raise ValueError("Модель не смогла получить положительную конечную цену для этих параметров.")
    price, mae = float(predicted[0]), float(metadata["test_metrics"]["mae_tenge"])
    return LocationPredictionResult(
        price_tenge=price, price_per_m2=price / float(basic.iloc[0]["area"]),
        lower_tenge=max(0.0, price - mae), upper_tenge=price + mae,
        warnings=tuple(notices), location_notes=tuple(notes),
    )

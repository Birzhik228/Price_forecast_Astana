"""Подготовить отдельные deployment-метаданные выбранной модели без обучения.

Старые модели, отчёты и config.py не меняются. Новые JSON публикуются атомарно
и не перезаписываются: повтор разрешён только при побайтовом совпадении.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from location_model import LOCATION_INPUT_COLUMNS


MODEL = "models/location_2026/complex_coords.joblib"
ORIGINAL_METADATA = "models/location_2026/complex_coords.json"
COMBINED_METADATA = "models/combined_2026/model_metadata.json"
DATA = "data/processed/combined_2026/listings_clean.csv"
COMBINED_SPLIT = "reports/combined_update/split_manifest.csv"
EXPERIMENT = "reports/location_experiment/"
DEPLOYMENT = "models/deployment/complex_coords.json"
ACTIVATION = "reports/model_activation_2026_10_13/activation.json"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _path(root: Path, name: str) -> Path:
    path = (root / name.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Путь вне проекта: {name}")
    return path


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Ожидается JSON-объект: {path}")
    return value


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _publish_new(path: Path, content: bytes) -> None:
    """Hard link публикует готовый файл атомарно и не заменяет чужой файл."""
    if path.exists():
        _require(path.read_bytes() == content, f"Отказ от перезаписи существующего файла: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".activation-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == content, f"Конкурирующая запись: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _partition(frame: pd.DataFrame, split: pd.DataFrame, label: str) -> pd.DataFrame:
    _require({"row_index", "subset"}.issubset(split), f"Неполное разбиение: {label}")
    _require(len(split) == len(frame) and split.row_index.is_unique
             and set(split.row_index) == set(frame.index)
             and split.subset.isin(["train", "test"]).all(), f"Некорректное разбиение: {label}")
    return split.set_index("row_index").sort_index()


def prepare_activation(root: Path | None = None) -> dict[str, Any]:
    """Проверить frozen-артефакты и создать metadata/report; не переключает config.

    root используется для изолированных тестов. Назначения JSON фиксированы,
    поэтому невозможно передать январский путь в качестве выходного файла.
    """
    root = (root or Path(__file__).resolve().parent).resolve()
    plan = _json(_path(root, EXPERIMENT + "experiment_plan.json"))
    signature = hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "signature"},
                                         sort_keys=True).encode()).hexdigest()
    _require(plan.get("signature") == signature and isinstance(plan.get("sources"), dict),
             "Неверная подпись плана эксперимента.")
    verification = _json(_path(root, EXPERIMENT + "verification.json"))
    _require(verification.get("status") == "passed" and verification.get("experiment_signature") == signature,
             "Нет успешной проверки данного эксперимента.")
    expected: dict[str, str] = {}
    for mapping in (plan["sources"], verification.get("source_sha256", {}),
                    verification.get("verified_artifact_sha256", {})):
        for name, checksum in mapping.items():
            name = name.replace("\\", "/")
            _require(name not in expected or expected[name] == checksum, f"Противоречивые SHA256: {name}")
            expected[name] = checksum
    required = {MODEL, ORIGINAL_METADATA, COMBINED_METADATA, DATA, COMBINED_SPLIT,
                "models/best_model.joblib", "models/model_metadata.json",
                EXPERIMENT + "experiment_plan.json", EXPERIMENT + "experiment_report.json",
                EXPERIMENT + "split_manifest.csv", EXPERIMENT + "holdout_predictions.csv"}
    _require(required.issubset(expected), "Проверка не связывает все необходимые исходные артефакты.")
    for name, checksum in expected.items():
        _require(_digest(_path(root, name)) == checksum, f"Изменён замороженный артефакт: {name}")
    expected[EXPERIMENT + "verification.json"] = _digest(_path(root, EXPERIMENT + "verification.json"))

    original = _json(_path(root, ORIGINAL_METADATA))
    combined = _json(_path(root, COMBINED_METADATA))
    report = _json(_path(root, EXPERIMENT + "experiment_report.json"))
    _require(original.get("schema_version") == 2 and original.get("variant") == "complex_coords"
             and original.get("feature_columns") == list(LOCATION_INPUT_COLUMNS)
             and original.get("target") == "asking_price" and original.get("currency") == "KZT"
             and original.get("geo_training_only") is True and original.get("selected_by_cv") is False
             and original.get("experiment_signature") == signature
             and original.get("model_sha256") == expected[MODEL], "Несовместимая исходная модель complex_coords.")
    _require(combined.get("schema_version") == 1 and combined.get("feature_columns") == list(FEATURE_COLUMNS)
             and combined.get("input_sha256") == expected[DATA]
             and combined.get("split_manifest_sha256") == expected[COMBINED_SPLIT],
             "Combined-метаданные не соответствуют данным и разбиению.")
    _require(report.get("signature") == signature and report.get("selected_variant") == "complex"
             and verification.get("selected_by_recomputed_oof") == "complex", "Не согласован победитель CV.")
    _require(verification.get("candidates", {}).get("complex_coords", {}).get("model_sha256") == expected[MODEL],
             "Проверенная модель отличается от выбранной.")

    frame = pd.read_csv(_path(root, DATA))
    base_split = _partition(frame, pd.read_csv(_path(root, COMBINED_SPLIT)), "combined")
    location_split = _partition(frame, pd.read_csv(_path(root, EXPERIMENT + "split_manifest.csv")), "location")
    _require(base_split.subset.equals(location_split.subset), "Обучающие строки location и combined различаются.")
    _require("record_id" in location_split and frame.record_id.is_unique
             and frame.record_id.equals(location_split.record_id.reset_index(drop=True)), "Нарушено соответствие record_id.")
    train_ids = location_split.index[location_split.subset.eq("train")]
    test_ids = location_split.index[location_split.subset.eq("test")]
    _require(len(train_ids) == original.get("train_rows") == combined.get("train_rows") == report.get("training_rows")
             and len(test_ids) == combined.get("test_rows") == report.get("original_test_rows"),
             "Число train/test не соответствует описаниям моделей.")
    for column in ("group", "experiment_group"):
        _require(column in location_split and not location_split[column].isna().any()
                 and not set(location_split.loc[train_ids, column]) & set(location_split.loc[test_ids, column]),
                 f"Группы пересекают train/test: {column}")
    train = frame.loc[train_ids]
    categories = {column: sorted(train[column].dropna().unique().tolist()) for column in CATEGORICAL_FEATURES}
    ranges = {column: {"min": float(train[column].min()), "max": float(train[column].max())}
              for column in NUMERIC_FEATURES if train[column].notna().any()}
    _require(categories == combined.get("categories_from_train") and ranges == combined.get("numeric_ranges_from_train"),
             "Категории или диапазоны не воспроизводятся на том же train.")

    predictions = pd.read_csv(_path(root, EXPERIMENT + "holdout_predictions.csv"))
    _require(predictions.record_id.is_unique and predictions.record_id.tolist() == frame.loc[test_ids, "record_id"].tolist(),
             "Прогнозы принадлежат другому holdout или имеют другой порядок.")
    _require(predictions.primary_evaluation.isin([True, False]).all(), "Некорректная маска primary_evaluation.")
    _require(predictions.primary_evaluation.tolist() == location_split.loc[test_ids, "primary_evaluation"].tolist(),
             "Маски holdout расходятся.")
    _require(np.allclose(predictions.price.to_numpy(), frame.loc[test_ids, "price"].to_numpy(), rtol=0, atol=.01),
             "Фактические цены holdout отличаются от frozen-данных.")
    primary = predictions.loc[predictions.primary_evaluation]
    actual = primary.price.to_numpy(float)
    predicted = primary.complex_coords_prediction.to_numpy(float)
    _require(len(actual) > 1 and np.isfinite(actual).all() and (actual > 0).all()
             and np.isfinite(predicted).all(), "Недопустимые цены или прогнозы.")
    error = predicted - actual
    scores = {"n": len(actual), "mae_tenge": float(np.abs(error).mean()),
              "rmse_tenge": float(np.sqrt(np.square(error).mean())),
              "r2": float(1 - np.square(error).sum() / np.square(actual - actual.mean()).sum()),
              "over_20_pct": float((np.abs(error) > actual * .2).mean() * 100)}
    matches = [item for item in report.get("comparison", []) if item.get("model") == "complex_coords"]
    _require(len(matches) == 1, "Нет единственной строки метрик complex_coords.")
    for published in (matches[0], verification.get("recomputed_holdout_metrics", {}).get("complex_coords", {})):
        _require(all(key in published and math.isclose(value, published[key], rel_tol=1e-10, abs_tol=1e-6)
                     for key, value in scores.items()), "Метрики complex_coords не воспроизводятся из прогнозов.")
    _require(scores["n"] == report.get("test_rows") == verification.get("primary_holdout_rows"), "Размер primary holdout различается.")
    # Сохраняем опубликованные значения (без дрейфа последних битов от CSV).
    metrics = {key: matches[0][key] for key in scores}
    metadata_path, activation_path = _path(root, DEPLOYMENT), _path(root, ACTIVATION)
    existing = _json(metadata_path) if metadata_path.exists() else None
    created_at = existing.get("created_at_utc") if existing else datetime.now(timezone.utc).isoformat()
    _require(isinstance(created_at, str) and bool(created_at), "Нет даты предыдущей подготовки metadata.")
    limitations = list(report.get("limitations", [])) + [
        "Свежая независимая оценка после марта не выполнена: пригодных новых наблюдений нет.",
        "Вариант с ЖК и координатами выбран пользователем; минимум CV MAE получил вариант только с ЖК.",
    ]
    metadata = {
        "schema_version": 2, "created_at_utc": created_at, "variant": "complex_coords",
        "model_name": "RandomForest + complex + coordinates", "model_label": "ЖК и координаты",
        "target": "asking_price", "currency": "KZT", "feature_columns": list(LOCATION_INPUT_COLUMNS),
        "base_feature_columns": list(FEATURE_COLUMNS), "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES), "categories_from_train": categories,
        "numeric_ranges_from_train": ranges, "model_path": MODEL, "model_sha256": expected[MODEL],
        "input_sha256": expected[DATA], "split_manifest_sha256": expected[COMBINED_SPLIT],
        "location_split_manifest_sha256": expected[EXPERIMENT + "split_manifest.csv"],
        "original_metadata_path": ORIGINAL_METADATA, "original_metadata_sha256": expected[ORIGINAL_METADATA],
        "experiment_signature": signature, "fitted_under_experiment_signature": original.get("fitted_under_experiment_signature", signature),
        "experiment_report_sha256": expected[EXPERIMENT + "experiment_report.json"],
        "train_rows": len(train_ids), "test_rows": scores["n"], "test_metrics": metrics,
        "approximate_range": {"half_width_tenge": metrics["mae_tenge"], "source": "reused_march_holdout",
                              "interpretation": "prediction +/- MAE; descriptive, not a confidence or prediction interval",
                              "guaranteed_coverage": None},
        "data_period": "2026-01_to_2026-03", "data_period_label": combined["data_period_label"],
        "activation_reason": "explicit_user_request", "selected_by_cv": False, "cv_selected_variant": "complex",
        "deployment_role": "user_selected_application_model", "geo_training_only": True,
        "holdout_previously_analyzed": True, "fresh_validation_completed": False,
        "limitations": limitations, "source_sha256": expected,
        "previous_january": {"model_path": "models/best_model.joblib", "metadata_path": "models/model_metadata.json",
                             "model_sha256": expected["models/best_model.joblib"], "metadata_sha256": expected["models/model_metadata.json"]},
    }
    content = _bytes(metadata)
    activation = {
        "schema_version": 1, "created_at_utc": created_at, "status": "deployment_metadata_prepared",
        "activation_reason": "explicit_user_request", "variant": "complex_coords", "fit_calls": 0,
        "configuration_changed_by_this_script": False, "models_copied_or_overwritten": False,
        "selected_by_cv": False, "cv_selected_variant": "complex", "experiment_signature": signature,
        "deployment_metadata": {"path": DEPLOYMENT, "sha256": hashlib.sha256(content).hexdigest()},
        "model": {"path": MODEL, "sha256": expected[MODEL]}, "original_metadata_sha256": expected[ORIGINAL_METADATA],
        "experiment_report_sha256": expected[EXPERIMENT + "experiment_report.json"],
        "previous_january": metadata["previous_january"], "source_sha256": expected,
        "train_rows": len(train_ids), "test_rows": scores["n"], "test_metrics": metrics,
        "comparison_on_reused_march_holdout": report["comparison"],
        "fresh_validation_completed": False, "limitations": limitations,
        "checks": ["experiment_signature", "all_pinned_source_and_verified_artifact_sha256", "original_model_and_metadata",
                   "identical_combined_and_location_train_rows", "disjoint_apartment_groups", "train_categories_and_ranges_recomputed",
                   "holdout_record_ids_and_prices", "holdout_metrics_recomputed", "no_training_or_original_artifact_writes"],
    }
    outputs = ((metadata_path, content), (activation_path, _bytes(activation)))
    # Не публиковать описание, если вход изменился между проверкой и чтением.
    for name, checksum in expected.items():
        _require(_digest(_path(root, name)) == checksum, f"Артефакт изменился во время подготовки: {name}")
    for path, value in outputs:
        _require(not path.exists() or path.read_bytes() == value, f"Отказ от перезаписи существующего файла: {path}")
    for path, value in outputs:
        _publish_new(path, value)
    return activation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = prepare_activation()
    print(json.dumps({key: result[key] for key in ("status", "variant", "train_rows", "test_rows", "test_metrics")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

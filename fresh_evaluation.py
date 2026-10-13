"""Сравнение трёх замороженных моделей на заранее проверенной выборке без fit.

Свежесть источника и отсутствие исторических объектов устанавливает внешний
протокол подготовки. Этот модуль не подбирает модели, не исключает неудобные
строки и не меняет приложение; все прогнозы относятся к одинаковым объектам.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from building_catalog import canonical_house_key, normalize_complex_name
from clean import atomic_write
from comparables import safe_listing_url
from config import PROJECT_ROOT
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from location_model import LOCATION_INPUT_COLUMNS, prepare_location_input
from prediction import load_artifacts


MODEL_NAMES = ("january", "combined", "complex")
MODEL_LABELS = {"january": "Январская (ориентир)", "combined": "Январь + март", "complex": "Январь + март + ЖК"}
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, "price", "record_id", "url", "collected_at", "residential_complex", "address")
PRICE_BANDS = ("≤20 млн ₸", "20–40 млн ₸", "40–70 млн ₸", ">70 млн ₸")
OUTPUT_NAMES = ("predictions.csv", "segments.csv", "evaluation_report.json", "report.html")


@dataclass(frozen=True)
class FrozenEvaluationConfig:
    """В обычном запуске root — проект; тесты используют отдельные временные bundles."""

    root: Path = PROJECT_ROOT
    bootstrap_repetitions: int = 2000
    seed: int = 42


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Нужен JSON-объект: {path.name}")
    return value


def _root_path(root: Path, name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError("Путь замороженного входа должен быть строкой.")
    relative = Path(name)
    candidate = (root / relative).resolve()
    if relative.is_absolute() or relative.drive or not candidate.is_relative_to(root):
        raise ValueError("Замороженный вход должен находиться внутри root проекта.")
    return candidate


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Нужен DataFrame с уникальными колонками.")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing or len(frame) < 2:
        raise ValueError(f"Нужно минимум два пригодных объявления и полная схема; отсутствуют: {sorted(missing)}")
    if "primary_eligible" in frame and not frame.primary_eligible.eq(True).fillna(False).all():
        raise ValueError("В оценку переданы строки, не допущенные внешним протоколом.")
    result = frame.copy(deep=True).reset_index(drop=True)
    ids = result.record_id
    if not ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("record_id должны быть непустыми строками.")
    if ids.str.strip().duplicated().any():
        raise ValueError("Повторные record_id не допускаются.")
    urls = result.url.map(lambda value: safe_listing_url(value) or (value.strip() if isinstance(value, str) else None)).dropna()
    if urls[urls.ne("")].duplicated().any():
        raise ValueError("Повторные URL не являются независимыми пригодными строками.")
    for column in (*NUMERIC_FEATURES, "price"):
        if result[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"Логическое значение вместо числа: {column}")
        try:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Нечисловое значение: {column}") from error
        values = result[column].to_numpy()
        if np.isinf(values).any() or (column in {"price", "area", "rooms"} and (not np.isfinite(values).all() or (values <= 0).any())):
            raise ValueError(f"Некорректные значения: {column}")
    if not np.equal(result.rooms, np.floor(result.rooms)).all():
        raise ValueError("Число комнат должно быть целым.")
    for column in CATEGORICAL_FEATURES:
        values = result[column]
        if not values.dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Категории должны быть строками: {column}")
        result[column] = values.astype(object).where(values.notna(), np.nan)
    return result


def _load_models(root: Path) -> tuple[dict[str, Pipeline], dict[str, dict], dict[str, str]]:
    """Проверить источники по плану и каждую пару model/metadata, затем загрузить."""
    plan_path = root / "reports/location_experiment/experiment_plan.json"
    plan = _json(plan_path)
    signature = hashlib.sha256(json.dumps({key: value for key, value in plan.items() if key != "signature"}, sort_keys=True).encode()).hexdigest()
    if plan.get("signature") != signature or not isinstance(plan.get("sources"), dict):
        raise ValueError("План location повреждён или не имеет корректной подписи.")
    protected = dict(plan["sources"])
    for name, expected in protected.items():
        if _digest(_root_path(root, name)) != expected:
            raise ValueError(f"Изменён замороженный вход location: {name}")
    paths = {
        "january": ("models/best_model.joblib", "models/model_metadata.json"),
        "combined": ("models/combined_2026/best_model.joblib", "models/combined_2026/model_metadata.json"),
        "complex": ("models/location_2026/complex.joblib", "models/location_2026/complex.json"),
    }
    other = ("data/processed/listings_clean.csv", "data/processed/january_2026/listings_clean.csv",
             "data/processed/combined_2026/listings_clean.csv", "reports/combined_update/split_manifest.csv",
             "reports/location_experiment/experiment_plan.json")
    for name in (*other, *(name for pair in paths.values() for name in pair)):
        current = _digest(_root_path(root, name))
        if name in protected and protected[name] != current:
            raise ValueError(f"Замороженный вход изменился во время загрузки: {name}")
        protected[name] = current
    models, metadata = {}, {}
    for name in ("january", "combined"):
        models[name], metadata[name] = load_artifacts(*(_root_path(root, path) for path in paths[name]))
    january_hash = metadata["january"].get("input_sha256")
    if not (january_hash == protected[other[0]] == protected[other[1]]):
        raise ValueError("Январские обучающие данные не соответствуют модели.")
    if metadata["combined"].get("input_sha256") != protected[other[2]] or metadata["combined"].get("split_manifest_sha256") != protected[other[3]]:
        raise ValueError("Combined данные или разбиение не соответствуют модели.")
    candidate = _json(_root_path(root, paths["complex"][1]))
    if (candidate.get("schema_version") != 2 or candidate.get("variant") != "complex"
            or candidate.get("selected_by_cv") is not True or candidate.get("feature_columns") != list(LOCATION_INPUT_COLUMNS)
            or candidate.get("target") != "asking_price" or candidate.get("currency") != "KZT"
            or candidate.get("experiment_signature") != signature
            or candidate.get("model_sha256") != protected[paths["complex"][0]]):
        raise ValueError("Артефакт +ЖК не соответствует выбранной модели схемы 2.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(_root_path(root, paths["complex"][0]))
    if (not isinstance(model, Pipeline) or list(getattr(model, "feature_names_in_", [])) != list(LOCATION_INPUT_COLUMNS)
            or getattr(model.named_steps.get("location_features"), "variant", None) != "complex"):
        raise ValueError("Pipeline +ЖК имеет несовместимую схему.")
    models["complex"], metadata["complex"] = model, candidate
    return models, metadata, protected


def evaluation_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual <= 0).any():
        raise ValueError("Для метрик нужны согласованные конечные цены и прогнозы.")
    if not len(actual):
        return dict(n=0, mae_tenge=None, rmse_tenge=None, r2=None, over_20_pct=None)
    error = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    return {"n": len(actual), "mae_tenge": float(np.abs(error).mean()),
            "rmse_tenge": float(np.sqrt(np.square(error).mean())),
            "r2": float(1 - np.square(error).sum() / denominator) if len(actual) >= 2 and denominator > 0 else None,
            "over_20_pct": float(np.mean(np.abs(error) > actual * .2) * 100)}


def _bootstrap(actual: np.ndarray, predictions: dict[str, np.ndarray], repetitions: int, seed: int) -> list[dict]:
    if not repetitions:
        return []
    pairs = (("combined", "january"), ("complex", "january"), ("complex", "combined"))
    differences = np.stack([np.abs(predictions[left] - actual) - np.abs(predictions[right] - actual) for left, right in pairs])
    rng, boot = np.random.default_rng(seed), []
    for start in range(0, repetitions, 128):
        selected = rng.integers(0, len(actual), size=(min(128, repetitions - start), len(actual)))
        boot.append(differences[:, selected].mean(axis=2))
    values = np.concatenate(boot, axis=1)
    return [{"candidate": left, "reference": right, "mae_difference_tenge": float(differences[index].mean()),
             "ci95_low_tenge": float(np.quantile(values[index], .025)), "ci95_high_tenge": float(np.quantile(values[index], .975)),
             "repetitions": repetitions, "seed": seed, "resampling_unit": "eligible_unique_row"}
            for index, (left, right) in enumerate(pairs)]


def _html_report(report: dict, segments: pd.DataFrame) -> str:
    comparison = pd.DataFrame(report["comparison"]).assign(model=lambda table: table.model.map(MODEL_LABELS))
    def table(value: pd.DataFrame) -> str:
        return value.to_html(index=False, border=0, escape=True, na_rep="—", float_format=lambda number: f"{number:,.3f}")
    return """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Сравнение замороженных моделей</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px/1.5 system-ui;color:#172635;background:#f5f7fa}section{background:white;padding:20px;margin:16px 0;border-radius:12px;overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{line-height:1.2}</style>
<h1>Сравнение замороженных моделей</h1><p>Одинаковые пригодные строки для января, объединённой модели и варианта + ЖК. Обучение и переключение приложения не выполнялись.</p>
""" + f"<section><h2>Общее качество: {report['rows']} объявлений</h2>{table(comparison)}</section>" + (
        f"<section><h2>Парные различия MAE</h2>{table(pd.DataFrame(report['paired_bootstrap']))}<p>Отрицательная разница означает меньшую ошибку кандидата. Интервалы описывают этот набор; они не доказывают репрезентативность рынка и не учитывают скрытую зависимость объявлений.</p></section>"
        if report["paired_bootstrap"] else ""
    ) + f"<section><h2>Сегменты</h2><p>Группы меньше 30 строк отмечены small_group; R² для единственной строки или постоянной цены не определён.</p>{table(segments)}</section>" + (
        "<section><h2>Ограничения и происхождение</h2><ul>" + "".join(f"<li>{html.escape(value)}</li>" for value in report["limitations"]) +
        "</ul><pre>" + html.escape(json.dumps(report["provenance"], ensure_ascii=False, indent=2)) + "</pre></section>"
        '<p><a href="predictions.csv">Построчные прогнозы</a> · <a href="segments.csv">Срезы CSV</a> · <a href="evaluation_report.json">Отчёт JSON</a></p></html>'
    )


def evaluate_frozen(frame: pd.DataFrame, *, output_dir: Path, provenance: dict,
                    config: FrozenEvaluationConfig | None = None) -> dict:
    """Оценить заранее допущенные строки; свежесть и исключение истории задаёт протокол."""
    config = config or FrozenEvaluationConfig()
    if type(config.bootstrap_repetitions) is not int or config.bootstrap_repetitions < 0 or type(config.seed) is not int:
        raise ValueError("Число повторов bootstrap и seed должны быть целыми; число повторов неотрицательно.")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Нужно непустое описание источника и протокола отбора.")
    provenance = json.loads(json.dumps(provenance, ensure_ascii=False, allow_nan=False))
    data = _validate_frame(frame)
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("Результат уже существует; укажите отдельную папку оценки.")
    root = Path(config.root).resolve()
    models, metadata, before = _load_models(root)
    legacy = data.loc[:, list(FEATURE_COLUMNS)].copy()
    location = legacy.assign(residential_complex=data.residential_complex,
                             house_key=[canonical_house_key(a, d, c) for a, d, c in
                                        zip(data.address, data.district, data.residential_complex)],
                             raw_latitude=np.nan, raw_longitude=np.nan, raw_coordinate_status="missing")
    location = prepare_location_input(location)
    actual, predictions = data.price.to_numpy(dtype=float), {}
    for name in MODEL_NAMES:
        predicted = np.asarray(models[name].predict(location if name == "complex" else legacy), dtype=float)
        if predicted.shape != actual.shape or not np.isfinite(predicted).all() or (predicted <= 0).any():
            raise ValueError(f"Некорректные прогнозы {name}; строки нельзя исключать по результатам модели.")
        predictions[name] = predicted
    known = {normalize_complex_name(value) for value in models["complex"].named_steps["preprocess"].named_transformers_["complex"].named_steps["encoder"].categories_[0]} - {None}
    complexes = data.residential_complex.map(normalize_complex_name)
    labels = {
        "district": data.district.fillna("Не указан"),
        "price_band": pd.cut(data.price, [0, 20e6, 40e6, 70e6, np.inf], labels=PRICE_BANDS, right=True),
        "completeness": legacy.notna().sum(axis=1).map(lambda value: f"{value}/8"),
        "complex_status": complexes.map(lambda value: "missing" if value is None else "known" if value in known else "new"),
    }
    segment_rows = []
    for dimension, values in labels.items():
        order = PRICE_BANDS if dimension == "price_band" else sorted(values.unique())
        for label in order:
            mask = values.eq(label).to_numpy()
            for name in MODEL_NAMES:
                segment_rows.append({"dimension": dimension, "segment": label, "model": name,
                                     **evaluation_metrics(actual[mask], predictions[name][mask]), "small_group": int(mask.sum()) < 30})
    segments = pd.DataFrame(segment_rows)
    exported = data.copy()
    exported.insert(0, "evaluation_row_index", np.arange(len(data)))
    exported["complex_status"] = labels["complex_status"]
    exported["completeness"] = legacy.notna().sum(axis=1)
    exported["price_band"] = labels["price_band"]
    for name, predicted in predictions.items():
        exported[name + "_prediction"] = predicted
        exported[name + "_signed_error"] = predicted - actual
        exported[name + "_absolute_error"] = np.abs(predicted - actual)
        exported[name + "_ape_pct"] = np.abs(predicted - actual) / actual * 100
    after = {name: _digest(_root_path(root, name)) for name in before}
    if before != after:
        raise RuntimeError("Модели, их описание или обучающие данные изменились во время оценки.")
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "completed",
              "rows": len(data), "record_ids": data.record_id.tolist(), "fit_calls": 0,
              "active_model": "january", "activated": False, "provenance": provenance,
              "actual_price_vector_sha256": hashlib.sha256(actual.astype("<f8").tobytes()).hexdigest(),
              "evaluation_frame_sha256": hashlib.sha256(data.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
              "comparison": [{"model": name, **evaluation_metrics(actual, predicted)} for name, predicted in predictions.items()],
              "paired_bootstrap": _bootstrap(actual, predictions, config.bootstrap_repetitions, config.seed),
              "segments": json.loads(segments.to_json(orient="records", force_ascii=False)),
              "artifact_hashes_before": before, "artifact_hashes_after": after,
              "model_metadata": metadata,
              "limitations": ["Свежесть источника, фактическая дата наблюдений и исключение исторических объектов подтверждаются внешним протоколом, а не самим вызовом оценщика.",
                               "Все модели оцениваются на одинаковых строках без обучения и выбора по этому набору; приложение не переключается.",
                               "Целевая переменная — цена предложения, а не цена сделки и не прогноз на шесть месяцев.",
                               "Парный bootstrap использует уникальные допущенные строки; скрытые повторы и зависимость домов могут сужать интервалы.",
                               "Даже независимый набор может не представлять весь рынок; небольшие сегменты нестабильны."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "predictions.csv", exported.to_csv(index=False, lineterminator="\n").encode())
    atomic_write(output_dir / "segments.csv", segments.to_csv(index=False, lineterminator="\n").encode())
    atomic_write(output_dir / "report.html", _html_report(report, segments).encode())
    report["output_sha256"] = {name: _digest(output_dir / name) for name in ("predictions.csv", "segments.csv", "report.html")}
    atomic_write(output_dir / "evaluation_report.json", (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    return report

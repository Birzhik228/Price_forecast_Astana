"""Build and evaluate one frozen OSM-enriched candidate; never activate or fit.

The registry and code are frozen before prediction.  The 590 March observations
were examined previously; this is a diagnostic comparison, not fresh validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from location_model import prepare_location_input
from verified_geo_model import CANDIDATE_ID, POLICY_VERSION, build_verified_candidate, load_verified_registry


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = Path("data/processed/verified_coordinates_2026_10_13/registry.json")
DEFAULT_MODELS = Path("models/verified_geo_2026")
DEFAULT_REPORTS = Path("reports/coordinate_enrichment_2026_10_13")
BASE_MODEL = Path("models/location_2026/complex_coords.joblib")
BASE_METADATA = Path("models/deployment/complex_coords.json")
GATE = {"mae": "candidate <= base", "over_20_count": "candidate <= base",
        "unaffected_predictions": "exact equality", "activation": "separate explicit action"}
LIMITATIONS = [
    "Это повторно использованные 590 мартовских наблюдений, не свежая независимая проверка.",
    "Регрессор и препроцессинг заморожены; новое заполнение координат меняет распределение входов.",
    "OSM сопоставляет адрес с объектом карты; это не гарантия точности до квартиры или подтверждение продавца.",
    "Снимок OSM октября 2026 года применяется к архивным объявлениям января–марта; историческая неизменность адресов не доказана.",
    "Старые координаты из train сохраняются и не становятся внешне подтверждёнными от наличия нового реестра.",
    "Правила и реестр зафиксированы до прогнозирования; адреса не отбираются по ошибке модели.",
    "Прогнозируется архивная цена предложения, не текущая цена сделки и не прогноз на шесть месяцев.",
    "Диапазон плюс/минус MAE описательный и не имеет гарантированного покрытия.",
]


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.abs(actual - predicted)
    return {"n": len(actual), "mae_tenge": float(error.mean()),
            "rmse_tenge": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
            "over_20_count": int((error / actual > .2).sum()),
            "over_20_pct": float((error / actual > .2).mean() * 100)}


def activation_gate(base: dict, candidate: dict, unchanged_exact: bool) -> dict:
    checks = {"mae_not_worse": candidate["mae_tenge"] <= base["mae_tenge"],
              "over_20_count_not_worse": candidate["over_20_count"] <= base["over_20_count"],
              "unchanged_rows_identical": bool(unchanged_exact)}
    return {**checks, "passed": all(checks.values()), "rules": copy.deepcopy(GATE)}


def evaluate_candidates(base: Pipeline, candidate: Pipeline, frame: pd.DataFrame,
                        archived_predictions: np.ndarray | None = None) -> tuple[dict, pd.DataFrame]:
    """Compare identical ordered observations; no fit, target-driven choice or tuning."""
    if (len(frame) < 2 or "record_id" not in frame or not frame.record_id.is_unique
            or not frame.record_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise ValueError("Нужны как минимум две строки с уникальными непустыми record_id.")
    actual = pd.to_numeric(frame.price, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError("Цены test должны быть положительными конечными числами.")
    x = prepare_location_input(frame)
    old_features = base.named_steps["location_features"].transform(x)
    new_features = candidate.named_steps["location_features"].transform(x)
    pd.testing.assert_frame_equal(old_features.drop(columns=["latitude", "longitude"]),
                                  new_features.drop(columns=["latitude", "longitude"]))
    old_geo = old_features.latitude.notna().to_numpy()
    new_geo = new_features.latitude.notna().to_numpy()
    pd.testing.assert_frame_equal(old_features.loc[old_geo], new_features.loc[old_geo])
    affected = ~old_geo & new_geo
    pd.testing.assert_frame_equal(old_features.loc[~affected], new_features.loc[~affected])
    # RF's threaded sum can differ by tiny rounding errors. Sequential reduction
    # makes the promised exact equality check meaningful without changing models.
    with joblib.parallel_backend("sequential"):
        old_prediction = np.asarray(base.predict(x), dtype=float)
        new_prediction = np.asarray(candidate.predict(x), dtype=float)
    for prediction in (old_prediction, new_prediction):
        if prediction.shape != actual.shape or not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError("Модель вернула некорректные прогнозы.")
    if archived_predictions is not None:
        np.testing.assert_allclose(old_prediction, archived_predictions, rtol=0, atol=.05,
                                   err_msg="Базовая модель не воспроизводит сохранённый March holdout.")
    unchanged_exact = bool(np.array_equal(old_prediction[~affected], new_prediction[~affected]))
    scores = {"base": _scores(actual, old_prediction), "candidate": _scores(actual, new_prediction)}
    provenance = candidate.named_steps["location_features"].coordinate_provenance(x)
    exported = frame[[column for column in ("record_id", "group", "house_key", "district", "price") if column in frame]].reset_index(drop=True).copy()
    exported["base_prediction"] = old_prediction
    exported["candidate_prediction"] = new_prediction
    exported["base_absolute_error"] = np.abs(actual - old_prediction)
    exported["candidate_absolute_error"] = np.abs(actual - new_prediction)
    exported["prediction_delta_tenge"] = new_prediction - old_prediction
    exported["base_geo_available"] = old_geo
    exported["candidate_geo_available"] = new_geo
    exported["new_geo_available"] = affected
    exported["coordinate_source"] = provenance.source_status.to_numpy()
    exported["osm_url"] = provenance.osm_url.to_numpy()
    result = {"comparison": scores, "gate": activation_gate(scores["base"], scores["candidate"], unchanged_exact),
              "test_rows": len(frame), "base_geo_rows": int(old_geo.sum()), "candidate_geo_rows": int(new_geo.sum()),
              "new_geo_rows": int(affected.sum()), "unchanged_rows": int((~affected).sum()),
              "changed_prediction_rows": int((old_prediction != new_prediction).sum()),
              "record_ids": frame.record_id.tolist(),
              "actual_price_vector_sha256": hashlib.sha256(actual.astype("<f8").tobytes()).hexdigest(),
              "new_geo_segment": {name: _scores(actual[affected], prediction[affected]) for name, prediction in
                                  (("base", old_prediction), ("candidate", new_prediction))} if affected.any() else None,
              "fit_calls": 0, "prediction_reduction": "sequential for exact reproducibility"}
    return result, exported


def _resolved(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _label(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _verify_sources(root: Path, expected: dict[str, str]) -> None:
    for name, checksum in expected.items():
        path = _resolved(root, Path(name))
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Изменился зафиксированный источник: {name}.")


def _render_report(report: dict) -> str:
    rows = "".join("<tr><td>" + "</td><td>".join(html.escape(str(value)) for value in
                   (name, metric["n"], f'{metric["mae_tenge"]:,.2f}', f'{metric["rmse_tenge"]:,.2f}',
                    metric["over_20_count"])) + "</td></tr>" for name, metric in report["comparison"].items())
    caveats = "".join(f"<li>{html.escape(text)}</li>" for text in report["limitations"])
    status = "Пройден" if report["gate"]["passed"] else "Не пройден"
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Координаты OSM: диагностическое сравнение</title><style>body{{font:16px/1.55 system-ui;max-width:1100px;margin:32px auto;padding:0 18px;color:#142536}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbc7d2;padding:8px;text-align:left}}.table{{overflow:auto}}li{{margin:8px 0}}</style>
<h1>Координаты OSM: диагностическое сравнение</h1><p>Те же {report['test_rows']} мартовских квартир. Новых координат в test: {report['new_geo_rows']}; вызовов fit: 0.</p>
<p>Критерий допуска: {status}. Кандидат сохранён отдельно; этот запуск не переключает приложение.</p>
<div class="table"><table><thead><tr><th>Модель</th><th>Строк</th><th>MAE, ₸</th><th>RMSE, ₸</th><th>Ошибка &gt;20%</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Неизменность прогнозов незатронутых строк: {report['gate']['unchanged_rows_identical']}.</p>
<ul>{caveats}</ul><p><a href="model_comparison.json">Полный JSON</a> · <a href="model_comparison.csv">Метрики CSV</a> · <a href="holdout_predictions.csv">Прогнозы по строкам</a> · <a href="protocol.json">Зафиксированный протокол</a></p></html>'''


def run_enrichment(registry_path: Path = DEFAULT_REGISTRY, *, project_root: Path = PROJECT_ROOT,
                   models_dir: Path = DEFAULT_MODELS, reports_dir: Path = DEFAULT_REPORTS) -> dict:
    """Production runner: checks the archived split, freezes protocol, writes a candidate."""
    from comparables import load_catalog
    from location_experiment import prepare_partition
    from location_prediction import load_app_artifacts

    root = Path(project_root).resolve()
    models, reports = _resolved(root, Path(models_dir)), _resolved(root, Path(reports_dir))
    registry_file = _resolved(root, Path(registry_path))
    metadata_file, model_file = root / BASE_METADATA, root / BASE_MODEL
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("geo_enrichment"):
        raise ValueError("Базовый deployment уже обогащён; нужен исходный frozen complex_coords.")
    protected = copy.deepcopy(metadata["source_sha256"])
    _verify_sources(root, protected)
    required = [metadata_file, model_file, registry_file, root / "features.py", root / "location_prediction.py",
                root / "comparables.py", root / "clean.py", root / "config.py",
                Path(__file__).resolve(), Path(__file__).resolve().with_name("verified_geo_model.py")]
    sources = {**protected, **{_label(root, path): digest(path) for path in required}}
    registry = load_verified_registry(registry_file)
    if registry.sha256 != sources[_label(root, registry_file)]:
        raise ValueError("Реестр изменился во время чтения.")
    catalogue = load_catalog(root / "data/processed/property_catalog")
    bundle = prepare_partition(pd.read_csv(root / "data/processed/combined_2026/listings_clean.csv", low_memory=False),
                               catalogue.listings, pd.read_csv(root / "reports/combined_update/split_manifest.csv"))
    frame = bundle["frame"].loc[bundle["evaluation_ids"]].copy()
    saved = pd.read_csv(root / "reports/location_experiment/holdout_predictions.csv")
    if (len(frame) != 590 or len(saved) != 590 or not saved.primary_evaluation.eq(True).all()
            or saved.record_id.tolist() != frame.record_id.tolist()):
        raise ValueError("Изменились состав или порядок исходного holdout из 590 квартир.")
    np.testing.assert_array_equal(saved.price.to_numpy(float), frame.price.to_numpy(float))
    for directory in (models, reports):
        directory.mkdir(parents=True, exist_ok=True)
    outputs = [models / "complex_coords.joblib", models / "complex_coords.json"] + [reports / name for name in
               ("protocol.json", "model_comparison.json", "model_comparison.csv", "holdout_predictions.csv", "model_comparison.html")]
    if any(path.exists() for path in outputs):
        raise ValueError("Выходные файлы уже существуют: сохраните предыдущий запуск и выберите новый каталог.")
    _verify_sources(root, sources)
    protocol = {"schema_version": 1, "candidate_id": CANDIDATE_ID, "policy_version": POLICY_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sources,
                "registry_sha256": registry.sha256, "registry_entries": len(registry.entries),
                "base_model_sha256": metadata["model_sha256"], "record_ids": frame.record_id.tolist(),
                "fit_calls": 0, "holdout_previously_analyzed": True, "fresh_validation_completed": False,
                "policy": {"preserve_fitted_points": True, "block_training_trait_conflicts": True,
                           "block_query_trait_conflicts": True, "price_independent_registry": True,
                           "allow_external_replacement_of_rejected_raw_gps": True},
                "gate": copy.deepcopy(GATE), "limitations": LIMITATIONS}
    _save_json(reports / "protocol.json", protocol)
    protocol_sha = digest(reports / "protocol.json")
    base, loaded_metadata = load_app_artifacts(model_file, metadata_file)
    if loaded_metadata != metadata:
        raise ValueError("Метаданные изменились после фиксации протокола.")
    candidate = build_verified_candidate(base, registry)
    result, predictions = evaluate_candidates(base, candidate, frame, saved.complex_coords_prediction.to_numpy(float))
    overlay = candidate.named_steps["location_features"]
    joblib.dump(candidate, models / "complex_coords.joblib", compress=3)
    restored = joblib.load(models / "complex_coords.joblib")
    with joblib.parallel_backend("sequential"):
        np.testing.assert_array_equal(restored.predict(prepare_location_input(frame)), predictions.candidate_prediction.to_numpy())
    _verify_sources(root, sources)
    if digest(reports / "protocol.json") != protocol_sha:
        raise ValueError("Протокол изменился во время оценки.")
    new_sha = digest(models / "complex_coords.joblib")
    candidate_metadata = {**metadata, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": CANDIDATE_ID, "geo_enrichment": True, "geo_training_only": False,
        "policy_version": POLICY_VERSION, "registry_path": _label(root, registry_file), "registry_sha256": registry.sha256,
        "base_model_path": BASE_MODEL.as_posix(), "base_model_sha256": metadata["model_sha256"],
        "base_metadata_sha256": sources[BASE_METADATA.as_posix()],
        "model_path": _label(root, models / "complex_coords.joblib"), "model_sha256": new_sha,
        "model_label": "ЖК и координаты с адресами OSM", "test_metrics": result["comparison"]["candidate"],
        "approximate_range": {"half_width_tenge": result["comparison"]["candidate"]["mae_tenge"],
                              "source": "reused_march_holdout_after_osm_overlay",
                              "interpretation": "prediction +/- MAE; descriptive, not a confidence or prediction interval",
                              "guaranteed_coverage": None},
        "activation_reason": None, "deployment_role": "staged_candidate", "active": False,
        "activation_gate_passed": result["gate"]["passed"], "fit_calls": 0, "protocol_sha256": protocol_sha,
        "base_experiment_signature": metadata.get("experiment_signature"), "experiment_signature": protocol_sha,
        "base_experiment_report_sha256": metadata.get("experiment_report_sha256"),
        "previous_deployment": {"model_path": BASE_MODEL.as_posix(), "metadata_path": BASE_METADATA.as_posix(),
                                "model_sha256": metadata["model_sha256"], "metadata_sha256": sources[BASE_METADATA.as_posix()]},
        "base_coordinate_house_count": len(base.named_steps["location_features"].house_coordinate_map_),
        "external_coordinate_house_count": len(overlay.external_entries_),
        "source_sha256": sources, "limitations": LIMITATIONS}
    # The archived experiment report describes the base, not this derivative.
    candidate_metadata.pop("experiment_report_sha256", None)
    _save_json(models / "complex_coords.json", candidate_metadata)
    report = {**result, "candidate_id": CANDIDATE_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "policy_version": POLICY_VERSION, "registry_sha256": registry.sha256, "registry_entries": len(registry.entries),
              "base_model_sha256": metadata["model_sha256"], "model_sha256": new_sha,
              "protocol_sha256": protocol_sha, "source_sha256": sources,
              "base_coordinate_house_count": candidate_metadata["base_coordinate_house_count"],
              "external_coordinate_house_count": len(overlay.external_entries_),
              "registry_blocked_trait_conflict": len(set(registry.entries) & overlay.blocked_house_keys_),
              "registry_existing_base_house": len(set(registry.entries) & set(base.named_steps["location_features"].house_coordinate_map_)),
              "active_changed": False, "candidate_saved_inactive": True, "serialized_replay_exact": True,
              "protected_sources_unchanged": True, "holdout_previously_analyzed": True,
              "fresh_validation_completed": False, "limitations": LIMITATIONS}
    predictions.to_csv(reports / "holdout_predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame([{"model": name, **metric} for name, metric in result["comparison"].items()]).to_csv(
        reports / "model_comparison.csv", index=False, lineterminator="\n")
    report["prediction_csv_sha256"] = digest(reports / "holdout_predictions.csv")
    _save_json(reports / "model_comparison.json", report)
    (reports / "model_comparison.html").write_text(_render_report(report), encoding="utf-8")
    _verify_sources(root, sources)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    result = run_enrichment(args.registry, models_dir=args.models_dir, reports_dir=args.reports_dir)
    print(json.dumps({key: result[key] for key in ("comparison", "gate", "new_geo_rows", "fit_calls", "active_changed")},
                     ensure_ascii=False, indent=2))

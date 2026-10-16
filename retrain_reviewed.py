"""Retrain two predeclared candidates on reviewed data; preserve the active app.

The previously inspected March cohort is a retrospective diagnostic benchmark.
No model selection, tuning, activation, or calibration uses its outcomes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from config import APP_MODEL_METADATA_PATH, APP_MODEL_PIPELINE_PATH, PROJECT_ROOT
from error_analysis import sha256
from location_experiment import house_validation_groups, save_json
from location_model import LOCATION_INPUT_COLUMNS, make_location_pipeline, prepare_location_input
from location_prediction import load_app_artifacts, _known_complexes
from retraining_data import validate_retraining_partition
from retraining_report import compare_predictions, build_segments, render_html
from trainable_verified_geo import make_trainable_verified_pipeline
from verified_geo_model import build_verified_candidate, load_verified_registry


DATA_DIR = PROJECT_ROOT / "data/processed/reviewed_2026_10_16"
AUDIT_DIR = PROJECT_ROOT / "reports/active_error_audit_2026_10_16"
QUALITY_DIR = PROJECT_ROOT / "reports/data_quality_2026_10_16"
DEFAULT_REPORTS = PROJECT_ROOT / "reports/retraining_2026_10_16"
DEFAULT_MODELS = PROJECT_ROOT / "models/retrained_2026_10_16"
LABELS = {"current": "Текущая модель", "repair_only": "Исправленные данные",
          "trained_geo": "Исправленные данные + координаты при обучении"}
LIMITATIONS = [
    "Те же 590 мартовских квартир уже использовались при разработке и разборе ошибок: это архивное сравнение, а не независимая свежая проверка.",
    "Внешний реестр OSM заморожен, но его допуск ранее учитывал безценовые признаки полного каталога, включая test; эксперимент не является полностью изолированной проверкой на новых объектах.",
    "Снимок OSM октября 2026 года не подтверждает состояние здания в марте. Координаты старого источника проверены лишь на внутреннюю согласованность.",
    "Группировка повторов эвристическая; неизвестные дубли и зависимость между квартирами одного ЖК могут сохраняться.",
    "Парный bootstrap по адресным группам описывает чувствительность разницы на этом наборе; он не учитывает отбор данных и предыдущие решения по этому test.",
    "Редкие сегменты с n<30 показаны справочно. Сегменты пересекаются; различия полноты анкеты не доказывают причинный эффект пропусков.",
    "Цены — предложения января и марта 2026 года. Новых объявлений, подтверждённых сделок и проверки сегодняшнего рынка нет.",
    "Спорные 55 ранее удалённых записей не возвращены; подозрительные цены и площади оставлены без неподтверждённых исправлений.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def verify_hashes(expected: dict[str, str]) -> None:
    for name, checksum in expected.items():
        path = PROJECT_ROOT / name
        require(path.is_file() and sha256(path) == checksum, f"Изменился закреплённый файл: {name}")


def sequential_predict(pipeline, frame: pd.DataFrame) -> np.ndarray:
    with joblib.parallel_backend("sequential"):
        result = np.asarray(pipeline.predict(frame), dtype=float)
    require(result.shape == (len(frame),) and np.isfinite(result).all() and (result > 0).all(),
            "Модель вернула некорректные прогнозы.")
    return result


def coordinate_summary(pipeline, x: pd.DataFrame) -> dict:
    transformer = pipeline.named_steps["location_features"]
    provenance = transformer.coordinate_provenance(x)
    return {"rows": len(x), "available_rows": int(provenance.coordinate_available.sum()),
            "sources": {str(k): int(v) for k, v in provenance.source_status.value_counts().items()},
            "mapped_house_keys": len(transformer.house_coordinate_map_),
            "registry_sha256": transformer.registry_sha256_}


def run_experiment(reports_dir: Path = DEFAULT_REPORTS, models_dir: Path = DEFAULT_MODELS) -> dict:
    reports_dir, models_dir = reports_dir.resolve(), models_dir.resolve()
    # Each run is an immutable artifact set; even a partial run needs a new path.
    for path in (reports_dir, models_dir):
        relative(path)
        require(not path.exists(), f"Каталог уже существует; задайте новый путь: {path}")
    require(reports_dir != models_dir and reports_dir not in models_dir.parents
            and models_dir not in reports_dir.parents, "Каталоги отчёта и моделей должны быть раздельными.")

    quality = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads(APP_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    audit = json.loads((AUDIT_DIR / "audit_report.json").read_text(encoding="utf-8"))
    audit_protocol = json.loads((AUDIT_DIR / "protocol.json").read_text(encoding="utf-8"))
    require(quality["checks"]["targets_unchanged"] and quality["checks"]["row_order_preserved"],
            "Исправленная версия не сохранила цены/порядок.")
    expected = {**quality["sources_sha256"], **quality["files_sha256"],
                **quality["active_artifacts_sha256"], **audit_protocol["source_sha256"]}
    # Reconcile mappings separately so a conflicting digest cannot be hidden by merging.
    for mapping in (quality["sources_sha256"], quality["files_sha256"],
                    quality["active_artifacts_sha256"], audit_protocol["source_sha256"]):
        verify_hashes(mapping)
    verify_hashes({relative(AUDIT_DIR / "protocol.json"): audit["protocol_sha256"],
                   relative(AUDIT_DIR / "errors.csv"): audit["output_csv_sha256"]["errors.csv"],
                   relative(APP_MODEL_PIPELINE_PATH): metadata["model_sha256"],
                   metadata["registry_path"]: metadata["registry_sha256"]})
    require(sha256(DATA_DIR / "manifest.json") == sha256(QUALITY_DIR / "report.json"),
            "Manifest исправленной версии не совпадает с отчётом исправления.")
    require(audit["model_sha256"] == metadata["model_sha256"]
            and audit["metadata_sha256"] == sha256(APP_MODEL_METADATA_PATH),
            "Аудит относится к другой активной модели.")
    extra = [DATA_DIR / "manifest.json", QUALITY_DIR / "protocol.json", AUDIT_DIR / "protocol.json",
             APP_MODEL_PIPELINE_PATH, APP_MODEL_METADATA_PATH,
             *[PROJECT_ROOT / name for name in ("retrain_reviewed.py", "retraining_data.py", "retraining_report.py",
               "trainable_verified_geo.py", "verified_geo_model.py", "location_model.py", "location_experiment.py",
               "active_error_analysis.py", "error_analysis.py", "train.py", "features.py", "config.py")]]
    sources = {**expected, **{relative(path): sha256(path) for path in extra}}
    verify_hashes(sources)

    original = pd.read_csv(PROJECT_ROOT / "data/processed/combined_2026/listings_clean.csv", low_memory=False)
    corrected = pd.read_csv(DATA_DIR / "listings_clean.csv", low_memory=False)
    catalog = pd.read_csv(DATA_DIR / "listings.csv", low_memory=False)
    split = pd.read_csv(PROJECT_ROOT / "reports/combined_update/split_manifest.csv")
    location_split = pd.read_csv(PROJECT_ROOT / "reports/location_experiment/split_manifest.csv")
    saved = pd.read_csv(AUDIT_DIR / "errors.csv", low_memory=False, float_precision="round_trip")
    bundle = validate_retraining_partition(original, corrected, catalog, split, location_split, saved)
    frame, train_ids, test_ids = bundle["frame"], bundle["train_ids"], bundle["evaluation_ids"]
    x, y = prepare_location_input(frame), frame.price.astype(float)
    require(tuple(x.columns) == LOCATION_INPUT_COLUMNS, "Некорректная схема признаков.")
    current, loaded_metadata = load_app_artifacts(APP_MODEL_PIPELINE_PATH, APP_MODEL_METADATA_PATH)
    require(metadata == loaded_metadata, "Метаданные изменились во время загрузки.")
    registry = load_verified_registry(PROJECT_ROOT / metadata["registry_path"])
    require(registry.sha256 == metadata["registry_sha256"], "Сменился реестр координат.")
    parameters = current.named_steps["model"].get_params()
    require(parameters["random_state"] is not None, "Для воспроизводимости требуется фиксированный seed.")
    protocol = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_sha256": sources, "model_labels": LABELS, "parameters": parameters,
        "training_rows": len(train_ids), "holdout_rows": len(test_ids),
        "input_columns": list(LOCATION_INPUT_COLUMNS), "candidate_variants": ["repair_only", "trained_geo"],
        "variant_definitions": {
            "repair_only": "Fit original location pipeline on corrected train, attach fixed external registry after fitting the forest.",
            "trained_geo": "Fit the same pipeline/forest with fixed external registry applied before fitting preprocessing and forest.",
        },
        "selection": "No model selection or hyperparameter search; report both predeclared variants; keep current app model.",
        "split": "Exact historical train19269/test590; revalidate apartment groups on corrected keys; no exclusions allowed.",
        "learned_preprocessing": "Fit on train only; full-catalog coordinates/conflict aggregates are not input columns.",
        "external_registry": "Frozen previously built registry; its eligibility used non-price full-catalog traits including held-out rows.",
        "bootstrap": {"method": "paired cluster percentile; connected house/apartment groups", "repetitions": 2000, "seed": 42},
        "activation": False, "limitations": LIMITATIONS,
        "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                     for name in ("scikit-learn", "numpy", "pandas", "joblib")}},
    }
    reports_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    save_json(reports_dir / "protocol.json", protocol)
    protocol_sha = sha256(reports_dir / "protocol.json")
    verified_split = pd.DataFrame({"row_index": frame.index, "record_id": frame.record_id,
                                  "source_row": frame.source_row, "historical_group": frame.group,
                                  "reviewed_apartment_group": bundle["groups"], "house_key": frame.house_key,
                                  "subset": np.where(frame.index.isin(train_ids), "train", "test")})
    verified_split.to_csv(reports_dir / "split_manifest.csv", index=False)
    cohort = frame.loc[test_ids].copy()
    clusters = house_validation_groups(cohort, bundle["groups"].loc[test_ids])
    cohort_x = x.loc[test_ids]
    predictions = {"current": sequential_predict(current, cohort_x)}
    # Removing two false holdout keys must not silently change baseline predictions.
    old_catalog = pd.read_csv(PROJECT_ROOT / "data/processed/property_catalog/listings.csv", low_memory=False)
    old_x = prepare_location_input(old_catalog.set_index("record_id", drop=False).loc[cohort.record_id])
    old_predictions = sequential_predict(current, old_x)
    np.testing.assert_allclose(predictions["current"], old_predictions, rtol=0, atol=0)
    np.testing.assert_allclose(predictions["current"], saved.predicted_price, rtol=0, atol=.05)
    print(f"Verified: train={len(train_ids)}, identical holdout={len(test_ids)}, baseline replay max delta="
          f"{np.max(np.abs(predictions['current'] - saved.predicted_price)):.9f} KZT", flush=True)
    candidates, artifacts, timings = {}, {}, {}
    geography = {"current": {"train": coordinate_summary(current, x.loc[train_ids]),
                             "holdout": coordinate_summary(current, cohort_x)}}
    for variant in protocol["candidate_variants"]:
        print(f"Fitting {variant}: fixed RandomForest parameters, no holdout tuning", flush=True)
        pipeline = (make_location_pipeline(current.named_steps["model"], "complex_coords") if variant == "repair_only"
                    else make_trainable_verified_pipeline(current.named_steps["model"], registry))
        require(pipeline.named_steps["model"].get_params() == parameters, "Изменились параметры регрессора.")
        started = time.monotonic()
        with threadpool_limits(limits=2):
            pipeline.fit(x.loc[train_ids], y.loc[train_ids])
        if variant == "repair_only":
            pipeline = build_verified_candidate(pipeline, registry)
        timings[variant] = time.monotonic() - started
        predictions[variant] = sequential_predict(pipeline, cohort_x)
        model_path = models_dir / f"{variant}.joblib"
        joblib.dump(pipeline, model_path, compress=3)
        loaded = joblib.load(model_path)
        np.testing.assert_array_equal(predictions[variant], sequential_predict(loaded, cohort_x))
        candidates[variant] = pipeline
        artifacts[variant] = {"path": relative(model_path), "sha256": sha256(model_path)}
        geography[variant] = {"train": coordinate_summary(pipeline, x.loc[train_ids]),
                              "holdout": coordinate_summary(pipeline, cohort_x)}
        print(f"Saved {variant}; fit {timings[variant]:.1f}s; reload predictions identical", flush=True)

    comparison = compare_predictions(cohort, predictions, clusters)
    baseline_provenance = current.named_steps["location_features"].coordinate_provenance(cohort_x)
    common_geo = baseline_provenance.source_status.map({
        "training_consensus": "Координаты из train", "osm_exact_address": "Точный адрес OSM",
    }).where(baseline_provenance.coordinate_available, "Нет пригодных координат")
    segments = build_segments(cohort, predictions, _known_complexes(current), common_geo)
    prediction_rows = cohort[["record_id", "source_row", "group", "house_key", "district", "residential_complex",
                              "area", "rooms", "price"]].copy()
    prediction_rows.insert(0, "row_index", test_ids)
    prediction_rows["bootstrap_cluster"] = clusters
    for name, predicted in predictions.items():
        prediction_rows[f"{name}_prediction"] = predicted
        prediction_rows[f"{name}_absolute_error"] = np.abs(y.loc[test_ids].to_numpy() - predicted)
        prediction_rows[f"{name}_geo_available"] = ((current if name == "current" else candidates[name])
            .named_steps["location_features"].coordinate_provenance(cohort_x).coordinate_available.to_numpy())
        if name != "current":
            prediction_rows[f"{name}_error_delta"] = (prediction_rows[f"{name}_absolute_error"]
                                                      - prediction_rows.current_absolute_error)
    prediction_rows.to_csv(reports_dir / "holdout_predictions.csv", index=False)
    segments.to_csv(reports_dir / "segments.csv", index=False)
    verify_hashes(sources)
    require(sha256(reports_dir / "protocol.json") == protocol_sha, "Изменился протокол.")
    report = {"schema_version": 1, "created_at_utc": protocol["created_at_utc"], "protocol_sha256": protocol_sha,
              "model_labels": LABELS, "comparison": comparison, "training_rows": len(train_ids), "holdout_rows": len(test_ids),
              "record_ids": cohort.record_id.tolist(), "fit_seconds": timings, "artifacts": artifacts,
              "geography": geography, "holdout_previously_analyzed": True, "fresh_validation_completed": False,
              "application_switched": False, "limitations": LIMITATIONS,
              "segment_cohorts": "Shared corrected features, baseline training complex familiarity and baseline geographic availability.",
              "checks": {"sources_and_active_artifacts_unchanged": True, "targets_and_holdout_ids_unchanged": True,
                         "revalidated_apartment_groups_disjoint": True, "baseline_predictions_unchanged_by_test_repairs": True,
                         "serialized_candidate_predictions_identical": True, "holdout_tuning": False,
                         "max_baseline_replay_delta_tenge": float(np.max(np.abs(predictions["current"] - saved.predicted_price)))},
              "output_csv_sha256": {name: sha256(reports_dir / name) for name in
                                    ("split_manifest.csv", "holdout_predictions.csv", "segments.csv")}}
    for name, artifact in artifacts.items():
        save_json(models_dir / f"{name}.json", {"schema_version": 1, "candidate_id": name,
                  "status": "experimental_inactive", "model_path": artifact["path"], "model_sha256": artifact["sha256"],
                  "protocol_path": relative(reports_dir / "protocol.json"), "protocol_sha256": protocol_sha,
                  "report_path": relative(reports_dir / "comparison.json"), "train_rows": len(train_ids),
                  "test_rows": len(test_ids), "registry_sha256": registry.sha256,
                  "input_columns": list(LOCATION_INPUT_COLUMNS), "parameters": parameters,
                  "runtime": protocol["runtime"], "application_compatible_activation_validated": False})
    save_json(reports_dir / "comparison.json", report)
    (reports_dir / "report.html").write_text(render_html(report, segments), encoding="utf-8")
    (reports_dir / "ATTRIBUTION.txt").write_text(
        "Contains derived OpenStreetMap geographic information. © OpenStreetMap contributors; ODbL 1.0.\n"
        "https://www.openstreetmap.org/copyright\n"
        f"Frozen source registry: {metadata['registry_path']} SHA256 {registry.sha256}\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    args = parser.parse_args()
    run_experiment(args.reports_dir, args.models_dir)

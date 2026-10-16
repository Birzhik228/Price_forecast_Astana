"""Nested grouped comparison of native-categorical CatBoost MAE and log-price.

All new candidates use eight original apartment fields plus complex name.
Parameters and family selection use train only. The archived 590-row holdout
is inspected only after selection and final fits have been locked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from catboost_model import CatBoostPriceRegressor, CATBOOST_MODEL_COLUMNS
from catboost_report import compare_predictions, build_segments, render_html
from catboost_validation import make_nested_folds, export_fold_manifest
from config import APP_MODEL_METADATA_PATH, APP_MODEL_PIPELINE_PATH, PROJECT_ROOT
from error_analysis import sha256
from features import FEATURE_COLUMNS
from location_experiment import house_validation_groups, save_json
from location_model import make_location_pipeline, prepare_location_input, LOCATION_INPUT_COLUMNS
from location_prediction import load_app_artifacts, _known_complexes
from retraining_data import validate_retraining_partition
from retrain_reviewed import require, verify_hashes, sequential_predict, relative
from train import compute_metrics


DATA_DIR = PROJECT_ROOT / "data/processed/reviewed_2026_10_16"
PREVIOUS = PROJECT_ROOT / "reports/retraining_2026_10_16"
AUDIT = PROJECT_ROOT / "reports/active_error_audit_2026_10_16"
DEFAULT_REPORTS = PROJECT_ROOT / "reports/catboost_2026_10_16"
DEFAULT_MODELS = PROJECT_ROOT / "models/catboost_2026_10_16"
VARIANTS = ("catboost_mae", "catboost_log")
PRESETS = {
    "depth6_t600": {"iterations": 600, "depth": 6, "learning_rate": .05, "l2_leaf_reg": 5.},
    "depth8_t900": {"iterations": 900, "depth": 8, "learning_rate": .05, "l2_leaf_reg": 10.},
}
LABELS = {"current": "Текущая модель с координатами", "rf_control": "RandomForest: 8 полей + ЖК",
          "catboost_mae": "CatBoost: MAE цены", "catboost_log": "CatBoost: RMSE логарифма цены",
          "selected_procedure": "Выбор CatBoost только по внутренней проверке"}
LIMITATIONS = [
    "Внешний holdout содержит те же ранее изученные 590 мартовских квартир. Это архивное сравнение, не свежая независимая проверка рынка.",
    "Новые CatBoost и контрольный RandomForest используют 8 полей + ЖК. Координаты, OSM и house_key не являются признаками; house_key применяется только для разделения групп.",
    "Действующая модель использует прежнюю географию и реестр OSM с отбором по полному каталогу. Сопоставимый контроль RF без координат отделяет эту разницу от смены алгоритма.",
    "Вложенная проверка разделяет целые эвристические группы домов и квартир. Неизвестные повторы и разные ключи одного дома могут сохраняться.",
    "Архив уже участвовал в разработке и разборе ошибок. Вложенная проверка изолирует новый подбор, но не делает всю историю разработки независимой от этих данных.",
    "RandomForest служит фиксированным контролем; его прежние параметры не подбирались заново. Выбор семейства выполняется только между двумя CatBoost, по MAE в тенге.",
    "Рассмотрены два заранее заданных набора параметров на вариант; это ограниченный эксперимент, не исчерпывающий поиск лучшей модели.",
    "CatBoost с log1p(price) обучается на RMSE логарифмов, но все проверки используют expm1-прогнозы и MAE в тенге; корректировка по test и обрезка прогнозов не применяются.",
    "Парные bootstrap-интервалы по группам описательны, не учитывают предшествующий отбор моделей. Сегменты с n<30 не дают устойчивых выводов.",
    "Цены относятся к архивным предложениям января и марта 2026 года, не к подтверждённым сделкам и не к сегодняшнему рынку.",
]


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def score_prices(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    require(actual.ndim == 1 and actual.shape == predicted.shape and len(actual) > 1,
            "Несовместимые массивы цены и прогноза.")
    require(np.isfinite(actual).all() and np.isfinite(predicted).all() and (actual > 0).all(),
            "Некорректные цены или прогнозы.")
    absolute = np.abs(actual - predicted)
    return {"n": len(actual), **compute_metrics(actual, predicted),
            "median_ae_tenge": float(np.median(absolute)),
            "median_ape_pct": float(np.median(absolute / actual) * 100),
            "over_20_n": int((absolute > actual * .2).sum()),
            "over_20_pct": float((absolute > actual * .2).mean() * 100)}


def choose_best(rows: list[dict]) -> dict:
    require(bool(rows) and all(np.isfinite(row["mae_tenge"]) for row in rows), "Нет конечной MAE для выбора.")
    # Ties are deterministic and have no dependency on an external evaluation.
    return min(rows, key=lambda row: (row["mae_tenge"], row["variant"], row["params_id"]))


def load_inputs():
    previous = json.loads((PREVIOUS / "comparison.json").read_text(encoding="utf-8"))
    previous_protocol = json.loads((PREVIOUS / "protocol.json").read_text(encoding="utf-8"))
    verify_hashes(previous_protocol["sources_sha256"])
    verify_hashes({relative(PREVIOUS / "protocol.json"): previous["protocol_sha256"],
                   **{relative(PREVIOUS / name): checksum for name, checksum in previous["output_csv_sha256"].items()}})
    extra = [PREVIOUS / "protocol.json", PREVIOUS / "comparison.json", APP_MODEL_METADATA_PATH,
             APP_MODEL_PIPELINE_PATH, *[PROJECT_ROOT / name for name in (
                 "train_catboost_experiment.py", "catboost_model.py", "catboost_validation.py", "catboost_report.py",
                 "requirements.txt", "requirements.lock.txt")]]
    sources = {**previous_protocol["sources_sha256"], **{relative(path): sha256(path) for path in extra}}
    original = pd.read_csv(PROJECT_ROOT / "data/processed/combined_2026/listings_clean.csv", low_memory=False)
    corrected = pd.read_csv(DATA_DIR / "listings_clean.csv", low_memory=False)
    catalog = pd.read_csv(DATA_DIR / "listings.csv", low_memory=False)
    split = pd.read_csv(PROJECT_ROOT / "reports/combined_update/split_manifest.csv")
    location_split = pd.read_csv(PROJECT_ROOT / "reports/location_experiment/split_manifest.csv")
    saved = pd.read_csv(AUDIT / "errors.csv", low_memory=False, float_precision="round_trip")
    bundle = validate_retraining_partition(original, corrected, catalog, split, location_split, saved)
    current, metadata = load_app_artifacts(APP_MODEL_PIPELINE_PATH, APP_MODEL_METADATA_PATH)
    require(metadata["model_sha256"] == previous_protocol["sources_sha256"][relative(APP_MODEL_PIPELINE_PATH)],
            "Изменилась действующая модель.")
    return bundle, saved, current, metadata, sources


class Experiment:
    """Persist each validation prediction; reuse only identical verified fits."""

    def __init__(self, frame, x, y, current, reports, models, protocol_sha):
        self.frame, self.x, self.y, self.current = frame, x, y, current
        self.reports, self.models, self.protocol_sha = reports, models, protocol_sha
        self.events = []

    def estimator(self, variant: str, params_id: str):
        if variant == "rf_control":
            require(params_id == "fixed_current_rf", "RF должен иметь фиксированные параметры.")
            return make_location_pipeline(self.current.named_steps["model"], "complex")
        require(variant in VARIANTS and params_id in PRESETS, "Неизвестный вариант или параметры.")
        return CatBoostPriceRegressor(target_mode="price_mae" if variant == "catboost_mae" else "log_rmse",
                                     **PRESETS[params_id], random_seed=42, thread_count=2)

    def validation_fit(self, variant, params_id, train_ids, validation_ids, *, phase, outer_fold=0, fold=0):
        train_ids, validation_ids = np.asarray(train_ids), np.asarray(validation_ids)
        require(len(set(train_ids) & set(validation_ids)) == 0, "Train и validation пересекаются.")
        details = {"protocol_sha256": self.protocol_sha, "variant": variant, "params_id": params_id,
                   "train_index_sha256": signature(train_ids.tolist()),
                   "validation_index_sha256": signature(validation_ids.tolist())}
        key = signature(details)
        checkpoint = self.reports / "checkpoints" / key
        metadata_path, predictions_path = checkpoint.with_suffix(".json"), checkpoint.with_suffix(".csv")
        reused = metadata_path.exists()
        if reused:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            require(meta["identity"] == details and sha256(predictions_path) == meta["csv_sha256"],
                    "Checkpoint не соответствует входам.")
            saved = pd.read_csv(predictions_path, float_precision="round_trip")
            require(saved.row_index.tolist() == validation_ids.tolist()
                    and saved.record_id.tolist() == self.frame.loc[validation_ids, "record_id"].tolist(),
                    "Checkpoint изменил состав validation.")
            np.testing.assert_array_equal(saved.actual_price, self.y.loc[validation_ids])
            predicted = saved.predicted_price.to_numpy(float)
            seconds = meta["fit_seconds"]
        else:
            print(f"Fit {phase} outer={outer_fold} fold={fold}: {variant}/{params_id}; "
                  f"train={len(train_ids)}, validation={len(validation_ids)}", flush=True)
            estimator = self.estimator(variant, params_id)
            started = time.monotonic()
            with threadpool_limits(limits=2):
                estimator.fit(self.x.loc[train_ids], self.y.loc[train_ids])
            predicted = sequential_predict(estimator, self.x.loc[validation_ids])
            seconds = time.monotonic() - started
            saved = pd.DataFrame({"row_index": validation_ids,
                                  "record_id": self.frame.loc[validation_ids, "record_id"].to_numpy(),
                                  "actual_price": self.y.loc[validation_ids].to_numpy(), "predicted_price": predicted})
            saved.to_csv(predictions_path, index=False)
            save_json(metadata_path, {"identity": details, "csv_sha256": sha256(predictions_path),
                      "fit_seconds": seconds, "metrics": score_prices(saved.actual_price, predicted)})
        scores = score_prices(self.y.loc[validation_ids], predicted)
        self.events.append({"phase": phase, "outer_fold": outer_fold, "fold": fold, "variant": variant,
                            "params_id": params_id, "fit_seconds": seconds, "reused": reused,
                            "checkpoint": key, "train_rows": len(train_ids), **scores})
        save_json(self.reports / "progress.json", {"events": self.events, "completed_calls": len(self.events),
                  "unique_validation_fits": len({event["checkpoint"] for event in self.events})})
        print(f"  {'Reused' if reused else 'Done'} {variant}/{params_id}: MAE {scores['mae_tenge']:,.0f} KZT; "
              f"{seconds:.1f}s", flush=True)
        return pd.Series(predicted, index=validation_ids, dtype=float)

    def search(self, folds, scope_ids, phase, outer_fold=0):
        results = []
        for variant in VARIANTS:
            for params_id in PRESETS:
                out = pd.Series(np.nan, index=scope_ids, dtype=float)
                for fold in folds:
                    values = self.validation_fit(variant, params_id, fold["train_ids"], fold["validation_ids"],
                                                  phase=phase, outer_fold=outer_fold, fold=fold["fold"])
                    require(out.loc[values.index].isna().all(), "Повторное назначение OOF-прогноза.")
                    out.loc[values.index] = values
                require(out.notna().all(), "Не все строки получили OOF-прогноз.")
                results.append({"variant": variant, "params_id": params_id, "parameters": PRESETS[params_id],
                                **score_prices(self.y.loc[scope_ids], out)})
        by_family = {variant: choose_best([row for row in results if row["variant"] == variant]) for variant in VARIANTS}
        chosen = choose_best(results)
        for row in results:
            row["selected"] = row["params_id"] == by_family[row["variant"]]["params_id"]
            row["procedure_selected"] = (row["variant"], row["params_id"]) == (chosen["variant"], chosen["params_id"])
        return {"folds": len(folds), "selection_metric": "mae_tenge", "results": results,
                "by_family": by_family, "chosen_variant": chosen["variant"], "chosen_params_id": chosen["params_id"]}

    def final_fit(self, variant, params_id, train_ids):
        path = self.models / f"{variant}.joblib"
        meta_path = self.models / f"{variant}.json"
        identity = {"protocol_sha256": self.protocol_sha, "variant": variant, "params_id": params_id,
                    "train_index_sha256": signature(np.asarray(train_ids).tolist())}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            require(meta["identity"] == identity and sha256(path) == meta["model_sha256"], "Неверный final checkpoint.")
            return joblib.load(path), meta
        print(f"Final fit: {variant}/{params_id}; train={len(train_ids)}", flush=True)
        estimator = self.estimator(variant, params_id)
        started = time.monotonic()
        with threadpool_limits(limits=2):
            estimator.fit(self.x.loc[train_ids], self.y.loc[train_ids])
        seconds = time.monotonic() - started
        joblib.dump(estimator, path, compress=3)
        loaded = joblib.load(path)
        probe = self.x.loc[np.asarray(train_ids)[:100]]
        np.testing.assert_array_equal(sequential_predict(estimator, probe), sequential_predict(loaded, probe))
        meta = {"identity": identity, "status": "experimental_inactive", "model_path": relative(path),
                "model_sha256": sha256(path), "fit_seconds": seconds, "training_rows": len(train_ids),
                "actual_model_features": list(CATBOOST_MODEL_COLUMNS), "input_columns": list(LOCATION_INPUT_COLUMNS),
                "target_mode": "price_mae" if variant == "catboost_mae" else "log_rmse" if variant == "catboost_log" else "price_squared_error",
                "prediction_unit": "KZT", "uses_coordinates": False, "activation_compatibility_validated": False,
                "cb_effective_parameters": loaded.model_.get_all_params() if variant in VARIANTS else None}
        save_json(meta_path, meta)
        return loaded, meta


def run(reports=DEFAULT_REPORTS, models=DEFAULT_MODELS, *, resume=False):
    reports, models = Path(reports).resolve(), Path(models).resolve()
    for path in (reports, models):
        relative(path)
    require(reports != models and reports not in models.parents and models not in reports.parents,
            "Каталоги моделей и отчётов должны быть раздельными.")
    if resume:
        require((reports / "protocol.json").is_file() and models.is_dir(), "Нет начатого эксперимента для --resume.")
        require(not (reports / "comparison.json").exists(), "Эксперимент завершён; используйте новую папку.")
    else:
        require(not reports.exists() and not models.exists(), "Каталог уже существует; используйте новые пути или --resume.")
    bundle, saved, current, metadata, sources = load_inputs()
    frame, train_ids, test_ids = bundle["frame"], bundle["train_ids"], bundle["evaluation_ids"]
    x, y = prepare_location_input(frame), frame.price.astype(float)
    folds = make_nested_folds(frame.loc[train_ids], bundle["groups"].loc[train_ids], outer_splits=3, inner_splits=3)
    cv_manifest = export_fold_manifest(frame.loc[train_ids], folds)
    split_manifest = pd.DataFrame({"row_index": frame.index, "record_id": frame.record_id, "house_key": frame.house_key,
                                  "apartment_group": bundle["groups"],
                                  "subset": np.where(frame.index.isin(train_ids), "train", "test")})
    manifest_bytes = {"cv_folds.csv": cv_manifest.to_csv(index=False, lineterminator="\n").encode("utf-8"),
                      "split_manifest.csv": split_manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")}
    manifest_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in manifest_bytes.items()}
    spec = {"schema_version": 1, "sources_sha256": sources, "presets": PRESETS,
            "variants": list(VARIANTS), "target_modes": {"catboost_mae": "MAE(price)", "catboost_log": "RMSE(log1p(price))"},
            "features": list(CATBOOST_MODEL_COLUMNS), "uses_coordinates": False, "seed": 42, "threads": 2,
            "outer_folds": 3, "inner_folds": 3, "final_selection_folds": 3,
            "selection": "Minimum pooled validation MAE in KZT; CatBoost candidates only; fixed deterministic tiebreak.",
            "rf_control": {"role": "fixed matched-feature benchmark, excluded from parameter/family selection",
                           "parameters": current.named_steps["model"].get_params()},
            "test_rows": len(test_ids), "train_rows": len(train_ids), "early_stopping": False,
            "holdout_policy": "No holdout prediction/evaluation until selection is persisted and all final models are fitted.",
            "split_signature": signature(cv_manifest.to_dict("records")),
            "persisted_split_sha256": manifest_hashes,
            "grouping": {"components": int(folds["groups"].nunique()),
                         "train_rows_without_house_key": int(frame.loc[train_ids, "house_key"].isna().sum()),
                         "missing_house_policy": "Only known apartment links protect rows with no house identity."},
            "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
                        ("catboost", "scikit-learn", "numpy", "pandas", "joblib")}},
            "limitations": LIMITATIONS, "application_switched": False}
    if resume:
        protocol = json.loads((reports / "protocol.json").read_text(encoding="utf-8"))
        require(protocol["spec"] == spec and protocol["signature"] == signature(spec), "Изменились входы, код или параметры; resume запрещён.")
    else:
        protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "signature": signature(spec), "spec": spec}
        reports.mkdir(parents=True)
        models.mkdir(parents=True)
        (reports / "checkpoints").mkdir()
        save_json(reports / "protocol.json", protocol)
        for name, raw in manifest_bytes.items():
            (reports / name).write_bytes(raw)
    verify_hashes({relative(reports / name): digest for name, digest in manifest_hashes.items()})
    protocol_sha = sha256(reports / "protocol.json")
    print(f"Protocol frozen: train={len(train_ids)}, holdout={len(test_ids)}; nested 3x3 house-group CV; "
          f"CatBoost {spec['runtime']['catboost']}", flush=True)
    experiment = Experiment(frame, x, y, current, reports, models, protocol_sha)
    oof = pd.DataFrame(index=train_ids, columns=[*VARIANTS, "rf_control", "selected_procedure"], dtype=float)
    outer_rows, nested_selections = [], []
    for outer in folds["outer"]:
        number, inner = outer["fold"], outer["inner"]
        selection = experiment.search(inner, outer["train_ids"], "inner", number)
        nested_selections.append({"outer_fold": number, **selection})
        save_json(reports / f"outer_{number}_selection.json", selection)
        for variant in (*VARIANTS, "rf_control"):
            params_id = selection["by_family"][variant]["params_id"] if variant in VARIANTS else "fixed_current_rf"
            values = experiment.validation_fit(variant, params_id, outer["train_ids"], outer["validation_ids"],
                                                phase="outer", outer_fold=number, fold=number)
            oof.loc[values.index, variant] = values
            outer_rows.append({"outer_fold": number, "variant": variant, "selected_params_id": params_id,
                               **score_prices(y.loc[values.index], values)})
        selected = selection["chosen_variant"]
        validation = outer["validation_ids"]
        oof.loc[validation, "selected_procedure"] = oof.loc[validation, selected]
        outer_rows.append({"outer_fold": number, "variant": "selected_procedure", "selected_variant": selected,
                           "selected_params_id": selection["chosen_params_id"],
                           **score_prices(y.loc[validation], oof.loc[validation, "selected_procedure"])})
    require(oof.notna().all().all(), "Внешняя проверка не покрыла весь train.")
    nested = {"outer_folds": 3, "inner_folds": 3, "selection_metric": "mae_tenge",
              "summary": [{"variant": variant, **score_prices(y.loc[train_ids], oof[variant])} for variant in oof],
              "folds": outer_rows, "inner_selection": nested_selections,
              "rf_role": "fixed control, excluded from family selection"}
    save_json(reports / "nested_cv.json", nested)
    oof.insert(0, "record_id", frame.loc[train_ids, "record_id"])
    oof.insert(1, "actual_price", y.loc[train_ids])
    oof.insert(2, "house_group", folds["groups"])
    oof.to_csv(reports / "nested_oof_predictions.csv", index_label="row_index")
    selection = experiment.search(folds["selection"], train_ids, "selection")
    # This is the irrevocable selection checkpoint written before holdout access.
    save_json(reports / "selection_cv.json", selection)
    selection_sha = sha256(reports / "selection_cv.json")
    fitted, artifacts = {}, {}
    for variant in (*VARIANTS, "rf_control"):
        params_id = selection["by_family"][variant]["params_id"] if variant in VARIANTS else "fixed_current_rf"
        fitted[variant], artifacts[variant] = experiment.final_fit(variant, params_id, train_ids)
    verify_hashes(sources)
    require(sha256(reports / "selection_cv.json") == selection_sha, "Изменился выбор параметров.")
    # First access to the external evaluation predictions is intentionally here.
    cohort, cohort_x = frame.loc[test_ids], x.loc[test_ids]
    predictions = {"current": sequential_predict(current, cohort_x), **{
                   variant: sequential_predict(model, cohort_x) for variant, model in fitted.items()}}
    np.testing.assert_allclose(predictions["current"], saved.predicted_price, rtol=0, atol=.05)
    for variant in fitted:
        np.testing.assert_array_equal(predictions[variant], sequential_predict(joblib.load(models / f"{variant}.joblib"), cohort_x))
    clusters = house_validation_groups(cohort, bundle["groups"].loc[test_ids])
    comparison = compare_predictions(cohort, predictions, clusters)
    provenance = current.named_steps["location_features"].coordinate_provenance(cohort_x)
    geo = provenance.source_status.map({"training_consensus": "Координаты из train", "osm_exact_address": "Точный адрес OSM"})
    geo = geo.where(provenance.coordinate_available, "Нет пригодных координат")
    segments = build_segments(cohort, predictions, _known_complexes(current), geo)
    errors = cohort[["record_id", "source_row", "group", "house_key", "district", "residential_complex", *[
                    column for column in FEATURE_COLUMNS if column != "district"], "price"]].copy()
    errors.insert(0, "row_index", test_ids)
    errors["bootstrap_cluster"] = clusters
    for variant, prediction in predictions.items():
        errors[f"{variant}_prediction"] = prediction
        errors[f"{variant}_absolute_error"] = np.abs(prediction - y.loc[test_ids].to_numpy())
    errors.to_csv(reports / "holdout_predictions.csv", index=False)
    segments.to_csv(reports / "segments.csv", index=False)
    pd.DataFrame(experiment.events).to_csv(reports / "fit_log.csv", index=False)
    verify_hashes(sources)
    require(sha256(reports / "protocol.json") == protocol_sha, "Изменился протокол.")
    verify_hashes({relative(reports / name): digest for name, digest in manifest_hashes.items()})
    csvs = ("cv_folds.csv", "split_manifest.csv", "nested_oof_predictions.csv", "holdout_predictions.csv", "segments.csv", "fit_log.csv")
    report = {"schema_version": 1, "created_at_utc": protocol["created_at_utc"], "protocol_sha256": protocol_sha,
              "selection_sha256": selection_sha, "model_labels": LABELS, "training_rows": len(train_ids), "holdout_rows": len(test_ids),
              "comparison": comparison, "nested_cv": nested, "selection_cv": selection,
              "selected_variant": selection["chosen_variant"], "artifacts": artifacts,
              "fit_calls": {"unique_validation": len({event["checkpoint"] for event in experiment.events}), "final": 3},
              "limitations": LIMITATIONS, "application_switched": False, "fresh_validation_completed": False,
              "checks": {"sources_and_active_model_unchanged": True, "holdout_ids_targets_unchanged": True,
                         "nested_groups_disjoint": True, "holdout_used_for_tuning": False,
                         "prediction_reload_exact": True, "candidate_geo_input": False,
                         "baseline_replay_max_delta_tenge": float(np.max(np.abs(predictions["current"] - saved.predicted_price)))},
              "output_csv_sha256": {name: sha256(reports / name) for name in csvs}}
    (reports / "report.html").write_text(render_html(report, segments), encoding="utf-8")
    save_json(reports / "comparison.json", report)
    print(json.dumps({"nested": nested["summary"], "selected_variant": report["selected_variant"],
                      "holdout": comparison["overall"], "paired": comparison["paired"]}, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--resume", action="store_true", help="Continue only with identical sources, code and settings")
    args = parser.parse_args()
    run(args.reports_dir, args.models_dir, resume=args.resume)

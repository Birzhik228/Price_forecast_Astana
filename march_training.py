"""Групповое обучение March и January+March с внешним защищённым holdout."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from config import RANDOM_STATE
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from prediction import load_artifacts, validate_metadata
from train import (
    SCORING, TrainingConfig, compute_metrics, cv_feature_importance, cv_summary,
    file_sha256, make_models, save_figures, search_space, write_json,
)


DATA_PERIOD_LABEL = "конца марта 2026 года (по описанию источника)"
CV_FOLDS = 3
SEARCH_ITERATIONS = 8
MODEL_JOBS = 2


@dataclass(frozen=True)
class TrainingContext:
    """Происхождение train и holdout; задаётся отдельно от неизменного ML-поиска."""

    data_period: str
    data_period_label: str
    training_scope: str
    candidate_role: str
    holdout_description: str
    limitations: tuple[str, ...]
    holdout_previously_analyzed: bool = False


MARCH_CONTEXT = TrainingContext(
    data_period="2026-03",
    data_period_label=DATA_PERIOD_LABEL,
    training_scope="March train only; no January rows; not refitted on test or full dataset",
    candidate_role="march_candidate",
    holdout_description="Predefined March holdout with no January URL/signature matches found by the caller; both models evaluated on exactly the same rows.",
    limitations=(
        "March dates come from the source description, not independently verified per-listing timestamps.",
        "The caller defines the March-only dataset and removes January URL/signature overlaps from holdout eligibility.",
    ),
)

COMBINED_CONTEXT = TrainingContext(
    data_period="2026-01_to_2026-03",
    data_period_label="января и конца марта 2026 года (март — по описанию источника)",
    training_scope="Combined January and March train, including the former January holdout; protected March holdout excluded; not refitted on protected holdout or full combined dataset",
    candidate_role="combined_candidate",
    holdout_description="Previously analyzed protected March holdout; the caller removes its duplicate matches and whole groups from pooled January/March train; both models evaluated on exactly the same protected rows.",
    limitations=(
        "March dates come from the source description, not independently verified per-listing timestamps.",
        "The caller pools and deduplicates January and March sources, excluding all protected March holdout groups and their matches from training.",
        "The former January holdout is included in combined training and no longer validates this model.",
        "The protected March holdout was already analyzed and informed the decision to pool data; this is diagnostic reuse, not fresh independent validation.",
        "The same eight features, model families and search budget are retained; no new log-target, loss or geographic features are introduced. Pooling, deduplication and sample composition change, and CV may select a different family or parameters, so an improvement cannot be attributed to pooling alone.",
    ),
    holdout_previously_analyzed=True,
)


def _validate_partition(
    frame: pd.DataFrame, train_indices, test_indices, groups: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Все строки принадлежат ровно одной части; группы не пересекаются."""
    if not frame.columns.is_unique or not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("Нужны уникальные колонки и RangeIndex от 0 до n-1.")
    if not isinstance(groups, pd.Series) or not groups.index.equals(frame.index):
        raise ValueError("Группы должны иметь тот же индекс и порядок, что входной CSV.")
    if groups.isna().any() or not groups.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Каждая группа должна быть непустой строкой.")
    indices = []
    for name, values in (("train", train_indices), ("test", test_indices)):
        ids = np.asarray(values)
        if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu"
                or len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= len(frame)).any()):
            raise ValueError(f"Индексы {name} должны быть уникальными целыми в пределах CSV.")
        indices.append(ids.astype(np.int64))
    train_ids, test_ids = indices
    if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(frame.index):
        raise ValueError("Train/test должны быть непересекающимся полным разбиением входного CSV.")
    if set(groups.loc[train_ids]) & set(groups.loc[test_ids]):
        raise ValueError("Одна и та же группа обнаружена в train и test.")
    if len(test_ids) < 2 or len(train_ids) < 2 * CV_FOLDS or groups.loc[train_ids].nunique() < CV_FOLDS:
        raise ValueError("Недостаточно строк test или групп train для GroupKFold и R².")
    return train_ids, test_ids


def _feature_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = set((*FEATURE_COLUMNS, TARGET_COLUMN)) - set(frame.columns)
    if missing:
        raise ValueError(f"Во входном CSV отсутствуют поля: {sorted(missing)}")
    x = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in NUMERIC_FEATURES:
        x[column] = pd.to_numeric(x[column], errors="raise").astype(float)
        if np.isinf(x[column]).any():
            raise ValueError(f"Бесконечное значение в {column}.")
    if x.area.isna().any() or x.area.le(0).any():
        raise ValueError("Площадь должна быть заполнена и положительна.")
    for column in CATEGORICAL_FEATURES:
        x[column] = x[column].astype(object).where(x[column].notna(), np.nan)
        if not x[column].dropna().map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Категория {column} должна быть непустой строкой или пропуском.")
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(float)
    if not np.isfinite(y).all() or y.le(0).any():
        raise ValueError("Цена должна быть положительным конечным числом.")
    return x, y


def _checked_folds(x: pd.DataFrame, y: pd.Series, groups: pd.Series):
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(x, y, groups))
    details = []
    for number, (fit_ids, valid_ids) in enumerate(folds, start=1):
        fit_groups, valid_groups = set(groups.iloc[fit_ids]), set(groups.iloc[valid_ids])
        if fit_groups & valid_groups or len(valid_ids) < 2:
            raise ValueError("CV должен разделять группы и иметь хотя бы две строки валидации для R².")
        details.append({"fold": number, "fit_rows": len(fit_ids), "validation_rows": len(valid_ids),
                        "fit_groups": len(fit_groups), "validation_groups": len(valid_groups)})
    return folds, details


def _prediction_frame(frame: pd.DataFrame, x: pd.DataFrame, y: pd.Series, predicted) -> pd.DataFrame:
    # XGBoost возвращает float32. CSV должен сохранить точное значение этого числа,
    # а не его сокращённое float32-представление, которое иначе меняется при read_csv.
    predicted = np.asarray(predicted, dtype=float)
    result = pd.DataFrame({"row_index": x.index, "actual_price": y.to_numpy(),
                           "predicted_price": predicted,
                           "absolute_error": np.abs(y.to_numpy() - predicted)})
    if "source_row" in frame:
        result["source_row"] = frame.loc[x.index, "source_row"].to_numpy()
    return result


def train_grouped(
    frame: pd.DataFrame, train_indices, test_indices, groups: pd.Series,
    output_models_dir: Path, output_reports_dir: Path, input_path: Path,
    old_pipeline: Pipeline, old_metadata: dict[str, Any], *, context: TrainingContext,
) -> dict[str, Any]:
    """Обучить заданный train; выбор только по CV, происхождение явно в context.

    Caller готовит источники, дедупликацию и защищённый holdout. Эта функция
    проверяет полный group-disjoint split, но не устанавливает происхождение
    строк независимо от caller. Старый pipeline только предсказывает.
    """
    train_ids, test_ids = _validate_partition(frame, train_indices, test_indices, groups)
    x, y = _feature_target(frame)
    validate_metadata(old_metadata)
    if (not isinstance(old_pipeline, Pipeline)
            or list(getattr(old_pipeline, "feature_names_in_", [])) != list(FEATURE_COLUMNS)):
        raise ValueError("Старая модель должна быть обученным Pipeline с той же схемой признаков.")
    input_path, output_models_dir, output_reports_dir = map(Path, (input_path, output_models_dir, output_reports_dir))
    input_hash = file_sha256(input_path)
    try:
        pd.testing.assert_frame_equal(frame, pd.read_csv(input_path), check_dtype=False,
                                      check_index_type=False, check_column_type=False,
                                      check_exact=False, rtol=1e-12, atol=1e-6)
    except AssertionError as error:
        raise ValueError("Переданный frame не совпадает с input_path; перечитайте сохранённый CSV.") from error
    outputs = [output_models_dir / name for name in ("best_model.joblib", "model_metadata.json")]
    outputs += [output_reports_dir / name for name in (
        "training_report.json", "model_comparison.csv", "feature_importance.csv", "split_manifest.csv",
        "test_predictions.csv", "old_model_test_predictions.csv", "training_environment.txt",
    )]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Для запуска нужны отдельные каталоги без прежних артефактов.")
    x_train, x_test = x.loc[train_ids], x.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]
    train_groups = groups.loc[train_ids]
    folds, fold_details = _checked_folds(x_train, y_train, train_groups)
    config = TrainingConfig(input_path=input_path, models_dir=output_models_dir, reports_dir=output_reports_dir,
                            random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
                            search_iterations=SEARCH_ITERATIONS, jobs=MODEL_JOBS)
    print(f"{context.data_period}: train={len(x_train)}, test={len(x_test)}, train groups={train_groups.nunique()}.", flush=True)
    with threadpool_limits(limits=MODEL_JOBS):
        models = make_models(config)
        rows = []
        for name, pipeline in models.items():
            print(f"GroupKFold: {name}, {CV_FOLDS} folds", flush=True)
            scores = cross_validate(pipeline, x_train, y_train, cv=folds, scoring=SCORING,
                                    n_jobs=1, error_score="raise", verbose=2)
            rows.append({"model": name, "role": context.candidate_role, **cv_summary(scores)})
            print(f"{name}: CV MAE={rows[-1]['cv_mae_tenge']:,.0f} KZT", flush=True)
        best_base = min(rows, key=lambda row: row["cv_mae_tenge"])
        family = best_base["model"]
        space = search_space(family)
        candidates = min(SEARCH_ITERATIONS, int(np.prod([len(values) for values in space.values()])))
        print(f"Search {family}: {candidates} candidates x {CV_FOLDS} GroupKFold", flush=True)
        search = RandomizedSearchCV(
            models[family], param_distributions=space, n_iter=candidates, scoring=SCORING,
            refit="mae", cv=folds, random_state=RANDOM_STATE, n_jobs=1,
            error_score="raise", return_train_score=False, verbose=2,
        ).fit(x_train, y_train)
        tuning_rows = []
        for index, parameters in enumerate(search.cv_results_["params"]):
            tuning_rows.append({
                "parameters": parameters,
                "cv_mae_tenge": float(-search.cv_results_["mean_test_mae"][index]),
                "cv_mae_std_tenge": float(search.cv_results_["std_test_mae"][index]),
                "cv_rmse_tenge": float(-search.cv_results_["mean_test_rmse"][index]),
                "cv_rmse_std_tenge": float(search.cv_results_["std_test_rmse"][index]),
                "cv_r2": float(search.cv_results_["mean_test_r2"][index]),
            })
        tuned_name = f"{family} (tuned)"
        rows.append({"model": tuned_name, "role": context.candidate_role,
                     **{key: value for key, value in tuning_rows[search.best_index_].items() if key != "parameters"}})
        selected = tuned_name if -search.best_score_ < best_base["cv_mae_tenge"] else family
        models[tuned_name] = search.best_estimator_
        print(f"Selected by train CV: {selected}. Computing train-only importance before test.", flush=True)
        importance = cv_feature_importance(models[selected], x_train, y_train, folds, RANDOM_STATE)

        # Победитель уже зафиксирован. Ни одна test-метрика не участвует в выборе.
        predictions = {}
        for row in rows:
            name = row["model"]
            if name != tuned_name:
                models[name].fit(x_train, y_train)
            predicted = models[name].predict(x_test)
            predictions[name] = predicted
            row.update({f"test_{key}": value for key, value in compute_metrics(y_test, predicted).items()})
            row["selected"] = name == selected
            row["negative_test_predictions"] = int((predicted < 0).sum())
        old_predictions = old_pipeline.predict(x_test)
        old_metrics = compute_metrics(y_test, old_predictions)
        selected_metrics = compute_metrics(y_test, predictions[selected])
        median_baseline = compute_metrics(y_test, np.full(len(y_test), y_train.median()))
    old_name = f"January baseline: {old_metadata.get('model_name', 'saved model')}"
    old_row = {"model": old_name, "role": "january_frozen_reference", "selected": False,
               **{key: None for key in ("cv_mae_tenge", "cv_mae_std_tenge", "cv_rmse_tenge", "cv_rmse_std_tenge", "cv_r2")},
               **{f"test_{key}": value for key, value in old_metrics.items()},
               "negative_test_predictions": int((old_predictions < 0).sum())}
    comparison = pd.DataFrame([*rows, old_row])
    if file_sha256(input_path) != input_hash:
        raise ValueError("Входной CSV изменился во время обучения; артефакты не сохранены.")
    output_models_dir.mkdir(parents=True, exist_ok=True)
    output_reports_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = output_models_dir / "best_model.joblib"
    joblib.dump(models[selected], pipeline_path, compress=3)
    restored = joblib.load(pipeline_path)
    np.testing.assert_allclose(restored.predict(x_test), predictions[selected], rtol=1e-7, atol=.01)
    manifest = pd.DataFrame({"row_index": frame.index, "subset": "test", "cv_validation_fold": 0,
                             "group": groups.to_numpy()})
    manifest.loc[train_ids, "subset"] = "train"
    for number, (_, valid_ids) in enumerate(folds, start=1):
        manifest.loc[x_train.index[valid_ids], "cv_validation_fold"] = number
    if "source_row" in frame:
        manifest["source_row"] = frame.source_row
    manifest_path = output_reports_dir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    versions = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib")}
    selected_row = next(row for row in rows if row["selected"])
    metadata = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": selected, "target": "asking_price", "currency": "KZT",
        "feature_columns": list(FEATURE_COLUMNS), "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categories_from_train": {column: sorted(x_train[column].dropna().unique().tolist()) for column in CATEGORICAL_FEATURES},
        "numeric_ranges_from_train": {column: {"min": float(x_train[column].min()), "max": float(x_train[column].max())}
                                      for column in NUMERIC_FEATURES if x_train[column].notna().any()},
        "input_sha256": input_hash, "model_sha256": file_sha256(pipeline_path),
        "split_manifest_sha256": file_sha256(manifest_path), "split_strategy": "predefined_group_holdout",
        "random_state": RANDOM_STATE, "test_size": len(x_test) / len(frame),
        "train_rows": len(x_train), "test_rows": len(x_test), "cv_folds": CV_FOLDS,
        "cv_strategy": "GroupKFold", "train_group_count": int(train_groups.nunique()),
        "test_group_count": int(groups.loc[test_ids].nunique()), "cv_fold_details": fold_details,
        "selection_criterion": "minimum mean GroupKFold CV MAE on supplied train; test is diagnostic only",
        "selected_cv_mae_tenge": selected_row["cv_mae_tenge"], "test_metrics": selected_metrics,
        "approximate_range": {"half_width_tenge": selected_metrics["mae_tenge"], "source": "holdout_test_mae",
                              "interpretation": "prediction +/- MAE; descriptive, not a confidence or prediction interval",
                              "guaranteed_coverage": None},
        "data_period": context.data_period, "data_period_label": context.data_period_label,
        "training_scope": context.training_scope, "candidate_role": context.candidate_role,
        "holdout_description": context.holdout_description,
        "holdout_previously_analyzed": context.holdout_previously_analyzed,
        "python_version": platform.python_version(), "package_versions": versions,
    }
    validate_metadata(metadata)
    metadata_path = output_models_dir / "model_metadata.json"
    write_json(metadata_path, metadata)
    loaded, _ = load_artifacts(pipeline_path, metadata_path)
    np.testing.assert_allclose(loaded.predict(x_test), predictions[selected], rtol=1e-7, atol=.01)
    comparison.to_csv(output_reports_dir / "model_comparison.csv", index=False)
    importance.to_csv(output_reports_dir / "feature_importance.csv", index=False)
    _prediction_frame(frame, x_test, y_test, predictions[selected]).to_csv(output_reports_dir / "test_predictions.csv", index=False)
    _prediction_frame(frame, x_test, y_test, old_predictions).to_csv(output_reports_dir / "old_model_test_predictions.csv", index=False)
    report = {
        **metadata, "input_file": str(input_path), "comparison": [*rows, old_row],
        "tuned_family": family, "search_candidates": tuning_rows,
        "best_search_parameters": search.best_params_, "tuning_improved_cv": selected == tuned_name,
        "constant_train_median_test_metrics": median_baseline, "old_model_test_metrics": old_metrics,
        "old_model": {"model_name": old_metadata.get("model_name"), "model_sha256": old_metadata["model_sha256"],
                      "training_input_sha256": old_metadata.get("input_sha256"), "refitted": False,
                      "cv_on_training_data": None,
                      **({"cv_on_march_train": None} if context.candidate_role == "march_candidate" else {}),
                      "comparison_scope": context.holdout_description},
        "test_comparison": {"mae_change_tenge": selected_metrics["mae_tenge"] - old_metrics["mae_tenge"],
                            "mae_change_pct": (selected_metrics["mae_tenge"] / old_metrics["mae_tenge"] - 1) * 100
                            if old_metrics["mae_tenge"] else None,
                            "activation_decision": "not made by training runner"},
        "price_summary_by_subset": {name: {str(q): float(values.quantile(q)) for q in (0, .5, .95, .99, 1)}
                                    for name, values in (("train", y_train), ("test", y_test))},
        "feature_importance": importance.to_dict(orient="records"),
        "importance_method": "GroupKFold validation within supplied train, 5 permutations per fold; fixed selected parameters",
        "importance_caveat": "Descriptive, not nested CV; correlated features may share importance; std is not a confidence interval",
        "limitations": [
            *context.limitations,
            "Group disjointness is verified, but heuristic grouping does not prove uniqueness of physical apartments.",
            "CV used for selection is optimistic; holdout metrics are diagnostic for the fixed selected pipeline.",
            "Both models are evaluated on the same March holdout; this is not a six-month forecast test.",
            "Prediction +/- test MAE is descriptive, without guaranteed coverage or independent calibration.",
        ],
    }
    write_json(output_reports_dir / "training_report.json", report)
    environment = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in distributions())
    (output_reports_dir / "training_environment.txt").write_text(
        f"# Python {platform.python_version()}\n" + "\n".join(environment) + "\n", encoding="utf-8")
    # CV отсутствует у замороженной январской модели: график сравнивает только новых кандидатов.
    save_figures(pd.DataFrame(rows), y_test, predictions[selected], importance, selected, output_reports_dir / "figures")
    print(comparison.to_string(index=False), flush=True)
    print(f"Saved staged {context.data_period} artifacts: {output_models_dir}", flush=True)
    return report


def train_march(
    frame: pd.DataFrame, train_indices, test_indices, groups: pd.Series,
    output_models_dir: Path, output_reports_dir: Path, input_path: Path,
    old_pipeline: Pipeline, old_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Совместимый March wrapper: train содержит только мартовские строки."""
    return train_grouped(frame, train_indices, test_indices, groups, output_models_dir,
                         output_reports_dir, input_path, old_pipeline, old_metadata,
                         context=MARCH_CONTEXT)


def train_combined(
    frame: pd.DataFrame, train_indices, test_indices, groups: pd.Series,
    output_models_dir: Path, output_reports_dir: Path, input_path: Path,
    old_pipeline: Pipeline, old_metadata: dict[str, Any],
) -> dict[str, Any]:
    """January+March train, включая прежний January test; March holdout защищён.

    Каталоги models/combined_2026 и reports/combined_update задаёт caller.
    Повторно анализируемый March holdout не считается новой независимой проверкой.
    """
    return train_grouped(frame, train_indices, test_indices, groups, output_models_dir,
                         output_reports_dir, input_path, old_pipeline, old_metadata,
                         context=COMBINED_CONTEXT)

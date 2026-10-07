"""Сравнение трёх моделей, подбор по train-CV и однократная оценка test."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from config import (
    CLEAN_LISTINGS_PATH, CLEANING_REPORT_PATH, CV_FOLDS, LOGS_DIR,
    MODELS_DIR, PROJECT_ROOT, RANDOM_STATE, REPORTS_DIR, SEARCH_ITERATIONS, TEST_SIZE,
)
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN, build_pipeline


LOGGER = logging.getLogger(__name__)
SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
FEATURE_LABELS = {
    "area": "Площадь", "rooms": "Комнаты", "floor": "Этаж",
    "total_floors": "Этажность", "year_built": "Год постройки",
    "district": "Район", "building_type": "Тип дома", "condition": "Состояние",
}


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры одного воспроизводимого запуска."""

    input_path: Path = CLEAN_LISTINGS_PATH
    models_dir: Path = MODELS_DIR
    reports_dir: Path = REPORTS_DIR
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    search_iterations: int = SEARCH_ITERATIONS
    jobs: int = 2


def file_sha256(path: Path) -> str:
    """Контрольная сумма связывает отчёт с конкретным файлом."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """MAE и RMSE возвращаются в тенге, R² безразмерен."""
    return {
        "mae_tenge": float(mean_absolute_error(y_true, y_pred)),
        "rmse_tenge": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def split_data(
    frame: pd.DataFrame, random_state: int = RANDOM_STATE, test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Выбрать разрешённые колонки и разделить без обучения преобразований."""
    missing = set((*FEATURE_COLUMNS, TARGET_COLUMN)) - set(frame.columns)
    if missing or not frame.columns.is_unique or not frame.index.is_unique:
        raise ValueError(f"Нужны уникальные строки/колонки и все поля; отсутствуют: {sorted(missing)}")
    x = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in NUMERIC_FEATURES:
        x[column] = pd.to_numeric(x[column], errors="raise").astype(float)
        if np.isinf(x[column]).any():
            raise ValueError(f"Бесконечное значение в {column}; повторите очистку.")
    if x["area"].isna().any() or (x["area"] <= 0).any():
        raise ValueError("Площадь должна быть заполнена и положительна.")
    for column in CATEGORICAL_FEATURES:
        x[column] = x[column].astype(object).where(x[column].notna(), np.nan)
        if not x[column].dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Категория {column} должна быть строкой или пропуском.")
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Цена должна быть положительным конечным числом без пропусков.")
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def make_models(config: TrainingConfig) -> dict[str, Pipeline]:
    """Одинаковые входные данные и предобработка для трёх алгоритмов."""
    return {
        "LinearRegression": build_pipeline(LinearRegression()),
        "RandomForest": build_pipeline(RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2,
            random_state=config.random_state, n_jobs=config.jobs,
        )),
        "XGBoost": build_pipeline(XGBRegressor(
            objective="reg:squarederror", n_estimators=400, max_depth=4,
            learning_rate=0.05, subsample=0.85, colsample_bytree=0.9,
            tree_method="hist", random_state=config.random_state, n_jobs=config.jobs,
        )),
    }


def search_space(model_name: str) -> dict[str, list[Any]]:
    """Небольшие диапазоны: бюджет поиска ограничен числом кандидатов."""
    spaces = {
        "LinearRegression": {"model__fit_intercept": [True, False]},
        "RandomForest": {
            "model__n_estimators": [200, 350], "model__max_depth": [12, 24, None],
            "model__min_samples_leaf": [1, 2, 4], "model__max_features": [0.8, 1.0],
        },
        "XGBoost": {
            "model__n_estimators": [300, 600], "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.07], "model__min_child_weight": [1, 5],
            "model__reg_lambda": [1.0, 10.0],
        },
    }
    return spaces[model_name]


def cv_summary(scores: dict[str, Any]) -> dict[str, float]:
    """sklearn возвращает ошибки со знаком минус; в отчёте они положительные."""
    return {
        "cv_mae_tenge": float(-np.mean(scores["test_mae"])),
        "cv_mae_std_tenge": float(np.std(scores["test_mae"])),
        "cv_rmse_tenge": float(-np.mean(scores["test_rmse"])),
        "cv_rmse_std_tenge": float(np.std(scores["test_rmse"])),
        "cv_r2": float(np.mean(scores["test_r2"])),
    }


def cv_feature_importance(
    pipeline: Pipeline, x: pd.DataFrame, y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]], random_state: int,
) -> pd.DataFrame:
    """Перемешать исходные признаки на валидации внутри train, по 5 повторов."""
    values = []
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, start=1):
        LOGGER.info("Важность признаков: fold %s/%s", fold_number, len(folds))
        fitted = clone(pipeline).fit(x.iloc[fit_indices], y.iloc[fit_indices])
        result = permutation_importance(
            fitted, x.iloc[valid_indices], y.iloc[valid_indices],
            scoring="neg_mean_absolute_error", n_repeats=5,
            random_state=random_state + fold_number, n_jobs=1,
        )
        values.append(result.importances)
    joined = np.concatenate(values, axis=1)
    return pd.DataFrame({
        "feature": list(FEATURE_COLUMNS), "importance_mean_tenge": joined.mean(axis=1),
        "importance_std_tenge": joined.std(axis=1),
    }).sort_values("importance_mean_tenge", ascending=False, ignore_index=True)


def save_figures(
    comparison: pd.DataFrame, actual: pd.Series, predicted: np.ndarray,
    importance: pd.DataFrame, selected: str, figures_dir: Path,
) -> None:
    """Сохранить графики, не отбрасывая дорогие квартиры и большие ошибки."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax in axes:
        ax.scatter(actual / 1e6, predicted / 1e6, s=12, alpha=0.3, color="#087f8c")
        low = min(0.0, float(predicted.min()) / 1e6)
        high = max(float(actual.max()), float(predicted.max())) / 1e6 * 1.03
        ax.plot([low, high], [low, high], "--", color="#e97932", label="Идеальный прогноз")
        ax.set(xlabel="Реальная цена, млн тенге", ylabel="Прогноз, млн тенге")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_title("Полный диапазон цен")
    # symlog также показывает возможные отрицательные прогнозы линейной модели.
    axes[1].set(xscale="symlog", yscale="symlog", title="Симметричная логарифмическая шкала")
    fig.suptitle(f"{selected}: отложенная выборка, n = {len(actual):,}".replace(",", " "))
    fig.savefig(figures_dir / "training_actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    positions = np.arange(len(comparison))
    ax.barh(positions - 0.18, comparison["cv_mae_tenge"] / 1e6, height=0.34,
            color="#087f8c", label="CV train (критерий выбора)")
    ax.barh(positions + 0.18, comparison["test_mae_tenge"] / 1e6, height=0.34,
            color="#e97932", label="Test (итоговая оценка)")
    labels = [f"{row.model}{' *' if row.selected else ''}" for row in comparison.itertuples()]
    ax.set(yticks=positions, yticklabels=labels, xlabel="MAE, млн тенге",
           title="Сравнение моделей · * сохранённая модель")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(figures_dir / "training_model_comparison.png", dpi=180)
    plt.close(fig)

    ordered = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.barh(ordered["feature"].map(FEATURE_LABELS), ordered["importance_mean_tenge"] / 1e6,
            xerr=ordered["importance_std_tenge"] / 1e6, color="#087f8c", capsize=3)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set(xlabel="Увеличение MAE при перемешивании, млн тенге",
           title="Важность признаков на CV-валидации внутри train")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(figures_dir / "training_feature_importance.png", dpi=180)
    plt.close(fig)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Записать читаемый UTF-8 JSON без NaN и Infinity."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Выполнить этап 5; не использовать test для выбора модели и параметров."""
    if config.jobs < 1 or config.cv_folds < 2 or config.search_iterations < 1:
        raise ValueError("jobs >= 1, cv-folds >= 2, search-iterations >= 1 обязательны.")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size должен быть между 0 и 1.")
    input_hash = file_sha256(config.input_path)
    if config.input_path.resolve() == CLEAN_LISTINGS_PATH.resolve():
        cleaning = json.loads(CLEANING_REPORT_PATH.read_text(encoding="utf-8"))
        if input_hash != cleaning["output_sha256"]:
            raise ValueError("CSV не совпадает с отчётом очистки; повторите этап 3.")
    frame = pd.read_csv(config.input_path)
    x_train, x_test, y_train, y_test = split_data(frame, config.random_state, config.test_size)
    if len(x_test) < 2 or len(x_train) < 2 * config.cv_folds:
        raise ValueError("Недостаточно строк для test и CV с вычислением R².")
    LOGGER.info("Строк: %s; train: %s; test: %s", len(frame), len(x_train), len(x_test))
    folds = list(KFold(n_splits=config.cv_folds, shuffle=True,
                       random_state=config.random_state).split(x_train))
    models = make_models(config)
    rows: list[dict[str, Any]] = []
    for name, pipeline in models.items():
        LOGGER.info("CV: %s, %s folds", name, config.cv_folds)
        scores = cross_validate(pipeline, x_train, y_train, cv=folds, scoring=SCORING,
                                n_jobs=1, error_score="raise")
        row = {"model": name, **cv_summary(scores)}
        rows.append(row)
        LOGGER.info("%s: CV MAE = %.0f тенге", name, row["cv_mae_tenge"])

    best_base = min(rows, key=lambda row: row["cv_mae_tenge"])
    family = best_base["model"]
    space = search_space(family)
    candidates = min(config.search_iterations, int(np.prod([len(v) for v in space.values()])))
    LOGGER.info("Подбор %s: %s кандидатов × %s folds", family, candidates, config.cv_folds)
    search = RandomizedSearchCV(
        models[family], param_distributions=space, n_iter=candidates, scoring=SCORING,
        refit="mae", cv=folds, random_state=config.random_state, n_jobs=1,
        error_score="raise", return_train_score=False,
    ).fit(x_train, y_train)
    best_index = search.best_index_
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
    rows.append({"model": tuned_name, **{k: v for k, v in tuning_rows[best_index].items()
                                        if k != "parameters"}})
    selected = tuned_name if -search.best_score_ < best_base["cv_mae_tenge"] else family
    models[tuned_name] = search.best_estimator_
    LOGGER.info("Модель зафиксирована по CV: %s. Test ещё не использован.", selected)
    importance = cv_feature_importance(models[selected], x_train, y_train, folds, config.random_state)

    # После этой точки test используется только для итоговой диагностики.
    predictions: dict[str, np.ndarray] = {}
    for row in rows:
        name = row["model"]
        if name != tuned_name:
            models[name].fit(x_train, y_train)
        predicted = models[name].predict(x_test)
        predictions[name] = predicted
        row.update({f"test_{key}": value for key, value in compute_metrics(y_test, predicted).items()})
        row["selected"] = name == selected
        row["negative_test_predictions"] = int((predicted < 0).sum())
    comparison = pd.DataFrame(rows)
    selected_metrics = compute_metrics(y_test, predictions[selected])
    # Ориентир: константный прогноз медианы обучающих цен.
    median_baseline = compute_metrics(y_test, np.full(len(y_test), y_train.median()))
    pipeline = models[selected]
    if file_sha256(config.input_path) != input_hash:
        raise ValueError("Входной CSV изменился во время обучения; артефакты не сохранены.")

    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = config.models_dir / "best_model.joblib"
    joblib.dump(pipeline, pipeline_path, compress=3)
    restored = joblib.load(pipeline_path)
    np.testing.assert_allclose(restored.predict(x_test), predictions[selected], rtol=1e-7, atol=0.01)

    versions = {name: version(name) for name in
                ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib")}
    limits = {column: {"min": float(x_train[column].min()), "max": float(x_train[column].max())}
              for column in NUMERIC_FEATURES if x_train[column].notna().any()}
    categories = {column: sorted(x_train[column].dropna().unique().tolist())
                  for column in CATEGORICAL_FEATURES}
    selected_row = next(row for row in rows if row["selected"])
    metadata = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": selected, "target": "asking_price", "currency": "KZT",
        "feature_columns": list(FEATURE_COLUMNS), "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categories_from_train": categories, "numeric_ranges_from_train": limits,
        "input_sha256": input_hash, "model_sha256": file_sha256(pipeline_path),
        "random_state": config.random_state, "test_size": config.test_size,
        "train_rows": len(x_train), "test_rows": len(x_test), "cv_folds": config.cv_folds,
        "selection_criterion": "minimum mean CV MAE on train; test is diagnostic only",
        "selected_cv_mae_tenge": selected_row["cv_mae_tenge"],
        "test_metrics": selected_metrics,
        "approximate_range": {
            "half_width_tenge": selected_metrics["mae_tenge"], "source": "holdout_test_mae",
            "interpretation": "prediction +/- MAE; descriptive, not a confidence or prediction interval",
            "guaranteed_coverage": None,
        },
        "training_scope": "train only; not refitted on test or full dataset",
        "python_version": platform.python_version(), "package_versions": versions,
    }
    write_json(config.models_dir / "model_metadata.json", metadata)
    comparison.to_csv(config.reports_dir / "model_comparison.csv", index=False)
    importance.to_csv(config.reports_dir / "feature_importance.csv", index=False)
    manifest = pd.DataFrame({"row_index": frame.index, "subset": "test", "cv_validation_fold": 0})
    manifest.loc[x_train.index, "subset"] = "train"
    for number, (_, valid_indices) in enumerate(folds, start=1):
        manifest.loc[x_train.index[valid_indices], "cv_validation_fold"] = number
    if "source_row" in frame:
        manifest["source_row"] = frame["source_row"]
    manifest.to_csv(config.reports_dir / "split_manifest.csv", index=False)
    prediction_frame = pd.DataFrame({
        "row_index": x_test.index, "actual_price": y_test.to_numpy(),
        "predicted_price": predictions[selected],
        "absolute_error": np.abs(y_test.to_numpy() - predictions[selected]),
    })
    if "source_row" in frame:
        prediction_frame["source_row"] = frame.loc[x_test.index, "source_row"].to_numpy()
    prediction_frame.to_csv(config.reports_dir / "test_predictions.csv", index=False)
    report = {
        **metadata, "input_file": str(config.input_path),
        "comparison": rows, "tuned_family": family, "search_candidates": tuning_rows,
        "best_search_parameters": search.best_params_, "tuning_improved_cv": selected == tuned_name,
        "constant_train_median_test_metrics": median_baseline,
        "price_summary_by_subset": {
            name: {str(q): float(values.quantile(q)) for q in (0, 0.5, 0.95, 0.99, 1)}
            for name, values in (("train", y_train), ("test", y_test))
        },
        "feature_importance": importance.to_dict(orient="records"),
        "importance_method": "CV validation within train, 5 permutations per fold; fixed selected parameters",
        "importance_caveat": "Descriptive, not nested CV; correlated features may share importance; std is not a confidence interval",
        "limitations": [
            "Random split of one snapshot does not measure accuracy on future market data.",
            "Similar flats or repeated offers may remain after heuristic deduplication.",
            "CV scores used for selection are optimistic; use holdout metrics for the final assessment.",
            "MAE is an average and can hide large errors, especially for expensive properties.",
        ],
    }
    write_json(config.reports_dir / "training_report.json", report)
    environment = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in distributions())
    (config.reports_dir / "training_environment.txt").write_text(
        f"# Python {platform.python_version()}\n" + "\n".join(environment) + "\n", encoding="utf-8",
    )
    save_figures(comparison, y_test, predictions[selected], importance, selected,
                 config.reports_dir / "figures")
    print("\nСравнение моделей (все ошибки в тенге):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print("\nВажность признаков: рост MAE при перемешивании (тенге):")
    print(importance.to_string(index=False, float_format=lambda value: f"{value:,.0f}"))
    print(f"\nСохранена модель: {selected}\nПайплайн: {pipeline_path}")
    return report


def resolve_path(value: str) -> Path:
    """Относительные пути отсчитываются от корня проекта."""
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def main() -> int:
    """Командная строка для повторяемого обучения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=resolve_path, default=CLEAN_LISTINGS_PATH)
    parser.add_argument("--models-dir", type=resolve_path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=resolve_path, default=REPORTS_DIR)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--search-iterations", type=int, default=SEARCH_ITERATIONS)
    parser.add_argument("--jobs", type=int, default=2, help="Потоки модели; CV выполняется последовательно.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "train.log", encoding="utf-8")])
    config = TrainingConfig(
        input_path=args.input, models_dir=args.models_dir, reports_dir=args.reports_dir,
        cv_folds=args.cv_folds, search_iterations=args.search_iterations,
        jobs=args.jobs, random_state=args.random_state,
    )
    try:
        with threadpool_limits(limits=max(1, config.jobs)):
            run_training(config)
    except (ValueError, OSError, KeyError) as error:
        LOGGER.error("Обучение остановлено: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

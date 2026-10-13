"""Объединённое обучение January + March с защищённым мартовским test."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from clean import atomic_write
from config import PROJECT_ROOT
from error_analysis import metric_rows, summarize
from features import FEATURE_COLUMNS
from prediction import load_artifacts
from train import compute_metrics
from update_march import digest, save_json


DATA = PROJECT_ROOT / "data/processed/combined_2026"
MODELS = PROJECT_ROOT / "models/combined_2026"
REPORTS = PROJECT_ROOT / "reports/combined_update"
JANUARY = PROJECT_ROOT / "data/processed/january_2026/listings_clean.csv"
MARCH = PROJECT_ROOT / "data/processed/march_2026/listings_clean.csv"
MARCH_REPORTS = PROJECT_ROOT / "reports/march_update"


def protected_hashes() -> dict[str, str]:
    paths = ["models/best_model.joblib", "models/model_metadata.json",
             "data/processed/listings_clean.csv", "data/processed/listings_rejected.csv",
             "models/archive/january_2026/best_model.joblib",
             "models/archive/january_2026/model_metadata.json",
             "data/processed/january_2026/listings_clean.csv",
             "models/march_2026/best_model.joblib", "models/march_2026/model_metadata.json",
             "data/processed/march_2026/listings_clean.csv", "reports/march_update/split_manifest.csv"]
    return {name: digest(PROJECT_ROOT / name) for name in paths}


def write_csv_once(frame: pd.DataFrame, path: Path) -> None:
    """Разрешён повтор с теми же байтами, но не перезапись другого эксперимента."""
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Сохранённый файл отличается от подготовленного: {path}")
        return
    atomic_write(path, payload)


def paired_mae_difference(actual, old, new, iterations: int = 3000) -> dict:
    """Парный bootstrap по уникальным эвристическим группам; new MAE - old MAE."""
    actual, old, new = (np.asarray(values, dtype=float) for values in (actual, old, new))
    if (actual.ndim != 1 or len(actual) < 2 or actual.shape != old.shape or old.shape != new.shape
            or not all(np.isfinite(values).all() for values in (actual, old, new)) or iterations < 100):
        raise ValueError("Нужны одинаковые конечные массивы и минимум 100 повторов.")
    differences = np.abs(new - actual) - np.abs(old - actual)
    rng = np.random.default_rng(42)
    samples = np.array([rng.choice(differences, size=len(differences), replace=True).mean()
                        for _ in range(iterations)])
    low, high = np.quantile(samples, [.025, .975])
    return {"new_minus_january_mae_tenge": float(differences.mean()),
            "ci95_low_tenge": float(low), "ci95_high_tenge": float(high),
            "iterations": iterations, "seed": 42,
            "method": "paired percentile bootstrap over one representative per heuristic apartment group",
            "limitations": "Descriptive interval on a previously inspected holdout; heuristic groups may still be dependent. Does not account for model-selection or source bias."}


def evaluate(frame: pd.DataFrame, models: dict) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows, predictions = [], {}
    for name, (model, metadata) in models.items():
        predicted = np.asarray(model.predict(frame.loc[:, list(FEATURE_COLUMNS)]), dtype=float)
        rows.append({"model": name, "algorithm": metadata["model_name"],
                     **summarize(metric_rows(frame.price, predicted, metadata["test_metrics"]["mae_tenge"])),
                     "r2": compute_metrics(frame.price, predicted)["r2"]})
        predictions[name] = predicted
    return rows, predictions


def segment_comparisons(frame: pd.DataFrame, predictions: dict) -> dict[str, list[dict]]:
    labels = {
        "district": frame.district.fillna("Не указан"),
        "completeness": frame.loc[:, list(FEATURE_COLUMNS)].notna().sum(axis=1).astype(str) + " из 8",
        "area": pd.cut(frame.area, [0, 40, 70, 120, np.inf],
                       labels=["До 40 м²", "Более 40–70 м²", "Более 70–120 м²", "Более 120 м²"]),
    }
    result = {}
    for dimension, label in labels.items():
        entries = []
        for name in label.dropna().unique():
            mask = np.asarray(label == name)
            actual = frame.price.to_numpy()[mask]
            old, new = predictions["january"][mask], predictions["combined"][mask]
            old_error, new_error = np.abs(old - actual), np.abs(new - actual)
            entries.append({"group": str(name), "n": int(mask.sum()), "small_group": int(mask.sum()) < 30,
                            "january_mae_tenge": float(old_error.mean()), "combined_mae_tenge": float(new_error.mean()),
                            "mae_change_tenge": float(new_error.mean() - old_error.mean()),
                            "january_over_20_pct": float((old_error > actual * .2).mean() * 100),
                            "combined_over_20_pct": float((new_error > actual * .2).mean() * 100)})
        result[dimension] = entries
    return result


def run_update(*, resume: bool = False, prepare_only: bool = False) -> dict:
    from combined_data import build_combined
    from combined_report import render_report
    from march_training import train_combined

    before = protected_hashes()
    old, old_meta = load_artifacts(PROJECT_ROOT / "models/best_model.joblib", PROJECT_ROOT / "models/model_metadata.json")
    archive_meta = json.loads((PROJECT_ROOT / "models/archive/january_2026/model_metadata.json").read_text(encoding="utf-8"))
    if old_meta["model_sha256"] != archive_meta["model_sha256"] or digest(JANUARY) != old_meta["input_sha256"]:
        raise ValueError("Текущая модель должна соответствовать сохранённой январской версии.")
    march_model, march_meta = load_artifacts(PROJECT_ROOT / "models/march_2026/best_model.joblib",
                                           PROJECT_ROOT / "models/march_2026/model_metadata.json")
    if digest(MARCH) != march_meta["input_sha256"]:
        raise ValueError("Мартовские данные не соответствуют сохранённой модели.")
    old_manifest_path = MARCH_REPORTS / "split_manifest.csv"
    if digest(old_manifest_path) != march_meta["split_manifest_sha256"]:
        raise ValueError("Исходный защищённый мартовский split изменился.")
    sources = {"january_csv_sha256": digest(JANUARY), "march_csv_sha256": digest(MARCH),
               "protected_march_manifest_sha256": digest(old_manifest_path)}
    bundle = build_combined(pd.read_csv(JANUARY), pd.read_csv(MARCH), pd.read_csv(old_manifest_path))
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for key, path in (("frame", DATA / "listings_clean.csv"), ("rejected", DATA / "listings_duplicates.csv"),
                      ("reference_holdout", DATA / "reference_holdout.csv"), ("audit", REPORTS / "deduplication_audit.csv"),
                      ("manifest", REPORTS / "prepared_split.csv")):
        write_csv_once(bundle[key], path)
    preparation = {**bundle["summary"], "sources": sources, "output_sha256": digest(DATA / "listings_clean.csv")}
    save_json(REPORTS / "preparation_report.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2), flush=True)
    if prepare_only:
        if before != protected_hashes():
            raise AssertionError("Изменились защищённые артефакты.")
        return preparation

    frame = pd.read_csv(DATA / "listings_clean.csv")
    split = pd.read_csv(REPORTS / "prepared_split.csv").set_index("row_index")
    train_ids = split.index[split.subset.eq("train")].to_numpy()
    test_ids = split.index[split.subset.eq("test")].to_numpy()
    if resume:
        _, metadata = load_artifacts(MODELS / "best_model.joblib", MODELS / "model_metadata.json")
        if metadata["input_sha256"] != digest(DATA / "listings_clean.csv"):
            raise ValueError("Сохранённая combined-модель обучена на других данных.")
        saved_split = pd.read_csv(REPORTS / "split_manifest.csv").set_index("row_index")
        pd.testing.assert_frame_equal(saved_split[["subset", "group"]], split[["subset", "group"]])
        if digest(REPORTS / "split_manifest.csv") != metadata["split_manifest_sha256"]:
            raise ValueError("Разбиение сохранённой combined-модели было изменено.")
        training = json.loads((REPORTS / "training_report.json").read_text(encoding="utf-8"))
        if training["old_model"]["model_sha256"] != old_meta["model_sha256"]:
            raise ValueError("Январская модель для сравнения изменилась.")
    else:
        training = train_combined(frame, train_ids, test_ids, split.group, MODELS, REPORTS,
                                  DATA / "listings_clean.csv", old, old_meta)
    candidate, candidate_meta = load_artifacts(MODELS / "best_model.joblib", MODELS / "model_metadata.json")
    models = {"january": (old, old_meta), "march": (march_model, march_meta), "combined": (candidate, candidate_meta)}
    test = frame.loc[test_ids].copy()
    comparison, predicted = evaluate(test, models)
    saved = pd.read_csv(REPORTS / "test_predictions.csv")
    np.testing.assert_array_equal(saved.row_index, test_ids)
    np.testing.assert_array_equal(saved.source_row, test.source_row)
    np.testing.assert_allclose(saved.actual_price, test.price, rtol=0, atol=.01)
    np.testing.assert_allclose(saved.predicted_price, predicted["combined"], rtol=0, atol=.01)
    np.testing.assert_allclose(comparison[-1]["mae_tenge"], candidate_meta["test_metrics"]["mae_tenge"], rtol=0, atol=.01)
    reference = pd.read_csv(DATA / "reference_holdout.csv")
    reference_comparison, reference_predictions = evaluate(reference, models)
    # Прежние 594 строки подтверждаются исходными raw-позициями и прогнозами baseline.
    prior = pd.read_csv(MARCH_REPORTS / "old_model_test_predictions.csv").set_index("source_row")
    source_rows = reference.original_source_row.to_numpy()
    if set(source_rows) != set(prior.index) or len(source_rows) != len(prior):
        raise AssertionError("Повторное сравнение не соответствует исходному мартовскому test.")
    np.testing.assert_allclose(prior.loc[source_rows, "actual_price"], reference.price, rtol=0, atol=.01)
    np.testing.assert_allclose(prior.loc[source_rows, "predicted_price"], reference_predictions["january"], rtol=0, atol=.01)
    for subset_name, rows, predictions in (("unique", test, predicted), ("reference", reference, reference_predictions)):
        exported = rows.copy()
        for name, values in predictions.items():
            exported[f"{name}_prediction"] = values
            exported[f"{name}_absolute_error"] = np.abs(rows.price.to_numpy() - values)
        exported.to_csv(REPORTS / f"{subset_name}_error_comparison.csv", index=False, encoding="utf-8-sig")
    segments = segment_comparisons(test, predicted)
    for name, entries in segments.items():
        pd.DataFrame(entries).to_csv(REPORTS / f"by_{name}.csv", index=False, encoding="utf-8-sig")
    uncertainty = paired_mae_difference(test.price, predicted["january"], predicted["combined"])
    after = protected_hashes()
    if before != after:
        raise AssertionError("Изменились защищённые январские или мартовские артефакты.")
    counts = {"input_rows": len(bundle["audit"]), "unique_rows": len(frame),
              "removed_rows": len(bundle["rejected"]), "train_rows": len(train_ids),
              "test_rows": len(test_ids), "reference_test_rows": len(reference),
              "train_sources": frame.loc[train_ids].dataset_period.value_counts().to_dict()}
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "activated": False, "counts": counts,
              "active_model": "january", "preparation": preparation, "selected_model": candidate_meta["model_name"],
              "common_holdout_comparison": comparison, "original_holdout_comparison": reference_comparison,
              "mae_improvement_pct": (1 - comparison[-1]["mae_tenge"] / comparison[0]["mae_tenge"]) * 100,
              "paired_mae_difference": uncertainty, "segments": segments,
              "model_sha256": candidate_meta["model_sha256"], "input_sha256": candidate_meta["input_sha256"],
              "protected_artifacts_before": before, "protected_artifacts_after": after,
              "selection_criterion": training["selection_criterion"],
              "limitations": ["March holdout has already been inspected; this is a diagnostic comparison, not a fresh independent test.",
                              "January's former test rows may enter combined training. January historical test is not an evaluation of this candidate.",
                              "Duplicate groups use URL and price-independent attribute signatures; false matches and missed repeats remain possible.",
                              "March period is publisher-declared; there are no row timestamps. Asking prices, not transactions or a six-month forecast."]}
    save_json(REPORTS / "update_report.json", report)
    render_report(report, REPORTS / "update_report.html")
    print(json.dumps({"comparison": comparison, "reference": reference_comparison,
                      "paired_mae_difference": uncertainty, "activated": False}, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Пересчитать сравнение проверенной сохранённой модели без обучения.")
    parser.add_argument("--prepare-only", action="store_true", help="Только подготовить объединённые данные и защищённое разбиение.")
    args = parser.parse_args()
    run_update(resume=args.resume, prepare_only=args.prepare_only)

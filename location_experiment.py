"""Контролируемый эксперимент ЖК/географии с неизменными эталонными моделями."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from clean import atomic_write
from config import PROJECT_ROOT, PROPERTY_CATALOG_DIR
from features import FEATURE_COLUMNS
from prediction import load_artifacts
from train import compute_metrics


DATA = PROJECT_ROOT / "data/processed/combined_2026/listings_clean.csv"
SPLIT = PROJECT_ROOT / "reports/combined_update/split_manifest.csv"
MODELS = PROJECT_ROOT / "models/location_2026"
REPORTS = PROJECT_ROOT / "reports/location_experiment"
VARIANT_LABELS = {"baseline8": "Контроль: 8 признаков", "complex": "+ ЖК",
                  "coords": "+ координаты", "complex_coords": "+ ЖК и координаты"}
PROTECTED = ["models/best_model.joblib", "models/model_metadata.json",
             "models/combined_2026/best_model.joblib", "models/combined_2026/model_metadata.json",
             "models/march_2026/best_model.joblib", "models/march_2026/model_metadata.json",
             "data/processed/combined_2026/listings_clean.csv",
             "reports/combined_update/split_manifest.csv",
             "data/raw/kaggle_astana/astana_apartments.csv",
             "data/raw/turarr_march_2026/astana_apartments_raw.csv"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: dict) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _text(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def apartment_groups(frame: pd.DataFrame) -> pd.Series:
    """Связать старые группы и полные адресные сигнатуры, не используя цену."""
    parent = list(range(len(frame)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(frame.to_dict("records")):
        keys = [("original", _text(row.get("group")))]
        house = _text(row.get("house_key"))
        numbers = [row.get(name) for name in ("area", "rooms", "floor", "total_floors", "year_built")]
        if house and all(value is not None and pd.notna(value) and np.isfinite(float(value)) for value in numbers):
            keys.append(("signature", (house, *map(float, numbers))))
        for kind, key in keys:
            if key is None:
                continue
            key = (kind, key)
            if key in owners:
                parent[root(i)] = root(owners[key])
            else:
                owners[key] = i
    members = {}
    record_ids = frame.record_id.astype(str).tolist()
    for i in range(len(frame)):
        members.setdefault(root(i), []).append(record_ids[i])
    names = {key: "apartment_" + hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:20]
             for key, ids in members.items()}
    return pd.Series([names[root(i)] for i in range(len(frame))], index=frame.index)


def prepare_partition(combined: pd.DataFrame, enriched: pd.DataFrame, split: pd.DataFrame) -> dict:
    """Проверить исходные строки и сохранить общий честный test для всех моделей."""
    if not combined.index.equals(pd.RangeIndex(len(combined))) or not combined.record_id.is_unique:
        raise ValueError("Combined должен иметь RangeIndex и уникальные record_id.")
    if not enriched.record_id.is_unique or set(enriched.record_id) != set(combined.record_id):
        raise ValueError("Каталог должен покрывать ровно исходные record_id.")
    enriched = enriched.set_index("record_id", drop=False).loc[combined.record_id].reset_index(drop=True)
    for column in combined.columns:
        if column not in enriched:
            raise ValueError(f"В каталоге утрачен исходный столбец {column}.")
        try:
            pd.testing.assert_series_equal(combined[column], enriched[column], check_dtype=False,
                                           check_names=False, rtol=1e-12, atol=1e-6)
        except AssertionError as error:
            raise ValueError(f"Каталог изменил исходные значения: {column}.") from error
    required = {"row_index", "subset", "group", "source_row"}
    if not required.issubset(split.columns) or not split.row_index.is_unique:
        raise ValueError("Некорректная схема split.")
    if set(split.row_index) != set(combined.index):
        raise ValueError("Split не покрывает combined ровно один раз.")
    split = split.set_index("row_index").loc[combined.index]
    if split.source_row.tolist() != combined.source_row.tolist():
        raise ValueError("Порядок source_row в split нарушен.")
    if "record_id" in split and split.record_id.tolist() != combined.record_id.tolist():
        raise ValueError("Порядок record_id в split нарушен.")
    if split.group.tolist() != combined.group.tolist() or split.subset.isna().any() or not set(split.subset).issubset({"train", "test"}):
        raise ValueError("Группы или метки исходного split изменены.")
    train_ids, test_ids = (split.index[split.subset.eq(label)].to_numpy() for label in ("train", "test"))
    if not len(train_ids) or len(test_ids) < 2 or set(split.loc[train_ids, "group"]) & set(split.loc[test_ids, "group"]):
        raise ValueError("Исходное разбиение должно разделять группы и иметь train/test.")
    groups = apartment_groups(enriched)
    seen = set(groups.loc[train_ids])
    used, evaluation_ids, exclusions = set(), [], []
    for i in sorted(test_ids, key=lambda index: str(combined.loc[index, "record_id"])):
        group = groups.loc[i]
        reason = "new_alias_matches_training" if group in seen else "new_alias_duplicate_in_test" if group in used else None
        if reason:
            exclusions.append({"row_index": int(i), "record_id": combined.loc[i, "record_id"], "reason": reason, "group": group})
        else:
            evaluation_ids.append(int(i))
            used.add(group)
    evaluation_ids = np.array(sorted(evaluation_ids), dtype=int)
    if len(evaluation_ids) < 2:
        raise ValueError("После проверки адресных совпадений нет достаточного общего test.")
    return {"frame": enriched, "train_ids": train_ids, "test_ids": test_ids,
            "evaluation_ids": evaluation_ids, "groups": groups, "exclusions": exclusions}


def house_validation_groups(frame: pd.DataFrame, equivalence: pd.Series) -> pd.Series:
    """Разделять целые дома и связанные группы квартир, включая неизвестные адреса."""
    if not frame.index.equals(equivalence.index):
        raise ValueError("Порядок групп должен совпадать с frame.")
    parent, owners = list(range(len(frame))), {}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, (house, apartment) in enumerate(zip(frame.house_key, equivalence)):
        for key in (("house", _text(house)), ("apartment", _text(apartment))):
            if key[1] is None:
                continue
            if key in owners:
                parent[root(i)] = root(owners[key])
            else:
                owners[key] = i
    components = {}
    ids = frame.record_id.astype(str).tolist()
    for i in range(len(frame)):
        components.setdefault(root(i), []).append(ids[i])
    names = {key: "housecheck_" + hashlib.sha256("|".join(sorted(values)).encode()).hexdigest()[:20]
             for key, values in components.items()}
    return pd.Series([names[root(i)] for i in range(len(frame))], index=frame.index)


def metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Нужны одинаковые конечные векторы прогноза и цены.")
    return {"n": len(actual), **compute_metrics(actual, predicted),
            "over_20_pct": float(np.mean(np.abs(predicted - actual) > actual * .2) * 100)}


def geo_mask(pipeline, x: pd.DataFrame) -> np.ndarray:
    transformed = pipeline.named_steps["location_features"].transform(x)
    return transformed.latitude.notna().to_numpy() if "latitude" in transformed else np.zeros(len(x), bool)


def segment_metrics(frame: pd.DataFrame, predictions: dict, geography: np.ndarray, known_houses: set, known_complexes: set) -> pd.DataFrame:
    from building_catalog import normalize_complex_name
    labels = {
        "district": frame.district.fillna("Не указан"),
        "house_seen": frame.house_key.map(lambda value: "Адрес дома не определён" if _text(value) is None else "Дом встречался в train" if _text(value) in known_houses else "Новый дом"),
        "complex_seen": frame.residential_complex.map(lambda value: "ЖК не указан" if normalize_complex_name(value) is None else "ЖК встречался в train" if normalize_complex_name(value) in known_complexes else "Новый ЖК"),
        "geo_available": pd.Series(np.where(geography, "Координаты доступны", "Координаты неизвестны"), index=frame.index),
    }
    rows = []
    for dimension, values in labels.items():
        for label in sorted(values.unique()):
            mask = values.eq(label).to_numpy()
            for model, predicted in predictions.items():
                scores = metrics(frame.price.to_numpy()[mask], np.asarray(predicted)[mask]) if mask.sum() >= 2 else {
                    "n": int(mask.sum()), "mae_tenge": float(np.abs(frame.price.to_numpy()[mask] - np.asarray(predicted)[mask]).mean()),
                    "rmse_tenge": None, "r2": None, "over_20_pct": None}
                rows.append({"dimension": dimension, "segment": label, "model": model, **scores})
    return pd.DataFrame(rows)


def run_experiment(*, resume: bool = False, prepare_only: bool = False) -> dict:
    from comparables import load_catalog
    from location_model import LOCATION_INPUT_COLUMNS, LOCATION_VARIANTS, make_location_pipeline, prepare_location_input
    from location_experiment_report import render_report
    from building_catalog import normalize_complex_name

    before = {name: digest(PROJECT_ROOT / name) for name in PROTECTED}
    catalogue = load_catalog(PROPERTY_CATALOG_DIR)
    bundle = prepare_partition(pd.read_csv(DATA, low_memory=False), catalogue.listings, pd.read_csv(SPLIT))
    frame, train_ids, test_ids, evaluation_ids, groups = (bundle[key] for key in ("frame", "train_ids", "test_ids", "evaluation_ids", "groups"))
    x, y = prepare_location_input(frame), frame.price.astype(float)
    january, _ = load_artifacts(PROJECT_ROOT / "models/best_model.joblib", PROJECT_ROOT / "models/model_metadata.json")
    combined, combined_metadata = load_artifacts(PROJECT_ROOT / "models/combined_2026/best_model.joblib", PROJECT_ROOT / "models/combined_2026/model_metadata.json")
    if combined_metadata.get("input_sha256") != before[DATA.relative_to(PROJECT_ROOT).as_posix()] or combined_metadata.get("split_manifest_sha256") != before[SPLIT.relative_to(PROJECT_ROOT).as_posix()]:
        raise ValueError("Данные или split не соответствуют сохранённой объединённой модели.")
    if (combined_metadata.get("train_rows"), combined_metadata.get("test_rows")) != (len(train_ids), len(test_ids)):
        raise ValueError("Размеры разбиения не соответствуют объединённой модели.")
    estimator = clone(combined.named_steps["model"])
    sources = before | {path.relative_to(PROJECT_ROOT).as_posix(): digest(path) for path in (
        PROPERTY_CATALOG_DIR / "manifest.json", PROPERTY_CATALOG_DIR / "listings.csv",
        PROJECT_ROOT / "location_model.py", PROJECT_ROOT / "building_catalog.py", Path(__file__).resolve())}
    plan = {"sources": sources, "parameters": estimator.get_params(), "folds": 3, "seed": 42,
            "variants": list(LOCATION_VARIANTS), "selection": "minimum pooled apartment-group OOF MAE, no holdout selection"}
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan_path = REPORTS / "experiment_plan.json"
    if plan_path.exists():
        old_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if old_plan.get("signature") != signature:
            raise ValueError("Входы или код изменились: нужен новый каталог эксперимента, прежние результаты не перезаписываются.")
        if not resume and not prepare_only:
            raise FileExistsError("Эксперимент уже начат; используйте --resume.")
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    save_json(plan_path, plan | {"signature": signature})
    split_output = frame[["record_id", "house_key", "group"]].copy()
    split_output.insert(0, "row_index", frame.index)
    split_output["experiment_group"] = groups
    split_output["subset"] = np.where(frame.index.isin(train_ids), "train", "test")
    split_output["primary_evaluation"] = frame.index.isin(evaluation_ids)
    split_output.to_csv(REPORTS / "split_manifest.csv", index=False)
    save_json(REPORTS / "alias_overlap_audit.json", {"excluded_from_primary_test": bundle["exclusions"],
                                                       "original_test_rows": len(test_ids), "primary_test_rows": len(evaluation_ids)})
    print(f"Training {len(train_ids)}; original test {len(test_ids)}; common eligible test {len(evaluation_ids)}", flush=True)
    if prepare_only:
        return {"signature": signature, "train_rows": len(train_ids), "test_rows": len(evaluation_ids)}
    x_train, y_train = x.loc[train_ids], y.loc[train_ids]
    folds = list(GroupKFold(n_splits=3).split(x_train, y_train, groups.loc[train_ids]))
    cv_results = []
    with threadpool_limits(limits=2):
        for variant in LOCATION_VARIANTS:
            checkpoint = REPORTS / f"cv_{variant}.json"
            if resume and checkpoint.exists():
                result = json.loads(checkpoint.read_text(encoding="utf-8"))
                if result["signature"] != signature:
                    raise ValueError("Несовместимый CV checkpoint.")
                cv_results.append(result)
                continue
            oof = np.full(len(train_ids), np.nan)
            details = []
            for number, (fit, valid) in enumerate(folds, 1):
                if set(groups.loc[train_ids[fit]]) & set(groups.loc[train_ids[valid]]):
                    raise AssertionError("Пересечение квартир между CV частями.")
                print(f"{variant}: fold {number}/3, fit={len(fit)}, validation={len(valid)}", flush=True)
                started = time.monotonic()
                pipeline = make_location_pipeline(estimator, variant)
                pipeline.fit(x_train.iloc[fit], y_train.iloc[fit])
                predicted = pipeline.predict(x_train.iloc[valid])
                oof[valid] = predicted
                details.append({"fold": number, "fit_rows": len(fit), "validation_rows": len(valid),
                                "geo_available_rows": int(geo_mask(pipeline, x_train.iloc[valid]).sum()),
                                "elapsed_seconds": time.monotonic() - started, **metrics(y_train.iloc[valid], predicted)})
            result = {"signature": signature, "variant": variant, "metrics": metrics(y_train, oof),
                      "folds": details, "oof_predictions": oof.tolist()}
            save_json(checkpoint, result)
            cv_results.append(result)
            print(f"{variant}: CV MAE={result['metrics']['mae_tenge']:,.0f}", flush=True)
        selected = min(cv_results, key=lambda entry: entry["metrics"]["mae_tenge"])["variant"]
        print(f"Selected by CV: {selected}", flush=True)
        fitted, predictions = {}, {}
        test_x = x.loc[test_ids]
        for variant in LOCATION_VARIANTS:
            model_path, metadata_path = MODELS / f"{variant}.joblib", MODELS / f"{variant}.json"
            if resume and model_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata["experiment_signature"] != signature or metadata["model_sha256"] != digest(model_path):
                    raise ValueError("Модель checkpoint не соответствует эксперименту.")
                pipeline = joblib.load(model_path)
            else:
                print(f"Final fit {variant}: {len(train_ids)} rows", flush=True)
                pipeline = make_location_pipeline(estimator, variant).fit(x_train, y_train)
                joblib.dump(pipeline, model_path, compress=3)
                metadata = {"schema_version": 2, "variant": variant, "feature_columns": list(LOCATION_INPUT_COLUMNS),
                            "experiment_signature": signature, "model_sha256": digest(model_path),
                            "selected_by_cv": variant == selected, "activated": False, "train_rows": len(train_ids),
                            "geo_training_only": True, "target": "asking_price", "currency": "KZT"}
                save_json(metadata_path, metadata)
            fitted[variant] = pipeline
            predictions[variant] = np.asarray(pipeline.predict(test_x), float)
        predictions["january"] = np.asarray(january.predict(x.loc[test_ids, list(FEATURE_COLUMNS)]), float)
        predictions["combined"] = np.asarray(combined.predict(x.loc[test_ids, list(FEATURE_COLUMNS)]), float)
        np.testing.assert_allclose(predictions["baseline8"], predictions["combined"], rtol=1e-10, atol=.05,
                                   err_msg="Контрольный повтор baseline8 не воспроизводит frozen combined.")
        # Независимый от исходного test диагностический разрез по домам внутри train.
        # Не участвует в выборе варианта. Все отсутствующие адреса имеют отдельные ключи.
        houses = house_validation_groups(frame.loc[train_ids], groups.loc[train_ids])
        house_fit, house_valid = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(x_train, groups=houses))
        if set(houses.iloc[house_fit]) & set(houses.iloc[house_valid]):
            raise AssertionError("Дом пересекает разбиение для новых домов.")
        if set(groups.loc[train_ids[house_fit]]) & set(groups.loc[train_ids[house_valid]]):
            raise AssertionError("Группа квартиры пересекает разбиение для новых домов.")
        house_checkpoint = REPORTS / "unseen_house_check.json"
        if resume and house_checkpoint.exists():
            house_check = json.loads(house_checkpoint.read_text(encoding="utf-8"))
            if house_check["signature"] != signature:
                raise ValueError("Несовместимая проверка новых домов.")
        else:
            house_results = []
            for variant in dict.fromkeys(("baseline8", selected, "complex_coords")):
                print(f"Unseen-house diagnostic: {variant}", flush=True)
                pipeline = make_location_pipeline(estimator, variant).fit(x_train.iloc[house_fit], y_train.iloc[house_fit])
                predicted = pipeline.predict(x_train.iloc[house_valid])
                house_results.append({"variant": variant, **metrics(y_train.iloc[house_valid], predicted),
                                      "geo_available_rows": int(geo_mask(pipeline, x_train.iloc[house_valid]).sum())})
            house_check = {"signature": signature, "seed": 42, "fit_rows": len(house_fit), "validation_rows": len(house_valid),
                           "fit_record_ids": frame.loc[train_ids[house_fit], "record_id"].tolist(),
                           "validation_record_ids": frame.loc[train_ids[house_valid], "record_id"].tolist(), "results": house_results,
                           "selection_used": False, "unknown_address_caveat": "Unknown addresses cannot establish building independence.",
                           "selection_caveat": "Diagnostic reuse of training population; fixed parameters and variant were selected with CV that included these rows. Not independent model-selection validation."}
            save_json(house_checkpoint, house_check)
    eligible = np.isin(test_ids, evaluation_ids)
    scores = [{"model": name, **metrics(y.loc[test_ids].to_numpy()[eligible], predicted[eligible])} for name, predicted in predictions.items()]
    original_scores = [{"model": name, **metrics(y.loc[test_ids], predicted)} for name, predicted in predictions.items()]
    exported = frame.loc[test_ids, ["record_id", "group", "house_key", "district", "residential_complex", "price"]].copy()
    exported["primary_evaluation"] = eligible
    for name, predicted in predictions.items():
        exported[name + "_prediction"] = predicted
        exported[name + "_absolute_error"] = np.abs(exported.price - predicted)
    geography = geo_mask(fitted["complex_coords"], test_x)
    exported["geography_available"] = geography
    exported.to_csv(REPORTS / "holdout_predictions.csv", index=False)
    segments = segment_metrics(frame.loc[evaluation_ids], {name: value[eligible] for name, value in predictions.items()}, geography[eligible],
                               set(frame.loc[train_ids, "house_key"].dropna()),
                               set(frame.loc[train_ids, "residential_complex"].map(normalize_complex_name).dropna()))
    segments.to_csv(REPORTS / "segments.csv", index=False)
    sources_after = {name: digest(PROJECT_ROOT / name) for name in sources}
    if sources != sources_after:
        raise AssertionError("Входные данные, исходный код или эталонные модели изменены во время эксперимента.")
    after = {name: sources_after[name] for name in PROTECTED}
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "signature": signature, "active_model": "january",
              "activated": False, "selected_variant": selected, "training_rows": len(train_ids), "test_rows": len(evaluation_ids),
              "original_test_rows": len(test_ids), "excluded_test_rows": bundle["exclusions"], "comparison": scores,
              "original_holdout_comparison": original_scores, "cv_comparison": [{k: v for k, v in row.items() if k != "oof_predictions"} for row in cv_results],
              "unseen_house_check": {k: v for k, v in house_check.items() if not k.endswith("record_ids")},
              "holdout_geo_available_rows": int(geography[eligible].sum()), "catalogue_summary": catalogue.manifest["summary"],
              "protected_before": before, "protected_after": after,
              "limitations": ["Мартовский test уже изучался: это повторное диагностическое сравнение, а не свежая независимая оценка.",
                              "Проверки координат используют только train, но не подтверждают точность по внешнему реестру.",
                              "Координаты доступны только для домов, согласованно представленных в train; новым домам они не приписываются.",
                              "Группы квартир/домов основаны на консервативных адресных эвристиках; невыявленные повторы остаются возможными.",
                              "Восемь исходных параметров и цены сохраняются; конфликтные свойства дома не заменяют индивидуальную анкету.",
                              "Предсказывается архивная цена предложения, не цена сделки и не прогноз на шесть месяцев."]}
    save_json(REPORTS / "experiment_report.json", report)
    render_report(report, REPORTS / "report.html")
    print(json.dumps({"selected": selected, "comparison": scores, "geo_available": report["holdout_geo_available_rows"]}, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args()
    run_experiment(resume=options.resume, prepare_only=options.prepare_only)

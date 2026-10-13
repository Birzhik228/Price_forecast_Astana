"""Повторить прогнозы сохранённых location-моделей без обучения и изменения данных."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from config import PROJECT_ROOT, PROPERTY_CATALOG_DIR
from features import FEATURE_COLUMNS


DEFAULT_REPORTS = PROJECT_ROOT / "reports/location_experiment"
DEFAULT_MODELS = PROJECT_ROOT / "models/location_2026"
EXPECTED_HOLDOUT_ROWS = 590
VARIANTS = ("baseline8", "complex", "coords", "complex_coords")
PRICE_TOLERANCE_TENGE = .01


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scores(actual, predicted) -> dict:
    """Прямой пересчёт, независимо от функции метрик эксперимента."""
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    require(y.ndim == 1 and y.shape == p.shape and len(y) >= 2, "Неверные размеры векторов метрик.")
    require(np.isfinite(y).all() and np.isfinite(p).all() and (y > 0).all(), "Непригодная цена или прогноз.")
    residual, total = float(np.square(y - p).sum()), float(np.square(y - y.mean()).sum())
    r2 = 1 - residual / total if total else float(residual == 0)
    return {"n": len(y), "mae_tenge": float(np.abs(y - p).mean()),
            "rmse_tenge": float(np.sqrt(np.square(y - p).mean())), "r2": r2,
            "over_20_pct": float((np.abs(y - p) > y * .2).mean() * 100)}


def compare_scores(actual: dict, saved: dict, context: str) -> None:
    require(actual["n"] == saved["n"], f"{context}: разное число строк.")
    for key in ("mae_tenge", "rmse_tenge", "r2", "over_20_pct"):
        tolerance = PRICE_TOLERANCE_TENGE if key.endswith("tenge") else 1e-10
        require(np.isclose(actual[key], saved[key], rtol=0, atol=tolerance), f"{context}: не совпала метрика {key}.")


def _plan_path(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    require(path.is_relative_to(PROJECT_ROOT.resolve()), "Путь источника вышел за пределы проекта.")
    return path


def check_checkpoint_reuse(reports: Path, models: Path, plan: dict) -> dict:
    """Доказать допустимость сохранённых fit при двух исправленных predict-вызовах."""
    reuse_path = reports / "checkpoint_reuse.json"
    if not reuse_path.exists():
        for variant in VARIANTS:
            require("fitted_under_signature" not in read_json(reports / f"cv_{variant}.json")
                    and "fitted_under_experiment_signature" not in read_json(models / f"{variant}.json"),
                    "Есть следы переноса checkpoint без записи происхождения.")
        return {"status": "not_used"}

    reuse = read_json(reuse_path)
    require(reuse["status"] == "completed" and reuse["fit_calls"] == 0, "Миграция не подтверждена как inference-only.")
    backup = (reports / reuse["original_artifacts_directory"]).resolve()
    require(backup.is_relative_to(reports.resolve()), "Backup миграции вышел за каталог отчётов.")
    evidence = [reuse_path, backup / "experiment_plan.json", backup / "location_experiment.py"]
    old_plan = read_json(backup / "experiment_plan.json")
    old_signature = old_plan["signature"]
    calculated = hashlib.sha256(json.dumps({k: v for k, v in old_plan.items() if k != "signature"}, sort_keys=True).encode()).hexdigest()
    require(old_signature == calculated == reuse["original_signature"]
            and plan["signature"] == reuse["current_signature"], "Не совпали подписи миграции.")
    old_code = (backup / "location_experiment.py").read_bytes()
    expected_code = old_code
    for name in ("january", "combined"):
        previous = f"{name}.predict(frame.loc[test_ids, list(FEATURE_COLUMNS)])".encode()
        fixed = f"{name}.predict(x.loc[test_ids, list(FEATURE_COLUMNS)])".encode()
        require(expected_code.count(previous) == 1, "Не найден единственный исходный frozen predict.")
        expected_code = expected_code.replace(previous, fixed)
    current_code = _plan_path("location_experiment.py")
    require(expected_code == current_code.read_bytes(), "CLI изменён сверх двух разрешённых inference-вызовов.")
    require(hashlib.sha256(old_code).hexdigest() == old_plan["sources"]["location_experiment.py"],
            "Backup кода не соответствует исходному плану fit.")
    expected_plan = {**old_plan, "signature": plan["signature"],
                     "sources": {**old_plan["sources"], "location_experiment.py": digest(current_code)}}
    require(plan == expected_plan, "При миграции изменены параметры, данные или другие источники обучения.")
    for name, expected in plan["sources"].items():
        require(digest(_plan_path(name)) == expected, f"Источник миграции изменён: {name}")

    hashes = {}
    for variant in VARIANTS:
        old_cv_path = backup / f"cv_{variant}.json"
        old_metadata_path = backup / f"{variant}_metadata.json"
        evidence += [old_cv_path, old_metadata_path]
        old_cv, old_metadata = read_json(old_cv_path), read_json(old_metadata_path)
        require(old_cv["signature"] == old_metadata["experiment_signature"] == old_signature,
                f"{variant}: исходные checkpoint не связаны с планом fit.")
        require(read_json(reports / f"cv_{variant}.json") == {
            **old_cv, "signature": plan["signature"], "fitted_under_signature": old_signature,
        }, f"{variant}: изменены OOF-прогнозы, метрики или другая часть CV checkpoint.")
        require(read_json(models / f"{variant}.json") == {
            **old_metadata, "experiment_signature": plan["signature"],
            "fitted_under_experiment_signature": old_signature,
        }, f"{variant}: метаданные изменены сверх подписи и сохранения происхождения.")
        hashes[variant] = digest(models / f"{variant}.joblib")
        require(hashes[variant] == old_metadata["model_sha256"]
                == reuse["model_joblibs_unchanged_sha256"][variant], f"{variant}: обученный joblib изменился.")
    return {"status": "verified", "original_fit_signature": old_signature,
            "current_evaluation_signature": plan["signature"], "only_two_frozen_predict_inputs_changed": True,
            "oof_predictions_and_metrics_unchanged": True, "all_other_pinned_sources_unchanged": True,
            "model_joblibs_unchanged_sha256": hashes,
            "evidence_sha256": {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in evidence}}


def _check_coordinates(transformer, training: pd.DataFrame) -> pd.DataFrame:
    """Сверить сохранённые медианы с собственными исходными точками train."""
    from building_catalog import ASTANA_BBOX

    audit = transformer.coordinate_audit_.copy(deep=True)
    known = set(training.house_key.dropna())
    require(audit.house_key.is_unique and set(audit.house_key) == known, "Аудит координат не соответствует домам train.")
    require(transformer.training_geo_summary_["fit_rows"] == len(training), "Карта геоданных обучена на другом числе строк.")
    mapped = transformer.house_coordinate_map_
    require(set(mapped) == set(audit.loc[audit.coordinate_status.eq("plausible"), "house_key"]), "Карта и аудит геоданных расходятся.")
    for key, point in mapped.items():
        part = training.loc[training.house_key.eq(key)]
        require(not part.raw_coordinate_status.eq("invalid").any(), "В карте остался дом с невалидной исходной точкой.")
        points = part.loc[part.raw_coordinate_status.eq("unverified"), ["raw_latitude", "raw_longitude"]].astype(float)
        require(len(points) >= 2 and np.isfinite(points.to_numpy()).all(), "Медиана дома не имеет двух конечных исходных точек.")
        require(points.raw_latitude.between(ASTANA_BBOX["latitude_min"], ASTANA_BBOX["latitude_max"]).all()
                and points.raw_longitude.between(ASTANA_BBOX["longitude_min"], ASTANA_BBOX["longitude_max"]).all(), "Исходная точка за bbox.")
        np.testing.assert_allclose(point, points.median().to_numpy(), rtol=0, atol=1e-10,
                                   err_msg="Сохранённая геоточка отличается от train-only медианы.")
    audit["verification_note"] = "internally_consistent_not_external"
    audit["fit_scope"] = "primary_training_rows_only"
    return audit


def verify(reports: Path = DEFAULT_REPORTS, models: Path = DEFAULT_MODELS) -> dict:
    """Только загрузка/transform/predict; результаты пишутся после всех проверок."""
    plan = read_json(reports / "experiment_plan.json")
    report = read_json(reports / "experiment_report.json")
    signature = plan["signature"]
    calculated = hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "signature"}, sort_keys=True).encode()).hexdigest()
    require(signature == calculated == report["signature"], "Подписи плана и отчёта не совпали.")
    require(tuple(plan["variants"]) == VARIANTS, "Неожиданные варианты моделей.")
    require(report["activated"] is False and report["active_model"] == "january", "Отчёт неожиданно заявляет активацию.")
    verified_paths = [reports / name for name in (
        "experiment_plan.json", "experiment_report.json", "holdout_predictions.csv", "split_manifest.csv",
        "unseen_house_check.json", "alias_overlap_audit.json",
    )]
    verified_paths += [reports / f"cv_{variant}.json" for variant in VARIANTS]
    verified_paths += [models / f"{variant}{suffix}" for variant in VARIANTS for suffix in (".joblib", ".json")]
    artifact_hashes = {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in verified_paths}
    sources = plan["sources"]
    for name, expected in sources.items():
        require(digest(_plan_path(name)) == expected, f"Изменился источник плана: {name}")
    checkpoint_reuse = check_checkpoint_reuse(reports, models, plan)
    artifact_hashes.update(checkpoint_reuse.get("evidence_sha256", {}))
    require(report["protected_before"] == report["protected_after"], "Защищённые артефакты менялись в эксперименте.")
    require(all(sources[name] == expected for name, expected in report["protected_before"].items()), "Защищённые SHA отчёта отличаются от плана.")

    # Импортировать код модели только после проверки его закреплённого SHA.
    from building_catalog import canonical_house_key
    from comparables import load_catalog
    from location_model import LOCATION_INPUT_COLUMNS, prepare_location_input
    from location_experiment import apartment_groups, house_validation_groups
    from prediction import load_artifacts

    catalogue = load_catalog(PROPERTY_CATALOG_DIR)
    provenance = {}
    for name, relative in (("combined", "data/processed/combined_2026/listings_clean.csv"),
                           ("january_raw", "data/raw/kaggle_astana/astana_apartments.csv"),
                           ("march_raw", "data/raw/turarr_march_2026/astana_apartments_raw.csv")):
        actual = digest(PROJECT_ROOT / relative)
        require(catalogue.manifest["inputs"][name]["sha256"] == actual == sources[relative], f"Каталог использует другой исходник {name}.")
        provenance[name] = {"path": relative, "sha256": actual, "catalog_manifest_match": True}

    combined = pd.read_csv(PROJECT_ROOT / "data/processed/combined_2026/listings_clean.csv", low_memory=False)
    original_split = pd.read_csv(PROJECT_ROOT / "reports/combined_update/split_manifest.csv").set_index("row_index").loc[combined.index]
    experiment_split = pd.read_csv(reports / "split_manifest.csv").set_index("row_index").loc[combined.index]
    require(original_split.index.is_unique and experiment_split.index.is_unique
            and len(experiment_split) == len(combined), "Повторные или пропущенные строки split.")
    require(original_split.source_row.tolist() == combined.source_row.tolist(), "Исходный split не соответствует source_row.")
    require(experiment_split.record_id.tolist() == combined.record_id.tolist(), "Экспериментальный split не соответствует record_id.")
    require(experiment_split.subset.tolist() == original_split.subset.tolist(), "Изменены исходные train/test.")
    require(experiment_split.group.tolist() == combined.group.tolist() == original_split.group.tolist(), "Изменены исходные группы квартир.")
    require(catalogue.listings.record_id.is_unique and set(catalogue.listings.record_id) == set(combined.record_id), "Изменено покрытие каталога.")
    frame = catalogue.listings.set_index("record_id", drop=False).loc[combined.record_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame.loc[:, combined.columns], combined, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-6)
    expected_house = [canonical_house_key(row.address, row.district, row.residential_complex) for row in frame.itertuples()]
    require(frame.house_key.fillna("").tolist() == [value or "" for value in expected_house], "house_key не является row-local канонизацией.")
    require(experiment_split.house_key.fillna("").tolist() == frame.house_key.fillna("").tolist(), "House keys split изменены.")
    groups = apartment_groups(frame)
    require(experiment_split.experiment_group.tolist() == groups.tolist(), "Группы эксперимента не воспроизводятся.")
    train_ids = combined.index[original_split.subset.eq("train")].to_numpy()
    test_ids = combined.index[original_split.subset.eq("test")].to_numpy()
    primary = experiment_split.loc[test_ids, "primary_evaluation"].to_numpy(dtype=bool)
    require(len(test_ids) == EXPECTED_HOLDOUT_ROWS == report["original_test_rows"], "Ожидались прежние 590 строк holdout.")
    require(int(primary.sum()) == report["test_rows"] == EXPECTED_HOLDOUT_ROWS, "Изменился состав общего holdout.")
    require(len(train_ids) == report["training_rows"], "Изменился размер train.")
    require(not experiment_split.loc[train_ids, "primary_evaluation"].any(), "Строка train включена в evaluation.")
    require(set(groups.loc[train_ids]).isdisjoint(groups.loc[test_ids]), "Пересечение квартир train/test.")
    x = prepare_location_input(frame)
    x_train, x_test = x.loc[train_ids], x.loc[test_ids]
    actual = frame.loc[test_ids, "price"].to_numpy(dtype=float)
    saved = pd.read_csv(reports / "holdout_predictions.csv")
    require(saved.record_id.is_unique and saved.record_id.tolist() == frame.loc[test_ids, "record_id"].tolist(), "Не совпали 590 ID прогнозов.")
    displayed = ["group", "house_key", "district", "residential_complex"]
    pd.testing.assert_frame_equal(saved[displayed], frame.loc[test_ids, displayed].reset_index(drop=True), check_dtype=False)
    np.testing.assert_allclose(saved.price, actual, rtol=0, atol=0, err_msg="Не совпали целевые цены holdout.")
    require(saved.primary_evaluation.tolist() == primary.tolist(), "CSV и split не согласованы по evaluation.")

    folds = list(GroupKFold(n_splits=plan["folds"]).split(x_train, groups=groups.loc[train_ids]))
    cv_scores = {}
    for variant in VARIANTS:
        checkpoint = read_json(reports / f"cv_{variant}.json")
        require(checkpoint["signature"] == signature and checkpoint["variant"] == variant, "Неподходящий CV checkpoint.")
        oof = np.asarray(checkpoint["oof_predictions"], dtype=float)
        cv_scores[variant] = scores(frame.loc[train_ids, "price"], oof)
        compare_scores(cv_scores[variant], checkpoint["metrics"], f"{variant}: OOF")
        reported = next(row for row in report["cv_comparison"] if row["variant"] == variant)
        compare_scores(cv_scores[variant], reported["metrics"], f"{variant}: reported OOF")
        for number, (fit, valid) in enumerate(folds, 1):
            require(set(groups.loc[train_ids[fit]]).isdisjoint(groups.loc[train_ids[valid]]), "CV разделяет одну квартиру.")
            detail = next(row for row in checkpoint["folds"] if row["fold"] == number)
            require(detail["fit_rows"] == len(fit) and detail["validation_rows"] == len(valid), "CV fold имеет другой размер.")
            compare_scores(scores(frame.loc[train_ids[valid], "price"], oof[valid]), detail, f"{variant}: fold {number}")
    selected = min(VARIANTS, key=lambda name: cv_scores[name]["mae_tenge"])
    require(selected == report["selected_variant"], "Выбранный вариант не минимизирует OOF MAE.")

    predictions, candidate_checks = {}, {}
    coordinate_audit, geography = None, None
    for variant in VARIANTS:
        model_path, metadata_path = models / f"{variant}.joblib", models / f"{variant}.json"
        metadata = read_json(metadata_path)
        model_hash = digest(model_path)
        require(metadata["model_sha256"] == model_hash and metadata["experiment_signature"] == signature, f"Не совпал SHA/signature {variant}.")
        require(metadata["schema_version"] == 2 and metadata["variant"] == variant
                and metadata["feature_columns"] == list(LOCATION_INPUT_COLUMNS), "Несовместимая схема кандидата.")
        require(metadata["activated"] is False and metadata["geo_training_only"] is True
                and metadata["selected_by_cv"] == (variant == selected)
                and metadata["train_rows"] == len(train_ids), "Несогласованные метаданные кандидата.")
        pipeline = joblib.load(model_path)
        require(isinstance(pipeline, Pipeline) and list(pipeline.feature_names_in_) == list(LOCATION_INPUT_COLUMNS), "Неверная схема сохранённого Pipeline.")
        require(pipeline.named_steps["model"].get_params() == plan["parameters"], "Параметры регрессора отличаются от плана.")
        predicted = np.asarray(pipeline.predict(x_test), dtype=float)
        np.testing.assert_allclose(predicted, saved[f"{variant}_prediction"], rtol=0, atol=PRICE_TOLERANCE_TENGE)
        predictions[variant] = predicted
        candidate_checks[variant] = {"model_sha256": model_hash, "max_prediction_difference_tenge": float(np.max(np.abs(predicted - saved[f"{variant}_prediction"])))}
        if variant == "complex_coords":
            transformer = pipeline.named_steps["location_features"]
            coordinate_audit = _check_coordinates(transformer, x_train)
            geography = transformer.transform(x_test).latitude.notna().to_numpy()
            tampered = x_test.copy()
            tampered["raw_latitude"], tampered["raw_longitude"], tampered["raw_coordinate_status"] = -70.0, -130.0, "invalid"
            tampered["latitude"], tampered["longitude"], tampered["price"] = 999.0, 999.0, -999.0
            np.testing.assert_allclose(pipeline.predict(tampered), predicted, rtol=0, atol=PRICE_TOLERANCE_TENGE,
                                       err_msg="Test raw/catalog coordinates or target affected predictions.")
        del pipeline

    references = (
        ("january", "models", "data/processed/january_2026/listings_clean.csv", None),
        ("combined", "models/combined_2026", "data/processed/combined_2026/listings_clean.csv", "reports/combined_update/split_manifest.csv"),
        ("march", "models/march_2026", "data/processed/march_2026/listings_clean.csv", "reports/march_update/split_manifest.csv"),
    )
    reference_checks = {}
    for name, directory, input_file, split_file in references:
        pipeline, metadata = load_artifacts(PROJECT_ROOT / directory / "best_model.joblib", PROJECT_ROOT / directory / "model_metadata.json")
        require(metadata["input_sha256"] == digest(PROJECT_ROOT / input_file), f"Входные данные {name} изменены.")
        if split_file:
            require(metadata["split_manifest_sha256"] == digest(PROJECT_ROOT / split_file), f"Split {name} изменён.")
        if name == "january":
            archive = read_json(PROJECT_ROOT / "models/archive/january_2026/model_metadata.json")
            require(metadata["model_sha256"] == archive["model_sha256"] and metadata["input_sha256"] == archive["input_sha256"], "Активна другая модель, не frozen January.")
        if name in ("january", "combined"):
            predictions[name] = np.asarray(pipeline.predict(x_test.loc[:, list(FEATURE_COLUMNS)]), dtype=float)
            np.testing.assert_allclose(predictions[name], saved[f"{name}_prediction"], rtol=0, atol=PRICE_TOLERANCE_TENGE)
        reference_checks[name] = {"model_sha256": metadata["model_sha256"], "input_sha256": metadata["input_sha256"], "prediction_replayed": name in predictions}
        del pipeline
    np.testing.assert_allclose(predictions["baseline8"], predictions["combined"], rtol=0, atol=.05)
    for key in ("comparison", "original_holdout_comparison"):
        require(len(report[key]) == len(predictions) and {row["model"] for row in report[key]} == set(predictions), "Состав сравнительной таблицы не совпадает с моделями.")
    recomputed = {}
    for name, predicted in predictions.items():
        np.testing.assert_allclose(np.abs(actual - predicted), saved[f"{name}_absolute_error"], rtol=0, atol=PRICE_TOLERANCE_TENGE)
        recomputed[name] = scores(actual[primary], predicted[primary])
        compare_scores(recomputed[name], next(row for row in report["comparison"] if row["model"] == name), f"{name}: primary holdout")
        compare_scores(scores(actual, predicted), next(row for row in report["original_holdout_comparison"] if row["model"] == name), f"{name}: original holdout")
    require(saved.geography_available.tolist() == geography.tolist()
            and int(geography[primary].sum()) == report["holdout_geo_available_rows"], "Географическое покрытие отличается от отчёта.")

    house_check = read_json(reports / "unseen_house_check.json")
    require(house_check["signature"] == signature and house_check["selection_used"] is False, "Несогласованный house diagnostic.")
    fit_ids, valid_ids = house_check["fit_record_ids"], house_check["validation_record_ids"]
    require(len(set(fit_ids)) == len(fit_ids) and len(set(valid_ids)) == len(valid_ids)
            and set(fit_ids).isdisjoint(valid_ids)
            and set(fit_ids) | set(valid_ids) == set(frame.loc[train_ids, "record_id"]), "House diagnostic неверно разделяет train.")
    house_groups = house_validation_groups(frame.loc[train_ids], groups.loc[train_ids])
    houses_by_id = dict(zip(frame.loc[train_ids, "record_id"], house_groups))
    require({houses_by_id[key] for key in fit_ids}.isdisjoint(houses_by_id[key] for key in valid_ids), "Дом/квартира пересекает house diagnostic.")
    require(house_check["fit_rows"] == len(fit_ids) and house_check["validation_rows"] == len(valid_ids), "Неверные размеры house diagnostic.")

    for name, expected in sources.items():
        require(digest(_plan_path(name)) == expected, f"В процессе проверки изменился источник {name}.")
    for name, expected in artifact_hashes.items():
        require(digest(_plan_path(name)) == expected, f"В процессе проверки изменился артефакт {name}.")
    coordinate_path = reports / "training_coordinate_audit.csv"
    coordinate_audit.to_csv(coordinate_path, index=False, encoding="utf-8", lineterminator="\n")
    result = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "experiment_signature": signature, "fit_calls": 0, "active_model": "january", "active_unchanged": True,
              "source_sha256": sources, "catalog_source_provenance": provenance,
              "checkpoint_reuse": checkpoint_reuse,
              "verified_artifact_sha256": artifact_hashes,
              "record_ids": saved.record_id.tolist(), "original_holdout_rows": len(actual), "primary_holdout_rows": int(primary.sum()),
              "actual_price_vector_sha256": hashlib.sha256(actual.astype("<f8").tobytes()).hexdigest(),
              "price_tolerance_tenge": PRICE_TOLERANCE_TENGE, "candidates": candidate_checks,
              "protected_models": reference_checks, "recomputed_holdout_metrics": recomputed,
              "recomputed_oof_metrics": cv_scores, "selected_by_recomputed_oof": selected,
              "holdout_geo_available_rows": int(geography.sum()),
              "coordinate_audit": {"path": str(coordinate_path.relative_to(PROJECT_ROOT)), "sha256": digest(coordinate_path),
                                   "rows": len(coordinate_audit), "status_counts": {str(k): int(v) for k, v in coordinate_audit.coordinate_status.value_counts().items()},
                                   "note": "internally_consistent_not_external", "test_raw_coords_target_and_catalog_coords_ignored": True},
              "unseen_house_partition_verified": True,
              "limitations": ["CV metrics are recomputed from saved OOF predictions; models are not refitted to replay CV.",
                              "House diagnostic partition is checked; its models/predictions were not saved, so its metrics cannot be independently replayed without fitting.",
                              "March holdout was analyzed previously; replay confirms reproducibility, not independence or current-market accuracy.",
                              "Coordinate consistency is internal to archived training records, not external address verification."]}
    (reports / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "holdout_rows": len(actual), "replayed_models": list(predictions),
                      "selected_by_cv": selected, "geography_available": int(geography.sum()), "fit_calls": 0}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    arguments = parser.parse_args()
    verify(arguments.reports_dir.resolve(), arguments.models_dir.resolve())

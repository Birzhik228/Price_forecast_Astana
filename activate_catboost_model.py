"""Prepare deployment metadata for the user's CatBoost MAE choice, without fit.

Historical experiment files remain immutable. Only config.py and
location_prediction.py may change for application integration; their byte-exact
pre-activation snapshots must still match the experiment's pinned SHA256 values.
This command does not edit config.py or claim that UI integration was tested.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from activate_location_model import _bytes, _json, _path, _publish_new, _require
from catboost_model import (
    CATBOOST_CATEGORICAL_COLUMNS, CATBOOST_MODEL_COLUMNS, CatBoostPriceRegressor,
)
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from location_model import LOCATION_INPUT_COLUMNS, prepare_location_input


EXPERIMENT = "reports/catboost_2026_10_16"
ACTIVATION_DIR = "reports/catboost_activation_2026_10_16"
DEPLOYMENT = "models/deployment/catboost_mae.json"
MODEL = "models/catboost_2026_10_16/catboost_mae.joblib"
ORIGINAL_METADATA = "models/catboost_2026_10_16/catboost_mae.json"
DATA = "data/processed/reviewed_2026_10_16/listings_clean.csv"
CATALOG = "data/processed/reviewed_2026_10_16/listings.csv"
PREVIOUS_MODEL = "models/verified_geo_2026/complex_coords.joblib"
PREVIOUS_METADATA = "models/deployment/complex_coords_osm.json"
SNAPSHOT_SOURCES = frozenset(("config.py", "location_prediction.py"))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def verify_sources(root: Path, expected: dict[str, str]) -> dict[str, dict]:
    """Verify all frozen inputs, with a strict two-file snapshot allowlist."""
    evidence = {}
    _require(SNAPSHOT_SOURCES.issubset(expected), "Missing application source hashes.")
    for name, checksum in expected.items():
        verified_name = (f"{ACTIVATION_DIR}/source_snapshot/{name}"
                         if name in SNAPSHOT_SOURCES else name)
        path = _path(root, verified_name)
        _require(path.is_file() and _digest(path) == checksum,
                 f"Frozen source or original snapshot changed: {verified_name}")
        evidence[name] = {"sha256": checksum, "verified_path": verified_name,
                          "uses_original_snapshot": name in SNAPSHOT_SOURCES}
    return evidence


def _metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    _require(actual.ndim == 1 and actual.shape == predicted.shape and len(actual) > 1
             and np.isfinite(actual).all() and (actual > 0).all()
             and np.isfinite(predicted).all(), "Invalid holdout prices or predictions.")
    absolute = np.abs(actual - predicted)
    squared = np.square(actual - predicted)
    return {"n": len(actual), "mae_tenge": float(absolute.mean()),
            "rmse_tenge": float(np.sqrt(squared.mean())),
            "r2": float(1 - squared.sum() / np.square(actual - actual.mean()).sum()),
            "median_ae_tenge": float(np.median(absolute)),
            "median_ape_pct": float(np.median(absolute / actual) * 100),
            "over_20_n": int((absolute > .2 * actual).sum()),
            "over_20_pct": float((absolute > .2 * actual).mean() * 100)}


def _partition(frame: pd.DataFrame, split: pd.DataFrame) -> tuple[list[int], list[int]]:
    _require({"row_index", "record_id", "subset"}.issubset(split.columns)
             and len(frame) == len(split) and frame.record_id.is_unique
             and split.row_index.is_unique and set(split.row_index) == set(frame.index)
             and split.subset.isin(("train", "test")).all(), "Invalid frozen partition.")
    ordered = split.set_index("row_index").loc[frame.index]
    _require(ordered.record_id.tolist() == frame.record_id.tolist(), "Partition IDs differ.")
    return (ordered.index[ordered.subset.eq("train")].tolist(),
            ordered.index[ordered.subset.eq("test")].tolist())


def prepare_activation(root: Path | None = None) -> dict:
    root = (root or Path(__file__).resolve().parent).resolve()
    experiment = _path(root, EXPERIMENT)
    protocol_path = experiment / "protocol.json"
    protocol = _json(protocol_path)
    spec = protocol["spec"]
    _require(protocol.get("signature") == _signature(spec), "Invalid experiment signature.")
    evidence = verify_sources(root, spec["sources_sha256"])
    comparison = _json(experiment / "comparison.json")
    selection = _json(experiment / "selection_cv.json")
    original = _json(_path(root, ORIGINAL_METADATA))
    _require(comparison["protocol_sha256"] == original["identity"]["protocol_sha256"]
             == _digest(protocol_path), "Model and report refer to another protocol.")
    _require(comparison["selection_sha256"] == _digest(experiment / "selection_cv.json")
             and selection == comparison["selection_cv"]
             and selection["chosen_variant"] == comparison["selected_variant"] == "catboost_log",
             "CV selection evidence changed.")
    _require(original == comparison["artifacts"]["catboost_mae"]
             and original["model_path"] == MODEL
             and original["model_sha256"] == _digest(_path(root, MODEL))
             and original["identity"]["variant"] == "catboost_mae"
             and original["identity"]["params_id"] == selection["by_family"]["catboost_mae"]["params_id"]
             and original["target_mode"] == "price_mae"
             and original["actual_model_features"] == list(CATBOOST_MODEL_COLUMNS)
             and original["input_columns"] == list(LOCATION_INPUT_COLUMNS)
             and original["uses_coordinates"] is False, "Incompatible CatBoost artifact.")
    extra_sources = {f"{EXPERIMENT}/protocol.json": _digest(protocol_path),
                     f"{EXPERIMENT}/comparison.json": _digest(experiment / "comparison.json"),
                     f"{EXPERIMENT}/selection_cv.json": comparison["selection_sha256"],
                     ORIGINAL_METADATA: _digest(_path(root, ORIGINAL_METADATA)),
                     MODEL: original["model_sha256"]}
    for name, checksum in comparison["output_csv_sha256"].items():
        full_name = f"{EXPERIMENT}/{name}"
        _require(_digest(_path(root, full_name)) == checksum, f"Changed experiment output: {name}")
        extra_sources[full_name] = checksum
    _require(all(comparison["output_csv_sha256"].get(name) == checksum
                 for name, checksum in spec["persisted_split_sha256"].items()), "Split hashes disagree.")

    frame = pd.read_csv(_path(root, DATA), low_memory=False)
    catalog = pd.read_csv(_path(root, CATALOG), low_memory=False)
    split = pd.read_csv(experiment / "split_manifest.csv")
    train_ids, test_ids = _partition(frame, split)
    _require(len(train_ids) == original["training_rows"] == spec["train_rows"]
             == comparison["training_rows"] == 19269
             and len(test_ids) == spec["test_rows"] == comparison["holdout_rows"] == 590,
             "Unexpected train/holdout size.")
    _require(original["identity"]["train_index_sha256"] == _signature(train_ids),
             "Model training indices differ.")
    shared = ["record_id", "price", *FEATURE_COLUMNS, "residential_complex"]
    pd.testing.assert_frame_equal(frame[shared], catalog[shared], check_dtype=False)
    held = pd.read_csv(experiment / "holdout_predictions.csv", float_precision="round_trip")
    _require(held.row_index.tolist() == test_ids
             and held.record_id.tolist() == frame.loc[test_ids, "record_id"].tolist()
             == comparison["comparison"]["record_ids"], "Holdout IDs or order differ.")
    np.testing.assert_array_equal(held.price, frame.loc[test_ids, "price"])

    # Loading is allowed only after the project's own frozen artifact checksum passes.
    model = joblib.load(_path(root, MODEL))
    _require(isinstance(model, CatBoostPriceRegressor)
             and model.target_mode_ == model.target_mode == "price_mae"
             and model.training_loss_ == "MAE" and model.training_rows_ == len(train_ids)
             and list(model.feature_names_in_) == list(LOCATION_INPUT_COLUMNS)
             and list(model.model_feature_names_) == list(CATBOOST_MODEL_COLUMNS)
             and model.location_transformer_.variant == "complex"
             and model.model_.is_fitted()
             and model.model_.get_all_params()["loss_function"] == "MAE"
             and model.model_.feature_names_ == list(CATBOOST_MODEL_COLUMNS)
             and model.model_.get_cat_feature_indices()
             == [list(CATBOOST_MODEL_COLUMNS).index(name) for name in CATBOOST_CATEGORICAL_COLUMNS],
             "Loaded model is not the expected native CatBoost MAE without coordinates.")
    predicted = model.predict(prepare_location_input(catalog.loc[test_ids]))
    np.testing.assert_array_equal(predicted, held.catboost_mae_prediction.to_numpy(float))
    scores = _metrics(held.price, predicted)
    published = comparison["comparison"]["overall"]["catboost_mae"]
    _require(all(key in published and math.isclose(value, published[key], rel_tol=1e-10, abs_tol=1e-6)
                 for key, value in scores.items()), "Holdout metrics do not reproduce.")

    train = frame.loc[train_ids]
    categories = {name: sorted(train[name].dropna().unique().tolist()) for name in CATEGORICAL_FEATURES}
    ranges = {name: {"min": float(train[name].min()), "max": float(train[name].max())}
              for name in NUMERIC_FEATURES if train[name].notna().any()}
    complexes = sorted(prepare_location_input(catalog.loc[train_ids]).residential_complex.dropna().unique().tolist())
    path = _path(root, DEPLOYMENT)
    created = _json(path)["created_at_utc"] if path.exists() else datetime.now(timezone.utc).isoformat()
    previous = {"model_path": PREVIOUS_MODEL, "metadata_path": PREVIOUS_METADATA,
                "model_sha256": spec["sources_sha256"][PREVIOUS_MODEL],
                "metadata_sha256": spec["sources_sha256"][PREVIOUS_METADATA]}
    source_hashes = {**spec["sources_sha256"], **extra_sources}
    snapshot_policy = {
        "reason": "Application integration changes config and dispatch only; historical experiment source bytes are preserved.",
        "allowed_changed_paths": sorted(SNAPSHOT_SOURCES),
        "snapshot_paths": {name: evidence[name]["verified_path"] for name in sorted(SNAPSHOT_SOURCES)},
        "other_pinned_sources_must_match": True,
    }
    metadata = {
        "schema_version": 3, "created_at_utc": created, "variant": "catboost_mae",
        "model_name": "CatBoost MAE + ЖК", "model_label": "CatBoost MAE + ЖК",
        "target": "asking_price", "currency": "KZT", "prediction_unit": "KZT",
        "target_mode": "price_mae", "uses_coordinates": False,
        "feature_columns": list(LOCATION_INPUT_COLUMNS), "actual_model_features": list(CATBOOST_MODEL_COLUMNS),
        "base_feature_columns": list(FEATURE_COLUMNS), "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES), "categories_from_train": categories,
        "numeric_ranges_from_train": ranges, "known_complexes_from_train": complexes,
        "model_path": MODEL, "model_sha256": original["model_sha256"],
        "original_metadata_path": ORIGINAL_METADATA, "original_metadata_sha256": extra_sources[ORIGINAL_METADATA],
        "input_sha256": spec["sources_sha256"][DATA],
        "split_manifest_sha256": comparison["output_csv_sha256"]["split_manifest.csv"],
        "experiment_signature": protocol["signature"], "protocol_sha256": comparison["protocol_sha256"],
        "selection_sha256": comparison["selection_sha256"],
        "training_rows": len(train_ids), "train_rows": len(train_ids), "test_rows": len(test_ids),
        "test_metrics": published,
        "approximate_range": {"half_width_tenge": published["mae_tenge"],
                              "source": "reused_march_holdout_catboost_mae",
                              "interpretation": "prediction +/- MAE; descriptive, not a confidence or prediction interval",
                              "guaranteed_coverage": None},
        "data_period": "2026-01_to_2026-03",
        "data_period_label": "января и конца марта 2026 года (март — по описанию источника)",
        "activation_reason": "explicit_user_request_catboost_mae_after_report",
        "selected_by_cv": False, "cv_selected_variant": selection["chosen_variant"],
        "deployment_role": "application", "holdout_previously_analyzed": True,
        "fresh_validation_completed": False, "activation_compatibility_validated": False,
        "limitations": list(comparison["limitations"]) + [
            "CatBoost MAE подключён по выбору пользователя; перекрёстная проверка предпочла логарифм цены.",
            "Диапазон плюс/минус MAE описательный и не имеет гарантированного покрытия.",
        ],
        "source_sha256": source_hashes, "source_snapshot_policy": snapshot_policy,
        "previous_deployment": previous,
    }
    metadata_bytes = _bytes(metadata)
    activation = {
        "schema_version": 1, "status": "prepared_for_user_requested_activation", "created_at_utc": created,
        "variant": "catboost_mae", "model_path": MODEL, "model_sha256": original["model_sha256"],
        "metadata_path": DEPLOYMENT, "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "train_rows": len(train_ids), "test_rows": len(test_ids), "test_metrics": published,
        "selected_by_cv": False, "cv_selected_variant": selection["chosen_variant"],
        "activation_reason": metadata["activation_reason"], "fit_calls": 0,
        "prediction_replay_max_delta_tenge": float(np.max(np.abs(predicted - held.catboost_mae_prediction))),
        "application_config_modified_by_generator": False, "activation_compatibility_validated": False,
        "integration_audit": "A separate integration audit must confirm the application adapter and UI.",
        "source_snapshot_policy": snapshot_policy, "source_evidence": evidence,
        "artifact_sha256": extra_sources, "previous_deployment": previous,
        "checks": ["protocol_signature", "all_pinned_source_sha256_or_two_exact_snapshots",
                   "all_experiment_csv_sha256", "model_sha256_and_fitted_native_schema",
                   "same_holdout_ids_order_and_prices", "exact_saved_prediction_replay",
                   "holdout_metrics_recomputed", "train_only_categories_ranges_and_complexes",
                   "previous_deployment_preserved", "no_training"],
    }
    # Recheck evidence before publishing; no original artifact is overwritten.
    verify_sources(root, spec["sources_sha256"])
    for name, checksum in extra_sources.items():
        _require(_digest(_path(root, name)) == checksum, f"Artifact changed during activation: {name}")
    outputs = ((path, metadata_bytes), (_path(root, f"{ACTIVATION_DIR}/activation.json"), _bytes(activation)))
    for output_path, content in outputs:
        _require(not output_path.exists() or output_path.read_bytes() == content,
                 f"Refusing to overwrite existing activation: {output_path}")
    for output_path, content in outputs:
        _publish_new(output_path, content)
    return activation


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = prepare_activation()
    print(json.dumps({key: result[key] for key in ("status", "variant", "train_rows", "test_rows",
                                                  "test_metrics", "prediction_replay_max_delta_tenge")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

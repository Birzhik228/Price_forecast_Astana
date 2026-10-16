"""Validate a corrected archive against the frozen training/evaluation cohort.

This module does not fit, predict, write artifacts, or recompute house keys.
The caller must verify source/manifest hashes before passing the frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from location_experiment import prepare_partition


def _same(left: pd.DataFrame, right: pd.DataFrame, columns: list[str], message: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            left.loc[:, columns].reset_index(drop=True),
            right.loc[:, columns].reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(message) from error


def _split_by_row(split: pd.DataFrame, frame: pd.DataFrame, required: set[str], label: str) -> pd.DataFrame:
    if not split.columns.is_unique or not required.issubset(split.columns):
        raise ValueError(f"{label}: missing or duplicated columns")
    if split.row_index.isna().any() or not split.row_index.is_unique:
        raise ValueError(f"{label}: row_index must be unique and complete")
    positions = pd.to_numeric(split.row_index, errors="coerce")
    if (positions.isna().any() or not np.isfinite(positions).all()
            or positions.mod(1).ne(0).any() or set(positions) != set(frame.index)):
        raise ValueError(f"{label}: row_index must cover every original row exactly once")
    return split.set_index("row_index", drop=False).loc[frame.index].copy()


def validate_retraining_partition(
    original_clean: pd.DataFrame,
    corrected_clean: pd.DataFrame,
    corrected_catalog: pd.DataFrame,
    historical_combined_split: pd.DataFrame,
    historical_location_split: pd.DataFrame,
    saved_errors: pd.DataFrame,
    *,
    expected_holdout_rows: int | None = 590,
    expected_train_rows: int | None = 19_269,
) -> dict:
    """Return the frozen ``prepare_partition`` bundle with verified record IDs.

    Only training features may differ from the original archive. Row identity,
    provenance, targets, and all eight base holdout features must be unchanged.
    Historical location keys are audit data only: the corrected catalog supplies
    the current house keys. Any new alias overlap or dropped test row fails closed.
    Count expectations may be overridden (or disabled with None) for other cohorts.
    """
    required = {"record_id", "price", "source_row", "group", *FEATURE_COLUMNS}
    for frame, label in [(original_clean, "original"), (corrected_clean, "corrected")]:
        if (not frame.columns.is_unique or not required.issubset(frame.columns)
                or not frame.index.equals(pd.RangeIndex(len(frame)))
                or frame.record_id.isna().any() or not frame.record_id.is_unique
                or frame.record_id.astype(str).str.strip().eq("").any()):
            raise ValueError(f"{label}: unique IDs, canonical columns and RangeIndex are required")
        price = pd.to_numeric(frame.price, errors="coerce")
        if price.isna().any() or not np.isfinite(price).all() or price.le(0).any():
            raise ValueError(f"{label}: targets must be finite positive numbers")
    if original_clean.columns.tolist() != corrected_clean.columns.tolist():
        raise ValueError("Corrected clean schema changed")
    provenance = [name for name in ["record_id", "source_row", "group", "dataset_period",
                  "original_row_index", "original_source_row", "url", "address", "collected_at"]
                  if name in original_clean]
    _same(original_clean, corrected_clean, provenance, "Original row identity, order or provenance changed")
    _same(original_clean, corrected_clean, ["price"], "Original targets changed")
    for expected, label in [(expected_holdout_rows, "holdout"), (expected_train_rows, "train")]:
        if expected is not None and (type(expected) is not int or expected < 1):
            raise ValueError(f"Expected {label} count must be a positive integer or None")

    combined_split = _split_by_row(
        historical_combined_split, corrected_clean,
        {"row_index", "source_row", "group", "subset"}, "historical combined split",
    )
    location_split = _split_by_row(
        historical_location_split, corrected_clean,
        {"row_index", "record_id", "group", "subset", "primary_evaluation"}, "historical location split",
    )
    _same(location_split, corrected_clean, ["record_id", "group"], "Historical location identity does not match original rows")
    _same(location_split, combined_split, ["subset", "group"], "Historical splits disagree on membership")
    primary = location_split.primary_evaluation
    if not pd.api.types.is_bool_dtype(primary.dtype) or primary.isna().any():
        raise ValueError("Historical primary_evaluation must contain explicit boolean flags")
    historical_train = combined_split.index[combined_split.subset.eq("train")].to_numpy(dtype=int)
    historical_test = combined_split.index[combined_split.subset.eq("test")].to_numpy(dtype=int)
    historical_eval = location_split.index[primary].to_numpy(dtype=int)
    if not np.array_equal(historical_eval, historical_test):
        raise ValueError("Historical primary evaluation must equal the entire frozen test cohort")
    _same(original_clean.loc[historical_test], corrected_clean.loc[historical_test], list(FEATURE_COLUMNS),
          "Frozen holdout base features changed")

    # The original combined split has source_row. The location split does not,
    # and its old house_key must never overwrite a reviewed identity.
    bundle = prepare_partition(corrected_clean, corrected_catalog, historical_combined_split)
    if bundle["exclusions"]:
        raise ValueError("Corrected identity would silently exclude frozen holdout rows: " + str(bundle["exclusions"][:5]))
    for key, expected in [("train_ids", historical_train), ("test_ids", historical_test), ("evaluation_ids", historical_eval)]:
        if not np.array_equal(bundle[key], expected):
            raise ValueError(f"Frozen {key} membership or order changed")
    if expected_train_rows is not None and len(historical_train) != expected_train_rows:
        raise ValueError("Frozen train row count differs from expectation")
    if expected_holdout_rows is not None and len(historical_eval) != expected_holdout_rows:
        raise ValueError("Frozen holdout row count differs from expectation")
    train_record_ids = corrected_clean.loc[historical_train, "record_id"].tolist()
    evaluation_record_ids = corrected_clean.loc[historical_eval, "record_id"].tolist()
    if set(train_record_ids) & set(evaluation_record_ids):
        raise ValueError("A frozen evaluation record entered training")
    if set(bundle["groups"].loc[historical_train]) & set(bundle["groups"].loc[historical_eval]):
        raise ValueError("Apartment groups overlap between train and holdout")

    if (not saved_errors.columns.is_unique or not {"record_id", "actual_price"}.issubset(saved_errors.columns)
            or saved_errors.record_id.isna().any() or not saved_errors.record_id.is_unique):
        raise ValueError("Saved errors need unique record IDs and actual prices")
    if saved_errors.record_id.tolist() != evaluation_record_ids:
        raise ValueError("Saved error IDs/order do not match frozen evaluation")
    expected_targets = corrected_clean.loc[historical_eval, ["price"]].rename(columns={"price": "actual_price"})
    _same(saved_errors, expected_targets, ["actual_price"], "Saved evaluation targets changed")
    return {
        **bundle,
        "verified_train_record_ids": train_record_ids,
        "verified_evaluation_record_ids": evaluation_record_ids,
        "checks": {
            "original_row_identity_order_targets_preserved": True,
            "holdout_base8_features_preserved": True,
            "historical_split_membership_preserved": True,
            "evaluation_ids_targets_equal_saved_errors": True,
            "no_holdout_exclusions": True,
            "apartment_groups_disjoint": True,
            "historical_house_keys_ignored": True,
            "hash_verification_required_by_caller": True,
        },
    }

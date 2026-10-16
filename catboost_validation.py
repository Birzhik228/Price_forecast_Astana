"""Deterministic nested house/apartment CV, using training identities only.

Arrays contain original DataFrame index labels, never positions in a fold.
Unknown houses remain separate unless an apartment-equivalence group links them.
This module does not inspect targets, fit estimators or read/write artifacts.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from location_experiment import house_validation_groups


MANIFEST_COLUMNS = ("phase", "outer_fold", "fold", "row_index", "record_id", "group")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _validate_inputs(frame: pd.DataFrame, groups: pd.Series) -> None:
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or not {"record_id", "house_key"}.issubset(frame.columns) or frame.empty):
        raise ValueError("Training frame requires nonempty rows, record_id and house_key.")
    if isinstance(frame.index, pd.MultiIndex) or not frame.index.is_unique or frame.index.hasnans:
        raise ValueError("Original row index must be unique, nonmissing and one-dimensional.")
    if not frame.record_id.map(_nonempty_text).all() or not frame.record_id.is_unique:
        raise ValueError("record_id must be unique, nonempty strings without surrounding whitespace.")
    houses = frame.house_key.dropna()
    if not houses.map(_nonempty_text).all():
        raise ValueError("house_key must contain nonempty strings or missing values.")
    if (not isinstance(groups, pd.Series) or not groups.index.equals(frame.index)
            or not groups.map(_nonempty_text).all()):
        raise ValueError("Groups must be nonempty strings aligned exactly with the original row index.")


def _split_count(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
        raise ValueError(f"{name} must be an integer >= 2.")
    return int(value)


def _make_folds(ids: pd.Index, groups: pd.Series, n_splits: int, phase: str) -> list[dict]:
    scoped_groups = groups.loc[ids]
    if scoped_groups.nunique() < n_splits:
        raise ValueError(f"{phase}: insufficient independent groups for {n_splits} folds.")
    splitter = GroupKFold(n_splits=n_splits)
    return [
        {"fold": number, "train_ids": ids.take(train).to_numpy(copy=True),
         "validation_ids": ids.take(validation).to_numpy(copy=True)}
        for number, (train, validation) in enumerate(
            splitter.split(np.zeros(len(ids), dtype=np.uint8), groups=scoped_groups.to_numpy()), 1
        )
    ]


def _ids(values: Any, scope: pd.Index, label: str) -> pd.Index:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{label}: expected a nonempty one-dimensional array of original row IDs.")
    result = pd.Index(values)
    if not result.is_unique or result.hasnans or not result.isin(scope).all():
        raise ValueError(f"{label}: duplicate, missing or out-of-scope row IDs.")
    return result


def _validate_level(folds: Any, scope: pd.Index, groups: pd.Series, label: str) -> None:
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError(f"{label}: expected at least two folds.")
    validation_ids = []
    for number, fold in enumerate(folds, 1):
        if (not isinstance(fold, dict) or isinstance(fold.get("fold"), (bool, np.bool_))
                or fold.get("fold") != number):
            raise ValueError(f"{label}: folds must have consecutive one-based numbers.")
        train = _ids(fold.get("train_ids"), scope, f"{label}/{number}/train")
        validation = _ids(fold.get("validation_ids"), scope, f"{label}/{number}/validation")
        if train.isin(validation).any() or len(train) + len(validation) != len(scope):
            raise ValueError(f"{label}/{number}: train and validation must partition every scoped row.")
        if set(groups.loc[train]) & set(groups.loc[validation]):
            raise ValueError(f"{label}/{number}: house/apartment group leakage between train and validation.")
        validation_ids.extend(validation.tolist())
    covered = pd.Index(validation_ids)
    if not covered.is_unique or len(covered) != len(scope) or not scope.isin(covered).all():
        raise ValueError(f"{label}: every scoped row must occur in validation exactly once.")


def _validate_bundle(frame: pd.DataFrame, folds: dict) -> None:
    if not isinstance(folds, dict) or not {"groups", "outer", "selection"}.issubset(folds):
        raise ValueError("Fold bundle requires groups, outer and selection.")
    groups = folds["groups"]
    _validate_inputs(frame, groups)
    # Also reject a mutated bundle that separates one explicit house into groups.
    known = frame.house_key.notna()
    if groups.loc[known].groupby(frame.loc[known, "house_key"]).nunique().gt(1).any():
        raise ValueError("One house_key is assigned to multiple validation groups.")
    _validate_level(folds["outer"], frame.index, groups, "outer")
    for outer in folds["outer"]:
        _validate_level(outer.get("inner"), pd.Index(outer["train_ids"]), groups,
                        f"inner of outer {outer['fold']}")
    _validate_level(folds["selection"], frame.index, groups, "selection")


def make_nested_folds(
    frame_train: pd.DataFrame,
    apartment_groups_train: pd.Series,
    outer_splits: int = 3,
    inner_splits: int = 3,
) -> dict:
    """Make shared-component outer CV, nested inner CV and final selection CV.

    The transitive closure of house keys and apartment groups is computed once
    from the supplied training rows. No target or held-out population is read.
    Selection uses ``inner_splits`` on the entire supplied training population.
    GroupKFold uses its deterministic, unshuffled policy at every level.
    """
    _validate_inputs(frame_train, apartment_groups_train)
    outer_splits = _split_count(outer_splits, "outer_splits")
    inner_splits = _split_count(inner_splits, "inner_splits")
    identity = frame_train.loc[:, ["record_id", "house_key"]].copy(deep=True)
    groups = house_validation_groups(identity, apartment_groups_train.copy(deep=True))
    groups.name = "validation_group"
    outer = _make_folds(frame_train.index, groups, outer_splits, "outer")
    for fold in outer:
        fold["inner"] = _make_folds(pd.Index(fold["train_ids"]), groups, inner_splits,
                                     f"inner of outer {fold['fold']}")
    bundle = {"groups": groups, "outer": outer,
              "selection": _make_folds(frame_train.index, groups, inner_splits, "selection")}
    _validate_bundle(frame_train, bundle)
    return bundle


def export_fold_manifest(frame_train: pd.DataFrame, folds: dict) -> pd.DataFrame:
    """Export validation membership only, after validating every full split.

    ``outer_fold`` identifies the parent only for phase=inner; it is nullable
    otherwise. Train is the complement of a fold's validation IDs in the full
    frame for outer/selection, or in its parent's outer train for inner folds.
    Each row occurs once per phase/parent scope. A 3x3 nested design plus final
    selection emits 4 * len(frame_train) rows, with no outer-validation rows in
    that parent's inner manifest.
    No files are written. Columns outside identity/group membership are omitted.
    """
    _validate_bundle(frame_train, folds)
    chunks = []

    def append(phase: str, outer_fold: int | None, entries: list[dict]) -> None:
        for entry in entries:
            ids = entry["validation_ids"]
            chunks.append(pd.DataFrame({
                "phase": phase, "outer_fold": outer_fold, "fold": entry["fold"],
                "row_index": ids, "record_id": frame_train.loc[ids, "record_id"].to_numpy(),
                "group": folds["groups"].loc[ids].to_numpy(),
            }))

    append("outer", None, folds["outer"])
    for outer in folds["outer"]:
        append("inner", outer["fold"], outer["inner"])
    append("selection", None, folds["selection"])
    result = pd.concat(chunks, ignore_index=True).loc[:, list(MANIFEST_COLUMNS)]
    result["outer_fold"] = result["outer_fold"].astype("Int64")
    return result

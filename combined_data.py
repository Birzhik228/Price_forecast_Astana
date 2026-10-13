"""Объединение января и марта с дедупликацией и защищённым мартовским test."""

from __future__ import annotations

from collections import Counter
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from clean import COLUMNS
from features import FEATURE_COLUMNS
from update_march import apartment_keys


JANUARY = "2026-01"
MARCH = "2026-03"
PROVENANCE_COLUMNS = (
    "dataset_period", "original_row_index", "original_source_row", "record_id",
)
RESERVED_COLUMNS = {*PROVENANCE_COLUMNS, "group", "row_index"}


def _prepare_input(frame: pd.DataFrame, period: str, offset: int) -> pd.DataFrame:
    """Сохранить исходные идентификаторы, присвоить глобальную позицию записи."""
    required = {"source_row", "price", *FEATURE_COLUMNS}
    if not frame.columns.is_unique or (missing := required - set(frame.columns)):
        raise ValueError(f"{period}: нужны уникальные колонки и все входы модели; отсутствуют {sorted(required - set(frame.columns))}.")
    if conflicts := set(frame.columns) & RESERVED_COLUMNS:
        raise ValueError(f"{period}: зарезервированные колонки уже существуют: {sorted(conflicts)}.")
    if not frame.index.is_unique or any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 0
        for value in frame.index
    ):
        raise ValueError(f"{period}: индекс должен содержать уникальные неотрицательные целые позиции исходного CSV.")
    sources = pd.to_numeric(frame["source_row"], errors="raise")
    if sources.isna().any() or not np.isfinite(sources).all() or (sources <= 0).any() or (sources % 1 != 0).any() or not sources.is_unique:
        raise ValueError(f"{period}: source_row должен содержать уникальные положительные целые номера исходных записей.")
    result = frame.copy(deep=True)
    for column in COLUMNS:
        if column not in result:
            result[column] = pd.NA
    result["dataset_period"] = period
    result["original_row_index"] = frame.index.to_numpy(dtype="int64")
    result["original_source_row"] = sources.to_numpy(dtype="int64")
    result["record_id"] = [f"{period}:{int(value)}" for value in sources]
    result["source_row"] = np.arange(offset + 1, offset + len(frame) + 1, dtype="int64")
    return result.reset_index(drop=True)


def _validate_march_manifest(march: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"row_index", "source_row", "group", "subset"}
    if not manifest.columns.is_unique or not required.issubset(manifest.columns):
        raise ValueError("Мартовский manifest должен содержать уникальные row_index, source_row, group и subset.")
    indexed = manifest.copy(deep=True)
    positions = pd.to_numeric(indexed["row_index"], errors="raise")
    if positions.isna().any() or not np.isfinite(positions).all() or (positions < 0).any() or (positions % 1 != 0).any() or not positions.is_unique:
        raise ValueError("В мартовском manifest должны быть уникальные целые row_index.")
    indexed["row_index"] = positions.astype("int64")
    indexed = indexed.set_index("row_index")
    if len(indexed) != len(march) or set(indexed.index) != set(march.index):
        raise ValueError("Мартовский manifest должен покрывать каждую строку March ровно один раз.")
    indexed = indexed.loc[march.index]
    expected_sources = pd.to_numeric(march["source_row"], errors="raise").to_numpy()
    manifest_sources = pd.to_numeric(indexed["source_row"], errors="raise").to_numpy()
    if not np.array_equal(expected_sources, manifest_sources):
        raise ValueError("Мартовские source_row не совпадают с зафиксированным manifest.")
    if indexed["subset"].isna().any() or not set(indexed["subset"]).issubset({"train", "test"}):
        raise ValueError("Мартовский subset должен быть train или test.")
    if not indexed["subset"].eq("test").any():
        raise ValueError("Зафиксированный мартовский test отсутствует.")
    if indexed["group"].isna().any() or not indexed["group"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Мартовский manifest должен содержать непустые строковые group.")
    if indexed.groupby("group")["subset"].nunique().gt(1).any():
        raise ValueError("Зафиксированная мартовская группа пересекает train и test.")
    return indexed


def _key_union(keys: list[set[str]], positions: list[int]) -> set[str]:
    result: set[str] = set()
    for position in positions:
        result.update(keys[position])
    return result


def build_combined(
    january: pd.DataFrame, march: pd.DataFrame, march_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Вернуть объединённую выборку и полный аудит без записи файлов.

    Источники не изменяются. Глобальный source_row — позиция записи в
    конкатенации январь→март до удаления дублей, начиная с 1. record_id
    связывает период и исходный source_row. frame и manifest имеют RangeIndex;
    reference_holdout сохраняет все исходные мартовские test-наблюдения,
    их исходный row_index и provenance, даже если часть из них — дубли.
    """
    jan = _prepare_input(january, JANUARY, 0)
    mar = _prepare_input(march, MARCH, len(january))
    original_manifest = _validate_march_manifest(march, march_manifest)
    stacked = pd.concat([jan, mar], ignore_index=True)
    original_subsets = ["january_reference"] * len(jan) + original_manifest["subset"].tolist()
    original_groups: list[str | None] = [None] * len(jan) + original_manifest["group"].tolist()
    protected = np.array([False] * len(jan) + original_manifest["subset"].eq("test").tolist(), dtype=bool)
    keys = [apartment_keys(row) for row in stacked.to_dict(orient="records")]
    parents = list(range(len(stacked)))

    def find(position: int) -> int:
        while parents[position] != position:
            parents[position] = parents[parents[position]]
            position = parents[position]
        return position

    owners: dict[str, int] = {}
    for position, row_keys in enumerate(keys):
        for key in sorted(row_keys):
            if key in owners:
                left, right = find(position), find(owners[key])
                parents[max(left, right)] = min(left, right)
            else:
                owners[key] = position
    members: dict[int, list[int]] = {}
    for position in range(len(stacked)):
        members.setdefault(find(position), []).append(position)
    group_names = {root: f"combined_{root + 1:06d}" for root in members}
    protected_roots = {root for root, positions in members.items() if protected[positions].any()}
    for root in protected_roots:
        positions = members[root]
        january_positions = [position for position in positions if position < len(jan)]
        if january_positions:
            record_ids = stacked.loc[positions, "record_id"].tolist()
            raise ValueError(
                "Загрязнение baseline: защищённая мартовская test-группа связана с январскими строками "
                f"({group_names[root]}: {', '.join(record_ids[:8])}). Обучение запрещено."
            )

    filled_fields = stacked.loc[:, list(FEATURE_COLUMNS)].notna().sum(axis=1).to_numpy()
    originals = stacked["original_source_row"].to_numpy()
    periods = stacked["dataset_period"].to_numpy()
    representatives: dict[int, int] = {}
    for root, positions in members.items():
        candidates = [position for position in positions if protected[position]] if root in protected_roots else positions
        representatives[root] = min(candidates, key=lambda position: (
            0 if periods[position] == MARCH else 1,
            -int(filled_fields[position]), int(originals[position]), int(stacked.iloc[position]["source_row"]),
        ))
    audit_records = []
    for position, row in stacked.iterrows():
        root = find(position)
        kept_position = representatives[root]
        kept = position == kept_position
        if kept:
            reason = "kept_protected_holdout" if root in protected_roots else "kept_group_representative"
        elif root in protected_roots:
            reason = "duplicate_protected_holdout"
        elif periods[position] != periods[kept_position]:
            reason = "duplicate_older_period"
        elif filled_fields[position] < filled_fields[kept_position]:
            reason = "duplicate_less_complete"
        else:
            reason = "duplicate_later_source_row"
        common = keys[position] & keys[kept_position]
        match_type = "representative" if kept else (
            "url" if any(key.startswith("url:") for key in common) else
            "signature" if common else "transitive"
        )
        audit_records.append({
            "source_row": int(row["source_row"]), "record_id": row["record_id"],
            "dataset_period": row["dataset_period"], "original_row_index": int(row["original_row_index"]),
            "original_source_row": int(row["original_source_row"]), "url": row["url"],
            "group": group_names[root], "group_size": len(members[root]),
            "original_subset": original_subsets[position], "original_march_group": original_groups[position],
            "is_reference_holdout": bool(protected[position]), "n_filled_features": int(filled_fields[position]),
            "match_key_count": len(keys[position]), "kept": kept, "reason": reason,
            "match_type": match_type, "kept_record_id": stacked.iloc[kept_position]["record_id"],
            "kept_dataset_period": str(periods[kept_position]),
        })
    audit = pd.DataFrame(audit_records)
    kept_positions = sorted(representatives.values())
    frame = stacked.iloc[kept_positions].copy().reset_index(drop=True)
    frame["group"] = [group_names[find(position)] for position in kept_positions]
    subsets = ["test" if find(position) in protected_roots else "train" for position in kept_positions]
    manifest = pd.DataFrame({
        "row_index": np.arange(len(frame), dtype="int64"), "subset": subsets,
        "group": frame["group"], "source_row": frame["source_row"],
        **{column: frame[column] for column in PROVENANCE_COLUMNS},
    })
    kept_set = set(kept_positions)
    rejected_positions = [position for position in range(len(stacked)) if position not in kept_set]
    rejected = stacked.iloc[rejected_positions].copy().reset_index(drop=True)
    for column in ("group", "reason", "kept_record_id", "match_type"):
        rejected[column] = audit.iloc[rejected_positions][column].to_numpy()
    reference_positions = np.flatnonzero(protected).tolist()
    reference_holdout = stacked.iloc[reference_positions].copy().reset_index(drop=True)
    reference_holdout["row_index"] = reference_holdout["original_row_index"]
    reference_holdout["group"] = [group_names[find(position)] for position in reference_positions]
    reference_holdout["original_march_group"] = [original_groups[position] for position in reference_positions]
    reference_holdout["subset"] = "test"

    train_positions = [position for position in kept_positions if find(position) not in protected_roots]
    test_positions = [position for position in kept_positions if find(position) in protected_roots]
    train_groups = set(manifest.loc[manifest["subset"].eq("train"), "group"])
    test_groups = set(manifest.loc[manifest["subset"].eq("test"), "group"])
    if not train_positions or not test_positions:
        raise ValueError("Для объединённого набора нужны непустые train и test.")
    if train_groups & test_groups:
        raise AssertionError("Группы train/test пересекаются.")
    if _key_union(keys, train_positions) & _key_union(keys, test_positions):
        raise AssertionError("Ключи выбранных train/test записей пересекаются.")
    if _key_union(keys, train_positions) & _key_union(keys, reference_positions):
        raise AssertionError("Train имеет совпадения с исходным мартовским holdout.")
    train_ids = set(stacked.iloc[train_positions]["record_id"])
    reference_ids = set(reference_holdout["record_id"])
    if train_ids & reference_ids:
        raise AssertionError("Исходная test-запись попала в train.")
    if not all(protected[position] for position in test_positions):
        raise AssertionError("Представитель protected-группы не является исходной test-строкой.")
    if len(frame) + len(rejected) != len(stacked) or not frame["source_row"].is_unique or not frame["record_id"].is_unique:
        raise AssertionError("Нарушен баланс записей или уникальность provenance.")
    if not set(manifest.loc[manifest["subset"].eq("test"), "record_id"]).issubset(reference_ids):
        raise AssertionError("Итоговый test должен быть подмножеством исходного мартовского holdout.")
    cross_period = {
        root for root, positions in members.items() if len(set(periods[positions])) > 1
    }
    rejected_audit = audit.loc[~audit["kept"]]
    summary = {
        "rows_january_input": len(january), "rows_march_input": len(march),
        "rows_before": len(stacked), "rows_after": len(frame), "rows_removed": len(rejected),
        "groups_total": len(members), "duplicate_groups": sum(len(positions) > 1 for positions in members.values()),
        "cross_period_groups": len(cross_period),
        "cross_period_input_rows": sum(len(members[root]) for root in cross_period),
        "input_rows_without_match_keys": sum(not row_keys for row_keys in keys),
        "duplicates_by_source": {period: int(rejected_audit["dataset_period"].eq(period).sum()) for period in (JANUARY, MARCH)},
        "duplicates_by_reason": dict(sorted(Counter(rejected_audit["reason"]).items())),
        "duplicates_by_match_type": dict(sorted(Counter(rejected_audit["match_type"]).items())),
        "kept_by_source": {period: int(frame["dataset_period"].eq(period).sum()) for period in (JANUARY, MARCH)},
        "train_rows": len(train_positions), "test_rows": len(test_positions),
        "train_groups": len(train_groups), "test_groups": len(test_groups),
        "train_rows_by_source": {
            period: int((manifest["dataset_period"].eq(period) & manifest["subset"].eq("train")).sum())
            for period in (JANUARY, MARCH)
        },
        "reference_holdout_rows": len(reference_holdout),
        "reference_holdout_original_groups": int(original_manifest.loc[original_manifest["subset"].eq("test"), "group"].nunique()),
        "reference_holdout_combined_groups": int(reference_holdout["group"].nunique()),
        "reference_holdout_duplicate_rows_removed": len(reference_holdout) - len(test_positions),
        "protected_groups_with_original_march_train_rows": sum(
            any(original_subsets[position] == "train" for position in members[root]) for root in protected_roots
        ),
        "selection_policy": "Protect original March test; otherwise prefer March, then most nonmissing of eight model inputs, then lowest original source_row. Price is never used for grouping or representative selection.",
        "source_row_policy": "Unique 1-based position in January followed by March before deduplication; original CSV row index and raw source_row are retained separately.",
        "checks": [
            "March manifest covers all March records and original source_row values match.",
            "Union-find groups are transitive over both periods using URL and price-independent apartment_keys.",
            "No protected holdout group contains any January record.",
            "Exactly one record per combined group; every protected representative is an original March test record.",
            "Train/test groups and identity keys are disjoint; no original March test record or key appears in train.",
            "All input records appear exactly once in the representative frame or rejected table.",
        ],
        "limitations": [
            "Signatures are heuristic apartment candidates, not verified property identifiers; similar flats can merge and incomplete or changed descriptions can hide repeats.",
            "Connectivity is transitive: two records may be grouped through a third record without sharing a direct key.",
            "March is the publisher-declared period; individual March observation dates are unavailable. Preference for March does not prove a later observation date for each apartment.",
            "This pool combines asking prices and different sources; it is not a current-market or completed-sale dataset.",
            "The March holdout has already been inspected in previous experiments. It is a fixed benchmark, not a new untouched estimate for further model choices.",
            "Primary test has one representative per protected group; the complete original holdout is retained separately for comparison on the unchanged observation set.",
            "All January records may enter train, including the former January test; former January test scores cannot evaluate the combined model independently.",
        ],
    }
    return {
        "frame": frame, "audit": audit, "rejected": rejected,
        "manifest": manifest, "reference_holdout": reference_holdout, "summary": summary,
    }

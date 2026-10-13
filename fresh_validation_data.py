"""Подготовка независимого временного holdout без обучения и записи файлов.

Это проверка содержимого, а не происхождения дат. Вызывающая программа обязана
подтвердить дату наблюдения неизменяемым manifest и сохранённым источником.
Основная оценка использует только result['frame'][primary_eligible]. Неполные
идентификационные сведения остаются во frame как отдельно обозначенная когорта.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime, timezone
from typing import Any, Iterable

import pandas as pd

from building_catalog import canonical_house_key
from clean import COLUMNS, CleaningConfig, clean_dataframe, normalize_district, normalize_text, normalize_url, parse_numeric
from update_march import apartment_keys


IDENTITY_NUMBERS = ("area", "rooms", "floor", "total_floors", "year_built")
CONTENT_REJECTION_REASONS = {"missing_price_or_area", "price_out_of_bounds", "area_out_of_bounds"}
AUDIT_COLUMNS = [
    "record_id", "source_row", "fresh_group", "kept_record_id", "decision", "reason",
    "primary_eligible", "identity_status", "signature_available", "canonical_signature_available",
    "valid_url", "observed_date", "observed_precision", "date_status", "date_verified",
    "matched_reference_names", "direct_reference_names", "direct_reference_keys",
    "history_connected", "transitive_history_match", "fresh_group_size", "cleaning_reason",
]


def _text(value: object) -> str | None:
    normalized = normalize_text(value)
    return None if pd.isna(normalized) else str(normalized)


def _date_value(value: object) -> tuple[date | None, datetime | None, str]:
    """Do not invent a timezone for a source date or naive timestamp."""
    text = _text(value)
    if text is None:
        return None, None, "missing"
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return date.fromisoformat(text), None, "date_only"
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            return moment.date(), None, "timezone_unspecified"
        moment = moment.astimezone(timezone.utc)
        return moment.date(), moment, "timestamp_utc"
    except (ValueError, OverflowError):
        return None, None, "invalid"


def _identity(row: dict[str, Any]) -> tuple[set[str], bool, bool, bool]:
    """Price/condition and all model errors are deliberately absent from keys."""
    normalized = {name: row.get(name) for name in ("address", "residential_complex", "url")}
    normalized["district"] = normalize_district(row.get("district"))
    for field in IDENTITY_NUMBERS:
        value = parse_numeric(row.get(field), "area" if field == "area" else "number")
        normalized[field] = value if math.isfinite(value) and value > 0 else None
    keys = apartment_keys(normalized)
    url = normalize_url(normalized.get("url"))
    valid_url = pd.notna(url)
    if valid_url:
        # An explicit ID catches leading-zero/path variants as the same portal ID.
        keys.add("krisha_id:" + str(int(str(url).rsplit("/", 1)[1])))
    complete = all(normalized[field] is not None for field in IDENTITY_NUMBERS)
    house = canonical_house_key(normalized.get("address"), normalized.get("district"), normalized.get("residential_complex"))
    canonical_available = bool(house and complete)
    if canonical_available:
        numbers = "|".join(format(float(normalized[field]), ".12g") for field in IDENTITY_NUMBERS)
        keys.add(f"canonical_apartment:{house}|{numbers}")
    return keys, any(key.startswith("signature:") for key in keys), canonical_available, bool(valid_url)


def prepare_fresh_holdout(
    raw: pd.DataFrame,
    reference_frames: dict[str, pd.DataFrame],
    raw_reference_urls: Iterable[str] = (),
    *,
    as_of: str,
    cutoff: str = "2026-03-31",
    source_observed_column: str = "collected_at",
) -> dict[str, Any]:
    """Clean, exclude historical connections, deduplicate; never fit or write.

    * ``as_of`` is required: ISO date or timezone-aware timestamp supplied by the
      caller. ``cutoff`` is an inclusive calendar date; observations must be later.
    * Input record_id values are retained when present, otherwise fresh:000001…
      IDs are assigned in source order. source_row/original_source_row are 1-based
      positions in this exact raw frame. All original columns survive as input__*.
    * All normalized rows, including generic-cleaner duplicates, participate in
      the identity graph. Bad rooms/dates/prices cannot win representative choice.
    * Any transitive link to history excludes the complete fresh component.
      Among eligible remaining rows the earliest source position wins, without
      using price, model predictions, errors or feature completeness for ranking.
    * Primary eligibility additionally requires a valid canonical Krisha URL and
      a full canonical house+area/rooms/floor/total_floors/year signature. It is
      conditional on external provenance verification, never proof of identity.
    """
    if not raw.columns.is_unique:
        raise ValueError("В fresh raw повторяются имена колонок.")
    if not isinstance(source_observed_column, str) or not source_observed_column:
        raise ValueError("Нужно имя колонки наблюдения.")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(cutoff)):
        raise ValueError("cutoff должен быть календарной ISO-датой.")
    lower, _, _ = _date_value(cutoff)
    upper, upper_moment, upper_precision = _date_value(as_of)
    if lower is None or upper is None or upper_precision == "timezone_unspecified" or lower >= upper:
        raise ValueError("Нужны корректные cutoff < as_of; as_of — ISO-дата или timestamp с часовым поясом.")
    raw_positions = list(range(1, len(raw) + 1))
    if "record_id" in raw:
        record_ids = raw["record_id"].map(_text).tolist()
        if any(value is None for value in record_ids) or len(set(record_ids)) != len(record_ids):
            raise ValueError("Переданные fresh record_id должны быть непустыми и уникальными.")
    else:
        record_ids = [f"fresh:{position:06d}" for position in raw_positions]
    # Keep the entire raw payload in a separate namespace before normalization.
    originals = raw.copy(deep=True).reset_index(drop=True).add_prefix("input__")
    originals["input_row_index"] = [str(value) for value in raw.index]
    adapted = raw.copy(deep=True).reset_index(drop=True)
    adapted["collected_at"] = (raw[source_observed_column].to_numpy()
                              if source_observed_column in raw else pd.NA)
    # Avoid ALIASES ambiguity only for the explicitly selected observation field.
    if "parsed_at" in adapted:
        adapted = adapted.drop(columns="parsed_at")
    if adapted.empty:
        for field in ("price", "area", "rooms"):
            if field not in adapted and not (field == "price" and "price_local" in adapted):
                adapted[field] = pd.Series(dtype="object")
    cleaned, generic_rejected, cleaning_report = clean_dataframe(adapted, CleaningConfig())
    # Generic cleaning's dedup includes price; reconstruct normalized records and
    # replace its dedup selection with the target-independent graph below.
    normalized = pd.concat([cleaned.assign(cleaning_reason=None),
                            generic_rejected.rename(columns={"reason": "cleaning_reason"})], ignore_index=True)
    normalized = normalized.sort_values("source_row", kind="stable").reset_index(drop=True)
    if normalized["source_row"].tolist() != raw_positions:
        raise AssertionError("Очиститель не сохранил ровно одну запись для каждой позиции raw.")
    rows = pd.concat([normalized, originals], axis=1)
    rows["record_id"] = record_ids
    rows["original_source_row"] = raw_positions
    rows["date_verified"] = False  # Only an external immutable source manifest can verify this.

    fresh_keys, legacy_available, canonical_available, valid_urls = [], [], [], []
    observed_days, observed_precisions, date_statuses, quality_reasons = [], [], [], []
    for row in rows.to_dict(orient="records"):
        keys, legacy, canonical, valid_url = _identity(row)
        fresh_keys.append(keys); legacy_available.append(legacy)
        canonical_available.append(canonical); valid_urls.append(valid_url)
        day, moment, precision = _date_value(row["collected_at"])
        date_status = ("observed_date_missing" if precision == "missing" else
                       "observed_date_invalid" if day is None else
                       "observation_not_after_cutoff" if day <= lower else
                       "observation_after_as_of" if day > upper or (moment is not None and upper_moment is not None and moment > upper_moment) else
                       "valid_date_value")
        observed_days.append(day.isoformat() if day else None)
        observed_precisions.append(precision); date_statuses.append(date_status)
        reason = ("invalid_required_rooms" if pd.isna(row["rooms"]) else
                  str(row["cleaning_reason"]) if row["cleaning_reason"] in CONTENT_REJECTION_REASONS else
                  date_status if date_status != "valid_date_value" else None)
        quality_reasons.append(reason)
    rows["observed_date"] = observed_days
    rows["observed_precision"] = observed_precisions
    rows["date_status"] = date_statuses
    rows["signature_available"] = pd.Series(legacy_available, dtype=bool)
    rows["canonical_signature_available"] = pd.Series(canonical_available, dtype=bool)
    rows["valid_url"] = pd.Series(valid_urls, dtype=bool)

    all_keys = list(fresh_keys)
    reference_names: list[str | None] = [None] * len(rows)
    reference_summary = {}
    if any(not isinstance(name, str) or not name for name in reference_frames):
        raise ValueError("Источники reference должны иметь непустые строковые имена.")
    for name in sorted(reference_frames):
        reference = reference_frames[name]
        if not isinstance(reference, pd.DataFrame) or not reference.columns.is_unique:
            raise ValueError(f"Нужен reference DataFrame с уникальными колонками: {name}")
        details = [_identity(row) for row in reference.to_dict(orient="records")]
        all_keys.extend(item[0] for item in details)
        reference_names.extend([name] * len(details))
        reference_summary[name] = {
            "rows": len(reference), "valid_url_rows": sum(item[3] for item in details),
            "legacy_signature_rows": sum(item[1] for item in details),
            "canonical_signature_rows": sum(item[2] for item in details),
            "rows_without_identity_keys": sum(not item[0] for item in details),
        }
    raw_url_keys: set[str] = set()
    raw_url_records, raw_invalid_urls = 0, 0
    for value in raw_reference_urls:
        raw_url_records += 1
        keys, _, _, valid = _identity({"url": value})
        if not valid:
            raw_invalid_urls += 1
            continue
        raw_url_keys.update(keys)
        all_keys.append(keys)
        reference_names.append("raw_reference_urls")
    if len(all_keys) == len(rows):
        raise ValueError("Для независимой проверки необходима непустая прежняя reference-история.")
    if not any(all_keys[len(rows):]):
        raise ValueError("В прежней истории нет ни одного пригодного identity-ключа.")

    parents = list(range(len(all_keys)))

    def find(position: int) -> int:
        while parents[position] != position:
            parents[position] = parents[parents[position]]
            position = parents[position]
        return position

    owners: dict[str, int] = {}
    key_references: dict[str, set[str]] = {}
    for position, keys in enumerate(all_keys):
        for key in sorted(keys):
            if key in owners:
                left, right = find(position), find(owners[key])
                parents[max(left, right)] = min(left, right)
            else:
                owners[key] = position
            if reference_names[position] is not None:
                key_references.setdefault(key, set()).add(reference_names[position])
    component_history: dict[int, set[str]] = {}
    for position in range(len(rows), len(all_keys)):
        component_history.setdefault(find(position), set()).add(str(reference_names[position]))
    members: dict[int, list[int]] = {}
    for position in range(len(rows)):
        members.setdefault(find(position), []).append(position)
    representatives = {}
    for root, positions in members.items():
        eligible = [position for position in positions if quality_reasons[position] is None]
        if eligible and root not in component_history:
            representatives[root] = min(eligible)  # Original raw order, never price or predictions.

    audit = []
    for position, row in rows.iterrows():
        root = find(position)
        chosen = representatives.get(root)
        direct_keys = sorted(fresh_keys[position] & key_references.keys())
        direct_names = sorted({name for key in direct_keys for name in key_references[key]})
        history = component_history.get(root, set())
        identity_status = ("invalid_or_missing_url" if not valid_urls[position] else
                           "canonical_signature_no_known_overlap" if canonical_available[position] else
                           "legacy_signature_only" if legacy_available[position] else "insufficient_identity")
        reason = (quality_reasons[position] or
                  ("known_history_component" if history else
                   "duplicate_fresh_group" if chosen != position else None))
        if history:
            identity_status = "known_history_overlap"
        primary = reason is None and valid_urls[position] and canonical_available[position]
        audit.append({
            "record_id": row["record_id"], "source_row": int(row["source_row"]),
            "fresh_group": f"fresh_group_{min(members[root]) + 1:06d}",
            "kept_record_id": record_ids[chosen] if chosen is not None else None,
            "decision": "retained_primary_candidate" if primary else "retained_secondary" if reason is None else "rejected",
            "reason": reason, "primary_eligible": bool(primary), "identity_status": identity_status,
            "signature_available": legacy_available[position], "canonical_signature_available": canonical_available[position],
            "valid_url": valid_urls[position], "observed_date": observed_days[position],
            "observed_precision": observed_precisions[position], "date_status": date_statuses[position], "date_verified": False,
            "matched_reference_names": ";".join(sorted(history)), "direct_reference_names": ";".join(direct_names),
            "direct_reference_keys": "\n".join(direct_keys), "history_connected": bool(history),
            "transitive_history_match": bool(history and not direct_keys), "fresh_group_size": len(members[root]),
            "cleaning_reason": row["cleaning_reason"],
        })
    audit_frame = pd.DataFrame(audit, columns=AUDIT_COLUMNS)
    for column in ("primary_eligible", "signature_available", "canonical_signature_available", "valid_url",
                   "date_verified", "history_connected", "transitive_history_match"):
        audit_frame[column] = audit_frame[column].astype(bool)
    for column in ("fresh_group", "kept_record_id", "reason", "primary_eligible", "identity_status"):
        rows[column] = audit_frame[column].to_numpy()
    retained = rows.loc[audit_frame["decision"].ne("rejected")].copy().reset_index(drop=True)
    rejected = rows.loc[audit_frame["decision"].eq("rejected")].copy().reset_index(drop=True)
    retained_positions = audit_frame.index[audit_frame["decision"].ne("rejected")].tolist()
    seen: set[str] = set()
    for position in retained_positions:
        if fresh_keys[position] & set(key_references) or fresh_keys[position] & seen:
            raise AssertionError("У оставленных строк пересекаются доступные ключи с историей или другой группой.")
        seen.update(fresh_keys[position])
    if len(retained) + len(rejected) != len(raw) or not rows["record_id"].is_unique:
        raise AssertionError("Нарушен баланс raw/retained/rejected либо уникальность record_id.")
    summary = {
        "schema_version": 1, "rows_before": len(raw), "rows_after": len(retained), "rows_rejected": len(rejected),
        "primary_candidate_rows": int(retained["primary_eligible"].sum()),
        "secondary_rows": int((~retained["primary_eligible"].astype(bool)).sum()),
        "rejected_by_reason": dict(sorted(Counter(rejected["reason"].dropna()).items())),
        "identity_status_counts_retained": dict(sorted(Counter(retained["identity_status"]).items())),
        "date_status_counts": dict(sorted(Counter(date_statuses).items())),
        "observed_precision_counts": dict(sorted(Counter(observed_precisions).items())),
        "history_connected_rows": int(audit_frame["history_connected"].sum()),
        "transitive_history_match_rows": int(audit_frame["transitive_history_match"].sum()),
        "fresh_groups": len(members), "reference_frames": reference_summary,
        "raw_reference_url_records": raw_url_records, "raw_reference_invalid_urls": raw_invalid_urls,
        "raw_reference_unique_urls": sum(key.startswith("url:") for key in raw_url_keys),
        "cutoff_inclusive": cutoff, "as_of": str(as_of), "source_observed_column": source_observed_column,
        "source_dates_verification_required": True, "source_dates_verified_by_this_module": False,
        "original_columns": list(raw.columns), "original_column_prefix": "input__", "cleaning_report": cleaning_report,
        "policy": {
            "reference_scope": "Caller must include every frozen-model training pool; full previous January/March snapshots are a conservative superset.",
            "primary": "Valid observed-date value and required content; valid Krisha URL; complete canonical house+area/rooms/floor/total_floors/year signature; no known history component.",
            "representative": "First content/date-eligible original raw position in each nonhistorical transitive component.",
            "keys": "Canonical Krisha URL/ID, legacy apartment_keys and canonical house signature; never price, condition or model output.",
            "provenance": "All original fields retained under input__*, positions under source_row/original_source_row, caller IDs preserved.",
        },
        "limitations": [
            "No known-key overlap is not proof that physical apartments are independent: new IDs, address errors and incomplete/changed traits can hide repeats.",
            "Legacy address tokens erase punctuation and complex signatures may match different apartments; exclusion is intentionally conservative.",
            "Unclear addresses or missing structural fields are secondary only. This restriction changes the population represented by the primary score.",
            "Observation-date strings do not prove collection time; verify immutable source evidence and provenance before evaluating any candidate.",
            "Naive source timestamps retain unknown timezone and are compared as declared calendar dates; no current date is substituted.",
            "Prices are historical asking prices, not transaction values. No imputation, learned trimming, fitting or model-based record selection occurs.",
        ],
    }
    return {"frame": retained, "rejected": rejected, "audit": audit_frame, "summary": summary}

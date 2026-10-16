"""Evidence-backed corrections in a new archive version; no training or activation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from address_identity_audit import load_numbered_street_evidence
from building_catalog import build_property_catalog, canonical_house_key
from clean import COLUMNS, DISTRICTS, clean_dataframe, normalize_district, normalize_text, normalize_url
from config import APP_MODEL_METADATA_PATH, APP_MODEL_PIPELINE_PATH
from march_data import _clean_preserving_positions, adapt_march_schema


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data/processed/reviewed_2026_10_16"
DEFAULT_REPORTS = ROOT / "reports/data_quality_2026_10_16"
COMBINED = "data/processed/combined_2026/listings_clean.csv"
CATALOG = "data/processed/property_catalog/listings.csv"
JANUARY = "data/raw/kaggle_astana/astana_apartments.csv"
MARCH = "data/raw/turarr_march_2026/astana_apartments_raw.csv"
SPLIT = "reports/location_experiment/split_manifest.csv"
DEDUP = "reports/combined_update/deduplication_audit.csv"
OSM_MANIFEST = "data/raw/osm_2026_10_13/astana_addresses.source.json"
POLICY = "source_evidence_only_v1"
CHANGE_COLUMNS = ["record_id", "dataset_period", "original_source_row", "subset", "field",
                  "old_value", "new_value", "rule", "source_field", "source_value", "url"]
REVIEW_COLUMNS = ["record_id", "subset", "issue", "evidence", "decision", "url"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text(value) -> str | None:
    normalized = normalize_text(value)
    return None if pd.isna(normalized) else str(normalized)


def explicit_district_address(value) -> str | None:
    """Only an entire district label with an explicit marker, never a street name."""
    value = text(value)
    if value is None:
        return None
    value = value.casefold()
    match = re.fullmatch(r"(?:(?:район|р-н)\s+(.+)|(.+?)\s+(?:район|р-н))", value)
    if not match:
        return None
    district = normalize_district(match.group(1) or match.group(2))
    return str(district) if pd.notna(district) and district in set(DISTRICTS.values()) else None


def explicit_complex_address(value) -> str | None:
    value = text(value)
    if value is None:
        return None
    match = re.fullmatch(r"жк\s+([а-яёәғқңөұүһіa-z]+)", value.casefold())
    if not match or match[1] in {"рядом", "возле", "напротив", "неизвестно", "неизвестный", "название", "нет"}:
        return None
    return match[1]


def verify_replay(combined: pd.DataFrame, january: pd.DataFrame, march: pd.DataFrame) -> pd.DataFrame:
    """Rebuild historical source parsing and verify every retained original field."""
    jan, _, _ = clean_dataframe(january)
    adapted = adapt_march_schema(march)
    rejected = set(adapted.attrs["adapter_rejections"])
    mar, _, _ = _clean_preserving_positions(adapted.loc[~adapted.source_row.isin(rejected)].copy())
    parts = []
    for period, frame in [("2026-01", jan), ("2026-03", mar)]:
        frame = frame.copy()
        frame["record_id"] = period + ":" + frame.source_row.astype(str)
        parts.append(frame)
    replay = pd.concat(parts, ignore_index=True).set_index("record_id", verify_integrity=True)
    if not combined.record_id.is_unique or not combined.record_id.isin(replay.index).all():
        raise ValueError("Combined provenance is missing or duplicated in source replay")
    expected = replay.loc[combined.record_id].reset_index(drop=True)
    for field in COLUMNS:
        actual_field = "original_source_row" if field == "source_row" else field
        a = combined[actual_field].reset_index(drop=True)
        b = expected[field]
        if field in {"source_row", "price", "area", "rooms", "floor", "total_floors", "year_built"}:
            same = np.isclose(pd.to_numeric(a).to_numpy(dtype=float, na_value=np.nan),
                              pd.to_numeric(b).to_numpy(dtype=float, na_value=np.nan),
                              rtol=0, atol=0, equal_nan=True)
        else:
            same = a.map(text).fillna("").eq(b.map(text).fillna("")).to_numpy()
        if not same.all():
            ids = combined.loc[~same, "record_id"].head(5).tolist()
            raise ValueError(f"Source replay mismatch for {field}: {ids}")
    return replay


def correct_source_fields(combined: pd.DataFrame, january: pd.DataFrame, split: pd.DataFrame):
    required = {"record_id", "dataset_period", "original_source_row", "district", "address", "url", "residential_complex"}
    if not required.issubset(combined) or not combined.record_id.is_unique:
        raise ValueError("Missing or duplicate combined provenance")
    if not split.record_id.is_unique or set(split.record_id) != set(combined.record_id):
        raise ValueError("Split must cover every combined record exactly once")
    if split.subset.isna().any() or not split.subset.isin(["train", "test"]).all():
        raise ValueError("Unknown split subset")
    subsets = split.set_index("record_id").subset
    result = combined.copy(deep=True)
    changes, review = [], []
    for index, row in combined.loc[combined.dataset_period.eq("2026-01")].iterrows():
        position = row.original_source_row
        if not np.isfinite(position) or position != int(position) or not 1 <= position <= len(january):
            raise ValueError("Invalid original source row")
        if row.record_id != f"2026-01:{int(position)}":
            raise ValueError("record_id/source_row mismatch")
        raw = january.iloc[int(position) - 1]
        raw_url, row_url = text(normalize_url(raw.url)), text(normalize_url(row.url))
        if raw_url is None or raw_url != row_url:
            raise ValueError(f"URL/source row mismatch: {row.record_id}")
        complex_name = explicit_complex_address(raw.address)
        if complex_name is not None and text(row.residential_complex) is None:
            raw_complex = text(raw.get("residential_complex_name"))
            if raw_complex is not None and raw_complex.casefold() != complex_name:
                raise ValueError(f"Conflicting raw complex: {row.record_id}")
            result.loc[index, "residential_complex"] = complex_name
            changes.append({"record_id": row.record_id, "dataset_period": row.dataset_period,
                "original_source_row": int(position), "subset": subsets[row.record_id],
                "field": "residential_complex", "old_value": None, "new_value": complex_name,
                "rule": "missing_complex_from_own_explicit_complex_only_address",
                "source_field": "january_raw.address", "source_value": str(raw.address), "url": row.url})
        found = explicit_district_address(raw.address)
        if found is None:
            continue
        existing = text(row.district)
        if existing is None:
            # Do not override an explicit contradictory raw district lost elsewhere.
            raw_district = normalize_district(raw.district)
            if pd.notna(raw_district) and raw_district != found:
                raise ValueError(f"Conflicting raw district: {row.record_id}")
            result.loc[index, "district"] = found
            changes.append({"record_id": row.record_id, "dataset_period": row.dataset_period,
                "original_source_row": int(position), "subset": subsets[row.record_id],
                "field": "district", "old_value": None, "new_value": found,
                "rule": "missing_district_from_own_explicit_district_only_address",
                "source_field": "january_raw.address", "source_value": str(raw.address), "url": row.url})
        elif existing != found:
            review.append({"record_id": row.record_id, "subset": subsets[row.record_id],
                "issue": "district_conflicts_with_explicit_address_label",
                "evidence": f"district={existing}; raw.address={raw.address}",
                "decision": "retained_no_value_selected", "url": row.url})
    return result, pd.DataFrame(changes, columns=CHANGE_COLUMNS), pd.DataFrame(review, columns=REVIEW_COLUMNS)


def possible_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact complete signatures are review candidates, not proven apartment IDs."""
    columns = ["record_id", "group", "house_key", "rooms", "area", "floor", "total_floors", "year_built", "price", "url"]
    work = frame.copy()
    work["house_key"] = [canonical_house_key(r.address, r.district, r.residential_complex)
                         for r in work.itertuples()]
    fields = ["house_key", "rooms", "area", "floor", "total_floors", "year_built"]
    complete = work.dropna(subset=fields)
    candidates = complete.loc[complete.duplicated(fields, keep=False), columns].copy()
    if len(candidates):
        candidates["review_group"] = candidates.groupby(fields, sort=True).ngroup() + 1
    else:
        candidates["review_group"] = pd.Series(dtype="int64")
    candidates["decision"] = "retain_distinct_urls_identity_unconfirmed"
    return candidates.sort_values(["review_group", "record_id"])


def removed_address_conflicts(replay: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Expose earlier signature deletions with different houses on the same street."""
    from building_catalog import _HOUSE_ADDRESS, normalize_address
    columns = ["removed_record_id", "kept_record_id", "group", "original_subset", "match_type",
               "removed_address", "kept_address", "removed_house_key", "kept_house_key",
               "removed_url", "kept_url", "decision"]
    rows = []
    if not audit.kept.isin([True, False]).all():
        raise ValueError("Invalid dedup kept flag")
    for entry in audit.loc[~audit.kept].itertuples():
        old, kept = replay.loc[entry.record_id], replay.loc[entry.kept_record_id]
        if text(normalize_url(old.url)) == text(normalize_url(kept.url)):
            continue  # Same listing URL changing address is not a signature-only merge.
        left, ls = normalize_address(old.address, old.residential_complex)
        right, rs = normalize_address(kept.address, kept.residential_complex)
        if ls != "house" or rs != "house" or text(old.district) != text(kept.district) or text(old.district) is None:
            continue
        lm, rm = _HOUSE_ADDRESS.fullmatch(left), _HOUSE_ADDRESS.fullmatch(right)
        if lm["street"] != rm["street"] or (lm["house"], lm["unit"]) == (rm["house"], rm["unit"]):
            continue
        rows.append({"removed_record_id": entry.record_id, "kept_record_id": entry.kept_record_id,
            "group": entry.group, "original_subset": entry.original_subset, "match_type": entry.match_type,
            "removed_address": old.address, "kept_address": kept.address,
            "removed_house_key": canonical_house_key(old.address, old.district, old.residential_complex),
            "kept_house_key": canonical_house_key(kept.address, kept.district, kept.residential_complex),
            "removed_url": old.url, "kept_url": kept.url,
            "decision": "review_previous_deletion_do_not_restore_to_train_automatically"})
    return pd.DataFrame(rows, columns=columns)


def withdraw_street_only_house_keys(listings: pd.DataFrame, buildings: pd.DataFrame, evidence: pd.DataFrame):
    """Withdraw derived identities only; preserve the original address and raw GPS."""
    if not evidence.record_id.is_unique or not evidence.record_id.isin(listings.record_id).all():
        raise ValueError("Street evidence must refer to unique existing records")
    revised, houses = listings.copy(deep=True), buildings.copy(deep=True)
    mapped = revised.set_index("record_id").house_key
    for row in evidence.itertuples():
        if mapped[row.record_id] != row.old_house_key:
            raise ValueError("Street evidence refers to a different house key")
    affected = revised.record_id.isin(evidence.record_id)
    old_keys = set(evidence.old_house_key)
    # A mixed key must not erase a separate complete house record.
    if revised.loc[~affected, "house_key"].isin(old_keys).any():
        raise ValueError("Street-only key also belongs to an unreviewed record")
    revised.loc[affected, ["house_key", "building_id", "candidate_building_id"]] = pd.NA
    revised.loc[affected, "address_status"] = "street_only_osm_evidence"
    revised.loc[affected, "building_match_status"] = "address_unresolved"
    revised.loc[affected, "building_conflict_reasons"] = pd.NA
    revised.loc[affected, "coordinate_spread_m"] = np.nan
    revised.loc[affected, "latitude"] = revised.loc[affected, "raw_latitude"]
    revised.loc[affected, "longitude"] = revised.loc[affected, "raw_longitude"]
    revised.loc[affected, "coordinate_status"] = revised.loc[affected, "raw_coordinate_status"]
    revised.loc[affected, "coordinate_source"] = revised.loc[affected, "raw_coordinate_source"]
    houses = houses.loc[~houses.house_key.isin(old_keys)].copy()
    return revised, houses


def review_top_errors(errors: pd.DataFrame, march: pd.DataFrame, catalog: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Link top errors to original text; risk flags are not causal diagnoses."""
    top = errors.sort_values(["absolute_error", "record_id"], ascending=[False, True]).head(30).copy()
    train_ids = set(split.loc[split.subset.eq("train"), "record_id"])
    known_counts = catalog.loc[catalog.record_id.isin(train_ids), "residential_complex"].value_counts()
    rows = []
    for item in top.itertuples():
        if item.dataset_period != "2026-03":
            raise ValueError("Only archived March holdout expected")
        position = item.original_source_row
        if (not np.isfinite(position) or position != int(position) or not 1 <= position <= len(march)
                or item.record_id != f"2026-03:{int(position)}"):
            raise ValueError("Top error record_id/source row mismatch")
        matches = catalog.loc[catalog.record_id.eq(item.record_id)]
        if len(matches) != 1:
            raise ValueError("Top error must match exactly one catalog record")
        original = matches.iloc[0]
        for error_field, original_field in [("actual_price", "price"), ("area", "area"), ("rooms", "rooms"),
                                             ("floor", "floor"), ("total_floors", "total_floors"), ("year_built", "year_built")]:
            value, expected = getattr(item, error_field), original[original_field]
            if not ((pd.isna(value) and pd.isna(expected)) or (pd.notna(value) and pd.notna(expected) and value == expected)):
                raise ValueError(f"Top error source mismatch: {error_field}")
        for field in ["district", "building_type", "condition", "residential_complex", "address"]:
            if text(getattr(item, field)) != text(original[field]):
                raise ValueError(f"Top error source mismatch: {field}")
        source = march.iloc[int(item.original_source_row) - 1]
        if text(normalize_url(source.url)) != text(normalize_url(item.url)):
            raise ValueError("Top error raw URL/provenance mismatch")
        flags = []
        if item.area > 120: flags.append("area_over_120m2")
        if item.filled_base_fields < 8: flags.append("incomplete_base_fields")
        if not item.coordinate_available: flags.append("no_usable_geography")
        if pd.notna(item.building_conflict_reasons): flags.append("conflicting_house_traits")
        count = int(known_counts.get(item.residential_complex, 0))
        if count < 30: flags.append("complex_less_than_30_training_rows")
        rows.append({"record_id": item.record_id, "original_source_row": int(item.original_source_row),
            "url": item.url, "actual_price": item.actual_price, "predicted_price": item.predicted_price,
            "absolute_error": item.absolute_error, "signed_error": item.signed_error, "ape_pct": item.ape_pct,
            "district": item.district, "area": item.area, "complex": item.residential_complex,
            "train_complex_rows": count, "filled_base_fields": item.filled_base_fields,
            "risk_flags": ";".join(flags), "raw_title": source.title, "raw_price": source.price,
            "raw_area": source["Площадь"], "raw_floor": source["Этаж"], "raw_year": source["Год постройки"],
            "raw_city": source["Город"], "raw_complex": source["Жилой комплекс"],
            "finding": "source_parsing_reproduced; flags_do_not_prove_cause_or_wrong_source",
            "decision": "retain_price_and_features_no_confirmed_value_replacement"})
    return pd.DataFrame(rows)


def building_review(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    part = catalog.loc[catalog.building_match_status.eq("ambiguous_conflict")]
    for key, group in part.groupby("house_key", sort=True):
        values = lambda name: json.dumps(sorted(group[name].dropna().unique().tolist(), key=str), ensure_ascii=False)
        rows.append({"house_key": key, "listing_count": len(group), "record_ids": values("record_id"),
            "districts": values("district"), "complex_names": values("complex_name"),
            "years": values("year_built"), "total_floors": values("total_floors"),
            "building_types": values("building_type"), "conflict_reasons": values("building_conflict_reasons"),
            "decision": "retain_individual_records_no_majority_imputation"})
    return pd.DataFrame(rows)


def save_json(path: Path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require_new_outputs(root: Path, outputs: list[Path]):
    for output in outputs:
        if not output.is_relative_to(root) or output == root:
            raise ValueError("Output must be a new directory inside the project")
        for protected in [root / "data/raw", root / "models", root / "data/processed/property_catalog", root / "data/processed/combined_2026"]:
            if output.is_relative_to(protected) or protected.is_relative_to(output):
                raise ValueError("Output overlaps protected inputs")
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output already contains files: {output}")
    if outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        raise ValueError("Data/report directories must be separate")


def render_report(report: dict, changes: pd.DataFrame) -> str:
    counts = report["summary"]
    links = {"corrections.csv": "Все подтверждённые исправления", "review_queue.csv": "Сомнительные значения",
             "possible_duplicates.csv": "Похожие квартиры с разными ссылками",
             "previous_dedup_address_conflicts.csv": "Пересмотр прежнего удаления дублей",
             "building_review_queue.csv": "Конфликты характеристик домов", "numbered_street_evidence.csv": "Ошибочные привязки номерных улиц к домам",
             "top_error_review.csv": "Разбор 30 крупнейших ошибок по исходным полям", "report.json": "Итоги и контрольные суммы"}
    items = "".join(f'<li><a href="{name}">{label}</a></li>' for name, label in links.items())
    table = changes.loc[:, ["record_id", "field", "old_value", "new_value", "source_value"]].fillna("—").to_html(index=False, escape=True)
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка архивных данных · 16 октября 2026</title>
<style>body{{font:16px/1.6 system-ui;background:#f4f6fa;color:#17243c;margin:0}}main{{max-width:1150px;margin:32px auto;background:white;padding:32px;border-radius:16px}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dbe1ec;text-align:left}}.note{{padding:16px;background:#edf3ff;border-radius:8px}}a{{color:#1c58b6}}.scroll{{overflow-x:auto}}</style>
<main><h1>Исправления архивных данных</h1><p>16 октября 2026 · январские и мартовские объявления</p>
<p class="note">Изменённые данные сохранены отдельно. Исходные файлы, рабочая модель, её каталог и результаты проверки не перезаписаны. Переобучение и переключение приложения не выполнялись.</p>
<h2>Подтверждённые исправления</h2><p>Строк: <b>{counts['rows']:,}</b>. Исправлений района: <b>{counts['district_corrections']}</b>, ЖК: <b>{counts['complex_corrections']}</b>. Пропусков района: {counts['district_missing_before']} → {counts['district_missing_after']}. Изменённых строк прежнего test: {counts['holdout_changed_rows']}.</p>
<p>Район восстановлен только из собственного исходного адреса, целиком состоящего из названия района и явного «р-н»/«район». Одиночное «Сарайшык» может быть улицей и не используется. ЖК восстанавливается только из полного явного «ЖК + имя», без упоминаний соседних объектов. Номер дома не придумывается, цена и другие характеристики сохраняются.</p>
<div class="scroll">{table}</div>
<h2>Номерные улицы</h2><p>У {counts['street_only_records_corrected']} записей сняты ошибочные ключи домов ({counts['street_only_house_keys_withdrawn']} ключей): адрес содержит только название номерной улицы, подтверждённое сохранёнными названиями дорог OSM. Например, «Е 915» — название улицы; дом №915 на улице «Е» из этой записи не следует. Адрес и собственные исходные координаты сохранены; координаты конкретного дома не присвоены.</p>
<p>Данные карты: © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>, <a href="https://opendatacommons.org/licenses/odbl/1-0/">ODbL 1.0</a>. Снимок октября 2026 года. Совпадение имени дороги не подтверждает, что квартира находится именно на этой дороге.</p>
<h2>Что осталось на проверку</h2><p>Конфликтных адресных групп: {counts['unresolved_building_groups']}. Похожих записей по полной адресной сигнатуре: {counts['possible_duplicate_rows']}. Ранее удалённых записей с отличающимся номером дома на той же улице: {counts['previous_deletions_with_address_conflict']}.</p>
<p>Совпадение характеристик без номера квартиры не доказывает дубль. Различие адресов в старой группе требует пересмотра, но не доказывает, что оба исходных адреса правильны. Такие записи сохранены в очереди; они не удаляются и не возвращаются в train автоматически. Большая ошибка модели, необычная цена или площадь сами по себе не являются основанием исправить или убрать объявление.</p>
<h2>Файлы и воспроизведение</h2><ul>{items}</ul><p>Новая версия: <code>{html.escape(report['data_directory'])}</code>. Её split сохраняет прежние ID и распределение train/test. Цены всех строк неизменны.</p>
<p>Все исходные очищенные поля сверены с повторным разбором CSV; это проверяет перенос данных, но не истинность заявлений продавца. У 30 крупнейших ошибок числовые значения совпали с исходником; без независимого подтверждения цены и площади не исправлялись. Подробные признаки риска даны в отдельной таблице, они не доказывают причины ошибок.</p>
<p>Для обучения использовать новую версию отдельным экспериментом, с повторной проверкой групп внутри train. Файл reference_split_manifest.csv сохраняет историческое разбиение и старые ключи; его нельзя считать новым проверенным протоколом обучения. Старый мартовский test уже изучен и не подтверждает точность сегодняшнего рынка.</p></main></html>'''


def run(data_dir: Path = DEFAULT_DATA, reports_dir: Path = DEFAULT_REPORTS, root: Path = ROOT) -> dict:
    root, data_dir, reports_dir = root.resolve(), data_dir.resolve(), reports_dir.resolve()
    require_new_outputs(root, [data_dir, reports_dir])
    plan = json.loads((root / "reports/location_experiment/experiment_plan.json").read_text(encoding="utf-8"))
    paths = [COMBINED, CATALOG, JANUARY, MARCH, SPLIT, DEDUP, OSM_MANIFEST,
             "data/raw/osm_2026_10_13/astana_addresses.json", "address_identity_audit.py",
             "reports/active_error_audit_2026_10_16/errors.csv", "reports/active_error_audit_2026_10_16/audit_report.json",
             "reports/location_experiment/experiment_plan.json", "reports/property_catalog/conflict_audit.csv",
             "building_catalog.py", "location_model.py", "clean.py", "march_data.py", "repair_archived_data.py", "config.py"]
    hashes = {name: digest(root / name) for name in paths}
    for name in [COMBINED, CATALOG, JANUARY, MARCH, "building_catalog.py", "location_model.py"]:
        if hashes[name] != plan["sources"][name]:
            raise ValueError(f"Frozen source changed: {name}")
    model_paths = [Path(APP_MODEL_PIPELINE_PATH), Path(APP_MODEL_METADATA_PATH)]
    model_hashes = {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in model_paths}
    metadata = json.loads(model_paths[1].read_text(encoding="utf-8"))
    if metadata["model_sha256"] != digest(model_paths[0]):
        raise ValueError("Active model/metadata do not match")
    if hashes[SPLIT] != metadata["location_split_manifest_sha256"]:
        raise ValueError("Historical split does not match active model metadata")
    error_report = json.loads((root / "reports/active_error_audit_2026_10_16/audit_report.json").read_text(encoding="utf-8"))
    if (error_report["model_sha256"] != metadata["model_sha256"]
            or error_report["metadata_sha256"] != digest(model_paths[1])
            or error_report["output_csv_sha256"]["errors.csv"] != hashes["reports/active_error_audit_2026_10_16/errors.csv"]):
        raise ValueError("Error audit is stale or its CSV/model binding changed")
    combined = pd.read_csv(root / COMBINED)
    january = pd.read_csv(root / JANUARY, dtype=str, keep_default_na=False)
    march = pd.read_csv(root / MARCH, dtype=str, keep_default_na=False)
    split = pd.read_csv(root / SPLIT)
    if split.record_id.tolist() != combined.record_id.tolist():
        raise ValueError("Original split ordering changed")
    replay = verify_replay(combined, january, march)
    revised, corrections, queue = correct_source_fields(combined, january, split)
    catalog = build_property_catalog(revised, january, march)
    listings, buildings = catalog["listings"], catalog["buildings"]
    street_evidence = load_numbered_street_evidence(listings, root / OSM_MANIFEST)
    listings, buildings = withdraw_street_only_house_keys(listings, buildings, street_evidence)
    duplicates = possible_duplicates(revised)
    previous_deletions = removed_address_conflicts(replay, pd.read_csv(root / DEDUP))
    buildings_queue = building_review(listings)
    top_errors = review_top_errors(pd.read_csv(root / "reports/active_error_audit_2026_10_16/errors.csv"),
                                  march, listings, split)
    subsets = split.set_index("record_id").subset
    unusual = revised.loc[(revised.price / revised.area).lt(100_000)]
    extra = [{"record_id": r.record_id, "subset": subsets[r.record_id], "issue": "price_per_m2_below_100000",
              "evidence": f"price={r.price}; area={r.area}; original values agree with source",
              "decision": "retained_source_value_requires_external_confirmation", "url": r.url} for r in unusual.itertuples()]
    queue = pd.concat([queue, pd.DataFrame(extra, columns=REVIEW_COLUMNS)], ignore_index=True)
    # Check that no unapproved column, target, row or split was changed.
    pd.testing.assert_frame_equal(combined.drop(columns=["district", "residential_complex"]),
                                  revised.drop(columns=["district", "residential_complex"]))
    test_ids = set(split.loc[split.subset.eq("test"), "record_id"])
    changed_ids = set(corrections.record_id) | set(street_evidence.record_id)
    for output in [data_dir, reports_dir]: output.mkdir(parents=True, exist_ok=True)
    protocol = {"schema_version": 1, "policy": POLICY, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "sources_sha256": hashes, "active_artifacts_sha256": model_hashes,
                "rules": ["missing district only from own explicit district-only address",
                          "missing complex only from own explicit complex-only address", "no majority assignment",
                          "withdraw a derived house identity when the entire address is an OSM numbered street name",
                          "no target correction or outlier removal", "no training or activation", "preserve original split"]}
    save_json(reports_dir / "protocol.json", protocol)
    outputs = {data_dir / "listings_clean.csv": revised, data_dir / "listings.csv": listings,
               data_dir / "buildings.csv": buildings,
               reports_dir / "corrections.csv": corrections, reports_dir / "review_queue.csv": queue,
               reports_dir / "possible_duplicates.csv": duplicates,
               reports_dir / "previous_dedup_address_conflicts.csv": previous_deletions,
               reports_dir / "building_review_queue.csv": buildings_queue,
               reports_dir / "numbered_street_evidence.csv": street_evidence,
               reports_dir / "top_error_review.csv": top_errors}
    for path, frame in outputs.items():
        with path.open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
    reference_split = data_dir / "reference_split_manifest.csv"
    with reference_split.open("xb") as stream: stream.write((root / SPLIT).read_bytes())
    if hashes != {name: digest(root / name) for name in hashes} or model_hashes != {name: digest(root / name) for name in model_hashes}:
        raise ValueError("Protected sources changed while auditing; outputs are not validated")
    summary = {"rows": len(revised), "district_corrections": int(corrections.field.eq("district").sum()),
               "complex_corrections": int(corrections.field.eq("residential_complex").sum()),
               "street_only_records_corrected": len(street_evidence),
               "street_only_house_keys_withdrawn": int(street_evidence.old_house_key.nunique()),
               "street_only_match_types": street_evidence.match_type.value_counts().to_dict(),
               "district_correction_counts": corrections.loc[corrections.field.eq("district"), "new_value"].value_counts().to_dict(),
               "district_missing_before": int(combined.district.isna().sum()),
               "district_missing_after": int(revised.district.isna().sum()),
               "holdout_changed_rows": len(changed_ids & test_ids), "train_changed_rows": len(changed_ids - test_ids),
               "unresolved_building_groups": len(buildings_queue), "possible_duplicate_rows": len(duplicates),
               "possible_duplicate_groups": int(duplicates.review_group.nunique()),
               "previous_deletions_with_address_conflict": len(previous_deletions), "unusual_price_rows_retained": len(unusual)}
    report = {**protocol, "summary": summary, "data_directory": data_dir.relative_to(root).as_posix(),
              "files_sha256": {path.relative_to(root).as_posix(): digest(path) for path in [*outputs, reference_split]},
              "files": {name: {"path": f"{name}.csv", "sha256": digest(data_dir / f"{name}.csv"), "rows": len(frame)}
                        for name, frame in [("listings", listings), ("buildings", buildings)]},
              "checks": {"source_replay_fields_match": True, "original_files_unchanged": True,
                         "active_model_unchanged": True, "targets_unchanged": True, "row_order_preserved": True,
                         "reference_split_bytes_preserved": True, "requires_group_revalidation_before_training": True,
                         "rows_removed": 0, "rows_restored": 0, "fit_calls": 0,
                         "prediction_calls": 0, "application_switched": False},
              "limitations": ["Original source statements are not externally verified.",
                              "No effect on prediction accuracy is claimed without a new training experiment.",
                              "Conflicting house attributes and possible duplicate identity remain unresolved."]}
    save_json(reports_dir / "report.json", report)
    save_json(data_dir / "manifest.json", report)
    (reports_dir / "ATTRIBUTION.txt").write_text(
        "OSM-derived road names and object identifiers in numbered_street_evidence.csv and this report: "
        "© OpenStreetMap contributors, ODbL 1.0.\nhttps://www.openstreetmap.org/copyright\n"
        "https://opendatacommons.org/licenses/odbl/1-0/\n"
        "Source: https://download.geofabrik.de/asia/kazakhstan-261012.osm.pbf\n"
        "This attribution does not assign a license to apartment listings from other sources.\n", encoding="utf-8", newline="\n")
    (reports_dir / "report.html").write_text(render_report(report, corrections), encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    result = run(args.output_dir, args.reports_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

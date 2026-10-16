"""Replay and diagnose the active model on the frozen 590-row March holdout.

No fit, retraining, data repair, current-market inference or causal attribution.
Every output goes to a new audit directory; existing runs cannot be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from building_catalog import normalize_complex_name
from comparables import load_catalog
from config import APP_MODEL_METADATA_PATH, APP_MODEL_PIPELINE_PATH, PROJECT_ROOT
from error_analysis import metric_rows, sha256, summarize
from features import FEATURE_COLUMNS
from location_model import prepare_location_input
from location_prediction import _known_complexes, load_app_artifacts


DEFAULT_OUTPUT = PROJECT_ROOT / "reports/active_error_audit_2026_10_16"
COMBINED = Path("data/processed/combined_2026/listings_clean.csv")
CATALOG = Path("data/processed/property_catalog")
SPLIT = Path("reports/location_experiment/split_manifest.csv")
SAVED_PREDICTIONS = Path("reports/coordinate_enrichment_2026_10_13/holdout_predictions.csv")
SAVED_REPORT = Path("reports/coordinate_enrichment_2026_10_13/model_comparison.json")
ENRICHMENT_PROTOCOL = Path("reports/coordinate_enrichment_2026_10_13/protocol.json")
EXPECTED_ROWS = 590
MIN_SEGMENT_ROWS = 30
PRICE_TOLERANCE_TENGE = .05
MISSING = "Не указано"
LIMITATIONS = [
    "Те же 590 мартовских квартир использовались ранее при разработке; это диагностика архивного holdout, не свежая независимая оценка.",
    "Сегментные различия описательны: они не доказывают, что район, ЖК или пропуск поля стали причиной ошибки.",
    "Группы с n<30 сохранены, но исключены из рейтингов; n>=30 само по себе не гарантирует устойчивости результата.",
    "Цена и площадь разделены по заранее заданным границам; ценовые сегменты используют фактическую цену объявления.",
    "Полнота — число непустых исходных полей из восьми до заполнения пропусков моделью; ЖК и координаты в это число не входят.",
    "Знакомый ЖК означает присутствие названия в train; редкое название могло кодироваться общей категорией.",
    "Координаты OSM октября применяются к архивным мартовским объявлениям; историческое состояние здания не подтверждено.",
    "Оценивается цена предложения, не цена сделки. Ссылки относятся к архивным объявлениям и могут быть недоступны.",
    "Диапазон плюс/минус MAE описательный, без гарантии покрытия; сегменты могут включать связанные объявления одного дома.",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_series_equal(left: pd.Series, right: pd.Series, label: str) -> None:
    try:
        left, right = left.reset_index(drop=True), right.reset_index(drop=True)
        # CSV readers use either NaN or pd.NA for the same missing value. Check
        # missing positions separately, then compare all present values exactly.
        np.testing.assert_array_equal(left.isna().to_numpy(), right.isna().to_numpy())
        pd.testing.assert_series_equal(left.loc[left.notna()].astype(object).reset_index(drop=True),
                                       right.loc[right.notna()].astype(object).reset_index(drop=True),
                                       check_dtype=False, check_names=False, check_exact=True)
    except AssertionError as error:
        raise ValueError(f"Не совпали значения {label}.") from error


def align_holdout(combined: pd.DataFrame, catalog: pd.DataFrame, split: pd.DataFrame,
                  saved: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> pd.DataFrame:
    """Join by checked identities, never by the physical order of catalogue rows."""
    _require(combined.index.equals(pd.RangeIndex(len(combined))), "Combined должен иметь RangeIndex.")
    for name, table in (("combined", combined), ("catalog", catalog), ("saved predictions", saved)):
        _require("record_id" in table and table.record_id.notna().all() and table.record_id.is_unique,
                 f"Повторный или пустой record_id: {name}.")
        _require(table.record_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
                 f"Некорректный record_id: {name}.")
    _require(set(catalog.record_id) == set(combined.record_id), "Каталог не покрывает исходные record_id.")
    _require(set(combined.columns).issubset(catalog.columns), "Каталог потерял исходные колонки.")
    frame = catalog.set_index("record_id", drop=False).loc[combined.record_id].reset_index(drop=True)
    for column in combined.columns:
        _assert_series_equal(combined[column], frame[column], f"catalog.{column}")
    required = {"row_index", "record_id", "group", "experiment_group", "house_key", "subset", "primary_evaluation"}
    _require(required.issubset(split.columns), "Неполная схема location split.")
    ids = split.row_index
    _require(pd.api.types.is_integer_dtype(ids) and ids.is_unique and ids.notna().all()
             and set(ids) == set(combined.index), "Split должен покрывать каждую исходную строку ровно один раз.")
    manifest = split.set_index("row_index").loc[combined.index]
    for column in ("record_id", "group", "house_key"):
        _assert_series_equal(manifest[column], frame[column], f"split.{column}")
    _require(manifest.subset.isin(["train", "test"]).all(), "Неизвестная метка train/test.")
    _require(pd.api.types.is_bool_dtype(manifest.primary_evaluation)
             and manifest.primary_evaluation.notna().all(), "primary_evaluation должна быть boolean без пропусков.")
    train = manifest.subset.eq("train")
    test = manifest.subset.eq("test")
    _require(train.any() and test.any() and manifest.primary_evaluation.equals(test.rename("primary_evaluation")),
             "Первичный holdout должен содержать ровно все test и ни одной train-строки.")
    for column in ("group", "experiment_group"):
        _require(manifest[column].notna().all() and
                 set(manifest.loc[train, column]).isdisjoint(manifest.loc[test, column]),
                 f"Группы train/test пересекаются или отсутствуют: {column}.")
    result = frame.loc[test].copy()
    result.insert(0, "row_index", result.index)
    result.reset_index(drop=True, inplace=True)
    _require(len(result) == len(saved) == expected_rows, f"Ожидалось ровно {expected_rows} строк holdout.")
    _assert_series_equal(saved.record_id, result.record_id, "saved record_id / порядок holdout")
    for column in ("price", "group", "house_key", "district"):
        _require(column in saved, f"В сохранённых прогнозах нет {column}.")
        _assert_series_equal(saved[column], result[column], f"saved.{column}")
    _require(result.source_row.notna().all() and result.source_row.is_unique, "Некорректный исходный source_row.")
    return result


def count_filled_base_fields(frame: pd.DataFrame) -> pd.Series:
    """Count the eight original features, before any fitted imputation."""
    original = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in original:
        original[column] = original[column].map(
            lambda value: np.nan if isinstance(value, str) and not value.strip() else value
        )
    return original.notna().sum(axis=1).rename("filled_base_fields")


def _summary(rows: pd.DataFrame, total_error: float | None = None) -> dict:
    result = summarize(rows, total_error)
    # Do not attach nominal independent-row intervals to related house records.
    result.pop("over_20_ci_low_pct")
    result.pop("over_20_ci_high_pct")
    result.update(median_signed_error_tenge=float(rows.signed_error.median()),
                  overprediction_pct=float(rows.signed_error.gt(0).mean() * 100),
                  underprediction_pct=float(rows.signed_error.lt(0).mean() * 100),
                  exact_prediction_pct=float(rows.signed_error.eq(0).mean() * 100),
                  total_absolute_error_tenge=float(rows.absolute_error.sum()))
    return result


def segment_labels(frame: pd.DataFrame, known_complexes: set[str], geography: pd.Series) -> dict[str, pd.Series]:
    _require(frame.index.equals(geography.index), "География должна иметь тот же индекс и порядок строк.")
    complexes = frame.residential_complex.map(normalize_complex_name)
    familiarity = pd.Series(np.where(complexes.isna(), "ЖК не указан",
                            np.where(complexes.isin(known_complexes), "ЖК встречался в train", "ЖК не встречался в train")),
                            index=frame.index)
    return {
        "district": frame.district.map(lambda value: value.strip() if isinstance(value, str) and value.strip() else MISSING),
        "complex": complexes.fillna(MISSING),
        "area_band": pd.cut(frame.area, [0, 40, 60, 90, 120, np.inf],
                            labels=["≤40 м²", "40–60 м²", "60–90 м²", "90–120 м²", ">120 м²"]).astype(object).fillna(MISSING),
        "price_band": pd.cut(frame.price, [0, 30e6, 50e6, 80e6, 120e6, np.inf],
                             labels=["≤30 млн ₸", "30–50 млн ₸", "50–80 млн ₸", "80–120 млн ₸", ">120 млн ₸"]).astype(object).fillna(MISSING),
        "completeness": count_filled_base_fields(frame).map(lambda number: f"{number}/8"),
        "complex_familiarity": familiarity,
        "geography": geography,
    }


def summarize_segments(rows: pd.DataFrame, labels: dict[str, pd.Series]) -> pd.DataFrame:
    """Only n>=30 groups receive within-dimension error ranks; small groups remain visible."""
    output = []
    total = float(rows.absolute_error.sum())
    for dimension, values in labels.items():
        _require(rows.index.equals(values.index), f"Метки сегмента {dimension} не совпадают с порядком ошибок.")
        values = values.astype(object).fillna(MISSING)
        for name in sorted(values.unique(), key=str):
            group = rows.loc[values.eq(name)]
            output.append({"dimension": dimension, "segment": str(name), **_summary(group, total),
                           "eligible_for_ranking": len(group) >= MIN_SEGMENT_ROWS})
    result = pd.DataFrame(output)
    for column, metric in (("rank_mae", "mae_tenge"), ("rank_over_20", "over_20_pct")):
        result[column] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        eligible = result.eligible_for_ranking
        result.loc[eligible, column] = result.loc[eligible].groupby("dimension")[metric].rank(method="min", ascending=False).astype("Int64")
    return result


def assemble_errors(frame: pd.DataFrame, pipeline: Any, predicted: np.ndarray, half_width: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    x = prepare_location_input(frame)
    transformer = pipeline.named_steps["location_features"]
    transformed = transformer.transform(x)
    provenance = transformer.coordinate_provenance(x)
    _require(transformed.index.equals(frame.index) and provenance.index.equals(frame.index), "Трансформер изменил порядок строк.")
    available = transformed[["latitude", "longitude"]].notna().all(axis=1)
    _require(np.isfinite(transformed.loc[available, ["latitude", "longitude"]].to_numpy(float)).all(),
             "Неконечные координаты модели.")
    _require(np.array_equal(available, provenance.coordinate_available.to_numpy()), "Не согласованы география и её источник.")
    geography = provenance.source_status.map({"training_consensus": "Координаты из train", "osm_exact_address": "Точный адрес OSM"})
    geography = geography.where(available, "Нет пригодных координат")
    _require(geography.notna().all(), "Неизвестный источник пригодных координат.")
    labels = segment_labels(frame, _known_complexes(pipeline), geography)
    metrics = metric_rows(frame.price, predicted, half_width)
    metrics.index = frame.index
    # The original catalogue coordinates are exported with an explicit prefix;
    # latitude/longitude below are the coordinates actually consumed by the model.
    columns = [column for column in ("row_index", "record_id", "source_row", "original_source_row", "group", "url",
                "dataset_period", *FEATURE_COLUMNS, "residential_complex", "complex_name", "address", "address_normalized",
                "house_key", "building_id", "building_match_status", "building_conflict_reasons",
                "raw_latitude", "raw_longitude", "raw_coordinate_status") if column in frame]
    errors = frame.loc[:, columns].copy()
    for column in ("latitude", "longitude", "coordinate_status", "coordinate_source"):
        if column in frame:
            errors["catalog_" + column] = frame[column]
    errors["latitude"] = transformed.latitude
    errors["longitude"] = transformed.longitude
    errors["coordinate_available"] = available
    errors["coordinate_source"] = provenance.source_status
    errors["osm_url"] = provenance.osm_url
    errors["filled_base_fields"] = count_filled_base_fields(frame)
    errors["complex_familiarity"] = labels["complex_familiarity"]
    errors["geography_segment"] = geography
    for column in metrics:
        errors[column] = metrics[column]
    errors["signed_ape_pct"] = errors.signed_error / errors.actual_price * 100
    errors["error_direction"] = np.select([errors.signed_error.gt(0), errors.signed_error.lt(0)],
                                         ["Завышение", "Занижение"], default="Точное совпадение")
    return errors, summarize_segments(metrics, labels), _summary(metrics)


def top_errors(errors: pd.DataFrame, metric: str, limit: int = 30) -> pd.DataFrame:
    _require(metric in ("absolute_error", "ape_pct"), "Неизвестная метрика ранжирования ошибок.")
    return errors.sort_values([metric, "record_id"], ascending=[False, True], kind="stable").head(limit).copy()


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", force_ascii=False, double_precision=15))


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _verify_hashes(root: Path, expected: dict[str, str]) -> None:
    for name, checksum in expected.items():
        path = root / name
        _require(path.is_file() and sha256(path) == checksum, f"SHA256 не совпал: {name}.")


def _display_table(frame: pd.DataFrame, columns: dict[str, str]) -> str:
    header = "".join(f"<th>{escape(label)}</th>" for label in columns.values())
    body = []
    for row in frame.loc[:, list(columns)].to_dict("records"):
        values = []
        for name, value in row.items():
            if pd.isna(value):
                text = "—"
            elif isinstance(value, (float, np.floating)):
                text = f"{value:,.2f}"
            else:
                text = str(value)
            values.append(f"<td>{escape(text)}</td>")
        body.append("<tr>" + "".join(values) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def render_html(report: dict, segments: pd.DataFrame, absolute: pd.DataFrame, relative: pd.DataFrame) -> str:
    overall = report["overall"]
    eligible = segments.loc[segments.eligible_for_ranking].sort_values(["dimension", "rank_over_20", "rank_mae", "segment"])
    segment_table = _display_table(eligible, {"dimension": "Разрез", "segment": "Сегмент", "n": "n",
        "mae_tenge": "MAE, ₸", "median_ae_tenge": "Медианная |ошибка|, ₸", "over_20_pct": "Ошибка >20%, %", "bias_tenge": "Среднее завышение, ₸"})
    top_columns = {"record_id": "ID", "district": "Район", "residential_complex": "ЖК", "actual_price": "Цена, ₸",
                   "predicted_price": "Прогноз, ₸", "signed_error": "Прогноз − цена, ₸", "ape_pct": "|Ошибка|, %"}
    limitations = "".join(f"<li>{escape(text)}</li>" for text in report["limitations"])
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ошибки активной модели — 590 мартовских квартир</title><style>body{{font:16px/1.6 system-ui;color:#17303c;background:#f3f6f8;margin:0;padding:24px}}main{{max-width:1200px;margin:auto;background:white;padding:28px;border-radius:12px}}h1{{line-height:1.2}}.note{{background:#eaf3f5;padding:14px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}th,td{{padding:8px;border-bottom:1px solid #dce5e9;text-align:left}}th{{background:#eaf3f5}}li{{margin:7px 0}}@media(max-width:600px){{body{{padding:8px}}main{{padding:14px}}}}</style>
<main><h1>Где ошибается активная модель</h1><p>{escape(report['model_label'])} · {overall['n']} прежних мартовских наблюдений · fit: 0.</p>
<p class="note">MAE {overall['mae_tenge']/1e6:.3f} млн ₸; медианная абсолютная ошибка {overall['median_ae_tenge']/1e6:.3f} млн ₸.
Ошибка превышает 20% у {overall['over_20_n']} квартир ({overall['over_20_pct']:.1f}%). Это повторная диагностика, не оценка свежего рынка.</p>
<p>Завышение: {overall['overprediction_pct']:.1f}%; занижение: {overall['underprediction_pct']:.1f}%; точное совпадение: {overall['exact_prediction_pct']:.1f}%.
30 крупнейших денежных ошибок дают {report['top30_absolute_error_share_pct']:.1f}% суммарной абсолютной ошибки.</p>
<h2>Сегменты с n≥30</h2><p>Ранги рассчитываются только внутри каждого разреза. Граница n≥30 не делает различия причинными или гарантированно устойчивыми.
Все меньшие группы сохранены в <a href="segments.csv">CSV сегментов</a>, без рангов.</p>{segment_table}
<h2>30 крупнейших ошибок в тенге</h2>{_display_table(absolute, top_columns)}
<h2>30 крупнейших относительных ошибок</h2>{_display_table(relative, top_columns)}
<h2>Как интерпретировать</h2><ul>{limitations}</ul>
<h2>Файлы аудита</h2><ul><li><a href="errors.csv">Все 590 ошибок с исходными признаками и географией</a></li>
<li><a href="segments.csv">Все сегменты и размеры групп</a></li><li><a href="top30_absolute.csv">Топ-30 по тенге</a> · <a href="top30_relative.csv">Топ-30 по проценту</a></li>
<li><a href="audit_report.json">Полный JSON</a> · <a href="protocol.json">Зафиксированный протокол и хеши</a></li></ul></main></html>'''


def run_audit(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    root = PROJECT_ROOT.resolve()
    output = Path(output_dir).resolve()
    filenames = ("protocol.json", "errors.csv", "segments.csv", "top30_absolute.csv", "top30_relative.csv", "audit_report.json", "report.html")
    _require(not any((output / name).exists() for name in filenames), "Файлы аудита уже существуют; выберите новую папку --output-dir.")
    metadata = json.loads(APP_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    _require(metadata.get("geo_enrichment") is True, "Этот аудит ожидает активный вариант с OSM из закреплённого сравнения.")
    historical = dict(metadata["source_sha256"])
    # config.py was intentionally changed when the already evaluated candidate
    # was activated. All other historical inputs and code remain pinned.
    historical_config_sha = historical.pop("config.py", None)
    _verify_hashes(root, historical)
    direct = {str(APP_MODEL_PIPELINE_PATH.relative_to(root)).replace("\\", "/"): metadata["model_sha256"],
              COMBINED.as_posix(): metadata["input_sha256"], SPLIT.as_posix(): metadata["location_split_manifest_sha256"],
              ENRICHMENT_PROTOCOL.as_posix(): metadata["protocol_sha256"], SAVED_REPORT.as_posix(): metadata["diagnostic_report_sha256"],
              metadata["registry_path"]: metadata["registry_sha256"],
              metadata["staged_metadata_path"]: metadata["staged_metadata_sha256"]}
    _verify_hashes(root, direct)
    saved_report = json.loads((root / SAVED_REPORT).read_text(encoding="utf-8"))
    _require(saved_report["model_sha256"] == metadata["model_sha256"], "Сохранённые прогнозы принадлежат другой модели.")
    _verify_hashes(root, {SAVED_PREDICTIONS.as_posix(): saved_report["prediction_csv_sha256"]})
    catalog = load_catalog(root / CATALOG)
    saved = pd.read_csv(root / SAVED_PREDICTIONS, float_precision="round_trip")
    frame = align_holdout(pd.read_csv(root / COMBINED, low_memory=False), catalog.listings,
                          pd.read_csv(root / SPLIT), saved)
    _require(frame.record_id.tolist() == saved_report["record_ids"], "Состав test не совпадает с диагностическим отчётом.")
    extra = [APP_MODEL_METADATA_PATH, root / "config.py", Path(__file__).resolve(), root / "error_analysis.py",
             root / CATALOG / "buildings.csv", root / SAVED_PREDICTIONS, root / "prediction.py"]
    before = {**historical, **direct, **{path.relative_to(root).as_posix(): sha256(path) for path in extra}}
    _verify_hashes(root, before)
    protocol = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": APP_MODEL_PIPELINE_PATH.relative_to(root).as_posix(),
                "metadata_path": APP_MODEL_METADATA_PATH.relative_to(root).as_posix(),
                "source_sha256": before, "record_ids": frame.record_id.tolist(),
                "source_rows": frame.source_row.tolist(), "holdout_rows": EXPECTED_ROWS,
                "historical_config_sha256": historical_config_sha, "current_config_sha256": before["config.py"],
                "config_difference_reason": "Previously evaluated model was activated; current APP_MODEL paths are frozen here.",
                "minimum_segment_rows_for_ranking": MIN_SEGMENT_ROWS, "price_tolerance_tenge": PRICE_TOLERANCE_TENGE,
                "prediction_reduction": "joblib sequential backend", "fit_calls": 0,
                "area_bins_right_closed": [40, 60, 90, 120], "price_bins_right_closed_tenge": [30e6, 50e6, 80e6, 120e6],
                "limitations": LIMITATIONS}
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "protocol.json", protocol)
    protocol_sha = sha256(output / "protocol.json")
    pipeline, loaded_metadata = load_app_artifacts(APP_MODEL_PIPELINE_PATH, APP_MODEL_METADATA_PATH)
    _require(loaded_metadata == metadata, "Метаданные изменились после фиксации протокола.")
    transformer = pipeline.named_steps["location_features"]
    _require(getattr(transformer, "registry_sha256_", None) == metadata["registry_sha256"], "Встроенный реестр не соответствует metadata.")
    x = prepare_location_input(frame)
    with joblib.parallel_backend("sequential"):
        predicted = np.asarray(pipeline.predict(x), dtype=float)
    _require(predicted.shape == (EXPECTED_ROWS,) and np.isfinite(predicted).all() and (predicted > 0).all(), "Некорректный прогноз модели.")
    np.testing.assert_allclose(predicted, saved.candidate_prediction, rtol=0, atol=PRICE_TOLERANCE_TENGE)
    errors, segments, overall = assemble_errors(frame, pipeline, predicted, metadata["test_metrics"]["mae_tenge"])
    np.testing.assert_allclose(errors.absolute_error, saved.candidate_absolute_error, rtol=0, atol=PRICE_TOLERANCE_TENGE)
    np.testing.assert_allclose(overall["mae_tenge"], metadata["test_metrics"]["mae_tenge"], rtol=0, atol=PRICE_TOLERANCE_TENGE)
    np.testing.assert_allclose(overall["mae_tenge"], saved_report["comparison"]["candidate"]["mae_tenge"], rtol=0, atol=PRICE_TOLERANCE_TENGE)
    np.testing.assert_array_equal(errors.coordinate_available, saved.candidate_geo_available)
    _assert_series_equal(errors.coordinate_source, saved.coordinate_source, "coordinate_source")
    absolute, relative = top_errors(errors, "absolute_error"), top_errors(errors, "ape_pct")
    total = overall["total_absolute_error_tenge"]
    eligible = segments.loc[segments.eligible_for_ranking]
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_label": metadata["model_label"], "candidate_id": metadata["candidate_id"],
              "model_sha256": metadata["model_sha256"], "metadata_sha256": before[APP_MODEL_METADATA_PATH.relative_to(root).as_posix()],
              "protocol_sha256": protocol_sha, "overall": overall, "test_rows": len(frame), "fit_calls": 0,
              "top30_absolute_error_share_pct": float(absolute.absolute_error.sum() / total * 100) if total else 0.,
              "top30_relative_absolute_error_share_pct": float(relative.absolute_error.sum() / total * 100) if total else 0.,
              "minimum_segment_rows_for_ranking": MIN_SEGMENT_ROWS, "segments": _json_records(segments),
              "top_segments_by_frequency": _json_records(eligible.sort_values(["dimension", "rank_over_20", "rank_mae", "segment"])),
              "top_segments_by_mae": _json_records(eligible.sort_values(["dimension", "rank_mae", "rank_over_20", "segment"])),
              "top30_absolute_record_ids": absolute.record_id.tolist(), "top30_relative_record_ids": relative.record_id.tolist(),
              "record_ids": frame.record_id.tolist(), "source_rows": frame.source_row.tolist(),
              "actual_price_vector_sha256": hashlib.sha256(frame.price.to_numpy(dtype="<f8").tobytes()).hexdigest(),
              "checks": {"model_sha_verified": True, "historical_sources_verified_except_activation_config": True,
                         "test_ids_and_targets_aligned": True, "train_test_groups_disjoint": True,
                         "saved_predictions_replayed": True, "mae_reproduced": True,
                         "geography_replayed": True, "max_prediction_difference_tenge": float(np.abs(predicted - saved.candidate_prediction).max()),
                         "active_model_and_sources_unchanged": True},
              "fresh_validation_completed": False, "holdout_previously_analyzed": True, "limitations": LIMITATIONS}
    _verify_hashes(root, before)
    _require(sha256(output / "protocol.json") == protocol_sha, "Протокол изменился во время аудита.")
    for filename, table in (("errors.csv", errors), ("segments.csv", segments),
                            ("top30_absolute.csv", absolute), ("top30_relative.csv", relative)):
        table.to_csv(output / filename, index=False, lineterminator="\n")
    report["output_csv_sha256"] = {name: sha256(output / name) for name in filenames if name.endswith(".csv")}
    _save_json(output / "audit_report.json", report)
    (output / "report.html").write_text(render_html(report, segments, absolute, relative), encoding="utf-8")
    _verify_hashes(root, before)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    audit = run_audit(arguments.output_dir)
    print(json.dumps({"overall": audit["overall"], "top30_absolute_error_share_pct": audit["top30_absolute_error_share_pct"],
                      "checks": audit["checks"], "fit_calls": 0}, ensure_ascii=False, indent=2))

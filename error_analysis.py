"""Диагностика сохранённой модели на исходном test без переобучения."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLEAN_LISTINGS_PATH, MODEL_METADATA_PATH, MODEL_PIPELINE_PATH, REPORTS_DIR, PROJECT_ROOT
from features import FEATURE_COLUMNS, NUMERIC_FEATURES
from prediction import FIELD_LABELS, load_artifacts, prepare_input


OUTPUT_DIR = REPORTS_DIR / "errors"
MIN_GROUP_SIZE = 30
RELATIVE_THRESHOLD = 20.0
ABSOLUTE_THRESHOLD = 10_000_000.0
OPTIONAL_FIELDS = tuple(c for c in FEATURE_COLUMNS if c not in ("area", "rooms"))
SCENARIOS = (
    ("minimal", "2 поля: площадь и комнаты", ("area", "rooms")),
    ("district", "3 поля: добавлен район", ("area", "rooms", "district")),
    ("building", "5 полей: добавлены год и тип дома", ("area", "rooms", "district", "year_built", "building_type")),
    ("floors", "7 полей: добавлены этаж и этажность", tuple(c for c in FEATURE_COLUMNS if c != "condition")),
    ("complete", "8 полей: полная анкета", FEATURE_COLUMNS),
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metric_rows(actual, predicted, half_width: float) -> pd.DataFrame:
    """Пороговые события считаются строго >, границы диапазона включены."""
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("Нужны непустые одномерные массивы одинаковой длины.")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y <= 0).any():
        raise ValueError("Цены должны быть конечными, фактические цены положительными.")
    if not np.isfinite(half_width) or half_width < 0:
        raise ValueError("Ширина диапазона должна быть конечной и неотрицательной.")
    signed = p - y
    error = np.abs(signed)
    return pd.DataFrame({
        "actual_price": y, "predicted_price": p, "signed_error": signed,
        "absolute_error": error, "ape_pct": error / y * 100,
        "over_20": error > y * (RELATIVE_THRESHOLD / 100),
        "over_10m": error > ABSOLUTE_THRESHOLD,
        "covered": (y >= np.maximum(0, p - half_width)) & (y <= p + half_width),
        "under_20": signed < -y * (RELATIVE_THRESHOLD / 100),
        "overprediction_20": signed > y * (RELATIVE_THRESHOLD / 100),
    })


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """Описательный 95% интервал доли; предполагает независимые строки."""
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return float(max(0, center - margin) * 100), float(min(1, center + margin) * 100)


def summarize(rows: pd.DataFrame, total_error: float | None = None) -> dict:
    n = len(rows)
    if not n:
        raise ValueError("Пустая группа не имеет метрик.")
    count = int(rows["over_20"].sum())
    low, high = wilson_interval(count, n)
    error_sum = float(rows["absolute_error"].sum())
    denominator = error_sum if total_error is None else total_error
    return {
        "n": n, "mae_tenge": float(rows["absolute_error"].mean()),
        "median_ae_tenge": float(rows["absolute_error"].median()),
        "rmse_tenge": float(np.sqrt(np.square(rows["signed_error"]).mean())),
        "median_ape_pct": float(rows["ape_pct"].median()),
        "mape_pct": float(rows["ape_pct"].mean()),
        "p90_ae_tenge": float(rows["absolute_error"].quantile(.9)),
        "bias_tenge": float(rows["signed_error"].mean()),
        "over_20_pct": count / n * 100, "over_20_n": count,
        "over_20_ci_low_pct": low, "over_20_ci_high_pct": high,
        "over_10m_pct": float(rows["over_10m"].mean() * 100),
        "coverage_pct": float(rows["covered"].mean() * 100),
        "under_20_pct": float(rows["under_20"].mean() * 100),
        "overprediction_20_pct": float(rows["overprediction_20"].mean() * 100),
        "error_share_pct": error_sum / denominator * 100 if denominator else 0.0,
        "small_group": n < MIN_GROUP_SIZE,
    }


def summarize_groups(rows: pd.DataFrame, labels: pd.Series) -> list[dict]:
    if len(rows) != len(labels) or not rows.index.equals(labels.index):
        raise ValueError("Метки групп должны соответствовать тем же строкам и порядку.")
    total = float(rows["absolute_error"].sum())
    grouped = rows.assign(_group=labels.astype(object).fillna("Не указано"))
    return [{"group": str(name), **summarize(part, total)}
            for name, part in grouped.groupby("_group", sort=False, observed=True)]


def mask_features(frame: pd.DataFrame, kept: tuple[str, ...]) -> pd.DataFrame:
    if not {"area", "rooms"}.issubset(kept) or not set(kept).issubset(FEATURE_COLUMNS):
        raise ValueError("Маска должна сохранять обязательные поля и известную схему.")
    masked = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in FEATURE_COLUMNS:
        if column not in kept:
            masked[column] = np.nan
    return masked


def load_verified_test(input_path: Path = CLEAN_LISTINGS_PATH,
                       model_path: Path = MODEL_PIPELINE_PATH,
                       metadata_path: Path = MODEL_METADATA_PATH,
                       reports_dir: Path = REPORTS_DIR):
    """Не соединять по позиции: проверяем IDs, исходный split, цены и SHA256."""
    pipeline, metadata = load_artifacts(model_path, metadata_path)
    input_hash = sha256(input_path)
    if input_hash != metadata["input_sha256"]:
        raise ValueError("Очищенные данные отличаются от обучающих.")
    frame = pd.read_csv(input_path)
    manifest_path = reports_dir / "split_manifest.csv"
    prediction_path = reports_dir / "test_predictions.csv"
    manifest = pd.read_csv(manifest_path)
    saved = pd.read_csv(prediction_path)
    for table in (manifest, saved):
        ids = table["row_index"]
        if ids.isna().any() or not ids.is_unique or not np.equal(ids, ids.astype(int)).all():
            raise ValueError("row_index должен быть уникальным целым числом.")
    if set(manifest.row_index) != set(frame.index):
        raise ValueError("Манифест должен описывать каждую строку данных ровно один раз.")
    if metadata.get("split_strategy") == "predefined_group_holdout":
        if sha256(manifest_path) != metadata.get("split_manifest_sha256"):
            raise ValueError("Групповой манифест не совпадает с SHA256 метаданных модели.")
        if "group" not in manifest or manifest.group.isna().any():
            raise ValueError("В групповом манифесте нет групп.")
        expected_train = manifest.loc[manifest.subset.eq("train"), "row_index"]
        expected_test = manifest.loc[manifest.subset.eq("test"), "row_index"]
        if set(manifest.loc[manifest.subset.eq("train"), "group"]) & set(manifest.loc[manifest.subset.eq("test"), "group"]):
            raise ValueError("Группы train и test пересекаются.")
        split_check = "SHA256 заранее зафиксированного группового разбиения совпал; группы train/test не пересекаются"
    else:
        expected_train, expected_test = train_test_split(
            frame.index, random_state=metadata["random_state"], test_size=metadata["test_size"],
        )
        split_check = "Исходный случайный train/test split восстановлен и совпал с манифестом"
    if (set(manifest.loc[manifest.subset.eq("test"), "row_index"]) != set(expected_test)
            or set(manifest.loc[manifest.subset.eq("train"), "row_index"]) != set(expected_train)
            or set(saved.row_index) != set(expected_test)
            or len(saved) != metadata["test_rows"] or len(expected_train) != metadata["train_rows"]):
        raise ValueError("Манифест, прогнозы и исходное train/test разбиение не совпадают.")
    for table in (manifest, saved):
        if "source_row" in frame:
            np.testing.assert_array_equal(table.source_row, frame.loc[table.row_index, "source_row"])
    test = frame.loc[saved.row_index].copy().reset_index(names="row_index")
    np.testing.assert_allclose(test.price, saved.actual_price, rtol=0, atol=.01)
    x = test.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in NUMERIC_FEATURES:
        x[column] = x[column].astype(float)
    predicted = pipeline.predict(x)
    np.testing.assert_allclose(predicted, saved.predicted_price, rtol=1e-10, atol=.01)
    rows = metric_rows(test.price, predicted, metadata["test_metrics"]["mae_tenge"])
    np.testing.assert_allclose(rows.absolute_error, saved.absolute_error, rtol=1e-10, atol=.01)
    np.testing.assert_allclose(rows.absolute_error.mean(), metadata["test_metrics"]["mae_tenge"], rtol=1e-10, atol=.01)
    provenance = {
        "input_sha256": input_hash, "model_sha256": metadata["model_sha256"],
        "test_predictions_sha256": sha256(prediction_path),
        "split_manifest_sha256": sha256(manifest_path),
        "checks": ["SHA256 данных и модели совпадают с метаданными обучения",
                   split_check,
                   "row_index и source_row проверены, строки train не включены",
                   "Повторные прогнозы совпали с сохранёнными с точностью 0,01 ₸",
                   "MAE совпала с исходным отчётом обучения"],
    }
    return pipeline, metadata, test, x, rows, provenance


def app_eligible(frame: pd.DataFrame, metadata: dict) -> np.ndarray:
    eligible = []
    for values in frame.to_dict("records"):
        cleaned = {k: None if pd.isna(v) else v for k, v in values.items()}
        try:
            prepare_input(cleaned, metadata)
            eligible.append(True)
        except ValueError:
            eligible.append(False)
    return np.asarray(eligible)


def make_charts(report: dict, output_dir: Path) -> list[dict]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    descriptions = []
    for key, title, data in (
        ("district", "Ошибки по районам", report["groups"]["district"]),
        ("price_band", "Ошибки по цене объявления", report["groups"]["price_band"]),
        ("masking", "Те же квартиры при разной полноте анкеты", report["controlled_masking"]),
    ):
        if not data:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4.0, len(data) * .52)), layout="constrained")
        labels = [f"{r['group']} (n={r['n']})" for r in data]
        positions = np.arange(len(data))
        axes[0].barh(positions, [r["mae_tenge"] / 1e6 for r in data], color="#0F766E")
        axes[0].set(yticks=positions, yticklabels=labels, xlabel="MAE, млн ₸")
        rates = np.array([r["over_20_pct"] for r in data])
        low = rates - np.array([r["over_20_ci_low_pct"] for r in data])
        high = np.array([r["over_20_ci_high_pct"] for r in data]) - rates
        axes[1].barh(positions, rates, color="#C66A2A", xerr=np.maximum(0, [low, high]), capsize=3)
        axes[1].set(yticks=positions, yticklabels=[], xlabel="Доля ошибок >20%, 95% интервал Уилсона", xlim=(0, 100))
        for ax in axes:
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle(title, fontsize=14, fontweight="bold")
        filename = f"{key}.png"
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
        descriptions.append({"title": title, "filename": filename,
                             "caption": "Денежная величина ошибки и её частота показаны отдельно. Интервалы долей предполагают независимость объявлений."})
    return descriptions


def main(input_path: Path = CLEAN_LISTINGS_PATH,
         models_dir: Path = MODEL_PIPELINE_PATH.parent,
         reports_dir: Path = REPORTS_DIR,
         output_dir: Path = OUTPUT_DIR) -> None:
    pipeline, metadata, test, x, rows, provenance = load_verified_test(
        input_path, models_dir / "best_model.joblib", models_dir / "model_metadata.json", reports_dir,
    )
    print(f"Проверены модель, контрольные суммы и {len(test)} строк test.", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    filled = x.notna().sum(axis=1)
    labels = {
        "district": test.district,
        "completeness": filled.map(lambda n: f"{n} из 8 полей"),
        "missing_pattern": x.apply(lambda r: ", ".join(FIELD_LABELS[c] for c in FEATURE_COLUMNS if pd.isna(r[c])) or "Все поля заполнены", axis=1),
        "price_band": pd.cut(test.price, [0, 30e6, 50e6, 80e6, 120e6, np.inf],
                             labels=["До 30 млн включительно", "Более 30–50 млн", "Более 50–80 млн", "Более 80–120 млн", "Более 120 млн"]),
        "area_band": pd.cut(test.area, [0, 40, 60, 90, 120, np.inf],
                            labels=["До 40 м² включительно", "Более 40–60 м²", "Более 60–90 м²", "Более 90–120 м²", "Более 120 м²"]),
        "rooms_band": test.rooms.map(lambda n: "Не указано" if pd.isna(n) else ("5 и более комнат" if n >= 5 else f"{int(n)} комн.")),
        "building_type": test.building_type,
        "condition": test.condition,
        "year_band": pd.cut(test.year_built, [0, 1990, 2000, 2010, 2020, np.inf],
                            labels=["До 1990 включительно", "1991–2000", "2001–2010", "2011–2020", "С 2021 года"]),
        "district_completeness": test.district.fillna("Не указано") + " · " + filled.astype(str) + "/8",
    }
    groups = {key: summarize_groups(rows, value) for key, value in labels.items()}
    for key in ("district", "missing_pattern", "district_completeness", "building_type", "condition"):
        groups[key].sort(key=lambda r: (-r["over_20_pct"], -r["n"]))
    groups["completeness"].sort(key=lambda r: r["group"])
    for key in ("price_band", "area_band", "year_band"):
        order = [*labels[key].cat.categories, "Не указано"]
        groups[key].sort(key=lambda r: order.index(r["group"]))
    groups["rooms_band"].sort(key=lambda r: r["group"])
    eligible = app_eligible(x, metadata)
    complete = x.notna().all(axis=1).to_numpy() & eligible
    cohort = x.loc[complete]
    if not len(cohort):
        raise ValueError("Нет полных валидных анкет для контролируемого сравнения.")
    width = metadata["test_metrics"]["mae_tenge"]
    full = summarize(metric_rows(test.loc[complete, "price"], pipeline.predict(cohort), width))
    controlled, ablation, scenario_frames = [], [], []
    for scenario_id, label, kept in SCENARIOS:
        predicted = pipeline.predict(mask_features(cohort, kept))
        scenario_rows = metric_rows(test.loc[complete, "price"], predicted, width)
        metrics = summarize(scenario_rows)
        controlled.append({"scenario_id": scenario_id, "group": label, "fields_kept": list(kept),
                           **metrics, "delta_mae_tenge": metrics["mae_tenge"] - full["mae_tenge"]})
        scenario_frames.append(scenario_rows.assign(row_index=test.loc[complete, "row_index"].to_numpy(), scenario_id=scenario_id))
    for field in OPTIONAL_FIELDS:
        kept = tuple(c for c in FEATURE_COLUMNS if c != field)
        metrics = summarize(metric_rows(test.loc[complete, "price"], pipeline.predict(mask_features(cohort, kept)), width))
        ablation.append({"group": FIELD_LABELS[field], **metrics,
                         "delta_mae_tenge": metrics["mae_tenge"] - full["mae_tenge"]})
    ablation.sort(key=lambda r: -r["delta_mae_tenge"])
    print(f"Контролируемое скрытие полей проверено на {len(cohort)} полных анкетах.", flush=True)
    details = pd.concat([test.drop(columns="price"), rows], axis=1).assign(filled_fields=filled, app_eligible=eligible)
    example_columns = ["row_index", "source_row", "district", "area", "rooms", "filled_fields", "actual_price", "predicted_price", "absolute_error", "ape_pct", "signed_error", "url", "residential_complex"]
    examples = {"top_absolute": details.nlargest(20, "absolute_error")[example_columns],
                "top_relative": details.nlargest(20, "ape_pct")[example_columns]}
    rankings = [{"dimension": key, **r} for key, values in groups.items()
                if key not in ("missing_pattern", "district_completeness")
                for r in values if not r["small_group"]]
    rankings.sort(key=lambda r: (-r["over_20_pct"], -r["n"]))
    baseline = summarize(rows)
    worst_district = max((r for r in groups["district"] if not r["small_group"]), key=lambda r: r["over_20_pct"])
    largest_district = max((r for r in groups["district"] if not r["small_group"]), key=lambda r: r["mae_tenge"])
    price_worst = max((r for r in groups["price_band"] if not r["small_group"]), key=lambda r: r["over_20_pct"])
    incomplete = max((r for r in groups["completeness"] if not r["small_group"]), key=lambda r: r["over_20_pct"])
    large = next(r for r in groups["area_band"] if r["group"] == "Более 120 м²")
    top_n = max(1, int(np.ceil(len(rows) * .1)))
    top_share = rows.absolute_error.nlargest(top_n).sum() / rows.absolute_error.sum() * 100
    findings = [
        f"На {len(test)} объявлениях MAE = {baseline['mae_tenge']/1e6:.2f} млн ₸. Ошибка больше 20% встречается в {baseline['over_20_pct']:.1f}% случаев ({baseline['over_20_n']} объявлений).",
        f"Среди районов с n ≥ {MIN_GROUP_SIZE} максимальная наблюдаемая доля ошибок >20% в этом test — {worst_district['group']}: {worst_district['over_20_pct']:.1f}% (n={worst_district['n']}). Самая большая MAE — {largest_district['group']}: {largest_district['mae_tenge']/1e6:.2f} млн ₸. Рейтинг описательный: интервалы долей перекрываются.",
        f"Ценовой сегмент с наибольшей частотой ошибки >20%: {price_worst['group']} — {price_worst['over_20_pct']:.1f}% (n={price_worst['n']}), MAE {price_worst['mae_tenge']/1e6:.2f} млн ₸, среднее смещение {price_worst['bias_tenge']/1e6:+.2f} млн ₸ (минус означает занижение).",
        f"Среди исходных анкет с n ≥ {MIN_GROUP_SIZE} наибольшая частота ошибки >20% у группы «{incomplete['group']}»: {incomplete['over_20_pct']:.1f}%, при небольшой группе n={incomplete['n']}. Квартиры площадью более 120 м²: {large['over_20_pct']:.1f}% ошибок >20%, MAE {large['mae_tenge']/1e6:.2f} млн ₸ (n={large['n']}).",
        f"При скрытии полей на тех же {len(cohort)} квартирах MAE составляет {controlled[0]['mae_tenge']/1e6:.2f} млн ₸ для площади и комнат и {full['mae_tenge']/1e6:.2f} млн ₸ для полной анкеты. Частота ошибки >20%: {controlled[0]['over_20_pct']:.1f}% и {full['over_20_pct']:.1f}% соответственно.",
        f"При скрытии одного поля наибольший рост MAE даёт поле «{ablation[0]['group']}»: {ablation[0]['delta_mae_tenge']/1e6:+.2f} млн ₸ на полной когорте.",
        f"На {top_n} самых трудных объявлений (верхние 10% по абсолютной ошибке, округление вверх) приходится {top_share:.1f}% суммы абсолютных ошибок. Диапазон этой модели ±MAE покрывает {baseline['coverage_pct']:.1f}% цен этого test.",
    ]
    dates = pd.to_datetime(test.collected_at, errors="raise").dropna()
    if len(dates) == len(test):
        data_period = f"{dates.min():%d.%m.%Y}–{dates.max():%d.%m.%Y} (даты сбора test)"
    else:
        data_period = metadata.get("data_period_label", "Период не указан") + "; даты отдельных объявлений отсутствуют"
    grouped = metadata.get("split_strategy") == "predefined_group_holdout"
    split_description = ("Заранее зафиксированный групповой holdout мартовских объявлений без найденных пересечений с январём"
                         if grouped else "Исходный случайный holdout объявлений января 2026 года")
    report = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": metadata["model_name"],
        "data_period": data_period, "split_description": split_description,
        "provenance": provenance, "n_train": metadata["train_rows"], "n_test": len(test),
        "n_app_eligible": int(eligible.sum()), "n_complete": len(cohort),
        "thresholds": {"relative_error_pct": RELATIVE_THRESHOLD, "absolute_error_tenge": ABSOLUTE_THRESHOLD, "min_group_size": MIN_GROUP_SIZE},
        "baseline": baseline, "groups": groups, "controlled_masking": controlled,
        "field_ablation": ablation, "ranked_segments": rankings,
        **{key: json.loads(value.to_json(orient="records", force_ascii=False)) for key, value in examples.items()},
        "findings": findings,
        "limitations": [
            f"Целевая величина — цена объявления, не сделки. Период: {data_period}. Этот срез не проверяет качество на текущем рынке или на горизонте 6 месяцев.",
            f"В ходе диагностики модель не переобучалась. {split_description}. Остаточные дубли и похожие квартиры могут делать результаты оптимистичными.",
            "Полнота — количество непустых значений восьми входных признаков до заполнения пропусков пайплайном. Исходные группы полноты состоят из разных квартир: сравнение не доказывает влияние пропусков.",
            "Контролируемые сценарии используют одни и те же полные валидные анкеты. Остальные поля заменяются пропусками только на входе модели. Это эксперимент со скрытием сведений, а не статистика поведения пользователей. Порядок добавления полей выбран явно: число полей само по себе не гарантирует улучшения.",
            f"Из {len(test)} test-строк {int(eligible.sum())} проходят текущую проверку формы; контролируемая когорта включает только полные и допустимые анкеты.",
            "Порог >20% выбран для диагностики, а не как обещание точности. Положительное смещение означает завышение, отрицательное — занижение. Высокая MAE и частая относительная ошибка — разные характеристики.",
            "Покрытие ±MAE измерено на том же test, по которому рассчитана ширина диапазона. Это описание существующего диапазона, не независимая калибровка и не гарантия покрытия.",
            "Группы с n < 30 показаны, но исключены из рейтинга. Это практический порог, не доказательство надёжности. 95% интервалы Уилсона описывают долю ошибок при предположении независимости объявлений и не учитывают поиск множества сегментов.",
            "Сегменты пересекаются. Ценовые группы определены по известной цене объявления и служат для анализа после получения результата. Они не являются доступным модели входным признаком.",
            "После выбора улучшений по этому отчёту старый test становится диагностическим. Итоговое качество следующей версии следует подтвердить на новых независимых данных.",
        ],
        "recommendations": [
            "Проверить исходные объявления с наибольшими денежными и относительными ошибками. Не удалять дорогие квартиры только потому, что модель ошиблась.",
            "Собирать свежие объявления с датами и устойчивыми идентификаторами квартир. Проверять новую модель по времени и отдельно на ранее не встречавшихся объектах.",
            "Изучить точную географию, ЖК и характеристики дорогих/больших квартир. Проверить, снижает ли их добавление ошибки выявленных сегментов на независимой выборке.",
            "Для неполной анкеты оценить отдельные диапазоны на выделенной калибровочной выборке и показывать фактическую полноту. Не обещать уменьшение ошибки от каждого дополнительного поля.",
        ],
    }
    exports = []
    for key, values in groups.items():
        name = f"by_{key}.csv"
        pd.DataFrame(values).to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        exports.append({"label": f"Метрики: {key}", "filename": name})
    for key, value in {"controlled_masking": pd.DataFrame(controlled), "field_ablation": pd.DataFrame(ablation),
                       "test_error_details": details, "masking_predictions": pd.concat(scenario_frames, ignore_index=True),
                       "ranked_segments": pd.DataFrame(rankings), **examples}.items():
        name = f"{key}.csv"
        value.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        exports.append({"label": key, "filename": name})
    report["exports"] = exports
    report["charts"] = make_charts(report, output_dir)
    (output_dir / "error_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from error_report import render_report
    render_report(report, output_dir / "error_report.html")
    print(json.dumps({"findings": findings, "output": str(output_dir / "error_report.html")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CLEAN_LISTINGS_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODEL_PIPELINE_PATH.parent)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(*(path if path.is_absolute() else PROJECT_ROOT / path for path in
           (args.input, args.models_dir, args.reports_dir, args.output_dir)))

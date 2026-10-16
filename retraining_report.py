"""Paired, descriptive comparisons on one previously inspected archive holdout.

The caller freezes sources, split, models and predictions. These pure functions
never fit a model, select a candidate or write files. Predictions are prices in
tenge, not log prices. Cohorts are fixed once and shared by all three models.
"""
from __future__ import annotations

from html import escape
from typing import Mapping

import numpy as np
import pandas as pd

from active_error_analysis import MIN_SEGMENT_ROWS, segment_labels


MODEL_KEYS = ("current", "repair_only", "trained_geo")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
TIE_TOLERANCE_TENGE = 0.05
MODEL_LABELS = {"current": "Текущая модель с OSM", "repair_only": "Исправленные данные",
                "trained_geo": "Исправленные данные и обучение с OSM"}
LIMITATIONS = [
    "Это повторное сравнение на ранее изученном мартовском holdout, а не свежая независимая проверка.",
    "Реестр OSM зафиксирован заранее; его отбор использовал адреса и признаки полного каталога, включая test. Это не проверка при полностью неизвестных объектах.",
    "Карта OSM октября 2026 года не подтверждает историческое состояние адресов в январе–марте.",
    "Парные 95% bootstrap-интервалы описательны: они учитывают зависимость внутри переданных групп, но не неопределённость обучения, последующий выбор модели или прежнее изучение holdout.",
    "Группы домов определяются эвристикой; неполные адреса могут скрывать связи между объявлениями.",
    "Сегменты одинаковы для всех моделей; знакомство с ЖК и география заданы относительно текущей модели. Группы с n<30 отмечены как малые.",
    "Оценивается архивная цена предложения. Результат не является оценкой текущего рынка или цены сделки.",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate(frame: pd.DataFrame, predictions: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique, "Нужен DataFrame с уникальными колонками")
    _require({"record_id", "price"}.issubset(frame.columns), "Нет record_id или price")
    _require(len(frame) > 0 and frame.index.is_unique, "Нужны непустые строки с уникальным индексом")
    _require(frame.record_id.notna().all() and frame.record_id.is_unique
             and frame.record_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
             "record_id должен быть непустым и уникальным")
    _require(isinstance(predictions, Mapping) and set(predictions) == set(MODEL_KEYS),
             "Нужны ровно current, repair_only, trained_geo")
    try:
        actual = pd.to_numeric(frame.price, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("price должен быть числовым") from error
    _require(np.isfinite(actual).all() and (actual > 0).all(), "price должен быть положительным и конечным")
    arrays = {}
    for model in MODEL_KEYS:
        values = predictions[model]
        if isinstance(values, pd.Series):
            _require(values.index.equals(frame.index), f"Индекс прогноза {model} не совпадает с frame")
        try:
            raw = np.asarray(values)
            _require(raw.dtype.kind != "b", f"Прогноз {model} не должен быть boolean")
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Прогноз {model} должен быть числовым") from error
        _require(array.ndim == 1 and len(array) == len(frame), f"Размер прогноза {model} не совпадает с frame")
        _require(np.isfinite(array).all(), f"Прогноз {model} содержит неконечные значения")
        arrays[model] = array.copy()
    return actual, arrays


def _aligned_labels(frame: pd.DataFrame, labels: pd.Series, name: str) -> None:
    _require(isinstance(labels, pd.Series) and labels.index.equals(frame.index),
             f"Индекс {name} не совпадает с frame")
    _require(labels.notna().all() and labels.map(lambda value: not isinstance(value, str) or bool(value.strip())).all(),
             f"Метки {name} должны быть непустыми")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    absolute = np.abs(error)
    squared = np.square(error)
    ape = absolute / actual * 100
    denominator = float(np.square(actual - actual.mean()).sum())
    return {
        "n": len(actual), "mae_tenge": float(absolute.mean()),
        "rmse_tenge": float(np.sqrt(squared.mean())),
        "r2": float(1 - squared.sum() / denominator) if len(actual) >= 2 and denominator > 0 else None,
        "median_ae_tenge": float(np.median(absolute)), "median_ape_pct": float(np.median(ape)),
        "over_20_n": int((ape > 20).sum()), "over_20_pct": float((ape > 20).mean() * 100),
    }


def compare_predictions(frame: pd.DataFrame, predictions: Mapping[str, np.ndarray], cluster_labels: pd.Series) -> dict:
    """Compare paired absolute errors using row-weighted whole-cluster bootstrap.

    Each replicate samples G of the G supplied clusters with replacement, then
    retains every member of each sampled cluster, including multiplicities.
    Cluster sizes may vary; the statistic remains per-listing MAE, not the mean
    of cluster MAEs. The same draws are used for both candidate comparisons.
    """
    actual, arrays = _validate(frame, predictions)
    _aligned_labels(frame, cluster_labels, "cluster_labels")
    try:
        codes, names = pd.factorize(cluster_labels, sort=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Метки cluster_labels должны быть хешируемыми") from error
    n_clusters = len(names)
    sizes = np.bincount(codes, minlength=n_clusters)
    absolute = {model: np.abs(arrays[model] - actual) for model in MODEL_KEYS}
    overall = {model: _metrics(actual, arrays[model]) for model in MODEL_KEYS}
    base_sums = np.bincount(codes, weights=absolute["current"], minlength=n_clusters)
    differences = {model: absolute[model] - absolute["current"] for model in MODEL_KEYS[1:]}
    delta_sums = {model: np.bincount(codes, weights=values, minlength=n_clusters)
                  for model, values in differences.items()}
    delta_samples = {model: np.empty(BOOTSTRAP_REPLICATES) for model in differences}
    percent_samples = {model: np.full(BOOTSTRAP_REPLICATES, np.nan) for model in differences}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for iteration in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        rows = sizes[sampled].sum()
        base = base_sums[sampled].sum()
        for model in differences:
            delta = delta_sums[model][sampled].sum()
            delta_samples[model][iteration] = delta / rows
            if base > 0:
                percent_samples[model][iteration] = delta / base * 100
    paired = {}
    baseline_mae = overall["current"]["mae_tenge"]
    for model, difference in differences.items():
        delta = float(difference.mean())
        improved = int((difference < -TIE_TOLERANCE_TENGE).sum())
        worsened = int((difference > TIE_TOLERANCE_TENGE).sum())
        valid_percent = np.isfinite(percent_samples[model])
        paired[model] = {
            "mae_delta_tenge": delta,
            "mae_delta_pct": delta / baseline_mae * 100 if baseline_mae > 0 else None,
            "mae_delta_ci95_tenge": np.quantile(delta_samples[model], [.025, .975]).tolist(),
            "mae_delta_ci95_pct": (np.quantile(percent_samples[model], [.025, .975]).tolist()
                                   if valid_percent.all() else None),
            "percent_bootstrap_valid_replicates": int(valid_percent.sum()),
            "improved_rows": improved, "worsened_rows": worsened,
            "tied_rows": len(frame) - improved - worsened,
        }
    return {
        "schema_version": 1, "n": len(frame), "record_ids": frame.record_id.tolist(),
        "overall": overall, "paired": paired,
        "bootstrap": {"unit": "house/apartment equivalence cluster", "clusters": n_clusters,
                      "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "confidence_level": .95, "row_weighted": True,
                      "interval": "paired percentile cluster bootstrap", "descriptive_only": True,
                      "too_few_clusters": n_clusters < 30},
        "tie_tolerance_tenge": TIE_TOLERANCE_TENGE, "limitations": list(LIMITATIONS),
    }


def build_segments(frame: pd.DataFrame, predictions: Mapping[str, np.ndarray], known_complexes: set[str],
                   geography: pd.Series) -> pd.DataFrame:
    """Use the active-model cohorts once, with all models on identical rows."""
    actual, arrays = _validate(frame, predictions)
    _aligned_labels(frame, geography, "geography")
    labels = segment_labels(frame, known_complexes, geography)
    rows = []
    for dimension, values in labels.items():
        _aligned_labels(frame, values, dimension)
        for name in sorted(values.unique(), key=str):
            mask = values.eq(name).to_numpy(dtype=bool)
            baseline = _metrics(actual[mask], arrays["current"][mask])["mae_tenge"]
            for model in MODEL_KEYS:
                scores = _metrics(actual[mask], arrays[model][mask])
                delta = scores["mae_tenge"] - baseline
                rows.append({"dimension": dimension, "segment": str(name), "model": model, **scores,
                             "mae_delta_tenge": delta,
                             "mae_delta_pct": delta / baseline * 100 if baseline > 0 else None,
                             "small_group": scores["n"] < MIN_SEGMENT_ROWS,
                             "eligible_for_ranking": scores["n"] >= MIN_SEGMENT_ROWS})
    return pd.DataFrame(rows)


def _format(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}".replace(",", " ").replace(".", ",")
    return str(value)


def _table(rows: list[dict], columns: dict[str, str]) -> str:
    head = "".join(f"<th>{escape(label)}</th>" for label in columns.values())
    body = "".join("<tr>" + "".join(f"<td>{escape(_format(row.get(key)))}</td>" for key in columns) + "</tr>"
                   for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_html(report: dict, segments: pd.DataFrame) -> str:
    """Return a self-contained Russian HTML report, escaping all supplied text."""
    comparison = report.get("comparison", report)
    model_labels = {**MODEL_LABELS, **report.get("model_labels", {})}
    overall = [{"model_label": model_labels[model], **comparison["overall"][model]} for model in MODEL_KEYS]
    paired = []
    for model in MODEL_KEYS[1:]:
        values = comparison["paired"][model]
        interval = values["mae_delta_ci95_tenge"]
        paired.append({"model_label": model_labels[model], **values,
                       "interval_tenge": " … ".join(_format(value) for value in interval)})
    selected = segments.loc[segments.dimension.isin(["district", "area_band", "completeness", "geography"])].copy()
    selected["model_label"] = selected.model.map(model_labels)
    selected["size_note"] = np.where(selected.small_group, "Малая группа: n<30", "n≥30")
    selected["dimension_label"] = selected.dimension.map({"district": "Район", "area_band": "Площадь",
                                                          "completeness": "Заполнено полей", "geography": "География текущей модели"})
    combined_limitations = list(dict.fromkeys([*comparison.get("limitations", LIMITATIONS), *report.get("limitations", [])]))
    limitations = "".join(f"<li>{escape(str(value))}</li>" for value in combined_limitations)
    boot = comparison["bootstrap"]
    cluster_note = " Число групп меньше 30; интервал особенно нестабилен." if boot.get("too_few_clusters") else ""
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Переобучение после проверки архивных данных</title><style>
body{{font:16px/1.55 system-ui,sans-serif;background:#f1f5f7;color:#17303c;margin:0;padding:24px}}main{{max-width:1280px;margin:auto;background:white;padding:28px;border-radius:14px}}h1{{line-height:1.2}}h2{{margin-top:32px}}.note{{padding:16px;background:#e9f2f5;border-left:4px solid #367485}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:16px 0}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #dae4e8}}th{{background:#e9f2f5}}li{{margin:8px 0}}@media(max-width:600px){{body{{padding:8px}}main{{padding:15px}}}}</style>
<main><h1>Переобучение после проверки данных</h1>
<p>Три модели · {int(comparison['n'])} одинаковых архивных объявлений · цены предложения в тенге.</p>
<p class="note">Это описательное сравнение на ранее изученном holdout. Отрицательная разность MAE означает меньшую ошибку кандидата; положительная — большую. Результат не подтверждает качество на свежем рынке.</p>
<h2>Общие ошибки</h2>{_table(overall, {'model_label':'Модель','n':'n','mae_tenge':'MAE, ₸','rmse_tenge':'RMSE, ₸','r2':'R²','median_ae_tenge':'Медианная |ошибка|, ₸','median_ape_pct':'Медианная |ошибка|, %','over_20_n':'Ошибка >20%, квартир','over_20_pct':'Ошибка >20%, %'})}
<h2>Парные различия относительно текущей модели</h2>
<p>Пересэмплируются целиком {int(boot['clusters'])} групп домов/связанных квартир: {int(boot['replicates'])} повторов, seed {int(boot['seed'])}. Сохраняется вес каждой квартиры, включая разные размеры домов.{escape(cluster_note)}</p>
{_table(paired, {'model_label':'Кандидат','mae_delta_tenge':'Δ MAE, ₸','mae_delta_pct':'Δ MAE, %','interval_tenge':'95% bootstrap-интервал Δ, ₸','improved_rows':'Ошибка уменьшилась','worsened_rows':'Ошибка выросла','tied_rows':'Без изменения'})}
<p>Изменения абсолютной ошибки до {escape(_format(comparison['tie_tolerance_tenge']))} ₸ включительно считаются равными только при подсчёте строк. Метрики и интервалы используют исходные значения.</p>
<h2>Одинаковые сегменты для всех моделей</h2>
<p>Малые группы показаны с предупреждением. ЖК, знакомство с ЖК и ценовые сегменты также сохранены в <a href="segments.csv">полном CSV</a>. Сегментная разность не доказывает причину улучшения или ухудшения.</p>
{_table(selected.to_dict('records'), {'dimension_label':'Разрез','segment':'Сегмент','model_label':'Модель','n':'n','mae_tenge':'MAE, ₸','mae_delta_tenge':'Δ MAE, ₸','over_20_pct':'Ошибка >20%, %','size_note':'Размер группы'})}
<h2>Ограничения</h2><ul>{limitations}</ul>
</main></html>'''

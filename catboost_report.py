"""Pure CatBoost/RF comparison reporting; no fitting, model selection or file writes.

All predictions must already be in tenge, including inverse-transformed log-price
predictions. Models share one frozen cohort and cluster bootstrap resamples.
"""
from __future__ import annotations

from html import escape
from typing import Mapping

import numpy as np
import pandas as pd

from active_error_analysis import MIN_SEGMENT_ROWS, segment_labels


MODEL_KEYS = ("current", "catboost_mae", "catboost_log", "rf_control")
REQUIRED_MODELS = frozenset(MODEL_KEYS[:3])
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
TIE_TOLERANCE_TENGE = 0.05
MODEL_LABELS = {"current": "Текущая модель с OSM", "catboost_mae": "CatBoost: MAE в тенге",
                "catboost_log": "CatBoost: логарифм цены", "rf_control": "Random Forest: 8 признаков + ЖК",
                "selected_procedure": "Внутренний выбор CatBoost"}
LIMITATIONS = [
    "590 мартовских объявлений уже изучались ранее. Это архивное сравнение, не свежая независимая проверка.",
    "CatBoost и RF-контроль используют восемь исходных признаков и ЖК, без координат. Текущая модель использует также OSM: сравнение с ней одновременно меняет алгоритм и набор признаков.",
    "RF-контроль на тех же признаках помогает сравнить алгоритмы. Он не участвует в выборе семейства CatBoost или его параметров.",
    "Внешние OOF-прогнозы nested CV оценивают процедуру внутреннего выбора на отделённых группах домов. Полная train CV служит окончательному выбору; её лучший результат не является независимой оценкой выбранной модели.",
    "Все денежные метрики, включая выбор log-варианта, должны вычисляться после обратного преобразования прогноза в тенге.",
    "Парные 95% bootstrap-интервалы описательны: учитывают переданные группы, но не неопределённость обучения, множественные сравнения и прежние решения по holdout.",
    "Группировка домов эвристическая; неполные адреса могут скрывать зависимости. Географические сегменты заданы по текущей модели и не означают использование координат кандидатами.",
    "Октябрьский OSM в текущей модели не подтверждает историческое состояние адресов. Прогнозируется архивная цена предложения, не сегодняшняя цена сделки.",
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
    _require(isinstance(predictions, Mapping) and REQUIRED_MODELS.issubset(predictions)
             and set(predictions).issubset(MODEL_KEYS),
             "Нужны current, catboost_mae, catboost_log; rf_control необязателен")
    try:
        actual = pd.to_numeric(frame.price, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("price должен быть числовым") from error
    _require(np.isfinite(actual).all() and (actual > 0).all(), "price должен быть положительным и конечным")
    arrays = {}
    for model in (key for key in MODEL_KEYS if key in predictions):
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


def compare_predictions(frame: pd.DataFrame, predictions: Mapping[str, np.ndarray], clusters: pd.Series) -> dict:
    """Compare paired absolute errors using row-weighted whole-cluster bootstrap.

    Each replicate samples G of the G supplied clusters with replacement, then
    retains every member of each sampled cluster, including multiplicities.
    Cluster sizes may vary; the statistic remains per-listing MAE, not the mean
    of cluster MAEs. The same draws are used for every candidate comparison.
    """
    actual, arrays = _validate(frame, predictions)
    _aligned_labels(frame, clusters, "clusters")
    try:
        codes, names = pd.factorize(clusters, sort=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Метки clusters должны быть хешируемыми") from error
    n_clusters = len(names)
    sizes = np.bincount(codes, minlength=n_clusters)
    absolute = {model: np.abs(arrays[model] - actual) for model in arrays}
    overall = {model: _metrics(actual, arrays[model]) for model in arrays}
    base_sums = np.bincount(codes, weights=absolute["current"], minlength=n_clusters)
    differences = {model: absolute[model] - absolute["current"] for model in arrays if model != "current"}
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
        "model_keys": list(arrays), "overall": overall, "paired": paired,
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
            for model in (key for key in MODEL_KEYS if key in predictions):
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


def _cv_rows(values: object, labels: dict[str, str]) -> list[dict]:
    """Accept agreed list rows or a variant->metrics mapping; never rank/select."""
    if values is None:
        return []
    if isinstance(values, Mapping):
        values = [{"variant": variant, **metrics} for variant, metrics in values.items()]
    _require(isinstance(values, list), "CV rows должны быть списком или mapping по variant")
    result = []
    for record in values:
        _require(isinstance(record, Mapping), "Некорректная строка CV")
        metrics = record.get("metrics", {})
        _require(isinstance(metrics, Mapping), "metrics CV должен быть объектом")
        row = {**metrics, **record}
        variant = row.get("variant", row.get("model", ""))
        row["model_label"] = labels.get(variant, str(variant))
        row["params_label"] = row.get("params_id", row.get("selected_params_id"))
        row["selection_note"] = "Да" if row.get("selected") is True else "Нет"
        result.append(row)
    return result


def render_html(report: dict, segments: pd.DataFrame) -> str:
    """Render already locked choices, nested CV and holdout as separate evidence."""
    comparison = report["comparison"]
    _require(report.get("holdout_rows", comparison["n"]) == comparison["n"], "Число holdout расходится с comparison")
    labels = {**MODEL_LABELS, **report.get("model_labels", {})}
    keys = [key for key in MODEL_KEYS if key in comparison["overall"]]
    _require(REQUIRED_MODELS.issubset(keys), "Comparison потерял обязательную модель")
    chosen = report["selected_variant"]
    _require(chosen in {"catboost_mae", "catboost_log"}, "Выбор допускается только среди CatBoost")
    selection = report.get("selection_cv", {})
    _require(selection.get("chosen_variant", chosen) == chosen, "selected_variant расходится с selection_cv")
    nested = report.get("nested_cv", {})
    summary = _cv_rows(nested.get("summary", []), labels)
    folds = _cv_rows(nested.get("folds", []), labels)
    selection_rows = _cv_rows(selection.get("results", []), labels)
    overall = [{"model_label": labels[key], **comparison["overall"][key]} for key in keys]
    paired = []
    for key in keys:
        if key == "current":
            continue
        row = comparison["paired"][key]
        paired.append({"model_label": labels[key], **row,
                       "interval_tenge": " … ".join(_format(value) for value in row["mae_delta_ci95_tenge"])})
    selected = segments.loc[segments.dimension.isin(["district", "area_band", "completeness", "geography"])].copy()
    selected["model_label"] = selected.model.map(labels)
    selected["size_note"] = np.where(selected.small_group, "Малая группа: n<30", "n≥30")
    selected["dimension_label"] = selected.dimension.map({"district": "Район", "area_band": "Площадь",
                                                          "completeness": "Заполнено полей", "geography": "География текущей модели"})
    limitations = list(dict.fromkeys([*comparison.get("limitations", LIMITATIONS), *report.get("limitations", [])]))
    limits_html = "".join(f"<li>{escape(str(value))}</li>" for value in limitations)
    boot = comparison["bootstrap"]
    outer = nested.get("outer_folds", 3)
    inner = nested.get("inner_folds", 3)
    fold_details = (_table(folds, {"outer_fold": "Внешний фолд", "model_label": "Вариант",
                                  "params_label": "Параметры из inner CV", "n": "n", "mae_tenge": "MAE, ₸"})
                    if folds else "<p>Детализация фолдов не передана.</p>")
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CatBoost: выбор внутри CV и архивное сравнение</title><style>
body{{font:16px/1.55 system-ui,sans-serif;background:#f1f5f7;color:#17303c;margin:0;padding:24px}}main{{max-width:1280px;margin:auto;background:white;padding:28px;border-radius:14px}}h1{{line-height:1.2}}h2{{margin-top:32px}}.note{{padding:16px;background:#e9f2f5;border-left:4px solid #367485}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:16px 0}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #dae4e8}}th{{background:#e9f2f5}}li{{margin:8px 0}}@media(max-width:600px){{body{{padding:8px}}main{{padding:15px}}}}</style>
<main><h1>CatBoost: обучение и проверка</h1>
<p>{int(report['training_rows'])} обучающих объявлений · {int(comparison['n'])} одинаковых контрольных объявлений · {len(keys)} модели в сравнении.</p>
<p class="note">До оценки holdout выбрано: <strong>{escape(str(labels[chosen]))}</strong>, по полной train CV. Выбор только среди двух семейств CatBoost. RF служит контролем; результаты holdout не используются для изменения выбора.</p>
<h2>Какие признаки сравниваются</h2><p>Оба CatBoost и RF-контроль получают восемь исходных признаков и ЖК, без координат. Текущая модель дополнительно использует OSM. Разница с текущей моделью отражает и алгоритм, и набор признаков; RF-контроль позволяет отдельно сопоставить алгоритмы на одинаковых признаках.</p>
<h2>Внешняя проверка nested CV</h2><p>{escape(str(outer))} внешних × {escape(str(inner))} внутренних фолда с группировкой домов и связанных квартир. Параметры подбираются во внутренней части. Строка «Внутренний выбор CatBoost» объединяет внешние прогнозы варианта, выбранного без доступа к этому внешнему фолду. Pooled OOF MAE взвешивает каждую квартиру одинаково.</p>
{_table(summary, {'model_label':'Процедура / вариант','n':'n OOF','mae_tenge':'MAE OOF, ₸','rmse_tenge':'RMSE OOF, ₸','median_ae_tenge':'Медианная |ошибка|, ₸','over_20_pct':'Ошибка >20%, %'})}
<details><summary>Результаты внешних фолдов</summary>{fold_details}</details>
<h2>Окончательный выбор на полном train</h2><p>Эти CV-результаты используются для выбора параметров и семейства CatBoost. Минимум этой таблицы не заменяет внешнюю OOF-оценку процедуры. Для log-варианта MAE также выражена в тенге после обратного преобразования.</p>
{_table(selection_rows, {'model_label':'Вариант','params_label':'Набор параметров','n':'n OOF','mae_tenge':'MAE, ₸','selection_note':'Выбранный набор'})}
<h2>Результаты на прежних контрольных объявлениях</h2><p>Holdout уже изучался раньше: это описательное архивное сравнение, не новая независимая проверка и не оценка сегодняшнего рынка.</p>
{_table(overall, {'model_label':'Модель','n':'n','mae_tenge':'MAE, ₸','rmse_tenge':'RMSE, ₸','r2':'R²','median_ae_tenge':'Медианная |ошибка|, ₸','median_ape_pct':'Медианная |ошибка|, %','over_20_n':'Ошибка >20%, квартир','over_20_pct':'Ошибка >20%, %'})}
<h2>Парные различия относительно текущей модели</h2><p>Δ = кандидат − текущая модель: отрицательное значение лучше. Пересэмплируются целиком {int(boot['clusters'])} групп, {int(boot['replicates'])} повторов, seed {int(boot['seed'])}; сохраняется вес каждой квартиры. Интервалы описательные, без поправки на множественные сравнения и прежнее изучение holdout.</p>
{_table(paired, {'model_label':'Кандидат','mae_delta_tenge':'Δ MAE, ₸','mae_delta_pct':'Δ MAE, %','interval_tenge':'95% bootstrap Δ, ₸','improved_rows':'Ошибка уменьшилась','worsened_rows':'Ошибка выросла','tied_rows':'Без изменения'})}
<p>Допуск при подсчёте неизменившихся ошибок: {escape(_format(comparison['tie_tolerance_tenge']))} ₸. Сами метрики не округляются перед сравнением.</p>
<h2>Общие сегменты</h2><p>Состав сегментов одинаков для всех моделей. География и знакомство с ЖК определены по текущей модели; это не признаки CatBoost. Малые группы отмечены. Все разрезы, включая ЖК и цену, сохранены в <a href="segments.csv">CSV</a>.</p>
{_table(selected.to_dict('records'), {'dimension_label':'Разрез','segment':'Сегмент','model_label':'Модель','n':'n','mae_tenge':'MAE, ₸','mae_delta_tenge':'Δ MAE, ₸','over_20_pct':'Ошибка >20%, %','size_note':'Размер группы'})}
<h2>Ограничения</h2><ul>{limits_html}</ul></main></html>'''



"""Автономный HTML-отчёт по заранее рассчитанным ошибкам модели."""

from __future__ import annotations

import base64
import html
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import quote, urlsplit


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int = 1, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    spec = f"{'+' if signed else ''},.{digits}f"
    return format(number, spec).replace(",", "\u202f").replace(".", ",")


def _money(value: Any, *, signed: bool = False) -> str:
    return _number(None if value is None else float(value) / 1_000_000, 2, signed=signed)


def _list(items: list[str], *, ordered: bool = False) -> str:
    tag = "ol" if ordered else "ul"
    return f"<{tag}>" + "".join(f"<li>{_text(item)}</li>" for item in items) + f"</{tag}>"


def _group_table(rows: list[dict[str, Any]], caption: str, *, delta: bool = False) -> str:
    if not rows:
        return '<p class="empty">Для этого разреза нет наблюдений.</p>'
    columns = [
        ("n", "n", "Число объявлений", "integer"),
        ("mae_tenge", "MAE", "Средняя абсолютная ошибка, млн ₸", "money"),
        ("median_ape_pct", "MedAPE", "Медианная абсолютная ошибка, % от реальной цены", "percent"),
        ("over_20_pct", "Ошибка >20%", "Доля объявлений с абсолютной ошибкой больше 20%", "percent"),
        ("bias_tenge", "Смещение", "Среднее (прогноз − реальная цена), млн ₸", "signed_money"),
        ("coverage_pct", "Внутри ±MAE", "Доля реальных цен внутри диапазона прогноз ± MAE исходного test", "percent"),
    ]
    if delta:
        columns.insert(2, ("delta_mae_tenge", "Δ MAE", "Изменение MAE относительно всех восьми полей на той же когорте, млн ₸", "signed_money"))
    header = '<th scope="col">Группа</th>' + "".join(
        f'<th scope="col" title="{_text(description)}">{_text(label)}</th>'
        for _, label, description, _ in columns
    )
    body = []
    for row in rows:
        badge = '<span class="badge">мало данных</span>' if row.get("small_group") else ""
        fields = row.get("fields_kept")
        field_note = ""
        if fields is not None:
            names = {
                "area": "площадь", "rooms": "комнаты", "floor": "этаж",
                "total_floors": "этажность", "year_built": "год постройки",
                "district": "район", "building_type": "тип дома", "condition": "состояние",
            }
            field_note = '<small class="field-note">Известны: ' + _text(
                ", ".join(names.get(field, field) for field in fields) or "нет"
            ) + "</small>"
        cells = [f'<th scope="row">{_text(row.get("group", "—"))}{badge}{field_note}</th>']
        for key, _, _, style in columns:
            value = row.get(key)
            if style == "money":
                displayed = _money(value)
            elif style == "signed_money":
                displayed = _money(value, signed=True)
            elif style == "integer":
                displayed = _number(value, 0)
            else:
                displayed = _number(value) + ("%" if value is not None else "")
            cells.append(f'<td class="numeric">{displayed}</td>')
        row_class = ' class="small-sample"' if row.get("small_group") else ""
        body.append(f"<tr{row_class}>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-scroll" tabindex="0" role="region" aria-label="' + _text(caption) + '">'
        '<table><caption>' + _text(caption) + '</caption><thead><tr>' + header + '</tr></thead><tbody>'
        + "".join(body) + '</tbody></table></div>'
    )


def _listing_link(value: Any, label: str) -> str:
    if not isinstance(value, str):
        return _text(label)
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _text(label)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {"krisha.kz", "www.krisha.kz"}:
        return _text(label)
    return f'<a href="{_text(value)}" target="_blank" rel="noopener noreferrer">{_text(label)} ↗</a>'


def _example_table(rows: list[dict[str, Any]], caption: str) -> str:
    if not rows:
        return '<p class="empty">Нет примеров.</p>'
    body = []
    for row in rows:
        source = _number(row.get("source_row"), 0)
        label = _listing_link(row.get("url"), f"Строка {source}")
        district = row.get("district") or "Район не указан"
        complex_name = row.get("residential_complex")
        context = _text(district) + (" · " + _text(complex_name) if complex_name else "")
        signed = row.get("signed_error")
        direction = "завышение" if signed is not None and signed > 0 else "занижение" if signed is not None and signed < 0 else "совпадение"
        body.append(
            '<tr><th scope="row">' + label + f'<small class="field-note">{context}</small></th>'
            f'<td class="numeric">{_number(row.get("area"))}</td>'
            f'<td class="numeric">{_number(row.get("rooms"), 0)}</td>'
            f'<td class="numeric">{_number(row.get("filled_fields"), 0)}/8</td>'
            f'<td class="numeric">{_money(row.get("actual_price"))}</td>'
            f'<td class="numeric">{_money(row.get("predicted_price"))}</td>'
            f'<td class="numeric emphasis">{_money(row.get("absolute_error"))}</td>'
            f'<td class="numeric">{_number(row.get("ape_pct"))}%</td>'
            f'<td>{direction}</td></tr>'
        )
    headers = ["Объявление", "м²", "Комнаты", "Поля", "Цена", "Прогноз", "Ошибка", "Ошибка, %", "Направление"]
    return (
        '<div class="table-scroll" tabindex="0" role="region" aria-label="' + _text(caption) + '">'
        '<table class="examples"><caption>' + _text(caption) + '</caption><thead><tr>'
        + "".join(f'<th scope="col">{_text(label)}</th>' for label in headers)
        + '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'
    )


def _charts(charts: list[dict[str, Any]], root: Path) -> str:
    if not charts:
        return ""
    figures = []
    for chart in charts:
        path = (root / chart["filename"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("График должен находиться в папке отчёта или её подпапке.")
        payload = path.read_bytes()
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Ожидается PNG-график: {path.name}")
        encoded = base64.b64encode(payload).decode("ascii")
        title = _text(chart["title"])
        caption = _text(chart.get("caption", ""))
        figures.append(
            f'<figure><h3>{title}</h3><img src="data:image/png;base64,{encoded}" '
            f'alt="{title}" loading="lazy"><figcaption>{caption}</figcaption></figure>'
        )
    return '<section id="charts"><div class="section-heading"><span>02</span><h2>Ошибки на графиках</h2></div><div class="charts">' + "".join(figures) + '</div></section>'


CSS = """
:root{--ink:#163047;--muted:#587081;--teal:#0f766e;--teal-soft:#e7f4f0;--paper:#fff;--bg:#f7f9fc;--line:#dce5eb;--amber:#875012;--amber-soft:#fff7e6;--orange:#c76831}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Segoe UI",Arial,sans-serif}a{color:var(--teal);text-underline-offset:3px}a:hover{color:#09564f}a:focus-visible,summary:focus-visible,.table-scroll:focus-visible{outline:3px solid var(--orange);outline-offset:4px}.container{max-width:1220px;margin:auto;padding:0 32px}.hero{border-top:7px solid var(--teal);background:#fff;padding:44px 0 32px;border-bottom:1px solid var(--line)}.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:700;color:var(--teal);margin:0 0 14px}h1{font-size:clamp(30px,4vw,48px);line-height:1.15;letter-spacing:-.035em;max-width:850px;margin:0 0 20px;font-weight:750}.lead{max-width:870px;font-size:18px;color:var(--muted);margin:0}.meta{display:flex;gap:10px 24px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin:24px 0 0}.meta strong{color:var(--ink)}nav{display:flex;gap:12px 24px;flex-wrap:wrap;margin-top:26px;font-size:14px}nav a{text-decoration:none;border-bottom:1px solid #a9cec8}main{padding-bottom:56px}section{margin-top:40px}.section-heading{display:flex;gap:12px;align-items:baseline;margin-bottom:18px}.section-heading>span{color:var(--teal);font-weight:700;font-size:13px;letter-spacing:.04em}h2{font-size:25px;line-height:1.25;letter-spacing:-.02em;margin:0}h3{font-size:18px;line-height:1.35;margin:0 0 12px}p{margin:0 0 15px}.muted,.section-intro{color:var(--muted)}.section-intro{max-width:960px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.metric{padding:20px 22px;background:#fff;border:1px solid var(--line);border-radius:12px}.metric.primary{background:var(--teal);border-color:var(--teal);color:white}.metric .label{font-size:13px;font-weight:600;line-height:1.4;min-height:36px}.metric .value{font-size:34px;font-weight:700;letter-spacing:-.035em;line-height:1.2;margin:12px 0 9px}.metric .unit{font-size:17px;font-weight:500;letter-spacing:0}.metric .detail{font-size:12px;line-height:1.5;color:var(--muted)}.metric.primary .detail{color:#d4efea}.insights{border-left:4px solid var(--teal);padding:20px 26px;background:var(--teal-soft);border-radius:0 10px 10px 0}.insights h3{color:#0b5852}ul,ol{margin:0;padding-left:22px}li+li{margin-top:9px}.note{padding:16px 20px;border:1px solid #e9d3a9;border-radius:10px;background:var(--amber-soft);color:#6e471f;font-size:14px;margin:20px 0}.note strong{color:#604016}.glossary{display:grid;grid-template-columns:repeat(2,1fr);gap:14px 28px;margin:24px 0}.glossary div{padding-top:12px;border-top:1px solid var(--line)}.glossary dt{font-weight:650;margin-bottom:3px}.glossary dd{margin:0;color:var(--muted);font-size:14px}.table-scroll{overflow-x:auto;background:#fff;border:1px solid var(--line);border-radius:10px;margin:16px 0 22px}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}caption{text-align:left;color:var(--muted);padding:13px 16px;font-size:12px;border-bottom:1px solid var(--line)}th,td{padding:12px 14px;vertical-align:top;border-bottom:1px solid #e9eef2}thead th{background:#edf3f6;color:#3e5667;font-size:12px;text-align:right;white-space:nowrap}thead th:first-child{text-align:left}tbody th{font-weight:600;text-align:left;min-width:170px;max-width:300px}tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#f3f8f7}.numeric{text-align:right;white-space:nowrap}.small-sample{background:#fffbf1}.badge{display:inline-block;margin-left:7px;margin-top:2px;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:600;background:#f9e8c2;color:var(--amber);white-space:nowrap}.field-note{display:block;font-size:11px;font-weight:400;color:var(--muted);line-height:1.45;margin-top:4px;min-width:140px}.emphasis{font-weight:700;color:#9b4828}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}figure{margin:0;padding:18px;background:#fff;border:1px solid var(--line);border-radius:12px}figure img{display:block;width:100%;height:auto;margin:10px 0}figcaption{font-size:13px;color:var(--muted);line-height:1.5}.charts figure:only-child{grid-column:1/-1;max-width:100%}details{border:1px solid var(--line);background:#fff;border-radius:10px;margin:14px 0;padding:0 20px}summary{cursor:pointer;padding:17px 0;font-weight:600;color:var(--teal)}details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:18px}.details-body{padding-bottom:5px}.examples{font-size:12px}.examples tbody th{min-width:210px}.exports{display:flex;flex-wrap:wrap;gap:10px}.export{display:inline-block;background:#fff;border:1px solid #a6c9c2;border-radius:7px;padding:9px 14px;font-size:13px;text-decoration:none}.technical{font-size:12px;color:var(--muted)}.hashes{display:grid;grid-template-columns:190px 1fr;gap:8px 14px;font-size:12px}.hashes dt{font-weight:600}.hashes dd{margin:0;overflow-wrap:anywhere}code{font:11px/1.5 Consolas,monospace}.footer{border-top:1px solid var(--line);padding:24px 0 34px;font-size:12px;color:var(--muted)}.empty{padding:16px;color:var(--muted)}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.method-grid>div{background:white;border:1px solid var(--line);border-radius:10px;padding:22px}.print-note{display:none}
@media(max-width:850px){.container{padding:0 20px}.hero{padding-top:32px}.cards{grid-template-columns:repeat(2,1fr)}.charts,.method-grid{grid-template-columns:1fr}h2{font-size:23px}.metric .value{font-size:30px}.hashes{grid-template-columns:1fr;gap:4px}.hashes dd{margin-bottom:9px}}
@media(max-width:520px){.container{padding:0 14px}.hero{padding-top:26px}.lead{font-size:16px}.cards{gap:9px}.metric{padding:16px 13px}.metric .value{font-size:27px}.metric .unit{font-size:13px}.metric .label{font-size:12px;min-height:34px}.glossary{grid-template-columns:1fr}.insights{padding:18px}nav{gap:10px 17px}.section-heading{gap:9px}th,td{padding:11px}section{margin-top:32px}}
@media print{@page{size:A4 landscape;margin:13mm}html{scroll-behavior:auto}body{font-size:10px;background:#fff;-webkit-print-color-adjust:exact;print-color-adjust:exact}.container{max-width:none;padding:0}.hero{padding:12px 0 18px;border-top-width:4px}h1{font-size:28px;max-width:none}h2{font-size:20px}h3{font-size:15px}.lead{font-size:12px}.meta{font-size:10px;margin-top:12px}nav,.exports,.print-hidden{display:none}.print-note{display:block;font-size:10px;color:var(--muted)}section{margin-top:22px}.cards{grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0}.metric{padding:12px}.metric .label{min-height:0;font-size:10px}.metric .value{font-size:25px;margin:5px 0}.metric .detail{font-size:9px}.metric .unit{font-size:12px}.table-scroll{overflow:visible;border-radius:0;break-inside:auto;margin:10px 0 15px}table,.examples{font-size:9px}caption{font-size:9px;padding:7px}thead{display:table-header-group}thead th{font-size:9px;white-space:normal}th,td{padding:6px 7px}tbody th{min-width:95px;max-width:230px}.examples tbody th{min-width:110px}.field-note{min-width:70px;font-size:8px}.badge{font-size:8px}.section-heading,h2,h3,summary{break-after:avoid}tr,.metric,figure,.note,.insights{break-inside:avoid}figure{padding:10px}figcaption{font-size:10px}.charts{gap:12px}.glossary{margin:12px 0}.glossary dd,.section-intro,.note{font-size:10px}details{padding:0 10px}details::details-content{content-visibility:visible}details:not([open])>:not(summary){display:block}summary{font-size:12px;padding:10px 0;list-style:none}details .details-body{display:block}.method-grid>div{padding:14px}.hashes{font-size:9px;grid-template-columns:160px 1fr}code{font-size:8px}.footer{font-size:9px;padding:15px 0}a{color:inherit;text-decoration:none}}
"""

# Каждый график уже содержит две панели: полная ширина сохраняет читаемость подписей.
CSS += "\n.charts{grid-template-columns:minmax(0,1fr)}\n"


def render_report(report: dict, output_path: Path) -> None:
    """Оформить готовые метрики и встроить PNG; никаких вычислений модели."""
    if report.get("schema_version") != 1:
        raise ValueError("Поддерживается schema_version=1.")
    output_path = Path(output_path)
    baseline = report["baseline"]
    groups = report["groups"]
    thresholds = report["thresholds"]
    min_n = _number(thresholds["min_group_size"], 0)
    cards = [
        ("Средняя абсолютная ошибка", _money(baseline.get("mae_tenge")), "млн ₸", "MAE · среднее отклонение прогноза от цены предложения", True),
        ("Ошибки больше 20%", _number(baseline.get("over_20_pct")), "%", f'{_number(baseline.get("over_20_n"), 0)} из {_number(report["n_test"], 0)} объявлений', False),
        ("Попадание в прогноз ±MAE", _number(baseline.get("coverage_pct")), "%", "Наблюдаемое покрытие на этом test; ширина взята из его же MAE", False),
        ("Медианная относительная ошибка", _number(baseline.get("median_ape_pct")), "%", "MedAPE · у половины объявлений ошибка не больше этого значения", False),
        ("90-й процентиль ошибки", _money(baseline.get("p90_ae_tenge")), "млн ₸", "P90 · примерно у 10% объявлений абсолютная ошибка выше", False),
        ("Среднее смещение прогноза", _money(baseline.get("bias_tenge"), signed=True), "млн ₸", "Прогноз − реальная цена: плюс — завышение, минус — занижение", False),
    ]
    card_html = "".join(
        f'<article class="metric{" primary" if primary else ""}"><div class="label">{_text(label)}</div>'
        f'<div class="value">{value} <span class="unit">{_text(unit)}</span></div>'
        f'<div class="detail">{_text(detail)}</div></article>'
        for label, value, unit, detail, primary in cards
    )
    findings = _list(report.get("findings", []))
    charts = _charts(report.get("charts", []), output_path.parent)
    table_caption = "Деньги — млн ₸. Проценты рассчитаны внутри группы. Смещение = прогноз − реальная цена."
    district_table = _group_table(groups.get("district", []), "Районы · " + table_caption)
    completeness_table = _group_table(groups.get("completeness", []), "Исходная полнота · " + table_caption)
    patterns = _group_table(groups.get("missing_pattern", []), "Какие поля отсутствовали · " + table_caption)
    district_completeness = _group_table(groups.get("district_completeness", []), "Район × исходная полнота · " + table_caption)
    masking = _group_table(report.get("controlled_masking", []), "Одна и та же полная когорта · Δ MAE относительно всех восьми полей, млн ₸", delta=True)
    ablation = _group_table(report.get("field_ablation", []), "Скрываем по одному полю · Δ MAE относительно всех восьми полей, млн ₸", delta=True)
    segment_titles = [
        ("price_band", "Цена предложения", True),
        ("area_band", "Площадь квартиры", True),
        ("rooms_band", "Количество комнат", False),
        ("building_type", "Тип дома", False),
        ("condition", "Состояние квартиры", False),
        ("year_band", "Год постройки", False),
    ]
    dimension_labels = {
        "district": "Район", "completeness": "Полнота", "missing_pattern": "Пропуски",
        "price_band": "Цена", "area_band": "Площадь", "rooms_band": "Комнаты",
        "building_type": "Тип дома", "condition": "Состояние", "year_band": "Год постройки",
        "district_completeness": "Район и полнота",
    }
    ranking_rows = [
        {**row, "group": f'{dimension_labels.get(row.get("dimension"), row.get("dimension", ""))} · {row["group"]}'}
        for row in report.get("ranked_segments", [])[:10]
    ]
    ranking = _group_table(ranking_rows, "10 сегментов с наибольшей частотой ошибки >20% · " + table_caption)
    segments = "".join(
        f'<details{" open" if opened else ""}><summary>{_text(title)}</summary><div class="details-body">'
        + _group_table(groups.get(key, []), title + " · " + table_caption) + '</div></details>'
        for key, title, opened in segment_titles
    )
    absolute_examples = _example_table(report.get("top_absolute", []), "Крупнейшие абсолютные ошибки · цена, прогноз и ошибка в млн ₸")
    relative_examples = _example_table(report.get("top_relative", []), "Крупнейшие относительные ошибки · цена, прогноз и ошибка в млн ₸")
    export_labels = {
        **{f"by_{key}": f"Метрики: {label.lower()}" for key, label in dimension_labels.items()},
        "controlled_masking": "Сценарии скрытия полей", "field_ablation": "Скрытие отдельных полей",
        "test_error_details": "Все ошибки test", "masking_predictions": "Прогнозы при скрытии полей",
        "ranked_segments": "Рейтинг сегментов", "top_absolute": "Крупнейшие ошибки в тенге",
        "top_relative": "Крупнейшие ошибки в процентах", "error_report": "Полный отчёт в JSON",
    }
    export_links = []
    for item in report.get("exports", []):
        filename = str(item["filename"]).replace(chr(92), "/")
        label = export_labels.get(Path(filename).stem, item["label"])
        export_links.append(
            f'<a class="export" href="{_text(quote(filename, safe="/"))}" download>{_text(label)} ↓</a>'
        )
    exports = "".join(export_links)
    provenance = report.get("provenance", {})
    hashes = "".join(
        f'<dt>{_text(label)}</dt><dd><code>{_text(provenance.get(key, "—"))}</code></dd>'
        for key, label in [
            ("input_sha256", "Очищенный CSV · SHA-256"),
            ("model_sha256", "Модель · SHA-256"),
            ("test_predictions_sha256", "Прогнозы test · SHA-256"),
            ("split_manifest_sha256", "Разбиение · SHA-256"),
        ]
    )
    checks = _list(provenance.get("checks", []))
    limitations = _list(report.get("limitations", []))
    recommendations = _list(report.get("recommendations", []), ordered=True)
    document = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><title>Где ошибается модель · Квартиры Астаны</title><style>{CSS}</style></head>
<body><header class="hero"><div class="container"><p class="eyebrow">Астана · диагностика сохранённой модели</p>
<h1>Где ошибается модель<br>стоимости квартир</h1>
<p class="lead">Районы, характеристики квартир и полнота ввода: подробная проверка прогнозов на отложенной выборке.</p>
<div class="meta"><span>Модель: <strong>{_text(report['model_name'])}</strong></span><span>Данные: <strong>{_text(report['data_period'])}</strong></span><span>Test: <strong>{_number(report['n_test'], 0)} объявлений</strong></span></div>
<nav aria-label="Разделы отчёта"><a href="#overview">Главное</a><a href="#districts">Районы</a><a href="#completeness">Полнота</a><a href="#masking">Скрытие полей</a><a href="#segments">Сегменты</a><a href="#examples">Примеры ошибок</a><a href="#method">Методика</a></nav>
</div></header><main class="container">
<section id="overview"><div class="section-heading"><span>01</span><h2>Главные результаты</h2></div>
<div class="insights"><h3>Что показывает проверка</h3>{findings}</div><div class="cards">{card_html}</div>
<p class="section-intro">{_text(report.get('split_description', 'Исходный случайный holdout объявлений января 2026 года'))}: {_number(report['n_train'], 0)} строк train и {_number(report['n_test'], 0)} строк test. Это проверка цен предложения того периода. Результаты не подтверждают точность на сегодняшнем рынке и не являются оценкой цены сделки.</p>
<dl class="glossary"><div><dt>MAE: насколько ошибаемся в деньгах</dt><dd>Среднее значение |прогноз − цена|. Большие ошибки дорогих объектов могут заметно поднять среднее.</dd></div>
<div><dt>Ошибка >20%: как часто ошибка заметна</dt><dd>Доля строк, где |прогноз − цена| / цена &gt; 20%. Это выбранный аналитический порог, а не отраслевой стандарт качества.</dd></div>
<div><dt>Смещение: в какую сторону ошибаемся</dt><dd>Среднее (прогноз − цена). Ошибки разных знаков компенсируются: смещение около нуля не означает малую MAE.</dd></div>
<div><dt>Внутри ±MAE: насколько полезен диапазон</dt><dd>Доля цен внутри max(0, прогноз − MAE) … прогноз + MAE. Одна ширина взята из MAE всего этого test; оценка описательная, без независимой калибровки и гарантии покрытия.</dd></div></dl>
<div class="note"><strong>Осторожно с малыми группами.</strong> При n &lt; {min_n} результат отмечен «мало данных». Отдельные объявления могут сильно менять метрики. Различия между группами описывают этот срез и сами по себе не доказывают влияние района, ремонта или пропусков.</div></section>
{charts}
<section id="districts"><div class="section-heading"><span>03</span><h2>В каких районах ошибки больше</h2></div>
<p class="section-intro">Сравнивайте денежную и относительную ошибку вместе с размером группы. Более высокая MAE может отражать более дорогие квартиры в районе.</p>{district_table}
<details><summary>Районы с учётом исходной полноты данных</summary><div class="details-body">{district_completeness}</div></details></section>
<section id="completeness"><div class="section-heading"><span>04</span><h2>Полнота исходных объявлений</h2></div>
<p class="section-intro">Число заполненных полей считается среди восьми входов модели до заполнения пропусков. Группы содержат разные квартиры: их сравнение не изолирует эффект пропусков. Полных допустимых анкет для эксперимента: <strong>{_number(report['n_complete'], 0)}</strong>.</p>{completeness_table}
<details><summary>Точные сочетания отсутствующих полей</summary><div class="details-body">{patterns}</div></details></section>
<section id="masking"><div class="section-heading"><span>05</span><h2>Что меняется, когда скрываем поля</h2></div>
<p class="section-intro">Контролируемая проверка на одних и тех же полностью заполненных и допустимых для формы строках. Цена квартиры остаётся прежней, а часть входов заменяется пропусками. Модель и её предобработка остаются обученными ранее.</p>
<div class="note"><strong>Число полей не задаёт универсальную точность.</strong> В сценариях используется определённый порядок добавления полей. Сравнение относится к выбранным сочетаниям на этой когорте; для другой квартиры или другого набора полей результат может отличаться. Положительная Δ MAE означает ухудшение относительно ввода 8/8.</div>{masking}
<details open><summary>Вклад отдельного поля: скрываем одно, оставляем остальные</summary><div class="details-body"><p class="muted">Сравнение на той же полной когорте. Изменение ошибки описывает чувствительность модели к отсутствию поля, а не его причинное влияние на цену.</p>{ablation}</div></details></section>
<section id="segments"><div class="section-heading"><span>06</span><h2>Какие квартиры сложнее оценить</h2></div>
<p class="section-intro">Сегменты по цене определены по известной цене test и предназначены для анализа ошибок после прогноза. В каждом разрезе используются исходные объявления без искусственного скрытия полей.</p>
<h3>Где ошибка больше 20% встречается чаще всего</h3><p class="muted">Рейтинг включает только группы с n ≥ {min_n}. Сегменты пересекаются: одна квартира может присутствовать в нескольких строках. Строки рейтинга нельзя суммировать.</p>{ranking}{segments}</section>
<section id="examples"><div class="section-heading"><span>07</span><h2>Объявления с крупнейшими ошибками</h2></div>
<p class="section-intro">Это примеры для проверки данных и поведения модели. Крупная ошибка сама по себе не доказывает ошибку в объявлении. «Строка» — source_row в исходном CSV без заголовка, «Поля» — исходная заполненность восьми входов. Ссылки ведут к старым объявлениям и могут быть недоступны.</p>
{absolute_examples}<details><summary>Крупнейшие ошибки в процентах от цены</summary><div class="details-body">{relative_examples}</div></details></section>
<section id="method"><div class="section-heading"><span>08</span><h2>Как читать результат и что проверить дальше</h2></div>
<div class="method-grid"><div><h3>Ограничения проверки</h3>{limitations}</div><div><h3>Следующие шаги</h3>{recommendations}</div></div>
<p class="technical" style="margin-top:18px">Строк test, допустимых для полей формы: {_number(report['n_app_eligible'], 0)} из {_number(report['n_test'], 0)}. Метрики приведены с округлением только в представлении; машиночитаемые файлы сохраняют точные значения. Обозначение n — число объявлений.</p>
<details><summary>Проверки происхождения данных и контрольные суммы</summary><div class="details-body technical">{checks}<dl class="hashes">{hashes}</dl><p>Сформировано: {_text(report['generated_at'])}. Версия схемы: 1.</p></div></details>
<h3 style="margin-top:24px">Таблицы и расчёты</h3><div class="exports">{exports}</div>
<p class="technical" style="margin-top:10px">HTML и графики можно читать без сети: изображения встроены в файл. Для скачивания таблиц нужны соседние файлы из папки отчёта.</p>
<p class="print-note">В печатной версии могут быть скрыты закрытые подробные таблицы. Для полного экспорта раскройте нужные разделы перед печатью.</p>
</section></main><footer class="footer"><div class="container">Учебная модель · цены предложения квартир в Астане · анализ ошибок без переобучения модели</div></footer></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

"""Автономный отчёт сравнения объединённой и сохранённых моделей."""

from html import escape
from pathlib import Path

from clean import atomic_write


def fmt(value, digits=2):
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def comparison_table(rows: list[dict]) -> str:
    labels = {"january": "Январская · активна", "march": "Только март", "combined": "Январь + март"}
    body = "".join(
        f"<tr{' class=selected' if r['model'] == 'combined' else ''}><th>{labels[r['model']]}<small>{escape(r['algorithm'])}</small></th>"
        f"<td>{fmt(r['mae_tenge']/1e6)}</td><td>{fmt(r['rmse_tenge']/1e6)}</td><td>{fmt(r['r2'],3)}</td>"
        f"<td>{fmt(r['over_20_pct'],1)}%</td><td>{fmt(r['bias_tenge']/1e6)}</td></tr>" for r in rows)
    return '<div class="scroll"><table><thead><tr><th>Модель</th><th>MAE, млн ₸</th><th>RMSE, млн ₸</th><th>R²</th><th>Ошибка &gt;20%</th><th>Смещение, млн ₸</th></tr></thead><tbody>' + body + '</tbody></table></div>'


def segment_table(rows: list[dict]) -> str:
    body = "".join(f"<tr><th>{escape(r['group'])}{'<small>Малая группа</small>' if r['small_group'] else ''}</th>"
                   f"<td>{r['n']}</td><td>{fmt(r['january_mae_tenge']/1e6)}</td><td>{fmt(r['combined_mae_tenge']/1e6)}</td>"
                   f"<td>{fmt(r['mae_change_tenge']/1e6)}</td></tr>" for r in rows)
    return '<div class="scroll"><table><thead><tr><th>Группа</th><th>n</th><th>Январь: MAE, млн ₸</th><th>Объединение: MAE, млн ₸</th><th>Разница, млн ₸</th></tr></thead><tbody>' + body + '</tbody></table></div>'


def render_report(report: dict, output_path: Path) -> None:
    c = report['counts']
    change = report['mae_improvement_pct']
    direction = "снизилась" if change >= 0 else "выросла"
    interval = report['paired_mae_difference']
    inconclusive = interval['ci95_low_tenge'] <= 0 <= interval['ci95_high_tenge']
    uncertainty = ("Интервал включает ноль: на этой выборке недостаточно оснований считать разницу устойчивой."
                   if inconclusive else "Интервал не включает ноль, однако повторное использование test и зависимость похожих квартир ограничивают вывод.")
    sources = ', '.join(f"{escape(str(key))}: {fmt(value,0)}" for key, value in c['train_sources'].items())
    document = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Январь + март: переобучение и сравнение</title><style>
body{{margin:0;background:#f6f8fb;color:#18364b;font:16px/1.65 "Segoe UI",Arial,sans-serif}}main{{max-width:1100px;margin:auto;padding:40px 24px}}h1{{font-size:38px;line-height:1.2}}h2{{font-size:25px;margin-top:35px}}.eyebrow,small,.muted{{color:#597185}}.lead{{font-size:20px}}.callout{{padding:20px 24px;background:#e5f2ef;border-left:5px solid #087f79;margin:24px 0}}.note{{background:#fff7e8;border-left:4px solid #ce9c37;padding:15px 20px}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:13px;border-bottom:1px solid #dde5eb;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}thead{{background:#eaf0f4}}small{{display:block;font-weight:400}}.selected{{background:#eef8f5}}a{{color:#087b78}}li{{margin:8px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.card{{background:white;border:1px solid #dce5ea;padding:20px 24px;border-radius:8px}}.card h2{{margin-top:0}}@media(max-width:680px){{.grid{{grid-template-columns:1fr}}h1{{font-size:29px}}}}@media print{{body{{background:white}}main{{padding:0}}}}
</style></head><body><main>
<p class="eyebrow">АСТАНА · ОБЪЕДИНЕНИЕ ДАННЫХ И ПОВТОРНОЕ ОБУЧЕНИЕ</p>
<h1>Январь + март 2026</h1>
<p class="lead">Объединённая модель обучена и сохранена отдельно. В приложении по-прежнему работает январская модель.</p>
<div class="callout">На общей контрольной выборке без найденных дублей — <strong>{c['test_rows']} квартир</strong> — средняя абсолютная ошибка объединённой модели <strong>{direction} на {fmt(abs(change),1)}%</strong> относительно текущей январской модели.</div>
<p class="note">Это повторно используемая мартовская контрольная выборка. Она полностью исключена из обучения, но её результаты уже анализировались ранее. Сравнение диагностическое; для подтверждения на новых условиях нужен следующий независимый снимок данных.</p>
<h2>Основное сравнение: одна строка на группу квартиры</h2>{comparison_table(report['common_holdout_comparison'])}
<p class="muted">MAE — средняя абсолютная ошибка. Смещение = прогноз − цена объявления. Все модели оцениваются на одних и тех же строках. Объединённая модель выбрана только по групповой cross-validation внутри train; test не использовался для подбора параметров.</p>
<p>Разница MAE «объединённая − январская»: <strong>{fmt(interval['new_minus_january_mae_tenge']/1e6)} млн ₸</strong>. Описательный 95% парный bootstrap-интервал: <strong>{fmt(interval['ci95_low_tenge']/1e6)} … {fmt(interval['ci95_high_tenge']/1e6)} млн ₸</strong>. {uncertainty}</p>
<div class="grid"><div class="card"><h2>Объединение и дубли</h2><ul>
<li>Исходных очищенных строк: {fmt(c['input_rows'],0)}.</li><li>Оставлено представителей групп: {fmt(c['unique_rows'],0)}.</li>
<li>Повторных строк исключено: {fmt(c['removed_rows'],0)}; исходные значения и причины сохранены.</li>
<li>В train: {fmt(c['train_rows'],0)} строк ({sources}).</li><li>В test: {c['test_rows']} представителей из прежних {c['reference_test_rows']} объявлений.</li></ul></div>
<div class="card"><h2>Правила подготовки</h2><ul><li>Группы строятся по ссылкам и полным наборам характеристик с адресом либо ЖК. Цена и состояние не участвуют в определении совпадения.</li>
<li>Сначала защищены все прежние тестовые группы. Для train предпочтение у мартовской записи, затем у более полной анкеты; цена не влияет на выбор.</li>
<li>У каждой строки сохранены исходный период, номер записи и группа. Январские строки не попадают в мартовский test.</li></ul></div></div>
<h2>Проверка на прежних {c['reference_test_rows']} объявлениях</h2>{comparison_table(report['original_holdout_comparison'])}
<p class="muted">Эта таблица сохраняет состав предыдущего эксперимента, включая {c['reference_test_rows']-c['test_rows']} дополнительных строк из повторных групп. Ни одна из этих групп не входила в обучение. Основной вывод опирается на таблицу без дублей выше.</p>
<h2>Как изменилась ошибка по районам</h2>{segment_table(report['segments']['district'])}
<h2>Как изменилась ошибка по площади</h2>{segment_table(report['segments']['area'])}
<h2>Как изменилась ошибка по полноте анкеты</h2>{segment_table(report['segments']['completeness'])}
<p class="muted">Отрицательная разница означает меньшую MAE объединённой модели. Группы меньше 30 строк описательны; количество заполненных полей не отделяет эффект полноты от состава квартир.</p>
<h2>Обучение и ограничения</h2><ul>
<li>Сравнены LinearRegression, RandomForest и XGBoost, использованы те же восемь признаков и функции потерь. Подбор лучшего семейства: 3 групповых fold и до 8 вариантов параметров. Все заполнения пропусков и кодирование обучаются внутри train.</li>
<li>Январский набор включён полностью до удаления повторов. Его прежний январский test теперь может входить в train и не служит проверкой объединённой модели.</li>
<li>Совпадения характеристик — эвристика: возможны ошибочные объединения и невыявленные повторы. Проверка не обеспечивает независимость по зданиям.</li>
<li>Март — период по описанию автора; дат по отдельным строкам нет. Оцениваются цены предложения, а не сделки, рынок октября или будущие шесть месяцев.</li>
<li>Bootstrap отражает вариацию на этом test при условной независимости групп; он не учитывает весь процесс выбора подхода. Диапазон ±MAE не является гарантированным интервалом.</li></ul>
<h2>Файлы и воспроизводимость</h2><ul>
<li><a href="update_report.json">Результаты и контрольные суммы (JSON)</a></li><li><a href="preparation_report.json">Подготовка данных и источники</a></li>
<li><a href="deduplication_audit.csv">Все исходные строки и решения об удалении дублей</a></li><li><a href="split_manifest.csv">Train/test и CV-fold</a></li>
<li><a href="training_report.json">Алгоритмы, параметры и метрики обучения</a></li><li><a href="unique_error_comparison.csv">Прогнозы и ошибки на уникальных группах</a></li>
<li><a href="reference_error_comparison.csv">Прогнозы на прежних {c['reference_test_rows']} строках</a></li>
<li><a href="figures/training_actual_vs_predicted.png">График прогнозов и цен объявлений</a></li>
<li><a href="../march_update/update_report.html">Предыдущее мартовское сравнение</a></li></ul>
<p class="muted">{escape(report['created_at_utc'])}. Проверено: действующая январская модель и исходные январские/мартовские артефакты не изменились.</p>
</main></body></html>'''
    atomic_write(output_path, document.encode('utf-8'))

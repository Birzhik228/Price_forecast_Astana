"""Самодостаточный отчёт эксперимента с ЖК и координатами."""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path

from clean import atomic_write


LABELS = {"baseline8": "Контроль: 8 признаков", "complex": "+ ЖК", "coords": "+ координаты",
          "complex_coords": "+ ЖК и координаты", "january": "Январская (активна)", "combined": "Объединённая (сохранённая)"}


def _table(headers: list, rows: list) -> str:
    head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _score_table(scores: list, *, key: str = "model") -> str:
    return _table(["Модель", "Квартир", "MAE, млн ₸", "RMSE, млн ₸", "R²", "Ошибка >20%"], [
        [LABELS.get(row[key], row[key]), row["n"], f"{row['mae_tenge']/1e6:.3f}",
         f"{row['rmse_tenge']/1e6:.3f}", f"{row['r2']:.3f}", f"{row['over_20_pct']:.1f}%"] for row in scores])


def render_report(report: dict, path: Path) -> None:
    selected = report["selected_variant"]
    scores = {row["model"]: row for row in report["comparison"]}
    candidate = scores["complex_coords"]
    old, combined = scores["january"], scores["combined"]
    improvement_old = (1 - candidate["mae_tenge"] / old["mae_tenge"]) * 100
    improvement_combined = (1 - candidate["mae_tenge"] / combined["mae_tenge"]) * 100
    def change(value):
        return f"снижение ошибки на {value:.1f}%" if value >= 0 else f"рост ошибки на {-value:.1f}%"
    cv_table = _score_table([{**row["metrics"], "variant": row["variant"]} for row in report["cv_comparison"]], key="variant")
    coordinate_rows = [[LABELS[row["variant"]], sum(fold["geo_available_rows"] for fold in row["folds"]), report["training_rows"]]
                       for row in report["cv_comparison"]]
    catalogue = report["catalogue_summary"]
    normalization = catalogue["normalization"]
    counts = _table(["Показатель каталога", "Значение"], [
        ["Объявлений", catalogue["listing_count"]], ["Однозначных адресных групп", catalogue["building_count"]],
        ["Объявлений с привязкой к дому", catalogue["listings_with_building_id"]],
        ["Объявлений с ЖК", catalogue["listings_with_complex_id"]],
        ["Парковка указана", catalogue["listings_with_parking"]]])
    limitations = "".join(f"<li>{escape(value)}</li>" for value in report["limitations"])
    hashes = _table(["Защищённый файл", "Не изменён"], [[name, "Да" if checksum == report["protected_after"][name] else "Нет"]
                                                       for name, checksum in report["protected_before"].items()])
    districts = ""
    segment_path = path.parent / "segments.csv"
    if segment_path.exists():
        with segment_path.open(encoding="utf-8", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["dimension"] == "district"]
        groups = {}
        for row in rows:
            groups.setdefault(row["segment"], {})[row["model"]] = row
        districts = _table(["Район", "Квартир", "Январская", "Объединённая", "+ ЖК", "+ ЖК и координаты"], [
            [district, values["january"]["n"], *[f"{float(values[model]['mae_tenge']) / 1e6:.2f}"
             for model in ("january", "combined", "complex", "complex_coords")]]
            for district, values in sorted(groups.items())])
    html = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ЖК и география · проверка модели</title><style>
body{{margin:0;background:#f5f8fb;color:#19394d;font:16px/1.65 "Segoe UI",Arial,sans-serif}}main{{max-width:1120px;margin:auto;padding:42px 24px}}h1{{font-size:38px;line-height:1.2}}h2{{font-size:25px;margin-top:34px}}.muted{{color:#526e80}}.lead{{font-size:21px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{background:white;padding:20px;border:1px solid #dfe8ed;border-radius:8px}}.card strong{{font-size:30px;display:block}}.notice{{padding:18px 22px;background:#fff4df;border-left:4px solid #d4a238;margin:22px 0}}.good{{padding:18px 22px;background:#e4f3ef;border-left:4px solid #0b8176;margin:22px 0}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}td,th{{padding:11px;text-align:left;border-bottom:1px solid #dce5ed;overflow-wrap:anywhere}}th{{background:#e7eff4}}.scroll{{overflow-x:auto}}a{{color:#087e78}}li{{margin:8px 0}}@media(max-width:720px){{main{{padding:24px 15px}}h1{{font-size:28px}}.cards{{grid-template-columns:1fr}}}}
</style></head><body><main><p class="muted">АСТАНА · ЭКСПЕРИМЕНТ С АДРЕСАМИ И НОВЫМИ ПРИЗНАКАМИ</p>
<h1>Дают ли ЖК и координаты более точную оценку?</h1>
<p class="lead">Сравнение на одних и тех же {report['test_rows']} квартирах. Январская модель остаётся активной; экспериментальные версии сохранены отдельно.</p>
<div class="cards"><div class="card"><strong>{old['mae_tenge']/1e6:.2f} млн ₸</strong>Январская: средняя абсолютная ошибка</div><div class="card"><strong>{combined['mae_tenge']/1e6:.2f} млн ₸</strong>Объединённая: средняя абсолютная ошибка</div><div class="card"><strong>{candidate['mae_tenge']/1e6:.2f} млн ₸</strong>Эксперимент: ЖК + координаты</div></div>
<p class="good">ЖК + координаты: {change(improvement_old)} относительно январской и {change(improvement_combined)} относительно объединённой. По обучающей перекрёстной проверке выбран вариант «{escape(LABELS[selected])}».</p>
<p class="notice">Мартовская проверочная выборка использовалась ранее. Эти результаты служат диагностическим сравнением; улучшение на свежем рынке пока не подтверждено. Координаты проверены на согласованность исходных объявлений, но не сверены с официальным реестром.</p>
<h2>Что изменено в данных</h2><p>Однозначные варианты букв в суффиксах номеров нормализованы с сохранением исходных записей и аудита решений. Противоречивые совпадения не получают уверенной привязки к дому. Индивидуальные анкеты и цены не заменяются значениями, выбранными большинством.</p>{counts}
<p>Написание исправлено в {normalization['changed_address_rows']} строках. Объединены {normalization['accepted_merge_candidates']} группы вариантов адресов; отклонены {normalization['rejected_merge_candidates']} неоднозначных объединений. Всего выявлено {catalogue['ambiguous_candidate_count']} адресных групп с противоречиями ({catalogue['ambiguous_listing_count']} объявлений); их записи сохранены отдельно.</p>
<p>Примеры принятых объединений: «Турар Рыскулов 9a/9а» и «Абылай-хана проспект 20a/20а». «Нажимеденов 44a/44а» оставлен без уверенной привязки к дому: в исходных объявлениях указана этажность 9, 10 и 12. Совпадения написания здесь недостаточно.</p>
<p><a href="../property_catalog/normalization_audit.csv">Аудит нормализации адресов</a> · <a href="../property_catalog/conflict_audit.csv">Противоречия домов</a> · <a href="../property_catalog/quality.json">Сводка качества</a></p>
<h2>Сравнение на общей контрольной выборке</h2>{_score_table(report['comparison'])}
<p>Исходный test: {report['original_test_rows']} строк. Исключено после дополнительной проверки адресных совпадений: {len(report['excluded_test_rows'])}. Для всех строк таблицы использованы одинаковые квартиры. Контрольный повтор обучения с восемью признаками воспроизводит сохранённую объединённую модель.</p>
<h2>Польза отдельных признаков: перекрёстная проверка</h2><p>Один RandomForest с теми же параметрами, включая 350 деревьев и random_state=42. Три одинаковые части с разделением групп похожих квартир; {report['training_rows']} обучающих строк. ЖК кодируется по частотам только обучающей части; редкие значения объединяются. Подбор по мартовскому test не выполнялся.</p>{cv_table}
<h2>Как исключено подглядывание в проверочные строки</h2><p>Для каждой части заново строится справочник координат только на её обучающих квартирах. Проверочная квартира получает координаты лишь из этого справочника; её собственные сырые точки и другие проверочные строки не участвуют в согласовании. Неизвестный дом и противоречащие сведения оставляют координаты пустыми. Цены и цены за м² не используются в нормализации и географических признаках.</p>
{_table(['Вариант','Проверочных строк CV с координатами','Всего строк CV'], coordinate_rows)}
<p>В общем мартовском test координаты доступны для {report['holdout_geo_available_rows']} из {report['test_rows']} квартир. Поэтому результат географического эксперимента нужно оценивать с учётом ограниченного покрытия. Раздельные показатели по району, известным/новым/неопределённым домам и ЖК, наличию координат — в <a href="segments.csv">таблице сегментов</a>.</p>
<h2>Ошибка по районам на том же test</h2><p>Средняя абсолютная ошибка, млн ₸. В малых группах результат чувствителен к отдельным квартирам.</p>{districts}
<h2>Дополнительная проверка на новых домах</h2><p>Отдельные модели заново обучены на подмножестве train; адресные группы другой части им не показывались. Дома с неизвестным адресом нельзя надёжно разделить. Эта диагностика не использовалась для выбора варианта, но её строки ранее участвовали в CV, поэтому она не является независимой проверкой выбора модели.</p>{_score_table(report['unseen_house_check']['results'], key='variant')}
<h2>Ограничения и следующий шаг</h2><ul>{limitations}</ul><p>Следующий шаг — независимые более свежие объявления с датами и проверка лучших вариантов до замены действующей модели.</p>
<h2>Воспроизводимость</h2>{hashes}<ul><li><a href="experiment_report.json">Полный числовой отчёт</a></li><li><a href="experiment_plan.json">Источники, параметры и контрольные суммы</a></li><li><a href="holdout_predictions.csv">Прогнозы каждой модели по квартирам</a></li><li><a href="alias_overlap_audit.json">Проверка пересечений после нормализации</a></li><li><a href="unseen_house_check.json">Разбиение проверки новых домов</a></li><li><a href="verification.json">Повторная проверка сохранённых моделей</a></li><li><a href="training_coordinate_audit.csv">Причины принятия и отклонения координат в train</a></li></ul><p class="muted">Сформировано: {escape(report['created_at_utc'])}</p></main></body></html>'''
    atomic_write(path, html.encode("utf-8"))

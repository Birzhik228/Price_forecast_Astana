"""Render the coordinate coverage audit, matching evidence and diagnostic result."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "coordinate_enrichment_2026_10_13"
REASONS = {
    "trait_conflict": "Противоречия в ЖК, годе или этажности",
    "invalid": "Есть невалидные исходные координаты",
    "reused_marker": "Один маркер используется для разных адресов",
    "missing": "Нет исходных координат",
    "insufficient_observations": "Меньше двух наблюдений координат",
    "coordinate_spread": "Разброс точек больше 125 м",
    "plausible": "Внутренне согласованные координаты",
    "unique_exact_osm_building": "Точный адрес одного здания; добавлен",
    "existing_model_coordinate_preserved": "Адрес найден; исходная точка модели сохранена",
    "catalog_identity_review_required": "Адрес найден; характеристики дома требуют разбора",
    "no_exact_building_address": "Нет точного подходящего здания в снимке OSM",
    "multiple_distinct_building_footprints": "Один адрес относится к разным контурам зданий",
    "candidate_district_ambiguous": "Один адрес указан в разных районах архива",
    "osm_district_conflict": "Район в OSM противоречит архиву",
    "invalid_candidate_address": "Адрес нельзя однозначно разобрать",
    "candidate_house_key_mismatch": "Ключ дома не соответствует адресу и району",
    "conflicting_or_unresolved_source_address_objects": "Есть конкурирующие или непроверяемые объекты OSM по адресу",
    "matched_footprint_multiple_candidate_districts": "Одно здание сопоставлено с разными районами архива",
    "administrative_boundary_missing": "Нет подтверждённой административной границы",
}


def table(headers, rows) -> str:
    return ("<table><thead><tr>" + "".join(f"<th>{escape(str(value))}</th>" for value in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row)
            + "</tr>" for row in rows) + "</tbody></table>")


def render(directory: Path = REPORTS) -> Path:
    audit = json.loads((directory / "audit_summary.json").read_text(encoding="utf-8"))
    matching = json.loads((directory / "matching_summary.json").read_text(encoding="utf-8"))
    model = json.loads((directory / "model_comparison.json").read_text(encoding="utf-8"))
    integration_path = directory / "integration.json"
    integration = json.loads(integration_path.read_text(encoding="utf-8")) if integration_path.exists() else {"activated": False}
    candidates = pd.read_csv(directory / "address_candidates.csv")
    matched = pd.read_csv(directory / "matched_addresses.csv")
    old_counts = candidates.loc[candidates.model_house_status.ne("unseen_house"), "model_house_status"].value_counts()
    coverage = audit["coverage"]["catalog"]
    stats = matching["matching"]
    if integration["activated"]:
        outcome = "В приложении подключён вариант с новым адресным справочником."
    elif not model["gate"]["passed"]:
        outcome = "Адресный справочник сохранён. Активная модель оставлена прежней: новые точки не прошли условие автоматического подключения."
    else:
        outcome = "Адресный справочник сохранён, критерий сравнения пройден. Подключение кандидата в приложении ещё не выполнено."
    comparison = table(["Модель на тех же 590 квартирах", "MAE, млн ₸", "Ошибка более 20%"], [
        ["До добавления координат" if key == "base" else "С новыми координатами",
         f"{metrics['mae_tenge'] / 1e6:.4f}", f"{metrics['over_20_count']} из {metrics['n']}"]
        for key, metrics in model["comparison"].items()])
    eligible = matched.loc[matched.eligible_for_enrichment.eq(True)].head(20)
    examples = "<table><thead><tr><th>Адрес</th><th>Район</th><th>Координаты</th><th>Источник</th></tr></thead><tbody>"
    for row in eligible.to_dict("records"):
        examples += (f"<tr><td>{escape(str(row['address_normalized']))}</td><td>{escape(str(row['district']))}</td>"
                     f"<td>{float(row['latitude']):.6f}, {float(row['longitude']):.6f}</td>"
                     f"<td><a href=\"{escape(row['osm_url'], quote=True)}\">OSM {escape(str(row['osm_type']))} {int(row['osm_id'])}</a></td></tr>")
    examples += "</tbody></table>"
    html = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Координаты домов Астаны — аудит и обогащение</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f3f6f8;color:#17303c;margin:0;padding:32px;line-height:1.6}}
main{{max-width:1080px;margin:auto;background:white;padding:36px;border-radius:14px}}h1{{line-height:1.2}}h2{{margin-top:32px}}
.lead{{font-size:19px}}.note{{background:#edf5f8;padding:16px;border-left:4px solid #16869a}}table{{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce5e9}}th{{background:#eaf2f5}}a{{color:#006b87}}.small{{font-size:13px;color:#516975}}
</style><main><p class="small">13 октября 2026 · квартиры Астаны · версия источника OSM: 12 октября 2026</p>
<h1>Почему было только 197 домов с координатами</h1>
<p class="lead">Модель распознала 4 000 адресов в обучающем наборе, но только у 197 координаты прошли все проверки — 4,93%.</p>
<p>Архивный маркер объявления не обязательно указывает на отдельное здание. Проверялись повторяемость, географическая область,
радиус 125 м и согласованность характеристик дома. Это внутренняя проверка, не внешнее подтверждение адреса.</p>
{table(["Основная причина", "Домов в обучении"], [[REASONS.get(key, key), int(value)] for key, value in old_counts.items()])}
<p>Показана одна основная причина на дом: проверки имеют приоритет. Дополнительные причины сохранены в аудите.
В полном каталоге {coverage['rows']:,} объявлений и {coverage['house_keys']:,} однозначных адресных ключей.
У {coverage['rows_without_house_key']:,} объявлений адрес или район неполон либо неоднозначен; им не назначались координаты наугад.</p>
<h2>Что удалось подтвердить по карте</h2>
<p class="lead">Точно сопоставлено {stats['exact_matches']} адресных ключей; отдельных зданий — {stats['unique_matched_buildings']}.
Для дополнения модели допущено {matching['registry_entries']} новых ключей; отдельных зданий — {stats['unique_eligible_new_buildings']}.</p>
<p>Разные подтверждённые написания одного адреса могут ссылаться на одно здание. Поэтому количество ключей и зданий приводится отдельно.</p>
<p>Использован один открытый снимок Казахстана из <a href="https://download.geofabrik.de/asia/kazakhstan.html">Geofabrik / OpenStreetMap</a>.
Улица и номер должны совпасть с одним жилым зданием внутри административной границы Астаны.
Переводы улиц разрешены только при наличии обоих названий на одном объекте дороги OSM. Координата лежит внутри контура здания.</p>
{table(["Результат сопоставления", "Адресов"], [[REASONS.get(key, key), value] for key, value in stats['audit_reason_counts'].items()])}
<p>Исходные 197 точек модели не заменены. Точное совпадение адреса в OSM не устраняет противоречия в ЖК, годе или этажности:
такие совпадения сохранены отдельно и не допущены в прогноз. Не использованы нечёткий поиск, центр ЖК или догадки о новом названии улицы.</p>
<h2>Влияние на прогноз</h2><p class="note">{escape(outcome)}</p>
{comparison}
<p>Новые координаты получили {model['new_geo_rows']} строк проверочного набора. Это повторная диагностика на прежних мартовских квартирах;
свежая независимая оценка точности не выполнена. Карта октября не доказывает историческое состояние объекта в марте.
Деревья модели и обработка остальных признаков сохранены, переобучения не было. Прогнозы строк без новых координат проверены на совпадение.</p>
<p>Условие подключения заранее зафиксировано: MAE и количество ошибок больше 20% не увеличиваются.
Подбор адресов не использует цены или ошибки модели. <a href="model_comparison.html">Подробное сравнение моделей</a>.</p>
<h2>Примеры добавленных адресов</h2>{examples}
<p>Это адреса зданий, в которых находятся квартиры. Координата не указывает на квартиру, подъезд или вход.
Данные OSM подтверждают совпадение адреса на открытой карте, но не являются кадастровой или выездной проверкой.</p>
<h2>Файлы и воспроизводимость</h2><ul>
<li><a href="matched_addresses.csv">Все совпавшие адреса, координаты и ссылки на здания</a></li>
<li><a href="address_matching_audit.csv">Решение по каждому из 4 068 адресов</a></li>
<li><a href="address_candidates.csv">Происхождение адресов и все исходные варианты</a></li>
<li><a href="audit_summary.json">Исходное покрытие и причины отказов</a></li>
<li><a href="address_matching_protocol.json">Зафиксированные источники и код сопоставления</a></li>
<li><a href="model_comparison.json">Метрики, проверки и контрольные суммы моделей</a></li></ul>
<p class="small">© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>.
Производный адресный справочник распространяется по <a href="https://opendatacommons.org/licenses/odbl/1-0/">ODbL 1.0</a>.
Для каждого объекта сохранены URL, версия, время правки и SHA-256 исходного снимка. Дата правки карты не является датой обследования здания.</p>
</main></html>"""
    destination = directory / "report.html"
    destination.write_text(html, encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(render())

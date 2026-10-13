"""Диагностика архивного каталога и поиска аналогов; без обучения модели."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from clean import atomic_write
from config import PROJECT_ROOT, PROPERTY_CATALOG_DIR
from features import FEATURE_COLUMNS


REPORT_DIR = PROJECT_ROOT / "reports/property_catalog"
PROTECTED_HASHES = {
    "models/best_model.joblib": "f40f892700324892ee0fde041f8a5feaca2e0c01fb4231185e5c3a751b073bac",
    "models/model_metadata.json": "c623c137cab6f093071a18a6c183d513ad311744378b1695ca4063dcd39ac1da",
    "models/combined_2026/best_model.joblib": "643bfe75014cb3929376a44e689905c1fe8c4fa1e72165ec7b9c0e30db4d3c8a",
    "data/processed/combined_2026/listings_clean.csv": "28ad11f14b56315ea2dd903ffc7f8ab9d5a788f7f5e070b5ec1b61c0711d98eb",
}
LOCAL_MATCHES = {"building", "complex", "nearby"}
MATCH_LABELS = {"building": "Тот же дом", "complex": "Тот же ЖК", "nearby": "Рядом по координатам", "district": "Тот же район"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def check_retrieval(
    listings: pd.DataFrame, finder: Callable, *, sample_size: int = 200, seed: int = 42,
) -> dict[str, Any]:
    """Случайные известные анкеты: проверка доступности, не точности оценки."""
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
        raise ValueError("Число запросов должно быть положительным целым.")
    required = {"record_id", "group", "url", "dataset_period", "area", "rooms"}
    if not required.issubset(listings.columns) or listings.empty or not listings.record_id.is_unique:
        raise ValueError("Для диагностики нужны непустой каталог и уникальные record_id.")
    from comparables import safe_listing_url

    ordered = listings.sort_values("record_id", kind="stable").reset_index(drop=True)
    count = min(sample_size, len(ordered))
    positions = np.random.default_rng(seed).choice(len(ordered), count, replace=False)
    queries = ordered.iloc[positions]
    rows = []
    result_locations: Counter = Counter()
    bins: Counter = Counter({"0": 0, "1–2": 0, "3–5": 0})
    query_locations: Counter = Counter({"with_local": 0, "district_only": 0, "empty": 0})
    source_counts: Counter = Counter()
    query_columns = [*FEATURE_COLUMNS, "building_id", "complex_id", "latitude", "longitude", "coordinate_status"]
    for _, row in queries.iterrows():
        query = {column: _value(row[column]) for column in query_columns if column in row.index}
        query.update(exclude_record_id=_value(row.record_id), exclude_group=_value(row.group), exclude_url=_value(row.url))
        candidates = finder(listings, query, limit=5)
        if len(candidates) > 5:
            raise AssertionError("Поиск вернул больше пяти аналогов.")
        if len(candidates):
            if candidates.record_id.eq(row.record_id).any() or candidates.group.eq(row.group).any():
                raise AssertionError("Поиск вернул исходную квартиру или её группу.")
            candidate_urls = candidates.url.map(safe_listing_url)
            original_url = safe_listing_url(row.url)
            if original_url is not None and candidate_urls.eq(original_url).any():
                raise AssertionError("Поиск вернул исходное объявление с другой ссылкой.")
            if not candidates.group.is_unique or candidate_urls.dropna().duplicated().any():
                raise AssertionError("Одна квартира или ссылка повторяется в аналогах.")
        matches = Counter(candidates.location_match.tolist())
        if set(matches) - {*LOCAL_MATCHES, "district"}:
            raise AssertionError("Неизвестный уровень пространственного совпадения.")
        local_count = sum(matches[level] for level in LOCAL_MATCHES)
        n = len(candidates)
        bins["0" if n == 0 else "1–2" if n <= 2 else "3–5"] += 1
        query_locations["with_local" if local_count else "district_only" if n else "empty"] += 1
        result_locations.update(matches)
        source_counts[str(row.dataset_period)] += 1
        rows.append({"record_id": str(row.record_id), "dataset_period": str(row.dataset_period),
                     "group": str(row.group), "n": n, "local_count": local_count,
                     "location_matches": dict(sorted(matches.items())),
                     "result_record_ids": candidates.record_id.astype(str).tolist()})
    return {"sample_size_requested": sample_size, "queries": count, "seed": seed, "limit": 5,
            "sampling": "Uniform sample without replacement after sorting record_id; existing catalog listings with their available location and structural fields.",
            "source_counts": dict(sorted(source_counts.items())), "result_count_bins": dict(bins),
            "query_location_counts": dict(query_locations), "result_location_counts": dict(sorted(result_locations.items())),
            "queries_with_results": count - bins["0"], "returned_rows": sum(row["n"] for row in rows),
            "self_and_group_exclusions_verified": True, "cases": rows,
            "limitations": [
                "Availability on catalog-derived queries, not a test of prediction MAE, fair value, or relevance judged by a person.",
                "Queries carry available location IDs and fields from the catalog; coverage for manually entered incomplete addresses may be lower.",
                "Nearby uses approximate source coordinates; same district can cover distant buildings.",
                "Listings are archived and their current availability is not verified.",
            ]}


def protected_status(root: Path = PROJECT_ROOT) -> dict[str, dict]:
    result = {}
    for relative, expected in PROTECTED_HASHES.items():
        actual = sha256(root / relative) if (root / relative).is_file() else None
        kind = "Исходные объединённые данные" if relative.startswith("data/") else "Метаданные модели" if relative.endswith(".json") else "Модель"
        result[relative] = {"kind": kind, "expected_sha256": expected, "actual_sha256": actual, "unchanged": actual == expected}
    return result


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    heading = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'


def _n(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def render_report(summary: dict, retrieval: dict, quality: dict, issues: pd.DataFrame, output: Path) -> None:
    """Самодостаточный HTML; весь текст из данных экранируется."""
    models_unchanged = all(item["unchanged"] for item in summary["protected_artifacts"].values())
    model_notice = ("Январская модель остаётся активной. Контрольные суммы активной модели, метаданных, отдельной объединённой модели и исходных объединённых данных совпали с зафиксированными значениями."
                    if models_unchanged else "Контрольные суммы не совпали с ранее зафиксированными значениями либо файл отсутствует. Неизменность моделей не подтверждена; см. таблицу ниже.")
    bins = retrieval["result_count_bins"]
    query_counts = retrieval["query_location_counts"]
    retrieval_table = _table(["Показатель", "Запросов"], [
        ["Нет аналогов", bins["0"]], ["Один–два аналога", bins["1–2"]], ["Три–пять аналогов", bins["3–5"]],
        ["Есть совпадение дома, ЖК или по близости координат", query_counts["with_local"]],
        ["Только совпадение района", query_counts["district_only"]],
    ])
    coverage_table = _table(["Поле / статус", "Объявлений"], [[key, _n(value)] for key, value in summary["coverage"].items()])
    source_table = _table(["Период источника", "В каталоге", "В диагностических запросах"],
                          [[period, _n(count), retrieval["source_counts"].get(period, 0)] for period, count in summary["source_counts"].items()])
    tiers_table = _table(["Основание совпадения", "Возвращённых строк"],
                         [[MATCH_LABELS.get(key, key), value] for key, value in retrieval["result_location_counts"].items()])
    model_table = _table(["Файл", "Тип", "Совпадение", "Фактический SHA-256"],
                         [[name, status["kind"], "Да" if status["unchanged"] else "Нет", status["actual_sha256"] or "Файл отсутствует"] for name, status in summary["protected_artifacts"].items()])
    preview = issues.head(20).fillna("")
    issues_table = _table(list(preview.columns), preview.astype(str).values.tolist()) if len(preview) else "<p>В таблице issues.csv нет записей.</p>"
    document = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Каталог домов и аналоги · Астана</title><style>
body{{margin:0;background:#f6f8fb;color:#18364b;font:16px/1.65 "Segoe UI",Arial,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 24px}}h1{{font-size:38px;line-height:1.2}}h2{{font-size:25px;margin-top:35px}}.muted,.eyebrow{{color:#597185}}.lead{{font-size:20px}}.note{{padding:18px 22px;background:#fff7e8;border-left:4px solid #ce9c37}}.callout{{padding:20px 24px;background:#e5f2ef;border-left:5px solid #087f79;margin:24px 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{background:white;padding:18px 22px;border:1px solid #dce5ea;border-radius:8px}}.card strong{{display:block;font-size:30px}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:11px;border-bottom:1px solid #dde5eb;text-align:left;vertical-align:top}}thead{{background:#eaf0f4}}td{{overflow-wrap:anywhere}}a{{color:#087b78}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px;font-size:13px}}li{{margin:8px 0}}@media(max-width:720px){{.grid{{grid-template-columns:1fr}}h1{{font-size:29px}}main{{padding:24px 15px}}}}@media print{{body{{background:white}}main{{padding:0}}}}
</style></head><body><main>
<p class="eyebrow">АСТАНА · АРХИВНЫЙ КАТАЛОГ И ПОИСК АНАЛОГОВ</p><h1>Дом, окружение и похожие объявления</h1>
<p class="lead">Каталог связывает исходные объявления с домами и ЖК, сохраняет происхождение дополнительных сведений и помогает найти похожие квартиры.</p>
<div class="grid"><div class="card"><strong>{_n(summary['listings'])}</strong>объявлений после удаления повторов</div><div class="card"><strong>{_n(summary['buildings'])}</strong>адресных групп домов</div><div class="card"><strong>{retrieval['queries_with_results']} / {retrieval['queries']}</strong>запросов нашли хотя бы один аналог</div></div>
<p class="note">Это архивные цены предложения января и конца марта 2026 года. Ссылки и наличие квартир сейчас не проверялись. Координаты приблизительны; число найденных аналогов не измеряет точность прогноза цены.</p>
<h2>Что добавлено</h2><ul>
<li>Исходные свойства восстановлены по периоду и номеру записи raw; совпадение ссылки проверяется. Глобальный номер объединённого CSV не используется как номер строки raw.</li>
<li>Полный адрес дома и название ЖК имеют разные идентификаторы. Корпуса, буквы и дробные номера сохраняются; неоднозначные адреса не получают уверенное совпадение дома.</li>
<li>При поиске соблюдаются одинаковое число комнат и площадь ±20%. Порядок совпадений: дом → ЖК → близость по допустимым координатам → район; цена не задаёт порядок аналогов.</li>
<li>В форме есть необязательное поле «Ссылка на оцениваемое объявление (если есть)». Допустимая ссылка на карточку Krisha исключает её и известные связанные повторы из аналогов; сетевые запросы не выполняются, ссылка не передаётся модели. Если ссылка не указана или исходная запись не распознана в каталоге, собственное объявление может остаться среди аналогов. Повторные квартиры внутри результатов объединяются.</li>
<li>Интерфейс показывает контекст выбранного дома и архивные аналоги отдельно от прогноза модели. Неизвестные удобства остаются неизвестными.</li></ul>
<h2>Источники и покрытие</h2>{source_table}{coverage_table}
<p class="muted">Идентификатор дома основан на текстовом адресе, а не на проверенном реестре недвижимости. Название ЖК может объединять несколько зданий. Отсутствие значения не означает отсутствие свойства.</p>
<h2>Воспроизводимая проверка поиска</h2><p>Выбрано {retrieval['queries']} объявлений из каталога: случайная выборка без возвращения, seed {retrieval['seed']}, исходный порядок по record_id. Для каждого запроса переданы его доступные характеристики и местоположение, а собственная запись, группа и ссылка исключены. Возвращается не более пяти аналогов.</p>
{retrieval_table}{tiers_table}
<p class="note">Это проверка доступности аналогов для уже известных каталогу анкет. У ручного ввода с неполным адресом покрытие может быть ниже. Совпадение только района не обещает соседний дом. Эта диагностика не оценивает MAE, справедливую цену или качество аналогов по экспертной разметке.</p>
<h2>Качество данных и ограничения</h2><ul>
<li>Январские даты наблюдения относятся к 3–6 января 2026 года; часовой пояс исходных значений не указан. Для марта известен только период по описанию автора, без даты отдельного объявления.</li>
<li>Выход за границы Астаны и конфликтующие координаты не используются для расчёта близости. Даже допустимая точка может обозначать центр ЖК и не подтверждает конкретный корпус.</li>
<li>Сведения о парковке взяты из объявления. Пропуск — «неизвестно», не «парковки нет»; упоминание паркинга не подтверждает включённое в цену машиноместо. Конфликтующие свойства дома требуют проверки.</li>
<li>Застройщик и лифт не восстанавливаются догадкой. Если источник их не содержит, поле остаётся неизвестным.</li>
<li>Возможны ошибочные объединения похожих адресов и пропущенные совпадения разных написаний. Дата скачивания не подменяет дату наблюдения.</li></ul>
<p>В журнале качества: {_n(summary['issues'])} записей; одна исходная строка может иметь несколько замечаний. Ниже показаны первые {len(preview)}.</p>{issues_table}
<details><summary>Полная сводка проверок качества</summary><pre>{escape(json.dumps(quality, ensure_ascii=False, indent=2))}</pre></details>
<h2>Модель и воспроизводимость</h2><div class="callout">{escape(model_notice)}</div>{model_table}
<p>Этот скрипт читает каталог и модели, записывает только диагностику и HTML. Обучение и активация модели не выполняются.</p>
<ul><li><a href="retrieval_check.json">Запросы, результаты, seed и контрольные суммы</a></li><li><a href="quality.json">Качество каталога</a></li><li><a href="issues.csv">Полный журнал замечаний</a></li><li><a href="../../data/processed/property_catalog/manifest.json">Манифест каталога</a></li><li><a href="../../data/processed/property_catalog/buildings.csv">Каталог домов</a></li><li><a href="../../data/processed/property_catalog/listings.csv">Каталог объявлений</a></li></ul>
<p class="muted">Сформировано: {escape(summary['created_at_utc'])}.</p></main></body></html>'''
    atomic_write(output, document.encode("utf-8"))


def run_report(catalog_dir: Path = PROPERTY_CATALOG_DIR, report_dir: Path = REPORT_DIR, sample_size: int = 200) -> dict:
    from comparables import find_comparables, load_catalog

    catalog = load_catalog(catalog_dir)
    quality_path, issues_path = report_dir / "quality.json", report_dir / "issues.csv"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    issues = pd.read_csv(issues_path)
    before = protected_status()
    retrieval = check_retrieval(catalog.listings, find_comparables, sample_size=sample_size)
    listings = catalog.listings
    coverage = {"Полный идентификатор дома": int(listings.building_id.notna().sum()),
                "Идентификатор ЖК": int(listings.complex_id.notna().sum()),
                "Дата отдельного наблюдения": int(listings.collected_at.notna().sum()),
                "Сведения о парковке в объявлении": int(listings.parking_type.notna().sum()),
                "Парковка неизвестна": int(listings.parking_type.isna().sum())}
    for status, count in listings.coordinate_status.fillna("unknown").value_counts().items():
        coverage[f"Координаты: {status}"] = int(count)
    after = protected_status()
    if before != after:
        raise AssertionError("Во время проверки изменились модельные артефакты.")
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "listings": len(listings),
               "buildings": len(catalog.buildings), "issues": len(issues), "coverage": coverage,
               "source_counts": {str(key): int(value) for key, value in listings.dataset_period.value_counts().sort_index().items()},
               "protected_artifacts": after}
    retrieval.update({"created_at_utc": summary["created_at_utc"], "catalog_manifest_sha256": sha256(catalog_dir / "manifest.json"),
                      "quality_sha256": sha256(quality_path), "issues_sha256": sha256(issues_path), "summary": summary})
    atomic_write(report_dir / "retrieval_check.json", (json.dumps(retrieval, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    render_report(summary, retrieval, quality, issues, report_dir / "report.html")
    print(json.dumps({"listings": summary["listings"], "buildings": summary["buildings"], "queries": retrieval["queries"],
                      "bins": retrieval["result_count_bins"], "query_location_counts": retrieval["query_location_counts"],
                      "protected_artifacts_unchanged": all(status["unchanged"] for status in after.values())}, ensure_ascii=False, indent=2))
    return retrieval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=PROPERTY_CATALOG_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()
    run_report(args.catalog_dir, args.report_dir, args.sample_size)

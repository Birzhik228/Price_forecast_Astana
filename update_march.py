"""Обновление на март 2026: очистка, общий новый holdout, обучение и активация."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from clean import atomic_write, normalize_text, normalize_url
from config import PROJECT_ROOT
from error_analysis import metric_rows, summarize
from features import FEATURE_COLUMNS
from prediction import load_artifacts


RAW = PROJECT_ROOT / "data/raw/turarr_march_2026/astana_apartments_raw.csv"
MARCH_DATA = PROJECT_ROOT / "data/processed/march_2026"
MARCH_MODELS = PROJECT_ROOT / "models/march_2026"
MARCH_REPORTS = PROJECT_ROOT / "reports/march_update"
JANUARY_DATA = PROJECT_ROOT / "data/processed/january_2026"
JANUARY_MODELS = PROJECT_ROOT / "models/archive/january_2026"
JANUARY_REPORTS = PROJECT_ROOT / "reports/archive/january_2026"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: dict) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def preserve_january() -> None:
    """Исторический снимок не перезаписывается при повторном запуске."""
    archive_meta = JANUARY_MODELS / "model_metadata.json"
    if archive_meta.exists():
        _, meta = load_artifacts(JANUARY_MODELS / "best_model.joblib", archive_meta)
        if digest(JANUARY_DATA / "listings_clean.csv") != meta["input_sha256"]:
            raise ValueError("Исторические январские данные повреждены.")
        return
    active_models = PROJECT_ROOT / "models"
    _, meta = load_artifacts(active_models / "best_model.joblib", active_models / "model_metadata.json")
    if meta.get("data_period") not in (None, "2026-01"):
        raise ValueError("Нельзя считать текущую неянварскую модель исходной январской.")
    if digest(PROJECT_ROOT / "data/processed/listings_clean.csv") != meta["input_sha256"]:
        raise ValueError("Активные данные не соответствуют исходной модели.")
    for folder in (JANUARY_DATA, JANUARY_MODELS, JANUARY_REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    for name in ("best_model.joblib", "model_metadata.json"):
        shutil.copy2(active_models / name, JANUARY_MODELS / name)
    for name in ("listings_clean.csv", "listings_rejected.csv"):
        source = PROJECT_ROOT / "data/processed" / name
        if source.exists():
            shutil.copy2(source, JANUARY_DATA / name)
    # Не копируем архив в самого себя или новый мартовский запуск.
    for source in (PROJECT_ROOT / "reports").iterdir():
        if source.name in {"archive", "march_update"}:
            continue
        target = JANUARY_REPORTS / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    save_json(JANUARY_REPORTS / "snapshot.json", {
        "period": "2026-01", "model_sha256": meta["model_sha256"],
        "input_sha256": meta["input_sha256"], "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Original January model, cleaned data, training/EDA/error reports; not retrained.",
    })


def canonical_token(value) -> str | None:
    normalized = normalize_text(value, lowercase=True)
    if pd.isna(normalized):
        return None
    token = re.sub(r"[\W_]+", "", str(normalized), flags=re.UNICODE)
    return token or None


def apartment_keys(row: dict) -> set[str]:
    """Эвристические совпадения без цены/ремонта; не доказанные ID квартир."""
    keys: set[str] = set()
    url = normalize_url(row.get("url"))
    if pd.notna(url):
        keys.add("url:" + str(url))
    numbers = []
    for column in ("area", "rooms", "floor", "total_floors", "year_built"):
        value = row.get(column)
        if pd.isna(value):
            return keys  # Неполное совпадение слишком неоднозначно.
        numbers.append(format(float(value), ".6g"))
    district = canonical_token(row.get("district"))
    if not district:
        return keys
    core = "|".join([district, *numbers])
    for field in ("address", "residential_complex"):
        location = canonical_token(row.get(field))
        if location:
            keys.add(f"signature:{field}:{core}|{location}")
    return keys


def create_split(march: pd.DataFrame, january: pd.DataFrame, random_state: int = 42):
    """20% новых групп в test; вся остальная мартовская выборка в train."""
    if not march.index.equals(pd.RangeIndex(len(march))):
        raise ValueError("March frame должен иметь RangeIndex.")
    old_keys = set().union(*(apartment_keys(r) for r in january.to_dict("records")))
    parent = list(range(len(march)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    owners: dict[str, int] = {}
    audit = []
    for index, row in enumerate(march.to_dict("records")):
        keys = apartment_keys(row)
        for key in keys:
            if key in owners:
                left, right = find(index), find(owners[key])
                parent[max(left, right)] = min(left, right)
            else:
                owners[key] = index
        matches = keys & old_keys
        audit.append({
            "row_index": index, "source_row": int(row["source_row"]), "url": row.get("url"),
            "january_url_match": any(k.startswith("url:") for k in matches),
            "january_signature_match": any(k.startswith("signature:") for k in matches),
            "signature_available": any(k.startswith("signature:") for k in keys),
        })
    groups = pd.Series([f"apartment_{find(i)}" for i in range(len(march))], index=march.index, name="group")
    audit_frame = pd.DataFrame(audit).assign(group=groups)
    known = audit_frame.january_url_match | audit_frame.january_signature_match
    known_groups = set(groups[known])
    audit_frame["novel_group"] = ~groups.isin(known_groups)
    novel_indices = march.index[audit_frame.novel_group]
    if groups.loc[novel_indices].nunique() < 20:
        raise ValueError("Слишком мало новых групп для независимого holdout.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=random_state)
    _, positions = next(splitter.split(march.loc[novel_indices], groups=groups.loc[novel_indices]))
    test_indices = novel_indices[positions].to_numpy()
    train_indices = march.index[~march.index.isin(test_indices)].to_numpy()
    audit_frame["subset"] = np.where(march.index.isin(test_indices), "test", "train")
    if set(groups.loc[train_indices]) & set(groups.loc[test_indices]):
        raise AssertionError("Одна группа оказалась и в train, и в test.")
    if not audit_frame.loc[test_indices, "novel_group"].all():
        raise AssertionError("В test попали возможные повторные январские квартиры.")
    summary = {
        "rows_march_clean": len(march), "rows_january_reference": len(january),
        "exact_url_overlap_rows": int(audit_frame.january_url_match.sum()),
        "possible_signature_overlap_rows": int(audit_frame.january_signature_match.sum()),
        "either_overlap_rows": int(known.sum()),
        "known_group_rows": int((~audit_frame.novel_group).sum()),
        "novel_rows": len(novel_indices), "groups_march": int(groups.nunique()),
        "rows_with_signature": int(audit_frame.signature_available.sum()),
        "train_rows": len(train_indices), "test_rows": len(test_indices),
        "random_state": random_state,
        "policy": "GroupShuffleSplit: 20% of March groups with no match anywhere in January; remaining March groups train only. No January rows in new training.",
        "identity_caveat": "URL matches are exact. Price-independent district/location/area/rooms/floors/year signatures are conservative candidates, not proven identity. Incomplete signatures, changed addresses and similar flats may leave undetected repeats.",
    }
    return train_indices, test_indices, groups, audit_frame, summary


def render_comparison(report: dict, path: Path) -> None:
    def num(value, digits=2):
        return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    s = report["overlap"]
    comparison = report["common_holdout_comparison"]
    rows = []
    for row in comparison:
        rows.append("<tr><th>" + html.escape(row["model"]) + "</th>" + "".join(f"<td>{cell}</td>" for cell in (
            num(row["mae_tenge"] / 1e6), num(row["rmse_tenge"] / 1e6), num(row["r2"], 3),
            num(row["over_20_pct"], 1) + "%", num(row["bias_tenge"] / 1e6),
        )) + "</tr>")
    change = report["mae_improvement_pct"]
    direction = "снизилась" if change >= 0 else "увеличилась"
    old = report["old_model_all_novel_metrics"]
    cleaning = report["cleaning"]
    activated = "Мартовская модель установлена в приложении." if report["activated"] else "Мартовская модель обучена и сохранена отдельно. В приложении оставлена январская модель."
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Обновление модели на март 2026</title><style>
body{{margin:0;background:#f7f9fc;color:#163047;font:16px/1.65 "Segoe UI",Arial,sans-serif}}main{{max-width:1050px;margin:auto;padding:45px 24px}}h1{{font-size:38px;line-height:1.2}}h2{{font-size:24px;margin-top:32px}}.lead{{font-size:20px}}.callout{{background:#e7f4f0;border-left:5px solid #0f766e;padding:20px;margin:24px 0}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:13px;text-align:right;border-bottom:1px solid #dce5eb;white-space:nowrap}}th:first-child{{text-align:left}}thead{{background:#e8f0f3}}a{{color:#0f766e}}.muted{{color:#587081}}li{{margin:8px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.card{{background:white;padding:22px;border:1px solid #dce5eb;border-radius:9px}}strong{{font-weight:650}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}h1{{font-size:30px}}}}@media print{{body{{background:white}}main{{padding:0}}}}
</style><main><p class="muted">Астана · обновление данных и проверка модели</p><h1>Март 2026: результат переобучения</h1>
<p class="lead">{activated}</p><p>Источник — Turarr, конец марта 2026 года по описанию автора. Дат сбора по строкам нет. Целевая величина — цена предложения, не цена сделки.</p>
<div class="callout">На общей независимой от обучения контрольной выборке из <strong>{s['test_rows']}</strong> мартовских объявлений средняя абсолютная ошибка новой модели <strong>{direction} на {num(abs(change),1)}%</strong> относительно январской.</div>
<h2>Сравнение на одних и тех же квартирах</h2><div class="table"><table><thead><tr><th>Модель</th><th>MAE, млн ₸</th><th>RMSE, млн ₸</th><th>R²</th><th>Ошибка &gt;20%</th><th>Смещение, млн ₸</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">MAE — средняя абсолютная ошибка. Смещение = прогноз − цена объявления. Порог 20% выбран для диагностики. Модель и параметры выбирались по групповой CV только внутри train; контрольная выборка использована после выбора.</p>
<div class="grid"><div class="card"><h2>Обновление данных</h2><ul><li>В сыром источнике: {cleaning['rows_before']} записей.</li><li>После очистки: {s['rows_march_clean']}.</li><li>Исключено: {cleaning['rows_removed']}, с сохранением причин.</li><li>Мартовских строк для обучения: {s['train_rows']}.</li><li>Мартовских строк для итоговой проверки: {s['test_rows']}.</li></ul></div>
<div class="card"><h2>Проверка пересечений</h2><ul><li>Совпадений ссылок с январём: {s['exact_url_overlap_rows']}.</li><li>Возможных совпадений по характеристикам: {s['possible_signature_overlap_rows']} (могут пересекаться с совпадениями ссылок).</li><li>Новых строк после исключения известных групп: {s['novel_rows']}.</li><li>Январская модель на всех этих новых строках: MAE {num(old['mae_tenge']/1e6)} млн ₸; ошибка &gt;20% в {num(old['over_20_pct'],1)}% случаев.</li></ul></div></div>
<h2>Как была проведена проверка</h2><ol><li>Исходный мартовский файл проверен по SHA256. Пропуски сохранены; цены «от …» не принимаются за точные цены. Сарайшык и Сарыарка остаются разными районами.</li>
<li>Совпадения искались по ссылкам и признакам без цены/состояния: район, площадь, комнаты, этаж, этажность, год и адрес либо ЖК. Все строки одной связанной группы попадают в одну часть.</li>
<li>До оценки и обучения зафиксированы 20% новых групп для test. Остальные мартовские строки используются для обучения с GroupKFold. Январские строки не добавлялись в обучение новой модели.</li>
<li>Сравнены LinearRegression, RandomForest и XGBoost. Семейство и параметры выбирались только по MAE групповой CV. Итоговые модели проверены на общем test без дообучения на нём.</li></ol>
<h2>Ограничения</h2><ul><li>Сравнение показывает переносимость январской модели на новый мартовский источник, а также результат обучения на мартовских объявлениях. Изменение ошибки нельзя целиком объяснить изменением рынка: изменились источник и состав выборки.</li>
<li>Совпадение характеристик не доказывает идентичность квартир. Неполные признаки и изменения адресов могут скрывать повторы. Проверка не обеспечивает независимость по зданиям.</li>
<li>Обновление до марта не делает модель оценкой рынка октября и не добавляет прогноз на шесть месяцев.</li><li>Диапазон ±MAE остаётся описательным. Его ширина измерена по итоговому test и не является гарантированным интервалом.</li></ul>
<h2>Файлы проверки</h2><ul><li><a href="update_report.json">Метрики, происхождение и итоги обновления (JSON)</a></li><li><a href="overlap_audit.csv">Пересечения и зафиксированное разбиение</a></li><li><a href="training_report.json">Сравнение алгоритмов и подбор параметров</a></li><li><a href="cleaning_report.json">Отчёт очистки</a></li><li><a href="errors/error_report.html">Диагностика новой модели по районам и полноте</a></li><li><a href="../archive/january_2026/errors/error_report.html">Сохранённый январский отчёт</a></li></ul>
<p class="muted">{html.escape(report['created_at_utc'])}. Январские модель, данные и отчёты сохранены в архиве проекта.</p></main></html>'''
    atomic_write(path, document.encode("utf-8"))


def activate_march() -> None:
    """Опубликовать проверенные парные артефакты, сохраняя историю января."""
    pipeline, meta = load_artifacts(MARCH_MODELS / "best_model.joblib", MARCH_MODELS / "model_metadata.json")
    if digest(MARCH_DATA / "listings_clean.csv") != meta["input_sha256"]:
        raise ValueError("Перед активацией изменились мартовские данные.")
    test = pd.read_csv(MARCH_REPORTS / "test_predictions.csv")
    frame = pd.read_csv(MARCH_DATA / "listings_clean.csv")
    predicted = pipeline.predict(frame.loc[test.row_index, list(FEATURE_COLUMNS)])
    np.testing.assert_allclose(predicted, test.predicted_price, rtol=1e-9, atol=.01)
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("Некорректные прогнозы новой модели: активация отменена.")
    for name in ("listings_clean.csv", "listings_rejected.csv"):
        atomic_write(PROJECT_ROOT / "data/processed" / name, (MARCH_DATA / name).read_bytes())
    for source in MARCH_REPORTS.iterdir():
        if source.name in {"cleaning_report.json", "training_report.json", "model_comparison.csv", "feature_importance.csv", "split_manifest.csv", "test_predictions.csv", "training_environment.txt"}:
            atomic_write(PROJECT_ROOT / "reports" / source.name, source.read_bytes())
    figures = MARCH_REPORTS / "figures"
    if figures.exists():
        for source in figures.glob("*.png"):
            atomic_write(PROJECT_ROOT / "reports/figures" / source.name, source.read_bytes())
    # Описание публикуется последним; load_artifacts отвергает промежуточную пару.
    atomic_write(PROJECT_ROOT / "models/best_model.joblib", (MARCH_MODELS / "best_model.joblib").read_bytes())
    atomic_write(PROJECT_ROOT / "models/model_metadata.json", (MARCH_MODELS / "model_metadata.json").read_bytes())


def run_update(activate: bool = False, resume: bool = False) -> dict:
    from march_data import prepare_march_data
    from march_training import train_march
    preserve_january()
    print("Январская версия сохранена и проверена.", flush=True)
    cleaning = prepare_march_data(RAW, MARCH_DATA / "listings_clean.csv", MARCH_DATA / "listings_rejected.csv", MARCH_REPORTS / "cleaning_report.json")
    march = pd.read_csv(MARCH_DATA / "listings_clean.csv")
    january = pd.read_csv(JANUARY_DATA / "listings_clean.csv")
    train_ids, test_ids, groups, audit, overlap = create_split(march, january)
    audit.to_csv(MARCH_REPORTS / "overlap_audit.csv", index=False, encoding="utf-8-sig")
    save_json(MARCH_REPORTS / "split_policy.json", {**overlap, "input_sha256": digest(MARCH_DATA / "listings_clean.csv")})
    print(json.dumps(overlap, ensure_ascii=False, indent=2), flush=True)
    old_pipeline, old_metadata = load_artifacts(JANUARY_MODELS / "best_model.joblib", JANUARY_MODELS / "model_metadata.json")
    novel = march.loc[audit.novel_group]
    old_novel_pred = old_pipeline.predict(novel.loc[:, list(FEATURE_COLUMNS)])
    old_novel_metrics = summarize(metric_rows(novel.price, old_novel_pred, old_metadata["test_metrics"]["mae_tenge"]))
    pd.DataFrame({"row_index": novel.index, "source_row": novel.source_row, "actual_price": novel.price, "predicted_price": old_novel_pred}).to_csv(MARCH_REPORTS / "old_model_all_novel_predictions.csv", index=False)
    print(f"Январская модель на {len(novel)} новых мартовских строках: MAE {old_novel_metrics['mae_tenge']:,.0f} ₸", flush=True)
    if resume:
        _, staged_meta = load_artifacts(MARCH_MODELS / "best_model.joblib", MARCH_MODELS / "model_metadata.json")
        if staged_meta["input_sha256"] != digest(MARCH_DATA / "listings_clean.csv"):
            raise ValueError("Сохранённая мартовская модель обучена на других данных.")
        saved_split = pd.read_csv(MARCH_REPORTS / "split_manifest.csv").set_index("row_index")
        expected_split = audit.set_index("row_index")
        pd.testing.assert_frame_equal(saved_split[["subset", "group"]], expected_split[["subset", "group"]])
        if digest(MARCH_REPORTS / "split_manifest.csv") != staged_meta["split_manifest_sha256"]:
            raise ValueError("Сохранённое разбиение было изменено.")
        training = json.loads((MARCH_REPORTS / "training_report.json").read_text(encoding="utf-8"))
        if training["old_model"]["model_sha256"] != old_metadata["model_sha256"]:
            raise ValueError("Старая модель для сравнения отличается от исходной.")
        print("Используются проверенные сохранённые артефакты; обучение не повторяется.", flush=True)
    else:
        training = train_march(march, train_ids, test_ids, groups, MARCH_MODELS, MARCH_REPORTS, MARCH_DATA / "listings_clean.csv", old_pipeline, old_metadata)
    new_pipeline, new_metadata = load_artifacts(MARCH_MODELS / "best_model.joblib", MARCH_MODELS / "model_metadata.json")
    comparison = []
    for label, pipeline, width in (("Январская модель", old_pipeline, old_metadata["test_metrics"]["mae_tenge"]),
                                   ("Мартовская модель", new_pipeline, new_metadata["test_metrics"]["mae_tenge"])):
        y = march.loc[test_ids, "price"]
        pred = pipeline.predict(march.loc[test_ids, list(FEATURE_COLUMNS)])
        from train import compute_metrics
        comparison.append({"model": label, **summarize(metric_rows(y, pred, width)), "r2": compute_metrics(y, pred)["r2"]})
    # Activation is an explicit CLI choice. It does not tune/select on test.
    if activate:
        activate_march()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "data_period": "2026-03",
        "activated": activate, "cleaning": cleaning, "overlap": overlap,
        "old_model_all_novel_metrics": old_novel_metrics, "common_holdout_comparison": comparison,
        "mae_improvement_pct": (1 - comparison[1]["mae_tenge"] / comparison[0]["mae_tenge"]) * 100,
        "old_model_sha256": old_metadata["model_sha256"], "new_model_sha256": new_metadata["model_sha256"],
        "input_sha256": new_metadata["input_sha256"], "selected_model": training["model_name"],
    }
    save_json(MARCH_REPORTS / "update_report.json", report)
    render_comparison(report, MARCH_REPORTS / "update_report.html")
    print(json.dumps({"common_holdout": comparison, "activated": activate, "report": str(MARCH_REPORTS / "update_report.html")}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate", action="store_true", help="Установить проверенную мартовскую модель и данные как текущие.")
    parser.add_argument("--resume", action="store_true", help="Использовать уже обученную мартовскую модель после проверки данных и разбиения.")
    arguments = parser.parse_args()
    run_update(activate=arguments.activate, resume=arguments.resume)

"""Веб-форма Streamlit: прогноз цены предложения квартиры в Астане."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import (
    APP_MODEL_METADATA_PATH as MODEL_METADATA_PATH,
    APP_MODEL_PIPELINE_PATH as MODEL_PIPELINE_PATH,
    PROPERTY_CATALOG_DIR,
)
from comparables import PropertyCatalog, find_comparables, load_catalog, safe_listing_url
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from prediction import FIELD_LABELS, INPUT_LIMITS
from location_prediction import load_app_artifacts as load_artifacts, predict_app_apartment as predict_apartment


LOGGER = logging.getLogger(__name__)


def format_tenge(value: float) -> str:
    """Округление применяется только при отображении."""
    return f"{value:,.0f}".replace(",", " ") + " ₸"


def category_label(value: str | None) -> str:
    """Подпись отличается от исходного значения, которое получает модель."""
    return "Не указано" if value is None else value[0].upper() + value[1:]


def file_signature(path: Path) -> tuple[int, int]:
    """Изменение файла сбрасывает кэш загрузки модели."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_artifacts(
    model_path: str, metadata_path: str,
    model_signature: tuple[int, int], metadata_signature: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    """Сигнатуры входят в ключ кэша; сами файлы читаются только при загрузке."""
    return load_artifacts(Path(model_path), Path(metadata_path))


@st.cache_data(show_spinner=False, max_entries=2)
def cached_catalog(
    directory: str, signatures: tuple[tuple[int, int], ...],
) -> PropertyCatalog:
    """Каталог обновляется независимо от модели и проверяется по SHA256."""
    return load_catalog(Path(directory))


def optional_value(value: Any) -> Any:
    """CSV-пропуски не попадают в состояние формы как nan или '<NA>'."""
    return None if value is None or pd.isna(value) else value


def autofill_complex_district(districts: dict[str, str]) -> None:
    """Обновлять район только при смене ЖК, сохраняя последующие ручные правки."""
    st.session_state["district"] = districts.get(st.session_state.get("location_complex"))


def location_selector(catalog: PropertyCatalog, metadata: dict[str, Any]) -> dict[str, Any]:
    """Местоположение для модели и аналогов; сведения дома можно перенести в форму."""
    listings, buildings = catalog.listings, catalog.buildings
    complexes = listings.loc[listings["complex_id"].notna(), ["complex_id", "complex_name"]]
    complex_labels = complexes.drop_duplicates("complex_id").set_index("complex_id")["complex_name"].to_dict()
    complex_ids = sorted(complex_labels, key=lambda key: str(complex_labels[key]).casefold())
    # Все объявления: каталог разрешённых домов может скрыть спорные районы ЖК.
    known_districts = listings.dropna(subset=["complex_id", "district"]).groupby("complex_id")["district"].unique()
    complex_districts = {
        complex_id: districts[0] for complex_id, districts in known_districts.items()
        if len(districts) == 1 and districts[0] in metadata["categories_from_train"]["district"]
    }
    if st.session_state.get("location_complex") not in complex_labels:
        st.session_state["location_complex"] = None
    location_model = metadata.get("schema_version") in (2, 3)
    uses_coordinates = metadata.get("schema_version") == 2
    with st.expander("ЖК и адрес квартиры", expanded=True):
        st.caption(
            "Необязательно. ЖК участвует в оценке цены. По адресу модель учитывает координаты, если они известны для этого дома. Выбор также уточняет поиск аналогов."
            if uses_coordinates else
            "Необязательно. ЖК участвует в оценке CatBoost. Адрес и координаты помогают подобрать похожие объявления из архива."
            if location_model else
            "Необязательно. Выбор места уточняет поиск аналогов. Модель цены использует восемь параметров ниже."
        )
        complex_id = st.selectbox(
            "Жилой комплекс", [None, *complex_ids], key="location_complex",
            placeholder="Выберите ЖК или оставьте пустым",
            format_func=lambda value: "Не указан" if value is None else str(complex_labels[value]),
            on_change=autofill_complex_district, args=(complex_districts,),
        )
        if complex_id and complex_id not in complex_districts:
            st.caption("Для этого ЖК район неизвестен или указан по-разному. Выберите район вручную в анкете ниже.")
        available = buildings if complex_id is None else buildings.loc[buildings["complex_id"].eq(complex_id)]
        house_labels = available.set_index("building_id")["display_label"].to_dict()
        house_ids = sorted(house_labels, key=lambda key: str(house_labels[key]).casefold())
        if st.session_state.get("location_building") not in house_labels:
            st.session_state["location_building"] = None
        building_id = st.selectbox(
            "Дом / адрес", [None, *house_ids], key="location_building",
            placeholder="Выберите дом или оставьте пустым",
            format_func=lambda value: "Не указан" if value is None else str(house_labels[value]),
        )
        location: dict[str, Any] = {"complex_id": complex_id, "building_id": building_id}
        if complex_id:
            location["complex_name"] = complex_labels[complex_id]
        if building_id:
            building = available.loc[available["building_id"].eq(building_id)].iloc[0]
            for field in ("complex_id", "complex_name", "address_normalized", "latitude", "longitude", "coordinate_status"):
                location[field] = optional_value(building.get(field))
            location["location_district"] = optional_value(building.get("district"))
            parking = optional_value(building.get("parking_type"))
            st.write(f"Парковка по архиву: {parking or 'неизвестно'}. Лифт и застройщик: сведений пока нет.")
            if location.get("coordinate_status") == "plausible":
                st.caption("Координаты прошли базовую проверку; точность до дома не подтверждена. Расстояния приблизительные.")
            else:
                st.caption("Надёжных координат для расчёта расстояния нет. Подбор возможен по дому, ЖК и району.")
            if st.button("Подставить известные параметры дома", key="fill_building"):
                filled = []
                for field in ("district", "year_built", "total_floors", "building_type"):
                    value = optional_value(building.get(field))
                    if value is None:
                        continue
                    if field in CATEGORICAL_FEATURES:
                        if value not in metadata["categories_from_train"][field]:
                            continue
                    else:
                        low, high = INPUT_LIMITS[field]
                        if not low <= float(value) <= high or not float(value).is_integer():
                            continue
                        value = int(value)
                    st.session_state[field] = value
                    filled.append(FIELD_LABELS[field].lower())
                st.session_state.pop("prediction", None)
                st.session_state.pop("last_input", None)
                st.session_state.pop("last_location", None)
                st.info("Перенесены сведения из архива: " + ", ".join(filled) + ". Проверьте их перед расчётом." if filled else "Для этого дома нет согласованных параметров для заполнения.")
        elif not complex_id:
            st.caption("Если ЖК и дом не указаны, для подбора аналогов понадобится район.")
        own_url = st.text_input(
            "Ссылка на оцениваемое объявление (если есть)", key="location_exclude_url",
            placeholder="https://krisha.kz/a/show/…",
            help="Позволяет исключить само объявление и его известные повторы из аналогов. Ссылка не открывается и не передаётся модели.",
        ).strip()
        location["exclude_url"] = safe_listing_url(own_url) if own_url else None
        location["invalid_listing_url"] = bool(own_url and not location["exclude_url"])
    return location


def manual_location_selector(location: dict[str, Any], uses_coordinates: bool = True) -> dict[str, Any]:
    """Allow a location outside the archive, preserving unknown values."""
    result = location.copy()
    if result.get("complex_name") and result.get("address_normalized"):
        return result
    with st.expander("ЖК или адреса нет в списке", expanded=not bool(location)):
        if not result.get("complex_name"):
            value = st.text_input("Название ЖК", key="location_manual_complex", max_chars=200,
                                  placeholder="Необязательно, например: название вашего ЖК").strip()
            result["complex_name"] = value or None
        if not result.get("address_normalized"):
            value = st.text_input("Адрес дома", key="location_manual_address", max_chars=240,
                                  placeholder="Улица и номер дома",
                                  help=("Укажите район в анкете ниже. Если адрес модели неизвестен, расчёт останется доступен без известных координат дома."
                                        if uses_coordinates else
                                        "Укажите район в анкете ниже. Адрес уточняет поиск архивных аналогов; CatBoost использует параметры квартиры и ЖК.")).strip()
            result["address_normalized"] = value or None
    return result


def render_comparables(catalog: PropertyCatalog, values: dict[str, Any], location: dict[str, Any]) -> None:
    """Архивные цены служат отдельным сравнением и не изменяют прогноз модели."""
    st.subheader("Похожие объявления из архива")
    st.caption("Январь и март 2026 года. Цены предложения; доступность объявлений сегодня не проверена. Сравнение не меняет прогноз модели.")
    matches = find_comparables(catalog.listings, values | location)
    if matches.empty:
        st.info("Подходящих аналогов нет. Укажите ЖК, дом или район. Ищем то же число комнат и площадь в пределах ±20%.")
        return
    st.caption("Подбор: тот же дом → ЖК → примерно до 2 км → район; одинаковое число комнат, площадь ±20%. Цены не влияют на порядок.")
    for _, row in matches.iterrows():
        with st.container(border=True):
            st.write(f"**{int(row['rooms'])}-комн. · {row['area']:g} м² · {format_tenge(row['price'])}**")
            st.caption(f"{format_tenge(row['price_per_m2'])} за м²")
            address = optional_value(row.get("address_normalized")) or optional_value(row.get("address")) or "Адрес не уточнён"
            complex_name = optional_value(row.get("complex_name"))
            st.text((f"ЖК {complex_name} · " if complex_name else "") + str(address))
            floor = optional_value(row.get("floor"))
            total = optional_value(row.get("total_floors"))
            if floor is not None:
                st.caption(f"Этаж {floor:g}" + (f" из {total:g}" if total is not None else ""))
            st.text(str(row["similarity_reason"]))
            distance = optional_value(row.get("distance_m"))
            if distance is not None:
                st.caption(f"Примерно {distance / 1000:.1f} км по прямой; координаты из объявлений.")
            st.caption(str(row["date_label"]))
            url = safe_listing_url(row.get("url"))
            if url:
                st.link_button("Открыть объявление", url)
    st.caption("Архив может содержать невыявленные повторы. Квартиры могут отличаться ремонтом, видом и другими деталями, которых нет в анкете.")


def main() -> None:
    """Форма отправляет все значения одной кнопкой, без переобучения модели."""
    st.set_page_config(page_title="Цена квартиры · Астана", page_icon="🏙️", layout="centered")
    st.caption("АСТАНА · УЧЕБНАЯ МОДЕЛЬ")
    st.title("Цена квартиры в Астане")
    st.write("Укажите параметры квартиры и получите оценку в тенге.")

    try:
        file_signatures = (file_signature(MODEL_PIPELINE_PATH), file_signature(MODEL_METADATA_PATH))
        signature = (str(MODEL_PIPELINE_PATH), str(MODEL_METADATA_PATH), *file_signatures)
        with st.spinner("Загружаем модель…"):
            pipeline, metadata = cached_artifacts(
                str(MODEL_PIPELINE_PATH), str(MODEL_METADATA_PATH), *file_signatures,
            )
    except FileNotFoundError:
        st.error("Файлы выбранной модели не найдены. Восстановите сохранённые артефакты и подготовьте модель для приложения по README.")
        return
    except Exception:
        LOGGER.exception("Не удалось загрузить модель")
        st.error("Не удалось загрузить модель и её описание. Проверьте, что оба файла из одного запуска обучения и зависимости установлены по README.")
        return

    data_period_label = metadata.get("data_period_label", "января 2026 года")
    uses_location = metadata.get("schema_version") in (2, 3)
    uses_coordinates = metadata.get("schema_version") == 2
    if uses_location:
        st.caption("Модель: " + metadata.get("model_label", "ЖК и координаты" if uses_coordinates else "CatBoost с MAE · параметры квартиры и ЖК"))
        if uses_coordinates and metadata.get("geo_enrichment"):
            st.caption("Координаты части домов дополнены по точным адресам "
                       "[OpenStreetMap](https://www.openstreetmap.org/copyright), снимок 12 октября 2026. "
                       "© OpenStreetMap contributors · ODbL 1.0.")
    st.caption(f"Модель обучена на объявлениях {data_period_label}. Это не оценка рынка на сегодня.")

    if st.session_state.get("artifact_signature") != signature:
        st.session_state.pop("prediction", None)
        st.session_state.pop("last_input", None)
        st.session_state.pop("last_location", None)
        st.session_state["artifact_signature"] = signature

    catalog: PropertyCatalog | None = None
    catalog_signature = None
    try:
        catalog_signature = tuple(file_signature(PROPERTY_CATALOG_DIR / name) for name in (
            "manifest.json", "listings.csv", "buildings.csv",
        ))
        catalog = cached_catalog(str(PROPERTY_CATALOG_DIR), catalog_signature)
    except FileNotFoundError:
        st.caption("Справочник домов ещё не подготовлен. Оценка доступна; поиск аналогов появится после подготовки каталога.")
    except Exception:
        LOGGER.exception("Не удалось проверить справочник домов")
        st.warning("Поиск аналогов временно недоступен: справочник не прошёл проверку. Оценка модели доступна.")
    catalog_state = (catalog is not None, catalog_signature)
    if st.session_state.get("catalog_signature") != catalog_state:
        st.session_state.pop("prediction", None)
        st.session_state.pop("last_input", None)
        st.session_state.pop("last_location", None)
        st.session_state["catalog_signature"] = catalog_state
    location = location_selector(catalog, metadata) if catalog is not None else {}
    if uses_location:
        location = manual_location_selector(location, uses_coordinates=uses_coordinates)
    location_selection = tuple(location.get(field) for field in (
        "complex_id", "building_id", "complex_name", "address_normalized", "location_district",
        "exclude_url", "invalid_listing_url",
    ))
    if st.session_state.get("location_selection") != location_selection:
        st.session_state.pop("prediction", None)
        st.session_state.pop("last_input", None)
        st.session_state.pop("last_location", None)
        st.session_state["location_selection"] = location_selection

    with st.form("apartment_form"):
        st.subheader("Параметры квартиры")
        left, right = st.columns(2)
        values: dict[str, Any] = {}
        with left:
            values["district"] = st.selectbox(
                "Район", [None, *metadata["categories_from_train"]["district"]],
                format_func=category_label, key="district",
            )
            values["area"] = st.number_input(
                "Площадь, м²", min_value=10.0, max_value=1000.0,
                value=None, step=0.5, format="%.2f", key="area", placeholder="Например, 60",
            )
            values["floor"] = st.number_input(
                "Этаж квартиры", min_value=1, max_value=100, value=None, step=1,
                key="floor", placeholder="Не указан", help="Если этаж неизвестен, оставьте поле пустым.",
            )
            values["building_type"] = st.selectbox(
                "Тип дома", [None, *metadata["categories_from_train"]["building_type"]],
                format_func=category_label, key="building_type",
            )
        with right:
            values["rooms"] = st.number_input(
                "Количество комнат", min_value=1, max_value=20, value=None, step=1,
                key="rooms", placeholder="Например, 2",
            )
            values["year_built"] = st.number_input(
                "Год постройки", min_value=1850, max_value=int(INPUT_LIMITS["year_built"][1]),
                value=None, step=1, key="year_built", placeholder="Не указан",
                help="Для строящегося дома укажите планируемый год. Неизвестный год можно оставить пустым.",
            )
            values["total_floors"] = st.number_input(
                "Этажность дома", min_value=1, max_value=100, value=None, step=1,
                key="total_floors", placeholder="Не указана", help="Если этажность неизвестна, оставьте поле пустым.",
            )
            values["condition"] = st.selectbox(
                "Состояние", [None, *metadata["categories_from_train"]["condition"]],
                format_func=category_label, key="condition",
            )
        st.caption("Площадь и количество комнат обязательны. Неизвестные параметры можно оставить пустыми.")
        submitted = st.form_submit_button("Рассчитать стоимость", type="primary", use_container_width=True)

    if submitted:
        # Ошибка нового ввода не должна оставлять на экране старый прогноз.
        st.session_state.pop("prediction", None)
        st.session_state.pop("last_input", None)
        st.session_state.pop("last_location", None)
        try:
            if location.get("invalid_listing_url"):
                raise ValueError("Укажите ссылку на карточку Krisha вида https://krisha.kz/a/show/123456789 или оставьте поле пустым.")
            house_district = location.get("location_district")
            if house_district and values.get("district") and house_district != values["district"]:
                raise ValueError("Район в анкете отличается от района выбранного дома. Исправьте район или выбор дома перед расчётом.")
            with st.spinner("Рассчитываем стоимость…"):
                result = predict_apartment(pipeline, metadata, values, location=location)
            st.session_state["prediction"] = result
            st.session_state["last_input"] = values.copy()
            st.session_state["last_location"] = location.copy()
        except ValueError as error:
            st.error(str(error))
        except Exception:
            LOGGER.exception("Ошибка предсказания")
            st.error("Не удалось рассчитать стоимость. Проверьте файлы модели и повторите попытку.")

    if "prediction" in st.session_state:
        result = st.session_state["prediction"]
        st.subheader("Результат последнего расчёта")
        st.caption("После изменения параметров нажмите «Рассчитать стоимость» ещё раз.")
        price_column, unit_column = st.columns(2)
        price_column.metric("Прогноз цены", format_tenge(result.price_tenge))
        unit_column.metric("Цена за м²", format_tenge(result.price_per_m2))
        st.info(f"Ориентировочный диапазон: **{format_tenge(result.lower_tenge)} — {format_tenge(result.upper_tenge)}**")
        st.caption(
            f"Диапазон = прогноз ± средняя абсолютная ошибка ({format_tenge(metadata['test_metrics']['mae_tenge'])}). "
            "Это не доверительный интервал и не гарантия: ошибка для конкретной квартиры может быть больше. "
            "Нижняя граница ограничена нулём."
        )
        for notice in result.warnings:
            st.warning(notice)
        for note in getattr(result, "location_notes", ()):
            st.caption(note)
        with st.expander("Параметры этого расчёта"):
            snapshot = st.session_state["last_input"]
            labels = []
            for column in (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES):
                value = snapshot[column]
                displayed = category_label(value) if column in CATEGORICAL_FEATURES else (
                    "Не указано" if value is None else f"{value:g}"
                )
                labels.append({"Параметр": FIELD_LABELS[column], "Значение": displayed})
            if uses_location:
                place = st.session_state["last_location"]
                labels.extend([
                    {"Параметр": "ЖК", "Значение": place.get("complex_name") or "Не указан"},
                    {"Параметр": "Адрес дома", "Значение": place.get("address_normalized") or "Не указан"},
                ])
            st.table(pd.DataFrame(labels).set_index("Параметр"))

        if catalog is not None:
            try:
                render_comparables(catalog, st.session_state["last_input"], st.session_state["last_location"])
            except Exception:
                LOGGER.exception("Не удалось подобрать архивные объявления")
                st.warning("Не удалось подобрать похожие объявления. Рассчитанная оценка модели доступна выше.")

    st.divider()
    st.write("**Прогноз цены предложения, а не цены сделки.**")
    with st.expander("Как понимать оценку"):
        st.write(
            "Модель сравнивает параметры с закономерностями в объявлениях о продаже квартир. "
            + ("Учитывает ЖК и доступные по адресу координаты дома. Фактическая цена сделки, торг и вид из окна в прогнозе не учитываются."
               if uses_coordinates else
               "CatBoost учитывает восемь основных параметров и ЖК. Адрес и координаты помогают подбирать архивные аналоги. Фактическая цена сделки, торг и вид из окна в прогнозе не учитываются."
               if uses_location else "Фактическая цена сделки, торг, точный адрес, вид из окна и другие детали в прогнозе не учитываются.")
        )
        st.write(
            f"Средняя абсолютная ошибка {'на повторно использованной мартовской выборке' if uses_location else 'на отложенной выборке'} — {format_tenge(metadata['test_metrics']['mae_tenge'])}. "
            "Это среднее по разным квартирам. Особенно дорогие или необычные квартиры могут оцениваться значительно хуже."
        )
        if uses_coordinates:
            st.write("Координаты взяты из обучающего справочника и проверены на внутреннюю согласованность; точность до дома внешним источником не подтверждена. Проверка на свежих данных ещё не проведена.")


if __name__ == "__main__":
    main()

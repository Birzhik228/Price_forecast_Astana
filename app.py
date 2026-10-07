"""Веб-форма Streamlit: прогноз цены предложения квартиры в Астане."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from config import MODEL_METADATA_PATH, MODEL_PIPELINE_PATH
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from prediction import FIELD_LABELS, INPUT_LIMITS, load_artifacts, predict_apartment


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
) -> tuple[Pipeline, dict[str, Any]]:
    """Сигнатуры входят в ключ кэша; сами файлы читаются только при загрузке."""
    return load_artifacts(Path(model_path), Path(metadata_path))


def main() -> None:
    """Форма отправляет все значения одной кнопкой, без переобучения модели."""
    st.set_page_config(page_title="Цена квартиры · Астана", page_icon="🏙️", layout="centered")
    st.caption("АСТАНА · УЧЕБНАЯ МОДЕЛЬ")
    st.title("Цена квартиры в Астане")
    st.write("Укажите параметры квартиры и получите оценку в тенге.")
    st.caption("Модель обучена на объявлениях января 2026 года. Это не оценка рынка на сегодня.")

    try:
        signature = (file_signature(MODEL_PIPELINE_PATH), file_signature(MODEL_METADATA_PATH))
        with st.spinner("Загружаем модель…"):
            pipeline, metadata = cached_artifacts(
                str(MODEL_PIPELINE_PATH), str(MODEL_METADATA_PATH), *signature,
            )
    except FileNotFoundError:
        st.error("Файлы обученной модели не найдены. Сначала выполните обучение из папки проекта.")
        st.code(r".\.venv\Scripts\python.exe -X utf8 train.py", language="powershell")
        return
    except Exception:
        LOGGER.exception("Не удалось загрузить модель")
        st.error("Не удалось загрузить модель и её описание. Проверьте, что оба файла из одного запуска обучения и зависимости установлены по README.")
        return

    if st.session_state.get("artifact_signature") != signature:
        st.session_state.pop("prediction", None)
        st.session_state.pop("last_input", None)
        st.session_state["artifact_signature"] = signature

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
        try:
            with st.spinner("Рассчитываем стоимость…"):
                result = predict_apartment(pipeline, metadata, values)
            st.session_state["prediction"] = result
            st.session_state["last_input"] = values.copy()
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
        with st.expander("Параметры этого расчёта"):
            snapshot = st.session_state["last_input"]
            labels = []
            for column in (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES):
                value = snapshot[column]
                displayed = category_label(value) if column in CATEGORICAL_FEATURES else (
                    "Не указано" if value is None else f"{value:g}"
                )
                labels.append({"Параметр": FIELD_LABELS[column], "Значение": displayed})
            st.table(pd.DataFrame(labels).set_index("Параметр"))

    st.divider()
    st.write("**Прогноз цены предложения, а не цены сделки.**")
    with st.expander("Как понимать оценку"):
        st.write(
            "Модель сравнивает параметры с закономерностями в объявлениях о продаже квартир. "
            "Фактическая цена сделки, торг, точный адрес, вид из окна и другие детали в прогнозе не учитываются."
        )
        st.write(
            f"Средняя абсолютная ошибка на отложенной выборке — {format_tenge(metadata['test_metrics']['mae_tenge'])}. "
            "Это среднее по разным квартирам. Особенно дорогие или необычные квартиры могут оцениваться значительно хуже."
        )


if __name__ == "__main__":
    main()

"""Предобработка восьми входных признаков без доступа к целевой цене."""

from __future__ import annotations

from typing import Final

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "area", "rooms", "floor", "total_floors", "year_built",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "district", "building_type", "condition",
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: Final[str] = "price"
MISSING_CATEGORY: Final[str] = "не указано"


def build_preprocessor() -> ColumnTransformer:
    """Создать необученный преобразователь: fit вызывается только на train."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True,
        )),
        # Масштабирование нужно линейной модели; для деревьев оно допустимо.
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant", fill_value=MISSING_CATEGORY,
            keep_empty_features=True,
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", numeric, list(NUMERIC_FEATURES)),
         ("categorical", categorical, list(CATEGORICAL_FEATURES))],
        # Цена, цена за м² и служебные колонки никогда не попадают в модель.
        remainder="drop", sparse_threshold=0,
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Сохранение такого пайплайна включает и предобработку, и модель."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])

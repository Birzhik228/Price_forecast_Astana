"""Native-categorical CatBoost candidates with preprocessing fitted on train only.

Every prediction and sklearn score uses prices in tenge.  The log candidate's
RMSE is a training objective, never the criterion returned by ``score``.
The default experiment uses eight base fields and complex name, with no
coordinates.  An explicit cloneable location transformer can support a separate
geographic experiment; this module never reads a catalogue or a registry.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted

from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from location_model import (
    LOCATION_INPUT_COLUMNS, LocationFeatureTransformer, prepare_location_input,
)


CATBOOST_TARGET_MODES = ("price_mae", "log_rmse")
CATBOOST_CATEGORICAL_COLUMNS = (*CATEGORICAL_FEATURES, "residential_complex")
CATBOOST_NUMERIC_COLUMNS = tuple(NUMERIC_FEATURES)
CATBOOST_MODEL_COLUMNS = (*FEATURE_COLUMNS, "residential_complex")
CATBOOST_COORDINATE_COLUMNS = ("latitude", "longitude")
CATBOOST_MISSING_CATEGORY = "__CATBOOST_MISSING__"


def _catboost_regressor():
    # Importing the experimental module does not require CatBoost.  Actual fit
    # does, and a missing installation must fail rather than change algorithms.
    from catboost import CatBoostRegressor
    return CatBoostRegressor


def _target(y: Any, index: pd.Index) -> np.ndarray:
    if isinstance(y, pd.Series) and not y.index.equals(index):
        raise ValueError("Индексы y и X не совпадают.")
    try:
        raw = np.asarray(y)
        values = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Цена должна быть числовой.") from error
    if (raw.dtype.kind == "b" or values.ndim != 1 or len(values) != len(index)
            or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError("Нужна одна положительная конечная цена на каждую строку X.")
    return values


def _categorical_value(value: Any) -> str:
    if pd.isna(value):
        return CATBOOST_MISSING_CATEGORY
    if not isinstance(value, str):
        raise ValueError("Категории CatBoost должны быть строками или пропусками.")
    return value if value.strip() else CATBOOST_MISSING_CATEGORY


def _native_frame(transformed: pd.DataFrame, index: pd.Index, expected_columns=None) -> pd.DataFrame:
    allowed = (set(CATBOOST_MODEL_COLUMNS), set((*CATBOOST_MODEL_COLUMNS, *CATBOOST_COORDINATE_COLUMNS)))
    if (not isinstance(transformed, pd.DataFrame) or not transformed.columns.is_unique
            or not transformed.index.equals(index)
            or set(transformed.columns) not in allowed):
        raise ValueError("Location transformer должен сохранять индекс и возвращать 8 полей + ЖК, с необязательной парой latitude/longitude.")
    columns = (*CATBOOST_MODEL_COLUMNS, *CATBOOST_COORDINATE_COLUMNS) if "latitude" in transformed else CATBOOST_MODEL_COLUMNS
    if expected_columns is not None and tuple(expected_columns) != columns:
        raise ValueError("Схема location transform изменилась после fit.")
    result = transformed.loc[:, list(columns)].copy(deep=True)
    for column in (*CATBOOST_NUMERIC_COLUMNS, *(name for name in CATBOOST_COORDINATE_COLUMNS if name in result)):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        if np.isinf(result[column].to_numpy()).any():
            raise ValueError(f"Бесконечное значение в {column}.")
    for column in CATBOOST_CATEGORICAL_COLUMNS:
        result[column] = result[column].map(_categorical_value).astype(object)
    return result


class CatBoostPriceRegressor(RegressorMixin, BaseEstimator):
    """Two explicit objectives, fixed iterations, no validation set accepted.

    ``location_transformer=None`` uses ``LocationFeatureTransformer('complex')``:
    neither raw coordinates nor house identity becomes a model feature. An
    injected transformer is cloned for each fit and receives
    only the 13 approved input columns, without y.  It must output exactly
    ``CATBOOST_MODEL_COLUMNS`` (optionally plus latitude/longitude as a pair)
    and preserve the row index.  Categorical target
    statistics, if used by CatBoost, are therefore learned inside that fit.

    ``score`` returns **negative** MAE in tenge, following sklearn's convention
    that larger scores are better.  ``predict`` always returns tenge, including
    for ``log_rmse``.  No smearing correction or output clipping is applied.
    """

    def __init__(
        self, target_mode: str = "price_mae", location_transformer: BaseEstimator | None = None,
        iterations: int = 600, depth: int = 6, learning_rate: float = .05,
        l2_leaf_reg: float = 3., random_seed: int = 42, thread_count: int = 2,
    ):
        self.target_mode = target_mode
        self.location_transformer = location_transformer
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.thread_count = thread_count

    def _validate_parameters(self) -> None:
        if self.target_mode not in CATBOOST_TARGET_MODES:
            raise ValueError(f"Неизвестный target_mode: {self.target_mode}")
        for name in ("iterations", "depth", "thread_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} должен быть положительным целым числом.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, Integral) or self.random_seed < 0:
            raise ValueError("random_seed должен быть неотрицательным целым числом.")
        for name in ("learning_rate", "l2_leaf_reg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Некорректный {name}.")
        if self.learning_rate == 0:
            raise ValueError("learning_rate должен быть положительным.")

    def fit(self, X: pd.DataFrame, y: Any) -> "CatBoostPriceRegressor":
        self._validate_parameters()
        frame = prepare_location_input(X)
        if frame.empty or not frame.index.is_unique:
            raise ValueError("Для fit нужны непустые строки с уникальным индексом.")
        target = _target(y, frame.index)
        transformer = clone(self.location_transformer) if self.location_transformer is not None else (
            LocationFeatureTransformer("complex")
        )
        # Do not use fit_transform(X, y): a geographic policy has no reason to
        # observe prices, including transformed prices or any held-out labels.
        transformer.fit(frame)
        native = _native_frame(transformer.transform(frame), frame.index)
        loss = "MAE" if self.target_mode == "price_mae" else "RMSE"
        fitted_target = target if self.target_mode == "price_mae" else np.log1p(target)
        model = _catboost_regressor()(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg, random_seed=self.random_seed,
            thread_count=self.thread_count, loss_function=loss, eval_metric=loss,
            cat_features=list(CATBOOST_CATEGORICAL_COLUMNS), task_type="CPU",
            nan_mode="Min", use_best_model=False, allow_writing_files=False, verbose=False,
        )
        model.fit(native, fitted_target)
        self.location_transformer_ = transformer
        self.model_ = model
        self.target_mode_ = self.target_mode
        self.training_loss_ = loss
        self.feature_names_in_ = np.asarray(LOCATION_INPUT_COLUMNS, dtype=object)
        self.n_features_in_ = len(LOCATION_INPUT_COLUMNS)
        self.model_feature_names_ = np.asarray(native.columns, dtype=object)
        self.training_rows_ = len(frame)
        return self

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inspect the exact native features; does not update fitted geography."""
        check_is_fitted(self, ("location_transformer_", "model_", "target_mode_"))
        frame = prepare_location_input(X)
        return _native_frame(self.location_transformer_.transform(frame), frame.index, self.model_feature_names_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        native = self.transform_features(X)
        if native.empty:
            return np.empty(0, dtype=float)
        result = np.asarray(self.model_.predict(native), dtype=float)
        if self.target_mode_ == "log_rmse":
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.expm1(result)
        if result.shape != (len(native),) or not np.isfinite(result).all():
            raise ValueError("CatBoost вернул некорректные или неконечные прогнозы.")
        return result

    def score(self, X: pd.DataFrame, y: Any, sample_weight: Any = None) -> float:
        """Negative MAE in tenge for both objectives, never log RMSE or R²."""
        actual = _target(y, X.index)
        return -float(mean_absolute_error(actual, self.predict(X), sample_weight=sample_weight))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "model_feature_names_")
        return self.model_feature_names_.copy()


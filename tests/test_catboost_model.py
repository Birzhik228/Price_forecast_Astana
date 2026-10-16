"""Tiny tests of objective scale, train-only features and portable CatBoost."""

from __future__ import annotations

import copy
import importlib.util
import io
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error

from catboost_model import (
    CATBOOST_CATEGORICAL_COLUMNS, CATBOOST_MISSING_CATEGORY, CATBOOST_MODEL_COLUMNS,
    CatBoostPriceRegressor,
)
from features import FEATURE_COLUMNS
from location_model import LOCATION_INPUT_COLUMNS, LocationFeatureTransformer


def row(house="astana|есиль|тестовая 1", **changes):
    return {"area": 60., "rooms": 2., "floor": 5., "total_floors": 9., "year_built": 2015.,
            "district": "Есиль", "building_type": "кирпичный", "condition": "хорошее",
            "residential_complex": "Пример", "house_key": house, "raw_latitude": np.nan,
            "raw_longitude": np.nan, "raw_coordinate_status": "missing", **changes}


class RecordingCatBoost:
    """Record the fitting boundary while tests avoid expensive model fitting."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.frame_ = X.copy(deep=True)
        self.target_ = np.asarray(y).copy()
        self.constant_ = float(self.target_.mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.constant_)


class TargetGuardTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, latitude=51.2):
        self.latitude = latitude

    def fit(self, X, y=None):
        if y is not None:
            raise AssertionError("A geographic transformer must never receive targets")
        if tuple(X.columns) != LOCATION_INPUT_COLUMNS:
            raise AssertionError("Forbidden columns reached geographic fitting")
        self.train_indices_ = X.index.tolist()
        return self

    def transform(self, X):
        result = X.loc[:, [*FEATURE_COLUMNS, "residential_complex"]].copy()
        result["latitude"], result["longitude"] = self.latitude, 71.4
        return result


class ReorderingTransformer(TargetGuardTransformer):
    def transform(self, X):
        return super().transform(X).iloc[::-1]


class CatBoostModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([row(area=40. + i * 10) for i in range(4)], index=[7, 2, 9, 1])
        self.target = pd.Series([20e6, 30e6, 40e6, 50e6], index=self.frame.index)
        self.fake = patch("catboost_model._catboost_regressor", return_value=RecordingCatBoost)
        self.fake.start()
        self.addCleanup(self.fake.stop)

    def test_targets_and_score_stay_on_the_declared_scale(self):
        for mode, expected_loss in (("price_mae", "MAE"), ("log_rmse", "RMSE")):
            with self.subTest(mode=mode):
                model = CatBoostPriceRegressor(target_mode=mode, iterations=11).fit(self.frame, self.target)
                expected = self.target if mode == "price_mae" else np.log1p(self.target)
                np.testing.assert_array_equal(model.model_.target_, expected)
                self.assertEqual(model.model_.params["loss_function"], expected_loss)
                self.assertEqual(model.model_.params["iterations"], 11)
                self.assertFalse(model.model_.params["use_best_model"])
                self.assertFalse(model.model_.params["allow_writing_files"])
                self.assertNotIn("early_stopping_rounds", model.model_.params)
                expected_prediction = float(expected.mean())
                if mode == "log_rmse":
                    expected_prediction = np.expm1(expected_prediction)
                np.testing.assert_allclose(model.predict(self.frame), expected_prediction, rtol=0, atol=0)
                self.assertEqual(model.score(self.frame, self.target),
                                 -mean_absolute_error(self.target, model.predict(self.frame)))
                # Changing a constructor parameter alone cannot silently change
                # the inverse transform of an already fitted model.
                model.set_params(target_mode="log_rmse" if mode == "price_mae" else "price_mae")
                np.testing.assert_allclose(model.predict(self.frame), expected_prediction, rtol=0, atol=0)

    def test_native_categories_missing_numeric_and_schema_exclusion(self):
        frame = self.frame.copy()
        frame["district"] = [None, np.nan, pd.NA, "   "]
        frame["condition"] = ["новое", None, "хорошее", "новое"]
        frame["year_built"] = np.nan
        frame["residential_complex"] = [None, "Новый ЖК", pd.NA, "Пример"]
        frame["price"] = self.target
        frame["latitude"], frame["longitude"] = 0., 0.
        before = frame.copy(deep=True)
        model = CatBoostPriceRegressor().fit(frame, self.target)
        native = model.model_.frame_
        self.assertEqual(tuple(native.columns), CATBOOST_MODEL_COLUMNS)
        self.assertEqual(model.model_.params["cat_features"], list(CATBOOST_CATEGORICAL_COLUMNS))
        self.assertTrue(native.district.eq(CATBOOST_MISSING_CATEGORY).all())
        self.assertTrue(native.year_built.isna().all())
        self.assertNotIn("latitude", native)  # no raw or full-catalog coordinates reach the default model
        self.assertNotIn("longitude", native)
        self.assertTrue(native.residential_complex.map(lambda value: isinstance(value, str)).all())
        self.assertEqual(tuple(model.feature_names_in_), LOCATION_INPUT_COLUMNS)
        pd.testing.assert_frame_equal(frame, before)

    def test_injected_geo_is_cloned_sees_no_y_or_extra_columns_and_used_during_fit(self):
        transformer = TargetGuardTransformer(latitude=51.19)
        frame = self.frame.assign(price=self.target, unrelated_target_proxy=123)
        model = CatBoostPriceRegressor(location_transformer=transformer).fit(frame, self.target)
        self.assertFalse(hasattr(transformer, "train_indices_"))
        self.assertIsNot(model.location_transformer_, transformer)
        self.assertEqual(model.location_transformer_.train_indices_, self.frame.index.tolist())
        self.assertTrue(model.model_.frame_.latitude.eq(51.19).all())
        self.assertTrue(model.transform_features(self.frame).latitude.eq(51.19).all())

    def test_heldout_points_traits_and_other_rows_do_not_change_fitted_maps(self):
        frame = pd.DataFrame([
            row(raw_latitude=51.12, raw_longitude=71.42, raw_coordinate_status="unverified"),
            row(raw_latitude=51.12, raw_longitude=71.42, raw_coordinate_status="unverified"),
            row("astana|есиль|тестовая 2", raw_latitude=51.2, raw_longitude=71.5, raw_coordinate_status="unverified"),
        ])
        model = CatBoostPriceRegressor(location_transformer=LocationFeatureTransformer("complex_coords")).fit(
            frame, np.array([30e6, 40e6, 50e6]))
        before = copy.deepcopy(model.location_transformer_.house_coordinate_map_)
        heldout = pd.DataFrame([row(), row(year_built=2020),
                               row("astana|есиль|тестовая 2", raw_latitude=51.2, raw_longitude=71.5,
                                   raw_coordinate_status="unverified"), row(None)])
        native = model.transform_features(heldout)
        self.assertEqual(native.latitude.iloc[0], 51.12)
        self.assertTrue(native.latitude.iloc[1:].isna().all())
        changed = heldout.assign(raw_latitude=0., raw_longitude=0., raw_coordinate_status="invalid")
        pd.testing.assert_frame_equal(model.transform_features(changed), native)
        pd.testing.assert_frame_equal(model.transform_features(heldout.iloc[[0]]), native.iloc[[0]])
        self.assertEqual(model.location_transformer_.house_coordinate_map_, before)
        model.fit(pd.DataFrame([row("astana|есиль|тестовая 99")]), [40e6])
        self.assertEqual(set(model.location_transformer_.house_traits_), {"astana|есиль|тестовая 99"})

    def test_default_mode_ignores_raw_coordinates_house_keys_and_catalog_geography(self):
        model = CatBoostPriceRegressor().fit(self.frame, self.target)
        self.assertEqual(model.location_transformer_.variant, "complex")
        self.assertEqual(model.location_transformer_.house_coordinate_map_, {})
        before = model.transform_features(self.frame)
        changed = self.frame.assign(house_key="astana|есиль|другой дом 999", raw_latitude=51.1,
                                    raw_longitude=71.4, raw_coordinate_status="unverified",
                                    latitude=51.3, longitude=71.7, coordinate_status="plausible")
        changed.loc[changed.index[1], "house_key"] = None
        pd.testing.assert_frame_equal(model.transform_features(changed), before)
        second = CatBoostPriceRegressor().fit(changed, self.target)
        pd.testing.assert_frame_equal(second.model_.frame_, model.model_.frame_)
        self.assertEqual(tuple(model.model_feature_names_), CATBOOST_MODEL_COLUMNS)
        self.assertEqual(len(model.model_feature_names_), 9)

    def test_unknown_categories_serialization_clone_and_empty_prediction(self):
        model = CatBoostPriceRegressor("log_rmse", TargetGuardTransformer(), iterations=4).fit(self.frame, self.target)
        copied = clone(model)
        self.assertTrue(is_regressor(copied))
        self.assertEqual(copied.get_params()["target_mode"], "log_rmse")
        self.assertIsNot(copied.location_transformer, model.location_transformer)
        with self.assertRaises(NotFittedError):
            copied.predict(self.frame)
        query = self.frame.assign(residential_complex="Никогда не виденный ЖК", district="Новый район")
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_array_equal(model.predict(query), restored.predict(query))
        self.assertEqual(restored.predict(query.iloc[:0]).shape, (0,))

    def test_alignment_invalid_targets_and_forbidden_validation_arguments(self):
        for target in (self.target.iloc[::-1], [1., 2.], [1., 2., np.nan, 4.], [1., 2., 0., 4.], [True] * 4):
            with self.subTest(target=repr(target)), self.assertRaises(ValueError):
                CatBoostPriceRegressor().fit(self.frame, target)
        with self.assertRaisesRegex(ValueError, "сохранять индекс"):
            CatBoostPriceRegressor(location_transformer=ReorderingTransformer()).fit(self.frame, self.target)
        with self.assertRaises(TypeError):
            CatBoostPriceRegressor().fit(self.frame, self.target, eval_set=(self.frame, self.target))
        with self.assertRaises(ValueError):
            CatBoostPriceRegressor().fit(self.frame.assign(area=np.inf), self.target)
        for params in ({"target_mode": "anything"}, {"iterations": 0}, {"random_seed": None}, {"thread_count": -1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                CatBoostPriceRegressor(**params).fit(self.frame, self.target)

    @unittest.skipUnless(importlib.util.find_spec("catboost"), "CatBoost is not installed")
    def test_real_tiny_catboost_both_objectives_and_joblib(self):
        self.fake.stop()
        frame = pd.DataFrame([row(area=40. + i, district=None if i % 3 == 0 else "Есиль",
                                  residential_complex=None if i % 2 == 0 else "Пример") for i in range(20)])
        target = frame.area * 400_000
        for mode in ("price_mae", "log_rmse"):
            with self.subTest(mode=mode):
                model = CatBoostPriceRegressor(mode, iterations=4, depth=2, thread_count=1).fit(frame, target)
                self.assertEqual(model.model_.tree_count_, 4)
                query = frame.iloc[:3].assign(residential_complex="Новый ЖК")
                predicted = model.predict(query)
                self.assertTrue(np.isfinite(predicted).all())
                self.assertTrue((predicted > 1e6).all())
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                buffer.seek(0)
                np.testing.assert_array_equal(predicted, joblib.load(buffer).predict(query))


if __name__ == "__main__":
    unittest.main()

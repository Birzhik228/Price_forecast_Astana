"""Visible CatBoost activation behaviour, without loading real model artifacts."""

from __future__ import annotations

from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from catboost_model import CATBOOST_MODEL_COLUMNS, CatBoostPriceRegressor
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from test_app import apartment_fixture
from test_catboost_prediction import MAE, catboost_metadata_fixture, synthetic_catboost
from test_property_app import catalog_fixture


class CatBoostApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.model = synthetic_catboost()
        self.predict = Mock(wraps=self.model.predict)
        self.model.predict = self.predict
        self.metadata = catboost_metadata_fixture()
        self.stack.enter_context(patch("app.file_signature", return_value=(1, 1)))
        self.stack.enter_context(patch("app.cached_artifacts", return_value=(self.model, self.metadata)))
        self.loader = self.stack.enter_context(patch("app.cached_catalog", return_value=catalog_fixture()))
        self.stack.enter_context(patch.object(CatBoostPriceRegressor, "fit", side_effect=AssertionError("App must not train")))
        self.app = AppTest.from_string("from app import main\nmain()", default_timeout=30).run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)

    def submit(self) -> None:
        values = apartment_fixture()
        for feature in NUMERIC_FEATURES:
            self.app.number_input(key=feature).set_value(values[feature])
        for feature in CATEGORICAL_FEATURES:
            self.app.selectbox(key=feature).set_value(values[feature])
        next(button for button in self.app.button if button.label == "Рассчитать стоимость").click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)

    def visible_text(self) -> str:
        return " ".join(value.value for collection in (
            self.app.caption, self.app.markdown, self.app.info, self.app.warning,
        ) for value in collection)

    def test_complex_selection_autofills_district_and_predicts_with_nine_native_fields(self) -> None:
        self.app.selectbox(key="location_complex").set_value("complex-a").run()
        self.assertEqual(self.app.selectbox(key="district").value, "Есиль")
        self.app.selectbox(key="location_building").set_value("house-a").run()
        self.submit()
        frame = self.predict.call_args.args[0]
        self.assertEqual(frame.residential_complex.iloc[0], "тестовый жк")
        self.assertEqual(list(self.model.transform_features(frame).columns), list(CATBOOST_MODEL_COLUMNS))
        self.assertEqual(self.predict.call_count, 1)
        self.assertEqual(len(self.app.metric), 2)
        self.assertTrue(any("Похожие объявления" in item.value for item in self.app.subheader))

    def test_model_identity_range_and_geography_explanation_are_correct(self) -> None:
        self.submit()
        visible = self.visible_text()
        self.assertIn("CatBoost", visible)
        self.assertIn("MAE", visible)
        self.assertIn("5 580 000", visible)
        self.assertIn(self.metadata["data_period_label"], visible)
        result = self.app.session_state["prediction"]
        self.assertEqual(result.upper_tenge - result.price_tenge, MAE)
        self.assertIn("не доверительный интервал", visible)
        self.assertNotIn("Модель: ЖК и координаты", visible)
        self.assertNotIn("По адресу модель учитывает координаты", visible)
        self.assertNotIn("Учитывает ЖК и доступные по адресу координаты", visible)
        self.assertNotIn("Модель цены использует восемь параметров ниже", visible)

    def test_no_catalog_accepts_manual_complex_without_losing_prediction(self) -> None:
        self.loader.side_effect = FileNotFoundError("Synthetic catalogue unavailable")
        self.app.run()
        self.app.text_input(key="location_manual_complex").set_value("Тестовый ЖК").run()
        self.submit()
        frame = self.predict.call_args.args[0]
        self.assertEqual(frame.residential_complex.iloc[0], "тестовый жк")
        self.assertTrue(frame[["raw_latitude", "raw_longitude"]].isna().all().all())
        self.assertEqual(len(self.app.metric), 2)
        self.assertFalse(any("Похожие объявления" in item.value for item in self.app.subheader))

    def test_optional_and_unknown_complex_remain_usable(self) -> None:
        self.submit()
        self.assertTrue(pd.isna(self.predict.call_args.args[0].residential_complex.iloc[0]))
        self.app.text_input(key="location_manual_complex").set_value("Новый ЖК").run()
        self.assertEqual(len(self.app.metric), 0)
        self.submit()
        self.assertTrue(any("не встречался" in item.value for item in self.app.warning))
        self.assertEqual(len(self.app.metric), 2)

    def test_selected_house_district_conflict_stops_prediction(self) -> None:
        self.app.selectbox(key="location_building").set_value("house-a").run()
        self.app.number_input(key="area").set_value(60.)
        self.app.number_input(key="rooms").set_value(2)
        self.app.selectbox(key="district").set_value("Нура")
        next(button for button in self.app.button if button.label == "Рассчитать стоимость").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Район" in item.value for item in self.app.error))
        self.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()

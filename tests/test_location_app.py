"""Streamlit scenarios for schema 2 using a tiny synthetic pipeline only."""

from __future__ import annotations

from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from building_catalog import canonical_house_key
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from location_model import LOCATION_INPUT_COLUMNS, make_location_pipeline
from test_app import apartment_fixture, metadata_fixture
from test_property_app import catalog_fixture


MAE = 6_442_701.30963452


def location_metadata_fixture() -> dict:
    metadata = metadata_fixture()
    metadata.update({
        "schema_version": 2,
        "variant": "complex_coords",
        "feature_columns": list(LOCATION_INPUT_COLUMNS),
        "model_label": "ЖК и координаты",
        "data_period_label": "января и конца марта 2026 года (март — по описанию источника)",
        "geo_training_only": True,
        "selected_by_cv": False,
        "cv_selected_variant": "complex",
        "activation_reason": "explicit_user_request",
        "test_metrics": {"mae_tenge": MAE, "rmse_tenge": 12_123_338.426170168, "r2": .8688594694564467},
        "approximate_range": {"half_width_tenge": MAE, "source": "reused_march_holdout", "guaranteed_coverage": None},
    })
    return metadata


def synthetic_pipeline() -> Pipeline:
    """Train an inexpensive test estimator, never loading or fitting real artifacts."""
    rows = []
    for number in range(12):
        house = number % 2 + 1
        rows.append(apartment_fixture() | {
            "area": 50.0 + number,
            "residential_complex": "тестовый жк",
            "house_key": canonical_house_key(f"Тестовая {house}", "Есиль", "Тестовый ЖК"),
            "raw_latitude": 51.12 + .01 * (house - 1),
            "raw_longitude": 71.43 + .01 * (house - 1),
            "raw_coordinate_status": "unverified",
        })
    return make_location_pipeline(
        DummyRegressor(strategy="constant", constant=42_000_000.0), "complex_coords",
    ).fit(pd.DataFrame(rows, columns=list(LOCATION_INPUT_COLUMNS)), np.full(len(rows), 42_000_000.0))


class LocationApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.pipeline = synthetic_pipeline()
        self.predict = Mock(wraps=self.pipeline.predict)
        self.pipeline.predict = self.predict
        self.metadata = location_metadata_fixture()
        self.catalog = catalog_fixture()
        # Catalogue coordinates must never become model inputs: the model owns
        # its independent map learned from synthetic training observations.
        self.catalog.buildings.loc[:, "latitude"] = -70.0
        self.catalog.buildings.loc[:, "longitude"] = -130.0
        self.catalog.buildings.loc[:, "coordinate_status"] = "plausible"
        self.stack.enter_context(patch("app.file_signature", return_value=(1, 1)))
        self.stack.enter_context(patch("app.cached_artifacts", return_value=(self.pipeline, self.metadata)))
        self.loader = self.stack.enter_context(patch("app.cached_catalog", return_value=self.catalog))
        self.stack.enter_context(patch.object(Pipeline, "fit", side_effect=AssertionError("App must never fit")))
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

    def prediction_input(self) -> pd.DataFrame:
        frame = self.predict.call_args.args[0]
        self.assertEqual(frame.shape, (1, 13))
        self.assertEqual(list(frame.columns), list(LOCATION_INPUT_COLUMNS))
        self.assertTrue(frame[["raw_latitude", "raw_longitude"]].isna().all().all())
        self.assertEqual(frame.raw_coordinate_status.iloc[0], "missing")
        return frame

    def test_selected_complex_and_house_reach_thirteen_column_model(self) -> None:
        self.app.selectbox(key="location_complex").set_value("complex-a").run()
        self.app.selectbox(key="location_building").set_value("house-a").run()
        self.submit()
        frame = self.prediction_input()
        self.assertEqual(frame.residential_complex.iloc[0], "тестовый жк")
        expected = canonical_house_key("Тестовая 1", "Есиль", "Тестовый ЖК")
        self.assertEqual(frame.house_key.iloc[0], expected)
        self.assertIn(expected, self.pipeline.named_steps["location_features"].house_coordinate_map_)
        self.assertEqual(self.predict.call_count, 1)
        self.assertEqual(next(value.value for value in self.app.metric if value.label == "Прогноз цены"), "42 000 000 ₸")

    def test_no_location_still_predicts_with_unknown_location_features(self) -> None:
        self.submit()
        frame = self.prediction_input()
        self.assertTrue(pd.isna(frame.residential_complex.iloc[0]))
        self.assertTrue(pd.isna(frame.house_key.iloc[0]))
        self.assertEqual(len(self.app.metric), 2)

    def test_changing_selected_house_clears_price_and_archive_cards_without_predicting(self) -> None:
        self.app.selectbox(key="location_building").set_value("house-a").run()
        self.submit()
        self.assertEqual(len(self.app.metric), 2)
        self.app.selectbox(key="location_building").set_value("house-b").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.metric), 0)
        self.assertFalse(any("Похожие объявления" in value.value for value in self.app.subheader))
        self.assertEqual(self.predict.call_count, 1)

    def test_missing_catalog_accepts_manual_optional_complex_and_address(self) -> None:
        self.loader.side_effect = FileNotFoundError("Synthetic catalogue unavailable")
        self.app.run()
        self.app.text_input(key="location_manual_complex").set_value("Тестовый ЖК").run()
        self.app.text_input(key="location_manual_address").set_value("Тестовая 1").run()
        self.submit()
        frame = self.prediction_input()
        self.assertEqual(frame.residential_complex.iloc[0], "тестовый жк")
        self.assertEqual(frame.house_key.iloc[0], canonical_house_key("Тестовая 1", "Есиль", "Тестовый ЖК"))
        self.assertEqual(len(self.app.metric), 2)
        self.assertFalse(any("Похожие объявления" in value.value for value in self.app.subheader))

    def test_clearing_manual_address_removes_old_price_before_resubmission(self) -> None:
        self.loader.side_effect = FileNotFoundError("Synthetic catalogue unavailable")
        self.app.run()
        self.app.text_input(key="location_manual_complex").set_value("Тестовый ЖК").run()
        self.app.text_input(key="location_manual_address").set_value("Тестовая 1").run()
        self.submit()
        self.app.text_input(key="location_manual_address").set_value("").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.metric), 0)
        self.assertEqual(self.predict.call_count, 1)
        self.submit()
        frame = self.prediction_input()
        self.assertTrue(pd.isna(frame.house_key.iloc[0]))
        self.assertEqual(frame.residential_complex.iloc[0], "тестовый жк")

    def test_model_label_and_error_range_use_schema_two_metadata(self) -> None:
        self.submit()
        visible = " ".join(value.value for collection in (
            self.app.caption, self.app.markdown, self.app.info,
        ) for value in collection)
        self.assertIn("ЖК и координаты", visible)
        self.assertIn(self.metadata["data_period_label"], visible)
        self.assertIn("6 442 701", visible)
        self.assertIn("35 557 299", visible)
        self.assertIn("48 442 701", visible)
        self.assertNotIn("Модель цены пока использует восемь", visible)
        self.assertNotIn("точный адрес, вид из окна и другие детали в прогнозе не учитываются", visible)
        self.prediction_input()


if __name__ == "__main__":
    unittest.main()

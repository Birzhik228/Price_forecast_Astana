"""Пользовательские сценарии архивных аналогов и согласованности формы."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from test_app import metadata_fixture


def catalog_fixture():
    rows = []
    for index, building in enumerate(("house-a", "house-b")):
        rows.append({
            "record_id": f"record-{index}", "group": f"group-{index}",
            "building_id": building, "complex_id": "complex-a", "complex_name": "Тестовый ЖК",
            "address_normalized": f"Тестовая {index + 1}", "address": f"Тестовая {index + 1}",
            "district": "Есиль", "area": 60.0, "rooms": 2, "floor": 5.0,
            "total_floors": 9.0, "year_built": 2015.0, "building_type": "монолитный",
            "condition": "свежий ремонт", "price": 40_000_000 + index * 1_000_000,
            "latitude": np.nan, "longitude": np.nan, "coordinate_status": "missing",
            "parking_type": np.nan, "date_label": "Март 2026: период источника; точная дата неизвестна",
            "url": f"https://krisha.kz/a/show/{1000000000 + index}", "dataset_period": "2026-03",
        })
    listings = pd.DataFrame(rows)
    buildings = listings.copy()
    buildings["listing_count"] = 1
    buildings["display_label"] = buildings["address_normalized"]
    return SimpleNamespace(listings=listings, buildings=buildings, manifest={"schema_version": 1})


class PropertyApplicationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.pipeline = Mock()
        self.pipeline.predict.return_value = np.array([42_000_000.0])
        self.stack.enter_context(patch("app.file_signature", return_value=(1, 1)))
        self.stack.enter_context(patch("app.cached_artifacts", return_value=(self.pipeline, metadata_fixture())))
        self.loader = self.stack.enter_context(patch("app.cached_catalog", return_value=catalog_fixture()))
        self.app = AppTest.from_string("from app import main\nmain()", default_timeout=30).run()

    def submit(self):
        self.app.number_input(key="area").set_value(60.0)
        self.app.number_input(key="rooms").set_value(2)
        self.app.selectbox(key="district").set_value("Есиль")
        next(button for button in self.app.button if button.label == "Рассчитать стоимость").click().run()

    def test_archive_cards_use_snapshot_and_do_not_change_prediction(self):
        self.submit()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)
        self.assertEqual(self.pipeline.predict.call_count, 1)
        self.assertTrue(any("Похожие объявления" in item.value for item in self.app.subheader))
        self.assertTrue(any("точная дата неизвестна" in item.value for item in self.app.caption))
        self.assertEqual(self.app.session_state["last_input"]["area"], 60.0)
        self.app.number_input(key="area").set_value(100.0).run()
        self.assertEqual(self.app.session_state["last_input"]["area"], 60.0)
        self.assertEqual(self.pipeline.predict.call_count, 1)

    def test_location_change_removes_old_result_and_cards(self):
        self.submit()
        self.app.selectbox(key="location_building").set_value("house-a").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.metric), 0)
        self.assertFalse(any("Похожие объявления" in item.value for item in self.app.subheader))

    def test_invalid_submission_removes_old_cards(self):
        self.submit()
        self.app.number_input(key="area").set_value(None)
        next(button for button in self.app.button if button.label == "Рассчитать стоимость").click().run()
        self.assertTrue(self.app.error)
        self.assertEqual(len(self.app.metric), 0)
        self.assertFalse(any("Похожие объявления" in item.value for item in self.app.subheader))

    def test_autofill_uses_only_known_house_values(self):
        self.app.selectbox(key="location_building").set_value("house-a").run()
        self.app.button(key="fill_building").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.selectbox(key="district").value, "Есиль")
        self.assertEqual(self.app.number_input(key="year_built").value, 2015)
        self.assertIsNone(self.app.number_input(key="floor").value)
        self.assertIsNone(self.app.number_input(key="area").value)

    def test_district_conflict_requires_consistent_input(self):
        self.app.selectbox(key="location_building").set_value("house-a").run()
        self.app.number_input(key="area").set_value(60.0)
        self.app.number_input(key="rooms").set_value(2)
        self.app.selectbox(key="district").set_value("Нура")
        next(button for button in self.app.button if button.label == "Рассчитать стоимость").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Район" in item.value for item in self.app.error))
        self.pipeline.predict.assert_not_called()

    def test_unavailable_catalog_preserves_model_prediction(self):
        self.loader.side_effect = ValueError("bad checksum")
        self.app.run()
        self.submit()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.metric), 2)
        self.assertTrue(any("Поиск аналогов временно недоступен" in item.value for item in self.app.warning))

    def test_own_listing_url_excludes_card_without_changing_model_inputs(self):
        self.app.text_input(key="location_exclude_url").set_value("https://krisha.kz/a/show/1000000000?source=test").run()
        self.submit()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)
        self.assertEqual(self.app.session_state["last_location"]["exclude_url"], "https://krisha.kz/a/show/1000000000")
        rendered = " ".join(item.value for item in self.app.markdown)
        self.assertNotIn("40 000 000", rendered)
        self.assertIn("41 000 000", rendered)
        self.assertEqual(self.pipeline.predict.call_args.args[0].shape[1], 8)

    def test_invalid_own_url_is_rejected_before_model_call(self):
        self.app.text_input(key="location_exclude_url").set_value("https://example.org/123").run()
        self.submit()
        self.assertTrue(any("Krisha" in item.value for item in self.app.error))
        self.pipeline.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()

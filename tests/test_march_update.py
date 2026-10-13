"""Автономные проверки групп и независимого мартовского holdout без обучения."""

from __future__ import annotations

from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from update_march import apartment_keys, create_split, run_update


def apartment(number: int, **changes) -> dict:
    row = {
        "source_row": number + 2,
        "url": f"https://krisha.kz/a/show/{100_000 + number}",
        "price": 25_000_000.0 + number * 100_000,
        "area": 50.0 + number / 10,
        "rooms": 2.0,
        "floor": 5.0,
        "total_floors": 12.0,
        "year_built": 2018.0,
        "district": "Есиль",
        "building_type": "монолитный",
        "condition": "свежий ремонт",
        "address": f"Улица Тестовая {number}",
        "residential_complex": f"Тестовый ЖК {number}",
    }
    return row | changes


def novel_frame(extra_rows: list[dict] | None = None) -> pd.DataFrame:
    return pd.DataFrame([*[apartment(i) for i in range(35)], *(extra_rows or [])])


class ApartmentIdentityTests(unittest.TestCase):
    def test_price_and_condition_are_not_part_of_apartment_identity(self) -> None:
        first = apartment(1)
        changed = first | {"price": 999_000_000.0, "condition": "черновая отделка"}
        self.assertEqual(apartment_keys(first), apartment_keys(changed))
        without_price = {key: value for key, value in first.items() if key != "price"}
        self.assertEqual(apartment_keys(first), apartment_keys(without_price))

    def test_incomplete_signatures_keep_only_url_identity(self) -> None:
        for column in ("area", "rooms", "floor", "total_floors", "year_built", "district"):
            for missing in (None, np.nan, pd.NA):
                with self.subTest(column=column, missing=str(missing)):
                    row = apartment(1, **{column: missing})
                    self.assertEqual(apartment_keys(row), {"url:https://krisha.kz/a/show/100001"})
        self.assertEqual(
            apartment_keys(apartment(1, address=None, residential_complex=None)),
            {"url:https://krisha.kz/a/show/100001"},
        )


class MarchSplitTests(unittest.TestCase):
    def test_relisted_apartment_with_new_url_and_price_stays_in_one_subset(self) -> None:
        relisted = apartment(2, url="https://krisha.kz/a/show/900002", price=31_000_000.0)
        march = novel_frame([relisted])
        train, test, groups, audit, _ = create_split(march, pd.DataFrame())
        self.assertEqual(groups.loc[2], groups.loc[35])
        self.assertEqual(audit.loc[2, "subset"], audit.loc[35, "subset"])
        self.assertFalse(set(groups.loc[train]) & set(groups.loc[test]))
        self.assertEqual(set(train) | set(test), set(march.index))
        self.assertFalse(set(train) & set(test))

    def test_january_url_and_signature_matches_exclude_their_entire_groups(self) -> None:
        january_url = apartment(800)
        january_signature = apartment(900)
        url_match = apartment(100, url=january_url["url"] + "?utm_source=test")
        url_group_relisting = apartment(100, url="https://krisha.kz/a/show/990001", price=33_000_000.0)
        signature_match = apartment(900, url="https://krisha.kz/a/show/990002", price=44_000_000.0)
        # Эта строка связана с известной группой URL, но её собственная сигнатура новая.
        signature_group_relisting = apartment(901, url=signature_match["url"])
        march = novel_frame([url_match, url_group_relisting, signature_match, signature_group_relisting])
        train, test, groups, audit, summary = create_split(
            march, pd.DataFrame([january_url, january_signature]),
        )
        self.assertTrue(audit.loc[35, "january_url_match"])
        self.assertFalse(audit.loc[36, ["january_url_match", "january_signature_match"]].any())
        self.assertTrue(audit.loc[37, "january_signature_match"])
        self.assertFalse(audit.loc[38, ["january_url_match", "january_signature_match"]].any())
        self.assertEqual(groups.loc[35], groups.loc[36])
        self.assertEqual(groups.loc[37], groups.loc[38])
        self.assertTrue({35, 36, 37, 38}.issubset(set(train)))
        self.assertTrue(set(test).isdisjoint({35, 36, 37, 38}))
        self.assertFalse(audit.loc[[35, 36, 37, 38], "novel_group"].any())
        self.assertTrue(audit.loc[test, "novel_group"].all())
        self.assertEqual(summary["known_group_rows"], 4)
        self.assertEqual(summary["novel_rows"], 35)

    def test_transitive_links_cannot_split_a_group(self) -> None:
        first = apartment(100)
        second = apartment(100, url="https://krisha.kz/a/show/990003")
        third = apartment(101, url=second["url"])
        march = novel_frame([first, second, third])
        train, test, groups, audit, _ = create_split(march, pd.DataFrame())
        self.assertEqual(groups.loc[[35, 36, 37]].nunique(), 1)
        self.assertEqual(audit.loc[[35, 36, 37], "subset"].nunique(), 1)
        self.assertFalse(set(groups.loc[train]) & set(groups.loc[test]))

    def test_incomplete_ambiguous_relistings_are_not_merged_by_signature(self) -> None:
        first = apartment(100, floor=np.nan)
        second = first | {"url": "https://krisha.kz/a/show/990004", "price": 30_000_000.0}
        march = novel_frame([first, second])
        _, _, groups, audit, _ = create_split(march, pd.DataFrame())
        self.assertNotEqual(groups.loc[35], groups.loc[36])
        self.assertFalse(audit.loc[[35, 36], "signature_available"].any())

    def test_membership_is_deterministic_and_unchanged_when_targets_change(self) -> None:
        march = novel_frame([apartment(2, url="https://krisha.kz/a/show/990005")])
        january = pd.DataFrame([apartment(800)])
        expected = create_split(march, january, random_state=42)
        changed = march.copy()
        changed["price"] = np.linspace(999_000_000, 5_000_000, len(changed))
        changed_january = january.assign(price=1.0)
        for actual in (
            create_split(march, january, random_state=42),
            create_split(changed, changed_january, random_state=42),
        ):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            pd.testing.assert_series_equal(actual[2], expected[2])
            pd.testing.assert_frame_equal(actual[3], expected[3])


class MarchUpdateOrchestrationTests(unittest.TestCase):
    def test_common_holdout_metrics_and_training_handoff_without_training(self) -> None:
        march = novel_frame()
        january = pd.DataFrame([apartment(800)])
        old_pipeline = Mock()
        old_pipeline.predict.side_effect = lambda frame: np.full(len(frame), 24_000_000.0)
        new_pipeline = Mock()
        new_pipeline.predict.side_effect = lambda frame: np.full(len(frame), 26_000_000.0)
        old_metadata = {"test_metrics": {"mae_tenge": 4_000_000.0}, "model_sha256": "old"}
        new_metadata = {
            "test_metrics": {"mae_tenge": 2_000_000.0}, "model_sha256": "new", "input_sha256": "data",
        }
        with ExitStack() as stack:
            stack.enter_context(patch("update_march.preserve_january"))
            stack.enter_context(patch("march_data.prepare_march_data", autospec=True, return_value={}))
            stack.enter_context(patch("update_march.pd.read_csv", side_effect=[march, january]))
            stack.enter_context(patch("pandas.DataFrame.to_csv"))
            stack.enter_context(patch("update_march.digest", return_value="data"))
            stack.enter_context(patch("update_march.save_json"))
            stack.enter_context(patch("update_march.render_comparison"))
            stack.enter_context(patch("builtins.print"))
            stack.enter_context(patch("update_march.load_artifacts", side_effect=[
                (old_pipeline, old_metadata), (new_pipeline, new_metadata),
            ]))
            trainer = stack.enter_context(patch(
                "march_training.train_march", autospec=True, return_value={"model_name": "Fixture model"},
            ))
            activate = stack.enter_context(patch("update_march.activate_march"))
            report = run_update(activate=False)

        activate.assert_not_called()
        trainer.assert_called_once()
        args = trainer.call_args.args
        pd.testing.assert_frame_equal(args[0], march)
        self.assertEqual(set(args[1]) | set(args[2]), set(march.index))
        self.assertFalse(set(args[3].loc[args[1]]) & set(args[3].loc[args[2]]))
        self.assertIs(args[7], old_pipeline)
        self.assertIs(args[8], old_metadata)
        self.assertEqual(report["selected_model"], "Fixture model")
        self.assertFalse(report["activated"])
        self.assertEqual(len(report["common_holdout_comparison"]), 2)
        for metrics in report["common_holdout_comparison"]:
            for name in ("mae_tenge", "rmse_tenge", "r2", "bias_tenge", "over_20_pct"):
                self.assertTrue(np.isfinite(metrics[name]))
        pd.testing.assert_frame_equal(
            old_pipeline.predict.call_args.args[0], new_pipeline.predict.call_args.args[0],
        )
        self.assertEqual(list(new_pipeline.predict.call_args.args[0].columns), list(FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()

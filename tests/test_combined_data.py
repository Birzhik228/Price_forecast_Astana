"""Дедупликация обеих выборок и сохранение исходного мартовского holdout."""

from __future__ import annotations

import unittest

import pandas as pd

from combined_data import build_combined


def listing(source_row: int, **changes: object) -> dict:
    row = {
        "source_row": source_row, "price": 35_000_000, "area": 60.0, "rooms": 2,
        "floor": 3, "total_floors": 9, "year_built": 2015,
        "district": "Есиль", "building_type": "кирпичный", "condition": "свежий ремонт",
        "address": f"Улица {source_row}", "residential_complex": None,
        "url": f"https://krisha.kz/a/show/{source_row}", "collected_at": None,
    }
    row.update(changes)
    return row


def manifest_for(march: pd.DataFrame, test_rows: list[int], groups: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame({
        "row_index": march.index, "source_row": march["source_row"].to_numpy(),
        "subset": ["test" if index in test_rows else "train" for index in march.index],
        "group": groups or [f"old_{index}" for index in march.index],
    })


class CombinedDataTests(unittest.TestCase):
    def build(self, january: list[dict], march: list[dict], test_rows: list[int], groups=None) -> dict:
        jan, mar = pd.DataFrame(january), pd.DataFrame(march)
        return build_combined(jan, mar, manifest_for(mar, test_rows, groups))

    def test_transitivity_runs_over_both_sources(self) -> None:
        # A↔B по URL; B↔C по адресу. A и C не имеют прямого ключа.
        result = self.build(
            [listing(1, address="Первый адрес"), listing(2, url="https://krisha.kz/a/show/1", address="Второй адрес")],
            [listing(101, address="Второй адрес"), listing(102, address="Holdout", area=80)], [1],
        )
        self.assertEqual(len(result["frame"]), 2)
        self.assertEqual(result["audit"].iloc[:3]["group"].nunique(), 1)
        self.assertEqual(result["audit"].iloc[0]["match_type"], "transitive")
        self.assertEqual(result["summary"]["cross_period_groups"], 1)

    def test_march_priority_precedes_completeness_and_ignores_price(self) -> None:
        result = self.build(
            [listing(1, price=900_000_000)],
            [listing(101, url="https://krisha.kz/a/show/1", condition=None, price=25_000_000),
             listing(102, address="Holdout", area=80)], [1],
        )
        kept = result["frame"].loc[result["frame"]["original_source_row"].eq(101)].iloc[0]
        self.assertEqual(kept["price"], 25_000_000)
        self.assertTrue(pd.isna(kept["condition"]))
        self.assertEqual(result["rejected"].iloc[0]["reason"], "duplicate_older_period")

    def test_most_complete_then_original_source_row_selects_within_period(self) -> None:
        result = self.build(
            [listing(1, condition=None), listing(3, address="Улица 1"), listing(2, address="Улица 1", price=5_000_000)],
            [listing(101, address="Holdout", area=80)], [0],
        )
        train = result["frame"].loc[result["manifest"]["subset"].eq("train")]
        self.assertEqual(train["original_source_row"].tolist(), [2])
        self.assertEqual(set(result["rejected"]["reason"]), {"duplicate_less_complete", "duplicate_later_source_row"})

    def test_missing_keys_do_not_merge(self) -> None:
        result = self.build(
            [listing(1, url=None, floor=None), listing(2, url=None, floor=None, address="Улица 1")],
            [listing(101, address="Holdout", area=80)], [0],
        )
        self.assertEqual(len(result["frame"]), 3)
        self.assertEqual(result["summary"]["input_rows_without_match_keys"], 2)

    def test_protected_group_uses_only_original_test_even_if_train_more_complete(self) -> None:
        result = self.build(
            [listing(1)],
            [listing(101, address="New property", area=80),
             listing(102, address="New property", area=80, condition=None)], [1],
        )
        test = result["frame"].loc[result["manifest"]["subset"].eq("test")]
        self.assertEqual(test["original_source_row"].tolist(), [102])
        self.assertEqual(result["rejected"].iloc[0]["original_source_row"], 101)
        self.assertEqual(result["rejected"].iloc[0]["reason"], "duplicate_protected_holdout")
        self.assertEqual(result["summary"]["protected_groups_with_original_march_train_rows"], 1)

    def test_reference_holdout_retains_all_old_observations(self) -> None:
        result = self.build(
            [listing(1)],
            [listing(101, address="New property", area=80),
             listing(102, address="New property", area=80, price=45_000_000)], [0, 1], ["old_test", "old_test"],
        )
        self.assertEqual(result["summary"]["test_rows"], 1)
        reference = result["reference_holdout"]
        self.assertEqual(reference["original_source_row"].tolist(), [101, 102])
        self.assertEqual(reference["price"].tolist(), [35_000_000, 45_000_000])
        self.assertEqual(reference["row_index"].tolist(), [0, 1])
        self.assertEqual(reference["group"].nunique(), 1)
        train_ids = set(result["manifest"].loc[result["manifest"]["subset"].eq("train"), "record_id"])
        self.assertTrue(train_ids.isdisjoint(reference["record_id"]))

    def test_any_january_link_to_protected_group_fails_transitively(self) -> None:
        with self.assertRaisesRegex(ValueError, "Загрязнение baseline"):
            self.build(
                [listing(1, address="Old address")],
                [listing(101, url="https://krisha.kz/a/show/1", address="Bridge", area=60),
                 listing(102, address="Bridge", area=60)], [1],
            )

    def test_provenance_unique_global_source_ids_and_determinism(self) -> None:
        jan = pd.DataFrame([listing(55)], index=[7])
        mar = pd.DataFrame([listing(55, url="https://krisha.kz/a/show/999", address="Holdout", area=80)], index=[11])
        manifest = manifest_for(mar, [11])
        before = [x.copy(deep=True) for x in (jan, mar, manifest)]
        first = build_combined(jan, mar, manifest)
        second = build_combined(jan, mar, manifest)
        for key in ("frame", "audit", "rejected", "manifest", "reference_holdout"):
            pd.testing.assert_frame_equal(first[key], second[key])
        self.assertEqual(first["summary"], second["summary"])
        result = first["frame"]
        self.assertTrue(result.index.equals(pd.RangeIndex(2)))
        self.assertEqual(result["source_row"].tolist(), [1, 2])
        self.assertEqual(result["original_source_row"].tolist(), [55, 55])
        self.assertEqual(result["original_row_index"].tolist(), [7, 11])
        self.assertEqual(result["record_id"].tolist(), ["2026-01:55", "2026-03:55"])
        for original, unchanged in zip((jan, mar, manifest), before):
            pd.testing.assert_frame_equal(original, unchanged)

    def test_manifest_source_mismatch_fails_before_combining(self) -> None:
        jan, mar = pd.DataFrame([listing(1)]), pd.DataFrame([listing(101, area=80)])
        manifest = manifest_for(mar, [0])
        manifest.loc[0, "source_row"] = 999
        with self.assertRaisesRegex(ValueError, "source_row"):
            build_combined(jan, mar, manifest)

    def test_signature_grouping_does_not_use_price(self) -> None:
        result = self.build(
            [listing(1, price=10_000_000), listing(2, address="Улица 1", price=100_000_000)],
            [listing(101, address="Holdout", area=80)], [0],
        )
        self.assertEqual(result["summary"]["rows_removed"], 1)
        self.assertEqual(result["rejected"].iloc[0]["match_type"], "signature")


if __name__ == "__main__":
    unittest.main()

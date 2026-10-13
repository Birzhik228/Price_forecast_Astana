"""Temporal/identity gates, transitive exclusions and unchanged source evidence."""

from __future__ import annotations

import unittest

import pandas as pd

from fresh_validation_data import prepare_fresh_holdout


def listing(number: int = 1, **changes) -> dict:
    result = {"record_id": f"observed:{number}", "url": f"https://krisha.kz/a/show/{number}",
              "price": 45_000_000, "area": 70.5, "rooms": 2, "floor": 4,
              "total_floors": 10, "year_built": 2020, "district": "Есиль",
              "building_type": "кирпичный", "condition": None,
              "address": f"Улица Свежая {number}", "residential_complex": f"ЖК {number}",
              "collected_at": "2026-10-12T12:00:00+05:00"}
    result.update(changes)
    return result


def references() -> dict[str, pd.DataFrame]:
    return {"prior_full_snapshot": pd.DataFrame([listing(900, collected_at="2026-01-04")])}


class FreshValidationDataTests(unittest.TestCase):
    def prepare(self, records, reference_frames=None, raw_reference_urls=(), **options):
        return prepare_fresh_holdout(pd.DataFrame(records), reference_frames if reference_frames is not None else references(),
                                     raw_reference_urls, as_of=options.pop("as_of", "2026-10-13"), **options)

    def test_primary_eligibility_and_external_date_proof_distinguished(self):
        result = self.prepare([listing()])
        row = result["frame"].iloc[0]
        self.assertTrue(row.primary_eligible)
        self.assertFalse(row.date_verified)
        self.assertTrue(result["summary"]["source_dates_verification_required"])
        self.assertEqual(row.collected_at, "2026-10-12T12:00:00+05:00")
        self.assertEqual(row.observed_date, "2026-10-12")

    def test_as_of_is_required_and_not_system_clock(self):
        with self.assertRaises(TypeError):
            prepare_fresh_holdout(pd.DataFrame([listing()]), references())
        for value in ["nonsense", "2026-03-01", "2026-10-13T12:00:00"]:
            with self.subTest(as_of=value), self.assertRaises(ValueError):
                self.prepare([listing()], as_of=value)

    def test_cutoff_entire_day_and_future_dates_rejected(self):
        records = [listing(1, collected_at="2026-03-31T23:59:59Z"),
                   listing(2, collected_at="2026-04-01"),
                   listing(3, collected_at="2026-10-14"), listing(4, collected_at=None),
                   listing(5, collected_at="2026-02-30")]
        result = self.prepare(records)
        self.assertEqual(result["frame"].record_id.tolist(), ["observed:2"])
        self.assertEqual(set(result["rejected"].reason), {"observation_not_after_cutoff", "observation_after_as_of",
                                                        "observed_date_missing", "observed_date_invalid"})

    def test_future_timestamp_compared_to_explicit_instant(self):
        result = self.prepare([listing(collected_at="2026-10-13T13:00:00Z")], as_of="2026-10-13T12:00:00Z")
        self.assertEqual(result["rejected"].iloc[0].reason, "observation_after_as_of")

    def test_no_fabricated_collection_from_publication_or_unselected_alias(self):
        raw = listing(collected_at=None, published_at="2026-10-12", parsed_at="2026-10-12")
        result = self.prepare([raw])
        self.assertTrue(result["frame"].empty)
        self.assertEqual(result["rejected"].iloc[0].input__parsed_at, "2026-10-12")
        selected = self.prepare([raw], source_observed_column="parsed_at")
        self.assertEqual(selected["frame"].iloc[0].collected_at, "2026-10-12")
        self.assertFalse(selected["frame"].iloc[0].date_verified)

    def test_rooms_gate_before_url_dedup_restores_later_valid_record(self):
        records = [listing(1, rooms="bad rooms"), listing(2, url="https://krisha.kz/a/show/1")]
        result = self.prepare(records)
        self.assertEqual(result["frame"].record_id.tolist(), ["observed:2"])
        self.assertEqual(result["rejected"].iloc[0].reason, "invalid_required_rooms")
        self.assertEqual(result["frame"].iloc[0].cleaning_reason, "duplicate_url")

    def test_exact_old_url_and_portal_id_variants_excluded_regardless_of_price(self):
        for url in ["http://www.krisha.kz/a/show/900/?utm_source=now", "https://krisha.kz/a/show/000900"]:
            with self.subTest(url=url):
                result = self.prepare([listing(1, url=url, price=900_000_000)])
                self.assertTrue(result["frame"].empty)
                self.assertEqual(result["rejected"].iloc[0].reason, "known_history_component")

    def test_raw_url_blocklist_includes_rows_omitted_from_clean_history(self):
        result = self.prepare([listing()], raw_reference_urls=["https://krisha.kz/a/show/1?x=previous"])
        self.assertTrue(result["frame"].empty)
        self.assertIn("raw_reference_urls", result["audit"].iloc[0].matched_reference_names)

    def test_normalized_house_alias_blocks_changed_url_without_using_price(self):
        old = {"history": pd.DataFrame([listing(90, address="Туран 44a", residential_complex=None)])}
        result = self.prepare([listing(1, address="Туран 44а", residential_complex=None, price=50_000_000)], old)
        self.assertTrue(result["frame"].empty)
        self.assertIn("canonical_apartment:", result["audit"].iloc[0].direct_reference_keys)

    def test_transitive_fresh_bridge_excludes_whole_group(self):
        records = [listing(1, address="Свежая 10", residential_complex="Один ЖК"),
                   listing(2, address="Свежая 10", residential_complex="Один ЖК")]
        result = self.prepare(records, raw_reference_urls=["https://krisha.kz/a/show/1"])
        self.assertTrue(result["frame"].empty)
        self.assertEqual(result["summary"]["history_connected_rows"], 2)
        self.assertTrue(result["audit"].iloc[1].transitive_history_match)

    def test_price_independent_fresh_duplicates_choose_earliest_source(self):
        first = listing(1, address="Улица 50", residential_complex="ЖК X", price=35_000_000)
        second = listing(2, address="Улица 50", residential_complex="ЖК X", price=85_000_000, condition="новый")
        result = self.prepare([first, second])
        self.assertEqual(result["frame"].record_id.tolist(), ["observed:1"])
        self.assertEqual(result["frame"].iloc[0].price, 35_000_000)
        self.assertEqual(result["rejected"].iloc[0].reason, "duplicate_fresh_group")
        reverse = self.prepare([second, first])
        self.assertEqual(reverse["frame"].record_id.tolist(), ["observed:2"])
        self.assertEqual(reverse["frame"].iloc[0].price, 85_000_000)

    def test_incomplete_and_legacy_only_records_retained_as_secondary(self):
        records = [listing(1, year_built=None), listing(2, address="Без номера"),
                   listing(3, url=None), listing(4, district=None)]
        result = self.prepare(records)
        self.assertEqual(len(result["frame"]), 4)
        self.assertFalse(result["frame"].primary_eligible.any())
        self.assertEqual(set(result["frame"].identity_status),
                         {"insufficient_identity", "legacy_signature_only", "invalid_or_missing_url"})
        self.assertTrue(any("not proof" in text for text in result["summary"]["limitations"]))

    def test_missing_key_records_not_merged_on_shared_nulls(self):
        result = self.prepare([listing(1, year_built=None, url=None, address=None, residential_complex=None),
                               listing(2, year_built=None, url=None, address=None, residential_complex=None)])
        self.assertEqual(len(result["frame"]), 2)
        self.assertEqual(result["frame"].fresh_group.nunique(), 2)

    def test_legacy_complex_signature_is_conservative_extra_block(self):
        old = {"history": pd.DataFrame([listing(90, address="Первый дом 10", residential_complex="Одинаковый ЖК")])}
        result = self.prepare([listing(1, address="Второй дом 20", residential_complex="Одинаковый ЖК")], old)
        self.assertTrue(result["frame"].empty)
        self.assertIn("signature:residential_complex:", result["audit"].iloc[0].direct_reference_keys)

    def test_preserves_original_rejections_positions_ids_and_input_frames(self):
        raw = pd.DataFrame([listing(1, source_row=900, price="от 30 млн", extra_note="Unchanged raw"),
                            listing(2, source_row=901)], index=[100, 200])
        refs = references()
        snapshots = raw.copy(deep=True), refs["prior_full_snapshot"].copy(deep=True)
        result = prepare_fresh_holdout(raw, refs, as_of="2026-10-13")
        rejected = result["rejected"].iloc[0]
        self.assertEqual(rejected.input__price, "от 30 млн")
        self.assertEqual(rejected.input__source_row, 900)
        self.assertEqual(rejected.input__extra_note, "Unchanged raw")
        self.assertEqual(rejected.input_row_index, "100")
        self.assertEqual(result["frame"].iloc[0].source_row, 2)
        pd.testing.assert_frame_equal(raw, snapshots[0])
        pd.testing.assert_frame_equal(refs["prior_full_snapshot"], snapshots[1])

    def test_unprovided_ids_generated_stably_and_duplicate_ids_fail(self):
        row = listing(); row.pop("record_id")
        self.assertEqual(self.prepare([row])["frame"].iloc[0].record_id, "fresh:000001")
        with self.assertRaisesRegex(ValueError, "record_id"):
            self.prepare([listing(), listing(2, record_id="observed:1")])

    def test_reference_history_must_be_present_and_usable(self):
        for refs in [{}, {"empty": pd.DataFrame()}, {"unidentifiable": pd.DataFrame([{"price": 1}])}]:
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                self.prepare([listing()], refs)

    def test_reference_mapping_order_does_not_change_output(self):
        refs = {"A": pd.DataFrame([listing(500)]), "B": pd.DataFrame([listing(600)])}
        first = self.prepare([listing(1), listing(2)], refs)
        second = self.prepare([listing(1), listing(2)], {"B": refs["B"], "A": refs["A"]})
        for key in ("frame", "rejected", "audit"):
            pd.testing.assert_frame_equal(first[key], second[key])
        self.assertEqual(first["summary"], second["summary"])

    def test_empty_raw_returns_boolean_primary_mask_and_full_output_schema(self):
        for raw in [pd.DataFrame(), pd.DataFrame(columns=listing().keys())]:
            with self.subTest(columns=list(raw.columns)):
                result = prepare_fresh_holdout(raw, references(), as_of="2026-10-13")
                frame = result["frame"]
                self.assertTrue(frame.empty)
                self.assertEqual(frame.primary_eligible.dtype, bool)
                self.assertEqual(list(frame.loc[frame.primary_eligible].columns), list(frame.columns))
                self.assertIn("price", frame.columns)
                self.assertIn("record_id", frame.columns)
                self.assertEqual(result["summary"]["primary_candidate_rows"], 0)

    def test_all_rejected_keeps_boolean_mask_and_original_fields(self):
        result = self.prepare([listing(1, collected_at=None), listing(2, collected_at="2026-03-31")])
        frame = result["frame"]
        self.assertTrue(frame.empty)
        self.assertEqual(frame.primary_eligible.dtype, bool)
        self.assertEqual(frame.loc[frame.primary_eligible].shape, frame.shape)
        self.assertEqual(len(result["rejected"]), 2)
        self.assertEqual(result["summary"]["secondary_rows"], 0)


if __name__ == "__main__":
    unittest.main()

"""Автономные проверки парсинга, докачки и поведения при ошибках HTTP."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from bs4 import BeautifulSoup

from scraper import (
    AccessBlocked,
    CsvStore,
    FetchError,
    FIELDNAMES,
    LayoutError,
    PoliteClient,
    normalize_listing_url,
    parse_catalog,
    parse_listing,
    run_scraper,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
CATALOG_URL = "https://krisha.kz/prodazha/kvartiry/astana/"
LISTING_URL = "https://krisha.kz/a/show/1014889872"
SECOND_URL = "https://krisha.kz/a/show/1014549478"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def response(status: int = 200, html: str | None = None) -> Mock:
    """Минимальный HTTP-ответ без сетевых запросов."""
    return Mock(
        status_code=status,
        text=fixture("listing.html") if html is None else html,
        headers={"Content-Type": "text/html; charset=utf-8"},
        url=LISTING_URL,
    )


class ParseTests(unittest.TestCase):
    def test_real_catalog_links_and_next_page(self) -> None:
        urls, next_page = parse_catalog(fixture("catalog.html"), page=1)
        self.assertEqual(urls, [LISTING_URL, SECOND_URL])
        self.assertEqual(next_page, 2)

    def test_last_page_without_next_link(self) -> None:
        soup = BeautifulSoup(fixture("catalog.html"), "html.parser")
        soup.select_one("nav.paginator").decompose()
        urls, next_page = parse_catalog(str(soup), page=1)
        self.assertEqual(len(urls), 2)
        self.assertIsNone(next_page)

    def test_duplicate_catalog_links_are_not_fetched_twice(self) -> None:
        html = fixture("catalog.html")
        html = html.replace(SECOND_URL.split(".kz")[1], "/a/show/1014889872?utm_source=test")
        urls, _ = parse_catalog(html, page=1)
        self.assertEqual(urls, [LISTING_URL])

    def test_no_cards_is_an_explicit_layout_error(self) -> None:
        with self.assertRaises(LayoutError):
            parse_catalog("<html><body>Разметка изменилась</body></html>", page=1)

    def test_real_listing_fields(self) -> None:
        row = parse_listing(fixture("listing.html"), LISTING_URL)
        expected = {
            "price": "66 500 000 ₸",
            "area": "107.4",
            "rooms": "4",
            "floor": "10",
            "total_floors": "16",
            "year_built": "2006",
            "building_type": "монолитный",
            "condition": "не новый, но аккуратный ремонт",
            "district": "Сарыарка",
            "residential_complex": "Болашак",
            "url": LISTING_URL,
        }
        self.assertEqual(set(row), set(FIELDNAMES))
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(row[name], value)
        collected = datetime.fromisoformat(row["collected_at"].replace("Z", "+00:00"))
        self.assertEqual(collected.utcoffset(), timedelta(0))

    def test_optional_features_remain_missing(self) -> None:
        soup = BeautifulSoup(fixture("listing.html"), "html.parser")
        for name in ("flat.building", "map.complex", "house.year", "flat.renovation"):
            soup.select_one(f'[data-name="{name}"]').decompose()
        row = parse_listing(str(soup), LISTING_URL)
        for name in ("building_type", "residential_complex", "year_built", "condition"):
            with self.subTest(field=name):
                self.assertEqual(row[name], "")

    def test_missing_price_does_not_produce_a_training_row(self) -> None:
        soup = BeautifulSoup(fixture("listing.html"), "html.parser")
        soup.select_one(".offer__price").decompose()
        with self.assertRaises(LayoutError):
            parse_listing(str(soup), LISTING_URL)

    def test_another_city_and_rental_category_are_rejected(self) -> None:
        html = fixture("listing.html")
        changes = (
            html.replace("Астана, Сарыарка р-н", "Алматы, Алмалинский р-н"),
            html.replace("/prodazha/kvartiry/", "/arenda/kvartiry/"),
        )
        for changed in changes:
            with self.subTest(html=changed[:30]):
                with self.assertRaises(LayoutError):
                    parse_listing(changed, LISTING_URL)

    def test_url_normalization_removes_tracking(self) -> None:
        self.assertEqual(
            normalize_listing_url("/a/show/1014889872?utm_source=test#photo"),
            LISTING_URL,
        )

    def test_offsite_and_unrelated_urls_are_rejected(self) -> None:
        for url in (
            "https://example.com/a/show/1014889872",
            "https://krisha.kz.evil.example/a/show/1014889872",
            "https://krisha.kz/a/show/not-an-id",
            "https://krisha.kz/arenda/kvartiry/",
        ):
            with self.subTest(url=url):
                with self.assertRaises(LayoutError):
                    normalize_listing_url(url)


class CsvStoreTests(unittest.TestCase):
    def test_saved_rows_are_readable_immediately_and_resume_without_duplicates(self) -> None:
        row = parse_listing(fixture("listing.html"), LISTING_URL)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw" / "listings.csv"
            with CsvStore(path) as store:
                self.assertTrue(store.append(row))
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    saved = list(csv.DictReader(stream))
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0], row)
            with CsvStore(path) as store:
                self.assertIn(LISTING_URL, store.seen)
                self.assertFalse(store.append(row))
            with path.open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_incompatible_header_is_rejected_without_overwriting(self) -> None:
        original = "url,unexpected_column\nhttps://krisha.kz/a/show/1,value\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "listings.csv"
            path.write_text(original, encoding="utf-8")
            with self.assertRaises(ValueError):
                CsvStore(path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)


class HttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = Mock()
        self.session.headers = {}
        # Паузы проверяем отдельно, все запросы остаются локальными mock-вызовами.
        self.sleep = patch("time.sleep").start()
        self.addCleanup(patch.stopall)

    def test_transient_server_error_is_retried(self) -> None:
        self.session.get.side_effect = [response(503), response()]
        client = PoliteClient(session=self.session, attempts=3)
        self.assertIn("offer__price", client.get(LISTING_URL))
        self.assertEqual(self.session.get.call_count, 2)

    def test_repeated_timeouts_have_a_finite_attempt_limit(self) -> None:
        self.session.get.side_effect = requests.Timeout("test timeout")
        client = PoliteClient(session=self.session, attempts=3)
        with self.assertRaises(FetchError):
            client.get(LISTING_URL)
        self.assertEqual(self.session.get.call_count, 3)

    def test_access_denied_is_not_retried(self) -> None:
        for status in (401, 403, 429):
            with self.subTest(status=status):
                self.session.get.reset_mock()
                self.session.get.return_value = response(status)
                client = PoliteClient(session=self.session, attempts=3)
                with self.assertRaises(AccessBlocked):
                    client.get(LISTING_URL)
                self.assertEqual(self.session.get.call_count, 1)

    def test_redirects_are_not_followed_automatically(self) -> None:
        self.session.get.return_value = response()
        client = PoliteClient(session=self.session)
        client.get(LISTING_URL)
        self.assertIs(self.session.get.call_args.kwargs.get("allow_redirects"), False)

    def test_successful_requests_have_a_one_to_two_second_pause(self) -> None:
        self.session.get.return_value = response()
        client = PoliteClient(session=self.session)
        client.get(LISTING_URL)
        client.get(LISTING_URL)
        self.assertEqual(self.sleep.call_count, 2)
        for call in self.sleep.call_args_list:
            self.assertGreaterEqual(call.args[0], 1.0)
            self.assertLessEqual(call.args[0], 2.0)

    def test_unrelated_host_is_rejected_before_network(self) -> None:
        client = PoliteClient(session=self.session)
        with self.assertRaises((LayoutError, FetchError, ValueError)):
            client.get("https://example.com/a/show/1")
        self.session.get.assert_not_called()

    def test_recaptcha_script_in_normal_footer_is_not_a_block(self) -> None:
        html = fixture("listing.html") + '<script src="https://www.google.com/recaptcha/api.js"></script>'
        self.session.get.return_value = response(html=html)
        client = PoliteClient(session=self.session)
        self.assertEqual(client.get(LISTING_URL), html)

    def test_human_verification_page_stops_without_retry(self) -> None:
        self.session.get.return_value = response(
            html="<html><h1>Подтвердите, что вы не робот</h1></html>"
        )
        client = PoliteClient(session=self.session)
        with self.assertRaises(AccessBlocked):
            client.get(LISTING_URL)
        self.assertEqual(self.session.get.call_count, 1)


class RunTests(unittest.TestCase):
    def test_block_after_first_save_keeps_progress_and_stops(self) -> None:
        client = Mock()
        client.get.side_effect = [
            fixture("catalog.html"),
            fixture("listing.html"),
            AccessBlocked("HTTP 429"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "listings.csv"
            with self.assertRaises(AccessBlocked):
                run_scraper(client, output=output, max_pages=10)
            with CsvStore(output) as store:
                self.assertEqual(store.seen, {LISTING_URL})
        self.assertEqual(client.get.call_count, 3)

    def test_failed_detail_is_not_marked_as_saved(self) -> None:
        soup = BeautifulSoup(fixture("catalog.html"), "html.parser")
        soup.select(".a-card")[1].decompose()
        client = Mock()
        client.get.side_effect = [str(soup), FetchError("HTTP 404", status_code=404)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "listings.csv"
            result = run_scraper(client, output=output, max_pages=1)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["saved"], 0)
            with CsvStore(output) as store:
                self.assertNotIn(LISTING_URL, store.seen)
            client.get.side_effect = [str(soup), fixture("listing.html")]
            resumed = run_scraper(client, output=output, max_pages=1)
            self.assertEqual(resumed["saved"], 1)

    def test_page_limit_and_resume_skip_saved_detail_requests(self) -> None:
        soup = BeautifulSoup(fixture("catalog.html"), "html.parser")
        soup.select(".a-card")[1].decompose()

        def get(url: str) -> str:
            if url in (CATALOG_URL, CATALOG_URL + "?page=1"):
                return str(soup)
            if url == LISTING_URL:
                return fixture("listing.html")
            self.fail(f"Неожиданный URL: {url}")

        client = Mock()
        client.get.side_effect = get
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "listings.csv"
            result = run_scraper(client, output=output, max_pages=1)
            self.assertEqual(result["pages"], 1)
            self.assertEqual(result["saved"], 1)
            self.assertEqual(result["failed"], 0)
            client.get.reset_mock()
            resumed = run_scraper(client, output=output, max_pages=1)
            self.assertEqual(resumed["saved"], 0)
            self.assertEqual(resumed["skipped"], 1)
            self.assertNotIn(LISTING_URL, [call.args[0] for call in client.get.call_args_list])

    def test_listing_limit_stops_before_the_next_detail(self) -> None:
        client = Mock()
        client.get.side_effect = [fixture("catalog.html"), fixture("listing.html")]
        with tempfile.TemporaryDirectory() as directory:
            result = run_scraper(
                client,
                output=Path(directory) / "listings.csv",
                max_pages=10,
                max_listings=1,
            )
        self.assertEqual(result["saved"], 1)
        self.assertEqual(client.get.call_count, 2)


if __name__ == "__main__":
    unittest.main()

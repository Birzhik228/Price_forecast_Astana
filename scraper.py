"""Сбор цен предложения квартир в Астане с сохранением прогресса в CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import random
import re
import time
from urllib.parse import parse_qs, urljoin, urlsplit

from bs4 import BeautifulSoup
import requests

from config import LOGS_DIR, PROJECT_ROOT, RAW_LISTINGS_PATH


BASE_URL = "https://krisha.kz"
CATALOG_PATH = "/prodazha/kvartiry/astana/"
CATALOG_URL = BASE_URL + CATALOG_PATH
ROBOTS_URL = BASE_URL + "/robots.txt"
TERMS_URL = BASE_URL + "/content/special/agreement"
USER_AGENT = "AstanaApartmentResearch/0.1 (educational project)"

# Проверены 07.10.2026 на HTML каталога и /a/show/1014889872.
# При изменении сайта сверяйте их вручную; не подменяйте случайными селекторами.
SELECTORS = {
    "catalog_title": "h1",
    "listing_links": ".a-card a.a-card__title[href]",
    "next_page": ".paginator a.paginator__btn--next[href]",
    "title": ".offer__advert-title h1",
    "price": ".offer__price",
    "location": ".offer__location span",
    "sale_category": '.breadcrumbs a[href="/prodazha/kvartiry/"]',
    "area": '[data-name="live.square"] .offer__advert-short-info',
    "floor": '[data-name="flat.floor"] .offer__advert-short-info',
    "year_built": '[data-name="house.year"] .offer__advert-short-info',
    "building_type": '[data-name="flat.building"] .offer__advert-short-info',
    "condition": '[data-name="flat.renovation"] .offer__advert-short-info',
    "residential_complex": '[data-name="map.complex"] .offer__advert-short-info',
}

FIELDNAMES = (
    "price", "area", "rooms", "floor", "total_floors", "year_built",
    "building_type", "condition", "district", "residential_complex",
    "url", "collected_at",
)
LOGGER = logging.getLogger(__name__)


class LayoutError(ValueError):
    """HTML не соответствует ожидаемому объявлению или каталогу."""


class AccessBlocked(RuntimeError):
    """Сайт ограничил доступ; автоматические попытки нужно прекратить."""


class FetchError(RuntimeError):
    """Запрос не удался даже после допустимых повторов."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def compact_text(value: str) -> str:
    """Убирает лишние пробелы, включая неразрывные."""
    return " ".join(value.split())


def selected_text(soup: BeautifulSoup, name: str) -> str:
    node = soup.select_one(SELECTORS[name])
    return compact_text(node.get_text(" ", strip=True)) if node else ""


def checked_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    headings = " ".join(node.get_text(" ", strip=True) for node in soup.find_all(["title", "h1"]))
    markers = ("captcha", "access denied", "just a moment", "доступ ограничен", "проверка безопасности")
    if any(marker in headings.lower() for marker in markers):
        raise AccessBlocked("Получена страница ограничения доступа или CAPTCHA.")
    # Обычный текст о reCAPTCHA в подвале не означает блокировку.
    if re.search(r"подтвердите[,\s]+что вы не робот", soup.get_text(" ").lower()):
        raise AccessBlocked("Сайт запрашивает подтверждение: автоматический сбор остановлен.")
    return soup


def normalize_listing_url(url: str) -> str:
    """Оставляет каноническую ссылку на карточку без меток и фрагмента."""
    parsed = urlsplit(urljoin(BASE_URL, url))
    if parsed.scheme not in {"http", "https"} or parsed.netloc != "krisha.kz":
        raise LayoutError(f"Ссылка ведёт за пределы krisha.kz: {url}")
    match = re.fullmatch(r"/a/show/(\d+)/?", parsed.path)
    if not match:
        raise LayoutError(f"Неожиданный адрес карточки: {url}")
    return f"{BASE_URL}/a/show/{match.group(1)}"


def parse_catalog(html: str, page: int) -> tuple[list[str], int | None]:
    """Извлекает только ссылки карточек и проверенную следующую страницу."""
    soup = checked_soup(html)
    if selected_text(soup, "catalog_title") != "Продажа квартир в Астане":
        raise LayoutError("Не найден заголовок каталога продажи квартир в Астане.")
    urls = list(dict.fromkeys(
        normalize_listing_url(str(node["href"]))
        for node in soup.select(SELECTORS["listing_links"])
    ))
    if not urls:
        raise LayoutError("Карточки не найдены: проверьте HTML, доступ и наличие объявлений.")
    next_link = soup.select_one(SELECTORS["next_page"])
    if next_link is None:
        return urls, None
    parsed = urlsplit(urljoin(BASE_URL, str(next_link["href"])))
    query = parse_qs(parsed.query)
    if (parsed.scheme != "https" or parsed.netloc != "krisha.kz"
            or parsed.path != CATALOG_PATH or query != {"page": [str(page + 1)]}):
        raise LayoutError("Неожиданная пагинация: сбор остановлен для проверки HTML.")
    return urls, page + 1


def parse_listing(html: str, url: str) -> dict[str, str]:
    """Извлекает исходные значения; очистка и отбор выбросов будут на этапе 3."""
    soup = checked_soup(html)
    if soup.select_one(SELECTORS["sale_category"]) is None:
        raise LayoutError("Карточка не относится к продаже квартир или разметка изменилась.")
    title = selected_text(soup, "title")
    price = selected_text(soup, "price")
    location = selected_text(soup, "location")
    city, _, district = location.partition(",")
    if city.strip() != "Астана":
        raise LayoutError("Не удалось подтвердить город Астана в карточке.")
    # Не принимаем цену аренды или цену за м² за полную цену продажи.
    if not re.fullmatch(r"\d[\d\s.,]*(?:млн|млрд)?\s*(?:₸|тг\.?|тенге)", price):
        raise LayoutError("Не найдена полная цена в тенге; проверьте селектор и формат цены.")
    rooms = re.match(r"(\d+)\s*-\s*комнатн", title)
    area = re.search(r"(\d+(?:[.,]\d+)?)\s*м[²2]", selected_text(soup, "area") or title)
    if rooms is None or area is None:
        raise LayoutError("Не удалось извлечь комнаты или площадь: проверьте HTML карточки.")
    floor_text = selected_text(soup, "floor")
    floors = re.search(r"(\d+)\s*(?:из|/)\s*(\d+)", floor_text)
    if floors is None:
        floors = re.search(r"(\d+)\s*/\s*(\d+)\s*этаж", title)
    single_floor = re.fullmatch(r"\d+", floor_text)
    row = {name: "" for name in FIELDNAMES}
    row.update({
        "price": price,
        "area": area.group(1),
        "rooms": rooms.group(1),
        "floor": floors.group(1) if floors else (single_floor.group() if single_floor else ""),
        "total_floors": floors.group(2) if floors else "",
        "district": re.sub(r"\s+(?:р-н|район)$", "", district.strip()),
        "url": normalize_listing_url(url),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    for name in ("year_built", "building_type", "condition", "residential_complex"):
        row[name] = selected_text(soup, name)
    return row


class PoliteClient:
    """Последовательные запросы с паузами и ограниченными повторами."""

    def __init__(self, session: requests.Session | None = None,
                 attempts: int = 3, timeout: float = 30.0) -> None:
        if attempts < 1 or timeout <= 0:
            raise ValueError("Число попыток и таймаут должны быть положительными.")
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru"})
        self.attempts = attempts
        self.timeout = timeout

    def get(self, url: str) -> str:
        parsed = urlsplit(url)
        is_catalog = parsed.path == CATALOG_PATH and (
            not parsed.query or re.fullmatch(r"page=[1-9]\d*", parsed.query)
        )
        is_listing = re.fullmatch(r"/a/show/\d+", parsed.path) and not parsed.query
        if parsed.scheme != "https" or parsed.netloc != "krisha.kz" or not (is_catalog or is_listing):
            raise FetchError(f"Запрос за пределами каталога и карточек запрещён: {url}")
        for attempt in range(1, self.attempts + 1):
            # Каждый запрос, в том числе повторный, получает отдельную паузу.
            time.sleep(random.uniform(1.0, 2.0))
            LOGGER.info("GET %s (попытка %s/%s)", url, attempt, self.attempts)
            try:
                response = self.session.get(url, timeout=self.timeout, allow_redirects=False)
            except (requests.Timeout, requests.ConnectionError) as error:
                if attempt == self.attempts:
                    raise FetchError(f"Исчерпаны попытки загрузки {url}: {error}") from error
                LOGGER.warning("Сетевая ошибка: %s", error)
                time.sleep(2 ** attempt)
                continue
            except requests.RequestException as error:
                raise FetchError(f"Ошибка запроса {url}: {error}") from error
            status = response.status_code
            if status in {401, 403, 429}:
                retry_after = response.headers.get("Retry-After", "не указан")
                raise AccessBlocked(f"HTTP {status}: сбор остановлен. Retry-After: {retry_after}.")
            if 500 <= status < 600 and attempt < self.attempts:
                LOGGER.warning("HTTP %s: повтор после дополнительной паузы.", status)
                time.sleep(2 ** attempt)
                continue
            if 300 <= status < 400:
                raise AccessBlocked("Сайт перенаправил запрос: проверьте доступ вручную.")
            if status != 200:
                raise FetchError(f"HTTP {status} при загрузке {url}", status_code=status)
            if "text/html" not in response.headers.get("Content-Type", "").lower():
                raise LayoutError("Вместо HTML получен другой формат ответа.")
            response.encoding = "utf-8"
            checked_soup(response.text)
            return response.text
        raise FetchError(f"Не удалось загрузить {url}")

    def close(self) -> None:
        self.session.close()


class CsvStore:
    """Дописывает по одной строке и использует CSV как источник прогресса."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.seen: set[str] = set()
        has_content = path.exists() and path.stat().st_size > 0
        if has_content:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source, strict=True)
                if reader.fieldnames != list(FIELDNAMES):
                    raise ValueError("Заголовок существующего CSV не совпадает со схемой. Файл не изменён.")
                for number, row in enumerate(reader, start=2):
                    if None in row or any(value is None for value in row.values()) or not row["url"]:
                        raise ValueError(f"Повреждена строка {number} CSV. Проверьте файл перед докачкой.")
                    self.seen.add(normalize_listing_url(row["url"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = False
        if has_content:
            with path.open("rb") as source:
                source.seek(-1, os.SEEK_END)
                needs_newline = source.read(1) not in {b"\n", b"\r"}
        self.file = path.open("a", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDNAMES)
        if needs_newline:
            self.file.write("\n")
        if not has_content:
            self.writer.writeheader()
        self._flush()

    def _flush(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())

    def append(self, row: dict[str, str]) -> bool:
        if set(row) != set(FIELDNAMES):
            raise ValueError("Набор полей объявления не совпадает со схемой CSV.")
        url = normalize_listing_url(row["url"])
        if url in self.seen:
            return False
        self.writer.writerow({**row, "url": url})
        self._flush()
        self.seen.add(url)
        return True

    def __enter__(self) -> CsvStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.file.close()


def run_scraper(client: PoliteClient, output: Path, max_pages: int,
                start_page: int = 1, max_listings: int | None = None) -> dict[str, int]:
    """Обходит ограниченное число страниц, сохраняя успехи до следующего запроса."""
    if max_pages < 1 or start_page < 1 or (max_listings is not None and max_listings < 1):
        raise ValueError("Лимиты и номер первой страницы должны быть положительными.")
    counts = {"pages": 0, "saved": 0, "skipped": 0, "failed": 0}
    page = start_page
    with CsvStore(output) as store:
        LOGGER.info("В CSV уже сохранено %s уникальных ссылок.", len(store.seen))
        for _ in range(max_pages):
            catalog_url = CATALOG_URL if page == 1 else f"{CATALOG_URL}?page={page}"
            urls, next_page = parse_catalog(client.get(catalog_url), page)
            counts["pages"] += 1
            for url in urls:
                if url in store.seen:
                    counts["skipped"] += 1
                    continue
                try:
                    row = parse_listing(client.get(url), url)
                except FetchError as error:
                    counts["failed"] += 1
                    LOGGER.warning("Карточка пропущена, при следующем запуске повторим: %s", error)
                    continue
                # Ошибки доступа и разметки не скрываем и не продолжаем обход.
                store.append(row)
                counts["saved"] += 1
                missing = [key for key, value in row.items() if not value]
                LOGGER.info("Сохранено %s; пустые поля: %s", url, ", ".join(missing) or "нет")
                if max_listings is not None and counts["saved"] >= max_listings:
                    return counts
            if next_page is None:
                break
            page = next_page
    return counts


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Нужно целое число не меньше 1.")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=positive_int, default=1, help="Лимит страниц каталога (по умолчанию 1).")
    parser.add_argument("--start-page", type=positive_int, default=1, help="Начальная страница (по умолчанию 1).")
    parser.add_argument("--max-listings", type=positive_int, help="Лимит новых сохранённых объявлений за запуск.")
    parser.add_argument("--output", type=Path, default=RAW_LISTINGS_PATH, help="CSV; относительный путь отсчитывается от проекта.")
    args = parser.parse_args()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "scraper.log", encoding="utf-8")],
    )
    LOGGER.warning("Перед сбором проверьте robots.txt: %s и условия: %s", ROBOTS_URL, TERMS_URL)
    LOGGER.warning("Пункт 5.3 соглашения требует письменного разрешения на автоматический сбор.")
    output = (PROJECT_ROOT / args.output).resolve()
    client = PoliteClient()
    try:
        counts = run_scraper(client, output, args.max_pages, args.start_page, args.max_listings)
        LOGGER.info("Итог: страниц=%s, новых=%s, уже были=%s, ошибок=%s. CSV: %s",
                    counts["pages"], counts["saved"], counts["skipped"], counts["failed"], output)
        return 1 if counts["failed"] else 0
    except KeyboardInterrupt:
        LOGGER.warning("Остановлено пользователем. Сохранённые строки остаются в %s.", output)
        return 130
    except (AccessBlocked, FetchError, ValueError, OSError, csv.Error) as error:
        LOGGER.error("Сбор остановлен: %s. Ранее сохранённые строки остаются в %s.", error, output)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

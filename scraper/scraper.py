import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from bs4 import BeautifulSoup

from database.indexer import index_documents


LOGGER = logging.getLogger(__name__)


USER_AGENT = (
    "MontesBotScraper/1.0 (+https://www.utad.pt; polite bot for academic chatbot)"
)

BASE_DOMAIN = "utad.pt"
BASE_URL = "https://www.utad.pt/"

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUT_FILE = DATA_DIR / "raw_documents.jsonl"
INDEX_METADATA_FILE = DATA_DIR / "index_metadata.json"

SCRAPE_TIMEOUT_SECONDS = 15
REQUEST_DELAY_SECONDS_MIN = 1.0
REQUEST_DELAY_SECONDS_MAX = 2.0
MAX_PAGES_DEFAULT = 2000

_scheduler: Optional[BackgroundScheduler] = None


@dataclass
class ScrapedDocument:
    """Container for a scraped web page with normalized content and metadata."""

    url: str
    title: str
    content: str
    date_scraped: str
    category: str


def _load_robots_parser(base_url: str) -> Optional[RobotFileParser]:
    """Load and parse robots.txt for the target site."""
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()

    try:
        LOGGER.info("Loading robots.txt from %s", robots_url)
        response = requests.get(
            robots_url,
            headers={"User-Agent": USER_AGENT},
            timeout=SCRAPE_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            LOGGER.warning(
                "robots.txt returned status %s, proceeding cautiously", response.status_code
            )
            return None

        parser.parse(response.text.splitlines())
        return parser
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load robots.txt: %s", exc)
        return None


def _is_allowed_by_robots(parser: Optional[RobotFileParser], url: str) -> bool:
    """Check whether a URL is allowed to be fetched according to robots.txt."""
    if parser is None:
        # If robots.txt could not be loaded, be conservative but do not block everything.
        return True

    try:
        return parser.can_fetch(USER_AGENT, url)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Error checking robots.txt for %s: %s", url, exc)
        return False


def _normalize_url(base_url: str, href: str) -> Optional[str]:
    """Resolve and normalize a link relative to the base URL."""
    if not href:
        return None

    href = href.strip()
    if href.startswith("mailto:") or href.startswith("tel:"):
        return None

    absolute = urljoin(base_url, href)
    absolute, _ = urldefrag(absolute)
    parsed = urlparse(absolute)

    if not parsed.scheme.startswith("http"):
        return None

    # Restrict to UTAD domain (including subdomains).
    if not parsed.netloc.endswith(BASE_DOMAIN):
        return None

    return absolute


def _clean_text(text: str) -> str:
    """Clean extracted text by normalizing whitespace and removing noise."""
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


def _categorize(url: str, text: str) -> str:
    """Best-effort heuristic categorization based on URL path and keywords."""
    url_lower = url.lower()
    text_lower = text.lower()

    try:
        if any(
            keyword in url_lower
            for keyword in [
                "curso",
                "cursos",
                "licenciatura",
                "mestrado",
                "doutoramento",
                "calendario-academico",
                "calendario",
                "propinas",
                "regulamento",
                "admiss",
                "candidatura",
                "bolsa",
                "bolsas",
            ]
        ):
            return "Academic"

        if any(
            keyword in url_lower
            for keyword in [
                "servicos",
                "servicos-academicos",
                "biblioteca",
                "residencia",
                "residencias",
                "cantina",
                "alimentacao",
                "saude",
                "desporto",
                "desporto",
                "contactos",
                "contacto",
            ]
        ):
            return "Services & Contacts"

        if any(
            keyword in url_lower
            for keyword in [
                "institucional",
                "sobre",
                "missao",
                "governa",
                "orgaos-de-governo",
                "departamento",
                "departamentos",
                "escola",
                "investigacao",
                "unidade-de-investigacao",
                "unidades-de-investigacao",
            ]
        ):
            return "Institutional"

        if any(
            keyword in url_lower
            for keyword in [
                "agenda",
                "noticia",
                "noticias",
                "evento",
                "eventos",
                "projeto",
                "projetos",
            ]
        ):
            return "Events & News"

        # Fallback to content-based hints.
        if "calendário académico" in text_lower or "calendario academico" in text_lower:
            return "Academic"
        if "serviços académicos" in text_lower or "servicos academicos" in text_lower:
            return "Services & Contacts"
        if "missão" in text_lower or "missao" in text_lower:
            return "Institutional"
        if "evento" in text_lower or "notícia" in text_lower or "noticia" in text_lower:
            return "Events & News"
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Error categorizing URL %s: %s", url, exc)

    return "Uncategorized"


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract a reasonable page title from HTML."""
    try:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Error extracting title: %s", exc)
    return "Página UTAD"


def _scrape_single_page(url: str, session: requests.Session) -> Optional[ScrapedDocument]:
    """Scrape a single page and return a ScrapedDocument or None on failure."""
    try:
        LOGGER.info("Fetching %s", url)
        response = session.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=SCRAPE_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            LOGGER.warning("Skipping %s due to status %s", url, response.status_code)
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        raw_text = soup.get_text(separator="\n")
        cleaned = _clean_text(raw_text)
        if not cleaned:
            LOGGER.info("No usable text content at %s", url)
            return None

        title = _extract_title(soup)
        category = _categorize(url, cleaned)
        date_scraped = datetime.now(timezone.utc).isoformat()

        return ScrapedDocument(
            url=url,
            title=title,
            content=cleaned,
            date_scraped=date_scraped,
            category=category,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Error scraping %s: %s", url, exc)
        return None


def _find_links(url: str, html: str) -> List[str]:
    """Extract and normalize links from a page HTML."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        links: List[str] = []
        for anchor in soup.find_all("a", href=True):
            normalized = _normalize_url(url, anchor["href"])
            if normalized:
                links.append(normalized)
        return links
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Error extracting links from %s: %s", url, exc)
        return []


def _polite_delay() -> None:
    """Sleep between 1–2 seconds to avoid overloading the site."""
    delay = REQUEST_DELAY_SECONDS_MIN + (
        (REQUEST_DELAY_SECONDS_MAX - REQUEST_DELAY_SECONDS_MIN) * 0.5
    )
    # Use a fixed mid-range delay for predictability.
    time.sleep(delay)


def scrape_site(
    base_url: str = BASE_URL,
    max_pages: int = MAX_PAGES_DEFAULT,
) -> List[ScrapedDocument]:
    """
    Crawl UTAD public pages recursively, respecting robots.txt and being polite.

    Returns a list of ScrapedDocument with normalized content and metadata.
    """
    robots = _load_robots_parser(base_url)

    visited: Set[str] = set()
    queue: Deque[str] = deque([base_url])
    documents: List[ScrapedDocument] = []

    session = requests.Session()

    pages_processed = 0

    while queue and pages_processed < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        if not _is_allowed_by_robots(robots, url):
            LOGGER.info("Skipping %s due to robots.txt rules", url)
            continue

        try:
            LOGGER.debug("Requesting page for content and links: %s", url)
            response = session.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=SCRAPE_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Error fetching %s for crawling: %s", url, exc)
            continue

        if response.status_code != 200:
            LOGGER.warning("Skipping %s due to status %s", url, response.status_code)
            continue

        # Scrape and store document from this page.
        page_doc = _scrape_single_page(url, session)
        if page_doc:
            documents.append(page_doc)

        # Discover new links for crawling.
        links = _find_links(url, response.text)
        for link in links:
            if link not in visited:
                queue.append(link)

        pages_processed += 1
        _polite_delay()

    LOGGER.info("Scraping finished: %s pages processed, %s documents collected", pages_processed, len(documents))
    return documents


def _append_documents_to_jsonl(documents: List[ScrapedDocument], output_file: Path) -> None:
    """Append scraped documents to a JSONL file for later inspection or reprocessing."""
    try:
        with output_file.open("a", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
        LOGGER.info("Wrote %s documents to %s", len(documents), output_file)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to write scraped documents to %s: %s", output_file, exc)


def _write_index_metadata(scraped: int, indexed: int) -> None:
    """Persist a tiny JSON with last scrape metadata for observability."""
    try:
        payload = {
            "last_scrape_date": datetime.now(timezone.utc).isoformat(),
            "scraped": int(scraped),
            "indexed": int(indexed),
        }
        with INDEX_METADATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        LOGGER.info("Updated index metadata file at %s", INDEX_METADATA_FILE)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to write index metadata: %s", exc)


def get_last_scrape_date() -> Optional[str]:
    """
    Return ISO timestamp string of the last successful scrape+index run, if known.
    """
    try:
        if not INDEX_METADATA_FILE.is_file():
            return None
        with INDEX_METADATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("last_scrape_date")
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to read index metadata: %s", exc)
        return None


def run_scraper_and_index() -> Dict[str, int]:
    """
    Run the scraper, persist raw documents, and index them in ChromaDB.

    Returns a small summary dict with document counts for observability.
    """
    try:
        LOGGER.info("Starting full scrape and index run")
        documents = scrape_site()
        if not documents:
            LOGGER.warning("Scraper did not collect any documents")
            return {"scraped": 0, "indexed": 0}

        _append_documents_to_jsonl(documents, RAW_OUTPUT_FILE)

        indexed_count = index_documents(documents)
        LOGGER.info("Scrape and index run complete: scraped=%s indexed=%s", len(documents), indexed_count)
        _write_index_metadata(scraped=len(documents), indexed=indexed_count)
        return {"scraped": len(documents), "indexed": indexed_count}
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Scrape and index process failed: %s", exc)
        return {"scraped": 0, "indexed": 0}


def start_scheduler() -> None:
    """
    Start the APScheduler background scheduler for daily scraping.

    By default runs once per day at 03:00 server time.
    If SCRAPE_INTERVAL_HOURS is set to a value other than 24, an interval-based
    schedule will be used instead.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        LOGGER.info("Scraper scheduler already running; skipping reinitialization")
        return

    try:
        # Use a stable, cross-platform timezone. For most scraping workloads
        # a fixed UTC scheduler is sufficient and avoids OS-specific names.
        scheduler = BackgroundScheduler(timezone="UTC")

        interval_hours_str = os.getenv("SCRAPE_INTERVAL_HOURS", "24")
        try:
            interval_hours = int(interval_hours_str)
        except ValueError:
            LOGGER.warning(
                "Invalid SCRAPE_INTERVAL_HOURS=%s, defaulting to 24h daily at 03:00",
                interval_hours_str,
            )
            interval_hours = 24

        if interval_hours == 24:
            scheduler.add_job(run_scraper_and_index, "cron", hour=3, minute=0, id="daily_scrape")
            LOGGER.info("Scheduled daily scraping at 03:00 with APScheduler")
        else:
            scheduler.add_job(
                run_scraper_and_index,
                "interval",
                hours=interval_hours,
                id="interval_scrape",
            )
            LOGGER.info(
                "Scheduled interval scraping every %s hours with APScheduler",
                interval_hours,
            )

        scheduler.start()
        _scheduler = scheduler
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to start scraper scheduler: %s", exc)


if __name__ == "__main__":
    # Allow running the scraper directly for manual debugging.
    logging.basicConfig(level=logging.INFO)
    summary = run_scraper_and_index()
    print(summary)


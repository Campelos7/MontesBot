"""Fixtures partilhadas: arranque da API sem scheduler/scrape reais."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("api.main.start_scheduler", lambda: None)
    monkeypatch.setattr("api.main.get_vector_store", lambda: None)
    monkeypatch.setattr("api.main.get_document_count", lambda: 1)
    monkeypatch.setattr("api.main.run_scraper_and_index", lambda: {"scraped": 0, "indexed": 0})

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client

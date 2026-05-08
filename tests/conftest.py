"""Fixtures partilhadas para testes da API local."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client

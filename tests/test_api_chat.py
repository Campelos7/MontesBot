from unittest.mock import patch


def test_chat_uses_get_answer_and_returns_payload(client):
    with patch("api.main.get_answer", return_value=("Resposta de teste", [])) as m:
        response = client.post(
            "/chat",
            json={"message": "Olá", "session_id": "test-session-1"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Resposta de teste"
    assert data["sources"] == []
    assert data["session_id"] == "test-session-1"
    m.assert_called_once()


def test_chat_empty_message_400(client):
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400


def test_chat_message_too_long_422(client):
    response = client.post("/chat", json={"message": "x" * 4001})
    assert response.status_code == 422

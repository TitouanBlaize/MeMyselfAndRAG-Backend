from datetime import UTC, datetime

from app.rag import AnswerGenerationError


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_happy_path(client, monkeypatch, fake_conn, make_get_conn):
    monkeypatch.setattr(
        "app.main.answer_question",
        lambda question, owner_name=None: {"answer": "42", "sources": []},
    )
    monkeypatch.setattr("app.main.get_conn", make_get_conn(fake_conn))

    resp = client.post("/chat", json={"question": "What is it?"})

    assert resp.status_code == 200
    assert resp.json() == {"answer": "42", "sources": []}
    fake_conn.execute.assert_called_once_with(
        "INSERT INTO chat_logs (question, answer, error) VALUES (%s, %s, %s)",
        ("What is it?", "42", None),
    )


def test_chat_empty_question_returns_400(client):
    resp = client.post("/chat", json={"question": "   "})
    assert resp.status_code == 400


def test_chat_upstream_failure_returns_502(
    client, monkeypatch, fake_conn, make_get_conn
):
    def raise_error(question, owner_name=None):
        raise AnswerGenerationError("boom")

    monkeypatch.setattr("app.main.answer_question", raise_error)
    monkeypatch.setattr("app.main.get_conn", make_get_conn(fake_conn))

    resp = client.post("/chat", json={"question": "hi"})

    assert resp.status_code == 502
    fake_conn.execute.assert_called_once_with(
        "INSERT INTO chat_logs (question, answer, error) VALUES (%s, %s, %s)",
        ("hi", None, "boom"),
    )


def test_chat_unexpected_failure_returns_500_without_leaking_detail(
    client, monkeypatch
):
    def raise_unexpected(question, owner_name=None):
        raise RuntimeError("something broke internally")

    monkeypatch.setattr("app.main.answer_question", raise_unexpected)
    resp = client.post("/chat", json={"question": "hi"})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "internal server error"}
    assert "something broke internally" not in resp.text


def test_chat_logging_failure_does_not_break_response(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.answer_question",
        lambda question, owner_name=None: {"answer": "42", "sources": []},
    )

    def broken_get_conn():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("app.main.get_conn", broken_get_conn)

    resp = client.post("/chat", json={"question": "What is it?"})

    assert resp.status_code == 200
    assert resp.json() == {"answer": "42", "sources": []}


def test_ingest_text_missing_api_key_header_is_422(client):
    resp = client.post("/ingest/text", json={"source": "a.md", "text": "# Q\nA"})
    assert resp.status_code == 422


def test_ingest_text_wrong_api_key_is_401(client):
    resp = client.post(
        "/ingest/text",
        json={"source": "a.md", "text": "# Q\nA"},
        headers={"x-api-key": "wrong"},
    )
    assert resp.status_code == 401


def test_ingest_text_success(authed_client, monkeypatch, fake_conn, make_get_conn):
    monkeypatch.setattr("app.main.embed_documents", lambda chunks: [None] * len(chunks))
    monkeypatch.setattr("app.main.get_conn", make_get_conn(fake_conn))

    resp = authed_client.post("/ingest/text", json={"source": "a.md", "text": "# Q\nA"})

    assert resp.status_code == 200
    assert resp.json() == {"chunks_ingested": 1}


def test_list_documents(authed_client, monkeypatch, fake_conn, make_get_conn):
    fake_conn.execute.return_value.fetchall.return_value = [
        (1, "a.md", "content", datetime(2024, 1, 1, tzinfo=UTC)),
    ]
    monkeypatch.setattr("app.main.get_conn", make_get_conn(fake_conn))

    resp = authed_client.get("/documents")

    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["source"] == "a.md"
    assert body[0]["content"] == "content"

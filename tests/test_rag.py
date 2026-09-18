from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.rag import AnswerGenerationError, answer_question, retrieve


def _fake_message(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def test_retrieve_builds_expected_rows(monkeypatch, fake_conn, make_get_conn):
    fake_conn.execute.return_value.fetchall.return_value = [
        ("some answer", "a.md", 0.87),
    ]
    monkeypatch.setattr("app.rag.get_conn", make_get_conn(fake_conn))
    monkeypatch.setattr("app.rag.embed_query", lambda q: [0.1, 0.2])

    results = retrieve("what?")

    assert results == [{"content": "some answer", "source": "a.md", "similarity": 0.87}]


def test_answer_question_uses_passed_owner_name(monkeypatch, fake_conn, make_get_conn):
    # Regression test for the bug found during the portfolio audit: main.py
    # used to call answer_question(question) with no owner_name, so the
    # system prompt's {owner} placeholder never resolved to a real name.
    fake_conn.execute.return_value.fetchall.return_value = [
        ("Titouan studied CS.", "resume.md", 0.9),
    ]
    monkeypatch.setattr("app.rag.get_conn", make_get_conn(fake_conn))
    monkeypatch.setattr("app.rag.embed_query", lambda q: [0.1])

    fake_claude = MagicMock()
    fake_claude.messages.create.return_value = _fake_message(
        "Titouan studied computer science."
    )
    monkeypatch.setattr("app.rag._claude", fake_claude)

    result = answer_question("What did they study?", owner_name="Titouan")

    system_prompt = fake_claude.messages.create.call_args.kwargs["system"]
    assert "sur Titouan" in system_prompt
    assert "the site owner" not in system_prompt
    assert result["answer"] == "Titouan studied computer science."


def test_answer_question_no_chunks_uses_fallback_context(
    monkeypatch, fake_conn, make_get_conn
):
    fake_conn.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr("app.rag.get_conn", make_get_conn(fake_conn))
    monkeypatch.setattr("app.rag.embed_query", lambda q: [0.1])

    fake_claude = MagicMock()
    fake_claude.messages.create.return_value = _fake_message("I don't know.")
    monkeypatch.setattr("app.rag._claude", fake_claude)

    result = answer_question("anything")

    call_kwargs = fake_claude.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert "No documents have been ingested yet." in user_content
    assert result["sources"] == []


def test_answer_question_wraps_claude_errors(monkeypatch, fake_conn, make_get_conn):
    fake_conn.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr("app.rag.get_conn", make_get_conn(fake_conn))
    monkeypatch.setattr("app.rag.embed_query", lambda q: [0.1])

    fake_claude = MagicMock()
    fake_claude.messages.create.side_effect = RuntimeError("api down")
    monkeypatch.setattr("app.rag._claude", fake_claude)

    with pytest.raises(AnswerGenerationError):
        answer_question("anything")

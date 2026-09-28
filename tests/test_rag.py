from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.rag import AnswerGenerationError, answer_question, retrieve, stream_answer


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


def _fake_stream(pieces, fail_after=None):
    """Stand-in for `_claude.messages.stream(...)`: a context manager whose
    .text_stream yields `pieces`, optionally raising after `fail_after` of
    them to simulate the API dropping mid-answer."""

    @contextmanager
    def _stream(**kwargs):
        def _text():
            for i, piece in enumerate(pieces):
                if fail_after is not None and i == fail_after:
                    raise RuntimeError("connection reset")
                yield piece

        yield SimpleNamespace(text_stream=_text())

    return _stream


def test_stream_answer_yields_text_and_sources(monkeypatch, fake_conn, make_get_conn):
    fake_conn.execute.return_value.fetchall.return_value = [
        ("Titouan studied CS.", "resume.md", 0.9),
    ]
    monkeypatch.setattr("app.rag.get_conn", make_get_conn(fake_conn))
    monkeypatch.setattr("app.rag.embed_query", lambda q: [0.1])

    fake_claude = MagicMock()
    fake_claude.messages.stream = MagicMock(
        side_effect=_fake_stream(["J'ai ", "étudié ", "l'info."])
    )
    monkeypatch.setattr("app.rag._claude", fake_claude)

    sources, text_chunks = stream_answer("What did they study?", owner_name="Titouan")

    assert sources == [{"source": "resume.md", "similarity": 0.9}]
    # Claude isn't called until the text iterator is consumed.
    fake_claude.messages.stream.assert_not_called()
    assert list(text_chunks) == ["J'ai ", "étudié ", "l'info."]
    assert "sur Titouan" in fake_claude.messages.stream.call_args.kwargs["system"]


def test_stream_answer_wraps_mid_stream_errors(monkeypatch, fake_conn, make_get_conn):
    fake_conn.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr("app.rag.get_conn", make_get_conn(fake_conn))
    monkeypatch.setattr("app.rag.embed_query", lambda q: [0.1])

    fake_claude = MagicMock()
    fake_claude.messages.stream = MagicMock(
        side_effect=_fake_stream(["partial ", "never"], fail_after=1)
    )
    monkeypatch.setattr("app.rag._claude", fake_claude)

    _, text_chunks = stream_answer("anything")

    assert next(text_chunks) == "partial "
    with pytest.raises(AnswerGenerationError):
        next(text_chunks)

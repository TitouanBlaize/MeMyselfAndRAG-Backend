from collections.abc import Iterator

import anthropic

from app.config import settings
from app.db import get_conn
from app.embeddings import embed_query

_claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)


class AnswerGenerationError(RuntimeError):
    """Raised when the Claude API call fails."""


SYSTEM_PROMPT = """Tu es un assistant répondant aux questions sur {owner} \
en utilisant uniquement les extraits de contexte fournis de leur cv, thèse et \
articles. Si le contexte ne contient pas la réponse, dites-le honnêtement plutôt \
que de deviner. Gardez les réponses concises et parlez de {owner} à la première \
personne."""


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    top_k = top_k or settings.top_k
    query_embedding = embed_query(query)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT content, source, 1 - (embedding <=> %s) AS similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        ).fetchall()

    return [{"content": r[0], "source": r[1], "similarity": r[2]} for r in rows]


def _build_request(query: str, owner_name: str) -> tuple[list[dict], dict]:
    """Retrieves context for `query` and returns (chunks, Claude request
    kwargs). Shared by the blocking and streaming paths so both send
    exactly the same prompt."""
    chunks = retrieve(query)

    if not chunks:
        context = "No documents have been ingested yet."
    else:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['content']}" for c in chunks
        )

    request = {
        "model": settings.claude_model,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT.format(owner=owner_name),
        "messages": [
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }
        ],
    }
    return chunks, request


def _sources(chunks: list[dict]) -> list[dict]:
    return [{"source": c["source"], "similarity": c["similarity"]} for c in chunks]


def answer_question(query: str, owner_name: str = "the site owner") -> dict:
    chunks, request = _build_request(query, owner_name)

    try:
        message = _claude.messages.create(**request)
    except Exception as e:
        raise AnswerGenerationError("failed to generate an answer") from e

    answer_text = "".join(
        block.text for block in message.content if block.type == "text"
    )

    return {"answer": answer_text, "sources": _sources(chunks)}


def stream_answer(
    query: str, owner_name: str = "the site owner"
) -> tuple[list[dict], Iterator[str]]:
    """Streaming variant of answer_question(). Retrieval runs eagerly, so
    embedding/DB failures raise here, before any response bytes are sent.
    Returns (sources, text_chunks); iterating text_chunks drives the Claude
    call and raises AnswerGenerationError if it fails, possibly mid-answer."""
    chunks, request = _build_request(query, owner_name)

    def _text_chunks() -> Iterator[str]:
        try:
            with _claude.messages.stream(**request) as stream:
                yield from stream.text_stream
        except Exception as e:
            raise AnswerGenerationError("failed to generate an answer") from e

    return _sources(chunks), _text_chunks()

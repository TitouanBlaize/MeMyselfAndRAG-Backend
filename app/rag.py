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


def answer_question(query: str, owner_name: str = "the site owner") -> dict:
    chunks = retrieve(query)

    if not chunks:
        context = "No documents have been ingested yet."
    else:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['content']}" for c in chunks
        )

    try:
        message = _claude.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT.format(owner=owner_name),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                }
            ],
        )
    except Exception as e:
        raise AnswerGenerationError("failed to generate an answer") from e

    answer_text = "".join(
        block.text for block in message.content if block.type == "text"
    )

    return {
        "answer": answer_text,
        "sources": [
            {"source": c["source"], "similarity": c["similarity"]} for c in chunks
        ],
    }

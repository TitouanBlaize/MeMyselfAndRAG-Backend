import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auto_ingest import auto_ingest_qa_file
from app.chunking import chunk_markdown_qa
from app.config import settings
from app.db import get_conn, init_db
from app.deps import require_api_key
from app.embeddings import EmbeddingError, embed_documents
from app.rag import AnswerGenerationError, answer_question

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    auto_ingest_qa_file()
    yield


app = FastAPI(title="Personal RAG API", lifespan=lifespan)

# Lock this down to your actual frontend domain(s) before going live —
# override via the CORS_ORIGINS env var (see app/config.py).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/health")
def health():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


def _log_chat(
    question: str, answer: str | None = None, error: str | None = None
) -> None:
    """Best-effort persistence of a chat Q&A to the chat_logs table. Never
    raises — a logging failure must not turn a successful (or already-failed)
    chat request into a 500."""
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO chat_logs (question, answer, error) VALUES (%s, %s, %s)",
                (question, answer, error),
            )
    except Exception:
        logger.exception("failed to write chat log")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    logger.info("chat question received: %r", req.question)
    try:
        result = answer_question(req.question, owner_name=settings.owner_name)
        logger.info("chat answered, %d sources", len(result["sources"]))
        _log_chat(req.question, answer=result["answer"])
        return result
    except AnswerGenerationError as e:
        logger.error("chat failed: %s", e)
        _log_chat(req.question, error=str(e))
        raise HTTPException(502, "failed to generate an answer") from e


class IngestTextRequest(BaseModel):
    source: str
    text: str  # markdown, one "# Question" heading per Q/A pair


class IngestTextResponse(BaseModel):
    chunks_ingested: int


@app.post(
    "/ingest/text",
    response_model=IngestTextResponse,
    dependencies=[Depends(require_api_key)],
)
def ingest_text(req: IngestTextRequest):
    """Ingests a markdown Q/A document. Each top-level (#) heading and the
    text below it (until the next # heading) becomes one chunk, so a
    question and its answer are always stored — and retrieved — together."""
    chunks = chunk_markdown_qa(req.text)
    if not chunks:
        return IngestTextResponse(chunks_ingested=0)

    try:
        vectors = embed_documents(chunks)
    except EmbeddingError as e:
        logger.error("ingest failed: %s", e)
        raise HTTPException(502, "failed to embed text") from e

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO documents (source, content, embedding) "
                "VALUES (%s, %s, %s)",
                [(req.source, c, v) for c, v in zip(chunks, vectors, strict=True)],
            )
    except Exception as e:
        logger.error("ingest DB write failed: %s", e)
        raise HTTPException(503, "database unavailable") from e

    return IngestTextResponse(chunks_ingested=len(chunks))


class DeleteSourceResponse(BaseModel):
    deleted: int


@app.delete(
    "/ingest/{source}",
    response_model=DeleteSourceResponse,
    dependencies=[Depends(require_api_key)],
)
def delete_source(source: str):
    try:
        with get_conn() as conn:
            result = conn.execute("DELETE FROM documents WHERE source = %s", (source,))
            deleted = result.rowcount
    except Exception as e:
        logger.error("delete failed: %s", e)
        raise HTTPException(503, "database unavailable") from e

    return DeleteSourceResponse(deleted=deleted)


class ChunkOut(BaseModel):
    id: int
    source: str
    content: str
    created_at: str


@app.get(
    "/documents",
    response_model=list[ChunkOut],
    dependencies=[Depends(require_api_key)],
)
def list_documents(
    source: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Lists stored chunks, most recent last. Filter with ?source=Titouan.md
    and page through results with ?limit=&offset=."""
    limit = max(1, min(limit, 500))  # guard against accidentally huge pulls
    offset = max(0, offset)

    query = "SELECT id, source, content, created_at FROM documents"
    params: list = []
    if source:
        query += " WHERE source = %s"
        params.append(source)
    query += " ORDER BY id LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    try:
        with get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
    except Exception as e:
        logger.error("list_documents failed: %s", e)
        raise HTTPException(503, "database unavailable") from e

    return [
        ChunkOut(
            id=r[0],
            source=r[1],
            content=r[2],
            created_at=r[3].isoformat(),
        )
        for r in rows
    ]


class SourceSummary(BaseModel):
    source: str
    chunk_count: int
    last_updated: str


@app.get(
    "/documents/sources",
    response_model=list[SourceSummary],
    dependencies=[Depends(require_api_key)],
)
def list_sources():
    """Summarizes what's in the database: each distinct source and how
    many chunks it currently has."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT source, COUNT(*) AS chunk_count, MAX(created_at) AS last_updated
                FROM documents
                GROUP BY source
                ORDER BY source
                """
            ).fetchall()
    except Exception as e:
        logger.error("list_sources failed: %s", e)
        raise HTTPException(503, "database unavailable") from e

    return [
        SourceSummary(source=r[0], chunk_count=r[1], last_updated=r[2].isoformat())
        for r in rows
    ]

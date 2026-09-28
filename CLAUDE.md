# AGENTS.md

Minimal FastAPI RAG backend: Postgres+pgvector, VoyageAI embeddings, Claude (Anthropic). See `README.md` for full endpoint/deploy docs; this file covers what's easy to miss.

## Run locally
```
pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL, ANTHROPIC_API_KEY, VOYAGE_API_KEY, INGEST_API_KEY
uvicorn app.main:app --reload
```
Needs a real Postgres with the `vector` extension (pgvector/pgvector Docker image, or point `DATABASE_URL` at Render's *external* connection string).

Or `docker compose up --build` — runs the app plus a `pgvector/pgvector` Postgres in one command. See README.md for details.

## Tests
`pytest` — all external services (Postgres, VoyageAI, Anthropic) are mocked in `tests/`, so no real DB or API keys are needed to run the suite. `pip install -r requirements-dev.txt` first.

## Architecture (`app/`)
- `main.py` — FastAPI routes only; auth/business logic lives elsewhere
- `deps.py` — `require_api_key()`, the shared auth dependency for `/ingest/*` and `/documents*`
- `chunking.py` — `chunk_markdown_qa()` splits on top-level `# ` headings only (`##`/`###` inside an answer do NOT split); each chunk = one question+answer together
- `embeddings.py` — VoyageAI; `embed_documents()` uses `input_type="document"`, `embed_query()` uses `input_type="query"` (asymmetric — don't mix them up); raises `EmbeddingError` on failure
- `db.py` — `init_db()` creates the `vector` extension + `documents`/`ingested_sources` tables; runs on every startup, safe to call repeatedly
- `rag.py` — retrieval (cosine distance via `<=>`) + Claude call; raises `AnswerGenerationError` on failure. `answer_question()` backs `/chat`; `stream_answer()` backs `/chat/stream` (SSE) — retrieval runs eagerly, the Claude call only when its text iterator is consumed, so its errors surface mid-stream as an in-band `error` event
- `auto_ingest.py` — re-ingests `settings.qa_markdown_path` (default `Titouan.md`) on every startup, but only if its SHA-256 hash changed since last run (tracked in `ingested_sources` table)

## Gotchas
- **Embeddings must be wrapped in pgvector's `Vector` type** before use as a query param — a raw Python list gets sent as `double precision[]`, which the `<=>` operator rejects (see `app/embeddings.py`).
- **`register_vector()` requires the `vector` extension to already exist** — `init_db()` creates the extension on a raw connection *before* calling `get_conn()` for anything else; don't reorder this.
- **`embedding_dim` in `app/config.py` must match the Voyage model's actual output dim.** Changing `voyage_model` without updating `embedding_dim` will break inserts/queries silently or with a dimension-mismatch error.
- **No approximate index (IVFFlat/HNSW) on `documents.embedding` by design** — exact scan is fine at hundreds of rows; IVFFlat was tried and removed because it hurt recall at this scale. Don't add one back without re-reading the comment in `app/db.py`.
- **Ingest is not idempotent for `/ingest/text` or `ingest.py`** — re-running inserts duplicate rows. Only `auto_ingest.py`'s startup path (content-hash based) replaces rows for a source. Clear first via `DELETE /ingest/{source}` if re-running manually.
- **`ingest.py` must be run against the external DB URL**, not the internal Render one (internal only resolves inside Render's network).
- All `/ingest/*` and `/documents*` endpoints require header `x-api-key: <INGEST_API_KEY>`, enforced by `app/deps.py`'s `require_api_key`. `INGEST_API_KEY` has no default — startup fails fast if it's unset, on purpose.
- CORS defaults to `allow_origins=["*"]` (set `CORS_ORIGINS`, comma-separated, in `.env`) — intentional default for dev only; must be locked down before real deployment.

## Deploy
`render.yaml` provisions both the web service and Postgres via Render Blueprint. Secrets (`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `INGEST_API_KEY`) are set manually in the Render dashboard, not in the repo.

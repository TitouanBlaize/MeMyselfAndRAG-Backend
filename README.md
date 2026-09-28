# Personal RAG Backend

A minimal RAG API: FastAPI + Postgres/pgvector (Render) + Voyage embeddings + Claude.

## Endpoints

- `GET /health` — liveness check
- `POST /chat` — `{"question": "..."}` → retrieves relevant chunks, asks Claude, returns `{answer, sources}`
- `POST /chat/stream` — same input, streams the answer as Server-Sent Events (see "Streaming the answer" below)
- `POST /ingest/text` — `{"source": "...", "text": "..."}` (header `x-api-key: <INGEST_API_KEY>`) → chunks, embeds, and stores text
- `DELETE /ingest/{source}` — removes all chunks for a given source (header `x-api-key: <INGEST_API_KEY>`)
- `GET /documents` — lists stored chunks (header `x-api-key: <INGEST_API_KEY>`); filter with `?source=`, page with `?limit=&offset=`
- `GET /documents/sources` — summarizes each distinct source with its chunk count and last update (header `x-api-key: <INGEST_API_KEY>`)

## Local development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys and a local Postgres URL
uvicorn app.main:app --reload
```

You'll need a local Postgres with the `vector` extension available (the
easiest way is the `pgvector/pgvector` Docker image), or just point
`DATABASE_URL` at your Render database's "External Connection String"
during development.

## Local development with Docker

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, INGEST_API_KEY
docker compose up --build
```

This builds the app image and starts a `pgvector/pgvector` Postgres
alongside it — no separate local Postgres install needed.
`docker-compose.yml` overrides `DATABASE_URL` to point at the `db` service,
so the value in your `.env` is ignored in this mode (it's only used for the
non-Docker path above). The schema is created automatically on startup via
the same `init_db()` hook used everywhere else — no separate migration
step. `docker compose down -v` resets the database volume if you want a
clean slate.

## Deploying to Render

1. Push this repo to GitHub.
2. In Render, choose **New > Blueprint** and point it at the repo —
   `render.yaml` provisions both the web service and the Postgres
   database in one go.
3. In the service's **Environment** tab, set the secret values Render
   left blank: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `INGEST_API_KEY`.
4. Deploy. Check `GET https://<your-service>.onrender.com/health`.

## Ingesting your documents

Content is ingested from a single **markdown Q/A file**, formatted like:

```markdown
# Question 1
Answer 1

# Question 2
Answer 2
```

Each top-level (`#`) heading and everything below it (up to the next `#`
heading) becomes one chunk — so a question is always stored, and
retrieved, together with its own answer. `##`/`###` headings inside an
answer are left alone and stay part of that answer's chunk.

Run the CLI script **from your own machine**, pointed at the database's
*external* connection string (find it on the database's page in the
Render dashboard — the internal one only resolves inside Render's
network):

```bash
export DATABASE_URL="<external connection string from Render>"
export VOYAGE_API_KEY="..."
python ingest.py qa.md
```

Re-running it will insert duplicate rows for unchanged questions —
either clear the source first with `DELETE /ingest/{source}` (source
will be the file path, e.g. `qa.md`) or add your own upsert logic if
you'll be editing the file repeatedly.

Or use the `/ingest/text` endpoint directly if you'd rather POST the
markdown content instead of running the script locally.

## Calling it from your frontend

```js
const res = await fetch("https://<your-service>.onrender.com/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "What did Antoine study?" }),
});
const { answer, sources } = await res.json();
```

### Streaming the answer

`POST /chat/stream` takes the same body but responds with Server-Sent
Events, so the answer can be displayed as Claude writes it:

| event     | data                              |
| --------- | --------------------------------- |
| `sources` | `[{source, similarity}, ...]` — sent first |
| `delta`   | `{"text": "..."}` — one per text chunk |
| `done`    | `{}` — answer complete            |
| `error`   | `{"detail": "..."}` — generation failed mid-stream (status is already 200) |

`EventSource` only supports GET, so read the body with `fetch`:

```js
const res = await fetch("https://<your-service>.onrender.com/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "What did Antoine study?" }),
});
const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
let buffer = "";
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += value;
  const events = buffer.split("\n\n");
  buffer = events.pop(); // keep the incomplete trailing event
  for (const raw of events) {
    const event = raw.match(/^event: (.*)$/m)[1];
    const data = JSON.parse(raw.match(/^data: (.*)$/m)[1]);
    if (event === "delta") answerEl.textContent += data.text;
    else if (event === "sources") renderSources(data);
    else if (event === "error") showError(data.detail);
  }
}
```

Remember to restrict CORS to your actual frontend domain before going
live — set `CORS_ORIGINS` (comma-separated) in your environment; it
defaults to `"*"`, which is fine for testing only.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

All external services (Postgres, VoyageAI, Anthropic) are mocked — no
real database or API keys are needed to run the suite.

## Notes & things to tune later

- **Free tier cold starts**: the free web service spins down after ~15
  min idle; first request after that takes 10–30s. Upgrade to a paid
  plan to avoid this, or ping `/health` periodically to keep it warm.
- **Chunking**: chunks are split at `#` headings in your markdown Q/A
  file, one question+answer per chunk. If some answers are very long,
  consider also enforcing a max chunk size so a single answer doesn't
  dominate the retrieved context.
- **Model names**: double-check the current Claude and Voyage model
  names/dimensions in each provider's docs before deploying — these
  change over time and `embedding_dim` in `app/config.py` must match
  whatever Voyage model you pick.

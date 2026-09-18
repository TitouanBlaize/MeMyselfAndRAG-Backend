from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings

# Render's internal Postgres URL works fine here; the pool keeps a handful
# of connections warm so requests don't pay connection-setup cost.
pool = ConnectionPool(
    conninfo=settings.database_url,
    min_size=1,
    max_size=5,
    kwargs={"autocommit": True},
)


@contextmanager
def get_conn():
    with pool.connection() as conn:
        register_vector(conn)
        yield conn


def init_db():
    """Create the extension and table if they don't exist yet.
    Safe to call on every startup.

    IMPORTANT: register_vector() (used by get_conn) looks up the 'vector'
    type's OID in pg_type, so it fails on a brand-new database where the
    extension hasn't been created yet. We create the extension first on a
    raw connection, then switch to get_conn() for everything else.
    """
    with pool.connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    with get_conn() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({settings.embedding_dim}) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        # Drop the IVFFlat index from earlier versions of this file — it
        # actively hurt recall at this row count (see comment below).
        conn.execute("DROP INDEX IF EXISTS documents_embedding_idx")

        # No approximate index (IVFFlat/HNSW) on purpose: at the scale of a
        # personal Q/A corpus (dozens to low hundreds of chunks), an exact
        # sequential scan is effectively instant, and IVFFlat's default of
        # probing only 1 of its `lists` buckets gives *worse* — sometimes
        # near-random — results when there are far more buckets than rows.
        # If this ever grows to 10k+ chunks, revisit with an HNSW index
        # (better recall than IVFFlat and doesn't need row-count tuning):
        #   CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingested_sources (
                source TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id BIGSERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT,
                error TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )

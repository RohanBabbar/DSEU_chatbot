"""Shared configuration, embedding and retrieval logic.

Everything that used to be copy-pasted across main.py, ingest.py, ingest_numbers.py
and test_retrieval.py lives here, so the debug tool can never drift from what the
server actually does.
"""
import os
import asyncio
from functools import lru_cache

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# --- Paths -------------------------------------------------------------------
PDF_PATH = os.path.join(BASE_DIR, "DSEU_Admission_Brochure_2026_updated 9.5.2026.pdf")
SHEET_PATH = os.path.join(BASE_DIR, "programs_updated.numbers")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# --- Database ----------------------------------------------------------------
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "chatbot")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DSN = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- LLM ---------------------------------------------------------------------
# The key was previously read from GEMINI_API_KEY; both names are accepted so an
# existing .env keeps working after the switch to OpenAI.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
PLACEHOLDER_KEYS = {"", "your_api_key_here", "your_openai_api_key_here"}

# --- Embeddings --------------------------------------------------------------
EMBED_MODEL_NAME = "all-mpnet-base-v2"
EMBED_DIM = 768

# all-mpnet-base-v2 silently truncates anything past 384 tokens, so every chunk
# must be built to fit. These budgets leave room for the context header that
# ingestion prepends to each chunk.
HEADER_TOKEN_RESERVE = 32
TEXT_MAX_TOKENS = 256
TEXT_OVERLAP_TOKENS = 48
TABLE_MAX_TOKENS = 288

SOURCE_BROCHURE = "brochure"
SOURCE_SHEET = "spreadsheet"

# Bump this whenever chunking, fee/campus resolution or the pathway chunks change.
# `ingest.py --if-empty` re-ingests when the stored version differs, so a tester who
# pulls new code cannot end up serving answers from chunks built by the old pipeline.
PIPELINE_VERSION = "2026-08-30.3"

_embedder = None


def get_embedder():
    """Loads the sentence-transformer once per process."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading local embedding model ({EMBED_MODEL_NAME})...")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
        dim_of = getattr(_embedder, "get_embedding_dimension", None) or \
            _embedder.get_sentence_embedding_dimension
        assert dim_of() == EMBED_DIM, f"expected {EMBED_DIM}-dim embeddings, got {dim_of()}"
    return _embedder


def max_embed_tokens() -> int:
    return get_embedder().max_seq_length


def count_tokens(text: str) -> int:
    """Token count as the embedding model sees it, including special tokens."""
    return len(get_embedder().tokenizer.encode(text, add_special_tokens=True))


def fits(text: str) -> bool:
    return count_tokens(text) <= max_embed_tokens()


def embed_many(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Batched encoding. Much faster than one call per chunk."""
    if not texts:
        return []
    vecs = get_embedder().encode(
        texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
    )
    return [v.tolist() for v in vecs]


@lru_cache(maxsize=512)
def _embed_cached(text: str) -> tuple[float, ...]:
    return tuple(embed_many([text])[0])


def embed_query(text: str) -> list[float]:
    """Embeds a search query, memoised so repeated questions are free."""
    return list(_embed_cached(text))


async def embed_query_async(text: str) -> list[float]:
    """Keeps the ~20ms of CPU work off the event loop."""
    return await asyncio.to_thread(embed_query, text)


def to_pgvector(vec) -> str:
    """pgvector accepts its text form, '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"


# --- Schema ------------------------------------------------------------------
# Mirrors db/init.sql. init.sql only runs on a brand-new volume, so ingestion
# applies the same statements itself to keep existing databases in step.
SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_text TEXT NOT NULL,
    section TEXT,
    is_table BOOLEAN DEFAULT false,
    page_number INT,
    source TEXT DEFAULT '{SOURCE_BROCHURE}',
    embedding VECTOR({EMBED_DIM}),
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
);

-- Empty chunks are meaningless but still carry a unit-norm vector that competes
-- in nearest-neighbour search. Make them impossible rather than merely unlikely.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'document_chunks_text_not_blank'
    ) THEN
        DELETE FROM document_chunks WHERE length(btrim(chunk_text)) = 0;
        ALTER TABLE document_chunks
            ADD CONSTRAINT document_chunks_text_not_blank
            CHECK (length(btrim(chunk_text)) > 0);
    END IF;
END $$;

-- Cosine index, matched to the '<=>' operator used by the search query. The
-- vectors are unit-norm, so cosine and L2 rank identically; using the operator
-- the index was built for is what lets the index actually get used.
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx
    ON document_chunks USING GIN (tsv);

CREATE INDEX IF NOT EXISTS document_chunks_source_idx
    ON document_chunks (source);

CREATE TABLE IF NOT EXISTS ingest_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


async def ensure_schema(conn):
    await conn.execute(SCHEMA_SQL)


async def get_meta(conn, key: str) -> str | None:
    return await conn.fetchval("SELECT value FROM ingest_meta WHERE key = $1", key)


async def set_meta(conn, key: str, value: str) -> None:
    await conn.execute(
        """
        INSERT INTO ingest_meta (key, value) VALUES ($1, $2)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """,
        key, value,
    )


# --- Retrieval ---------------------------------------------------------------
# Hybrid search: vector k-NN and full-text search, fused with Reciprocal Rank
# Fusion. Each branch selects ids only and ranks with ROW_NUMBER() over an
# already-limited subquery -- ranking inside the branch with a window function
# over the whole table would force a full sort and defeat the HNSW/GIN indexes.
SEARCH_SQL = """
WITH q AS (
    -- websearch_to_tsquery ANDs every term, so a natural-language question
    -- ("how do I get a BBA degree?") matches no chunk at all and the full-text
    -- half of the hybrid contributes nothing. Re-joining the query's own lexemes
    -- with OR lets ts_rank do the discriminating instead of an all-or-nothing
    -- filter. Lexemes come from to_tsvector and are quoted, so this is safe.
    SELECT to_tsquery('english', nullif(string_agg(quote_literal(lexeme), ' | '), '')) AS tsq
    FROM unnest(to_tsvector('english', $2))
),
vector_search AS (
    SELECT id, ROW_NUMBER() OVER () AS rank
    FROM (
        SELECT id
        FROM document_chunks
        ORDER BY embedding <=> $1::vector
        LIMIT $4
    ) v
),
fts_search AS (
    SELECT id, ROW_NUMBER() OVER () AS rank
    FROM (
        SELECT c.id
        FROM document_chunks c, q
        WHERE q.tsq IS NOT NULL AND c.tsv @@ q.tsq
        ORDER BY ts_rank(c.tsv, q.tsq) DESC
        LIMIT $4
    ) f
)
SELECT c.chunk_text, c.section, c.is_table, c.page_number, c.source,
       COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + f.rank), 0.0) AS rrf_score
FROM vector_search v
FULL OUTER JOIN fts_search f ON v.id = f.id
JOIN document_chunks c ON c.id = COALESCE(v.id, f.id)
ORDER BY rrf_score DESC
LIMIT $3;
"""


async def search(conn, query: str, top_k: int = 8, pool: int = 25) -> list[dict]:
    """Runs hybrid search and returns the fused top-k chunks."""
    vec = await embed_query_async(query)
    rows = await conn.fetch(SEARCH_SQL, to_pgvector(vec), query, top_k, pool)
    return [dict(r) for r in rows]


def source_label(row) -> str:
    """Human-readable provenance for a retrieved chunk."""
    if row["source"] == SOURCE_SHEET or not row["page_number"]:
        return "Programs Spreadsheet"
    return f"Brochure Page {row['page_number']}"


def build_context(rows: list[dict]) -> tuple[str, list[str]]:
    """Formats retrieved chunks for the prompt and collects their sources."""
    parts, sources = [], []
    for row in rows:
        label = source_label(row)
        if label not in sources:
            sources.append(label)
        kind = "TABLE" if row["is_table"] else "TEXT"
        parts.append(
            f"--- START {kind} CHUNK ({label}) ---\n"
            f"{row['chunk_text']}\n"
            f"--- END CHUNK ---"
        )
    sources.sort(key=lambda s: (s != "Programs Spreadsheet", s))
    return "\n\n".join(parts), sources

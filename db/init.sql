-- Runs only when the Postgres volume is created for the first time.
-- backend/core.py applies the same statements on every ingest, so an existing
-- database picks up changes too.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_text TEXT NOT NULL,
    section TEXT,
    is_table BOOLEAN DEFAULT false,
    page_number INT,
    source TEXT DEFAULT 'brochure',
    embedding VECTOR(768),
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    -- An empty chunk carries a unit-norm embedding that still competes in
    -- nearest-neighbour search, so reject it outright.
    CONSTRAINT document_chunks_text_not_blank CHECK (length(btrim(chunk_text)) > 0)
);

-- Cosine index, matching the '<=>' operator the search query uses. Embeddings are
-- unit-norm so cosine and L2 rank identically; the point of matching the operator
-- is that the index actually gets used instead of being skipped for a full scan.
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text search half of the hybrid retrieval.
CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx
    ON document_chunks USING GIN (tsv);

-- Ingestion replaces one source at a time rather than truncating the table.
CREATE INDEX IF NOT EXISTS document_chunks_source_idx
    ON document_chunks (source);

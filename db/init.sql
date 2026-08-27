CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_text TEXT NOT NULL,
    section TEXT,
    is_table BOOLEAN DEFAULT false,
    page_number INT,
    source TEXT DEFAULT 'brochure',
    embedding VECTOR(768),
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
);

-- Index for vector similarity search (using HNSW for performance)
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Index for full-text search
CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx ON document_chunks USING GIN (tsv);

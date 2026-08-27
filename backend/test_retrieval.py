import os
import asyncio
import asyncpg
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_DB = os.getenv("POSTGRES_DB", "chatbot")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DSN = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_DB}"

print("Loading local embedding model (all-mpnet-base-v2)...")
embedder = SentenceTransformer('all-mpnet-base-v2')

async def debug_retrieval(query: str, top_k: int = 10):
    """Bypasses Gemini and just prints exactly what the database finds."""
    print(f"\n--- SEARCHING DATABASE FOR: '{query}' ---\n")
    
    query_vector = embedder.encode(query).tolist()
    sql = """
    WITH vector_search AS (
        SELECT id, chunk_text, section, is_table, page_number,
               RANK() OVER (ORDER BY embedding <-> $1) AS vector_rank
        FROM document_chunks
        ORDER BY embedding <-> $1
        LIMIT 20
    ),
    fts_search AS (
        SELECT id, chunk_text, section, is_table, page_number,
               RANK() OVER (ORDER BY ts_rank(tsv, websearch_to_tsquery('english', $2)) DESC) AS fts_rank
        FROM document_chunks
        WHERE tsv @@ websearch_to_tsquery('english', $2)
        ORDER BY fts_rank
        LIMIT 20
    )
    SELECT
        COALESCE(vs.id, fs.id) AS id,
        COALESCE(vs.chunk_text, fs.chunk_text) AS chunk_text,
        COALESCE(vs.is_table, fs.is_table) AS is_table,
        COALESCE(vs.page_number, fs.page_number) AS page_number,
        COALESCE(1.0 / (60 + vs.vector_rank), 0.0) + COALESCE(1.0 / (60 + fs.fts_rank), 0.0) AS rrf_score
    FROM vector_search vs
    FULL OUTER JOIN fts_search fs ON vs.id = fs.id
    ORDER BY rrf_score DESC
    LIMIT $3;
    """
    
    conn = await asyncpg.connect(DSN)
    rows = await conn.fetch(sql, str(query_vector), query, top_k)
    await conn.close()
    
    if not rows:
        print("NO RESULTS FOUND!")
        return

    for i, row in enumerate(rows):
        source = "SPREADSHEET" if row['page_number'] == 0 else f"PDF PAGE {row['page_number']}"
        chunk_preview = row['chunk_text'][:300].replace('\n', ' ') + "..."
        print(f"[{i+1}] Source: {source} | Score: {row['rrf_score']:.4f}")
        print(f"    Preview: {chunk_preview}\n")

if __name__ == "__main__":
    import sys
    # Let the user run: python test_retrieval.py "my question"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = input("Enter a test question: ")
    asyncio.run(debug_retrieval(query))

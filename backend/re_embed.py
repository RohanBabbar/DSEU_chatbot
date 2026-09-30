import asyncio
import asyncpg
import os
import math
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "chatbot")
DB_HOST = "127.0.0.1"
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DSN = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# B.4: Re-embed the existing corpus
async def re_embed_chunks():
    print("Loading BAAI/bge-m3 model...")
    model = SentenceTransformer("BAAI/bge-m3")
    
    print(f"Connecting to database at {DSN}...")
    conn = await asyncpg.connect(DSN)
    
    # Check how many need embedding
    total = await conn.fetchval("SELECT count(*) FROM document_chunks WHERE embedding_multilingual IS NULL")
    if total == 0:
        print("All chunks already have multilingual embeddings.")
    else:
        print(f"Found {total} chunks missing multilingual embeddings. Processing...")
        
        # Process in batches
        BATCH_SIZE = 32
        cursor = 0
        while True:
            rows = await conn.fetch(
                "SELECT id, chunk_text FROM document_chunks WHERE embedding_multilingual IS NULL LIMIT $1",
                BATCH_SIZE
            )
            if not rows:
                break
                
            ids = [row["id"] for row in rows]
            texts = [row["chunk_text"] for row in rows]
            
            # Embed
            embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
            
            # Update database
            for i, chunk_id in enumerate(ids):
                vec_str = "[" + ",".join(f"{float(x):.7f}" for x in embeddings[i]) + "]"
                await conn.execute("UPDATE document_chunks SET embedding_multilingual = $1::vector WHERE id = $2", vec_str, chunk_id)
                
            cursor += len(rows)
            print(f"Processed {cursor}/{total}...")
            
    print("Verification:")
    total_old = await conn.fetchval("SELECT count(*) FROM document_chunks WHERE embedding IS NOT NULL")
    total_new = await conn.fetchval("SELECT count(*) FROM document_chunks WHERE embedding_multilingual IS NOT NULL")
    print(f"Old embeddings: {total_old}, Multilingual embeddings: {total_new}")
    
    # B.5: Rebuild the vector index
    print("Building HNSW index on embedding_multilingual...")
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS document_chunks_multilingual_idx
        ON document_chunks USING hnsw (embedding_multilingual vector_cosine_ops);
    """)
    print("Index built successfully.")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(re_embed_chunks())

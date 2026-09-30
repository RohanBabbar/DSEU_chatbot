import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "chatbot")
DB_HOST = "127.0.0.1" # Connecting from Mac host to docker container
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DSN = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

async def main():
    print(f"Connecting to {DSN}...")
    conn = await asyncpg.connect(DSN)
    print("Adding embedding_multilingual vector(1024) column...")
    await conn.execute("""
        ALTER TABLE document_chunks
        ADD COLUMN IF NOT EXISTS embedding_multilingual VECTOR(1024);
    """)
    print("Schema update complete.")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())

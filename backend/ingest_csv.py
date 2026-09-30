import os
import csv
import asyncio
import asyncpg
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_DB = os.getenv("POSTGRES_DB", "chatbot")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DSN = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_DB}"

print("Loading local embedding model (all-mpnet-base-v2)...")
embedder = SentenceTransformer('all-mpnet-base-v2')

async def insert_chunk(conn, chunk_text: str, source_name: str):
    """Inserts a chunk and its vector into the database."""
    vec = embedder.encode(chunk_text).tolist()
    await conn.execute(
        '''
        INSERT INTO document_chunks (chunk_text, is_table, page_number, section, source, embedding)
        VALUES ($1, $2, $3, $4, $5, $6)
        ''',
        chunk_text, False, -1, None, source_name, str(vec)
    )

def csv_to_text(file_path: str, source_name: str) -> list[str]:
    chunks = []
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}, file not found.")
        return chunks
        
    with open(file_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert the row dictionary into a readable text chunk
            text_parts = [f"Data from {source_name}:"]
            for key, value in row.items():
                if value and value.strip() and value.strip() != 'NULL':
                    clean_key = key.replace('_', ' ').title()
                    text_parts.append(f"{clean_key}: {value.strip()}")
            chunks.append("\n".join(text_parts))
    return chunks

async def main():
    conn = await asyncpg.connect(DSN)
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    # List of CSV files and their clean source names
    csv_files = {
        'programs.csv': 'Programs CSV',
        'tutiton_fee_categories.csv': 'Tuition Fee Categories CSV',
        'fee_rebates.csv': 'Fee Rebates CSV',
        'other_fees.csv': 'Other Fees CSV',
        'exit_awards.csv': 'Exit Awards CSV'
    }
    
    for filename, source_name in csv_files.items():
        file_path = os.path.join(base_dir, filename)
        print(f"Processing {filename}...")
        chunks = csv_to_text(file_path, source_name)
        
        for chunk in chunks:
            await insert_chunk(conn, chunk, source_name)
            
        print(f"  -> Inserted {len(chunks)} rows from {filename}")
        
    await conn.close()
    print("CSV ingestion complete!")

if __name__ == "__main__":
    asyncio.run(main())

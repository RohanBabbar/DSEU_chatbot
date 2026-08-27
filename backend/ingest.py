import os
import sys
import asyncio
import asyncpg
import pymupdf
import pandas as pd
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configuration
# Resolving path assuming script is run from project root or backend folder
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PDF_PATH = os.path.join(BASE_DIR, "DSEU_Admission_Brochure_2026_updated 9.5.2026.pdf")

if not os.path.exists(PDF_PATH):
    print(f"ERROR: Could not find PDF at {PDF_PATH}")
    sys.exit(1)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    print("ERROR: Please set a valid GEMINI_API_KEY in the .env file.")
    sys.exit(1)

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# Database connection settings
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_DB = os.getenv("POSTGRES_DB", "chatbot")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DSN = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_DB}"

import time
from sentence_transformers import SentenceTransformer

# Initialize local embedding model (768 dimensions)
print("Loading local embedding model (all-mpnet-base-v2)...")
embedder = SentenceTransformer('all-mpnet-base-v2')

async def get_embedding(text: str) -> list[float]:
    """Generates a 768-dimensional embedding vector locally."""
    # sentence-transformers returns a numpy array, convert to list
    return embedder.encode(text).tolist()

async def insert_chunk(conn, chunk_text: str, is_table: bool, page_num: int, embedding_text: str):
    """Generates an embedding for embedding_text, but stores chunk_text in the database."""
    try:
        # 1. Generate Vector Embedding Locally
        vec = await get_embedding(embedding_text)
        
        # 2. Insert into PostgreSQL
        await conn.execute(
            '''
            INSERT INTO document_chunks (chunk_text, is_table, page_number, embedding)
            VALUES ($1, $2, $3, $4)
            ''',
            chunk_text, is_table, page_num, str(vec)
        )
        print(f"  -> Inserted {'table' if is_table else 'text'} chunk (Page {page_num})")
    except Exception as e:
        print(f"  -> Error inserting chunk from page {page_num}: {e}")

async def process_pdf():
    print(f"Opening PDF: {PDF_PATH}")
    doc = pymupdf.open(PDF_PATH)
    
    # Connect to the Database
    print("Connecting to database...")
    conn = await asyncpg.connect(DSN)
    
    # Clear existing data so we don't duplicate if we run the script multiple times
    await conn.execute("TRUNCATE TABLE document_chunks;")
    print("Cleared existing document chunks.\n")

    # Process page by page
    for page_num in range(len(doc)):
        page = doc[page_num]
        actual_page = page_num + 1
        print(f"Processing Page {actual_page}/{len(doc)}...")
        
        # --- 1. EXTRACT AND PROCESS TABLES ---
        tabs = page.find_tables()
        if tabs and tabs.tables:
            for i, t in enumerate(tabs.tables):
                df = t.to_pandas()
                
                # Clean up the dataframe (remove completely empty columns/rows caused by PDF formatting)
                df.dropna(how='all', inplace=True)
                df.dropna(how='all', axis=1, inplace=True)
                
                if df.empty:
                    continue
                
                # Convert to Markdown so it's readable
                markdown_table = df.to_markdown(index=False)
                
                print(f"  -> Found Table {i+1}. Inserting raw markdown...")
                try:
                    # Insert (Store and embed raw markdown to bypass API limits)
                    await insert_chunk(conn, chunk_text=markdown_table, is_table=True, page_num=actual_page, embedding_text=markdown_table)
                except Exception as e:
                     print(f"  -> Failed to insert table: {e}")

        # --- 2. EXTRACT AND PROCESS TEXT ---
        text = page.get_text()
        
        # Basic chunking: split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        
        # Combine short paragraphs into larger chunks (~1000 chars) to give the LLM enough context
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) < 1000:
                current_chunk += p + "\n"
            else:
                await insert_chunk(conn, chunk_text=current_chunk.strip(), is_table=False, page_num=actual_page, embedding_text=current_chunk.strip())
                current_chunk = p + "\n"
        
        if current_chunk.strip():
            await insert_chunk(conn, chunk_text=current_chunk.strip(), is_table=False, page_num=actual_page, embedding_text=current_chunk.strip())

    await conn.close()
    print("\n✅ Ingestion complete! The database is now populated.")

if __name__ == "__main__":
    # Run the async ingestion pipeline
    asyncio.run(process_pdf())

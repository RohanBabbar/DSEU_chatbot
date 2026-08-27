import os
import json
import asyncio
import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_DB = os.getenv("POSTGRES_DB", "chatbot")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DSN = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_DB}"

# Initialize models
print("Loading local embedding model (all-mpnet-base-v2)...")
embedder = SentenceTransformer('all-mpnet-base-v2')
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

# Connection pool for the database
db_pool = None

@app.on_event("startup")
async def startup_event():
    global db_pool
    db_pool = await asyncpg.create_pool(DSN)

@app.on_event("shutdown")
async def shutdown_event():
    await db_pool.close()

# Request Models
class Message(BaseModel):
    role: str # 'user' or 'model'
    content: str

class ChatRequest(BaseModel):
    query: str
    history: list[Message] = []

async def retrieve_context(query: str, top_k: int = 10) -> dict:
    """Performs Hybrid Search and returns both the context string and a list of sources."""
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
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, str(query_vector), query, top_k)
    
    if not rows:
        return {"context_string": "", "sources": []}
    
    context_parts = []
    sources = []
    
    for row in rows:
        source_type = "TABLE" if row['is_table'] else "TEXT"
        source_name = "Programs Spreadsheet" if row['page_number'] == 0 else f"Brochure Page {row['page_number']}"
        
        if source_name not in sources:
            sources.append(source_name)
            
        context_parts.append(
            f"--- START {source_type} CHUNK ({source_name}) ---\n"
            f"{row['chunk_text']}\n"
            f"--- END CHUNK ---"
        )
    
    return {
        "context_string": "\n\n".join(context_parts),
        "sources": sources
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Retrieve relevant context from the database
        retrieval_data = await retrieve_context(request.query)
        context_str = retrieval_data["context_string"]
        sources = retrieval_data["sources"]
        
        # 2. Build the system prompt
        system_prompt = (
            "You are a friendly, helpful admissions assistant for DSEU (Delhi Skill and Entrepreneurship University). "
            "Your job is to answer prospective students' questions based STRICTLY on the provided chunks below, which are drawn from the Official Admission Brochure and the Updated Programs Spreadsheet. "
            "IMPORTANT INSTRUCTIONS:\n"
            "- Carefully read ALL provided chunks before answering. The information may be spread out.\n"
            "- If you see data that is labelled as 'Updated Program Information' or 'Program Summary Fact', treat it as the most recent and accurate data.\n"
            "- If you find partial information (e.g. general criteria but not branch-specific), provide the partial information you DO have rather than saying it's missing.\n"
            "- If the provided context contains tables (formatted in Markdown), carefully read the rows and columns to find the exact data.\n"
            "- If the answer is truly NOT found in the context below at all, gracefully inform the user that you don't have that specific information in your current documents.\n"
            "- Do NOT make up information or use outside knowledge.\n\n"
            "CONTEXT DOCUMENTS:\n"
            f"{context_str}"
        )
        
        # 3. Format chat history for Gemini
        gemini_contents = [
            {"role": "user", "parts": [{"text": system_prompt}]}
        ]
        
        gemini_contents.append({"role": "model", "parts": [{"text": "Understood. I will act as the DSEU admissions assistant and only use the provided context."}]})
        
        for msg in request.history:
            gemini_contents.append({
                "role": "user" if msg.role == 'user' else "model",
                "parts": [{"text": msg.content}]
            })
            
        gemini_contents.append({"role": "user", "parts": [{"text": request.query}]})
        
        # 4. Generate Answer
        response = gemini_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=gemini_contents
        )
        
        return {
            "answer": response.text,
            "sources": sources if response.text and "I don't have" not in response.text else []
        }
        
    except Exception as e:
        print(f"Error during chat generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend
frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir)

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dir, 'index.html'))

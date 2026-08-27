import os
import asyncio
import asyncpg
from numbers_parser import Document
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

async def insert_chunk(conn, chunk_text: str):
    """Inserts a chunk and its vector into the database."""
    vec = embedder.encode(chunk_text).tolist()
    # Using page_number = 0 to signify it comes from the spreadsheet
    await conn.execute(
        '''
        INSERT INTO document_chunks (chunk_text, is_table, page_number, embedding)
        VALUES ($1, $2, $3, $4)
        ''',
        chunk_text, False, 0, str(vec)
    )

async def main():
    conn = await asyncpg.connect(DSN)
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'programs_updated.numbers')
    doc = Document(file_path)
    
    print("Processing Programs Sheet...")
    programs_sheet = doc.sheets[0]
    programs_table = programs_sheet.tables[0]
    
    # We will group by Program Name
    current_program = None
    current_options = None
    current_chunk = ""
    
    rows = programs_table.rows()
    # Skip header
    for i in range(1, len(rows)):
        row_data = [cell.value if cell else None for cell in rows[i]]
        if not any(row_data):
            continue # Skip empty rows
            
        program_name = row_data[1]
        exit_options = row_data[2]
        year = row_data[3]
        exit_text = row_data[4]
        
        # Forward fill program name and exit options
        if program_name:
            # If we hit a new program, save the old one
            if current_program and current_chunk:
                print(f"  -> Inserting chunk for {current_program}")
                await insert_chunk(conn, current_chunk)
                
            current_program = program_name
            current_options = exit_options
            current_chunk = f"Updated Program Information:\nProgram Name: {current_program}\nGeneral Exit Options:\n{current_options}\n\nSpecific Year Exits:\n"
            
        if current_program and year and exit_text:
            current_chunk += f"- {year}: {exit_text}\n"
            
    # Insert the final program
    if current_program and current_chunk:
        print(f"  -> Inserting chunk for {current_program}")
        await insert_chunk(conn, current_chunk)

    print("\nProcessing Summary Sheet...")
    if len(doc.sheets) > 1:
        summary_sheet = doc.sheets[1]
        summary_table = summary_sheet.tables[0]
        rows = summary_table.rows()
        # Find header row (which has 'Program Name')
        header_idx = 1
        for i, row in enumerate(rows):
            vals = [c.value if c else "" for c in row]
            if "Program Name" in vals:
                header_idx = i
                break
                
        for i in range(header_idx + 1, len(rows)):
            row_data = [cell.value if cell else None for cell in rows[i]]
            if len(row_data) >= 6 and row_data[1]:
                prog_name = row_data[1]
                level = row_data[2]
                duration = row_data[3]
                mode = row_data[4]
                total_exits = row_data[5]
                
                chunk = f"Program Summary Fact:\nProgram Name: {prog_name}\nLevel: {level}\nDuration: {duration}\nMode: {mode}\nTotal Exit Options: {total_exits}"
                await insert_chunk(conn, chunk)
                print(f"  -> Inserting summary for {prog_name}")
                
    await conn.close()
    print("Spreadsheet ingestion complete!")

if __name__ == "__main__":
    asyncio.run(main())

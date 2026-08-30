"""FastAPI server for the DSEU admissions chatbot."""
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIStatusError, AsyncOpenAI
from pydantic import BaseModel, Field

import core
from core import FRONTEND_DIR, OPENAI_API_KEY, OPENAI_MODEL, PLACEHOLDER_KEYS

TOP_K = 10
MAX_HISTORY_MESSAGES = 8  # 4 exchanges; keeps the request small and the cost flat

SYSTEM_PROMPT = """You are a friendly, helpful admissions assistant for DSEU \
(Delhi Skill and Entrepreneurship University). Answer prospective students' \
questions using ONLY the context documents provided below, which are drawn from \
the Official Admission Brochure and the Updated Programs Spreadsheet.

Instructions:
- Read ALL the provided chunks before answering. Relevant details are often split \
across several chunks.
- USE BOTH SOURCES TOGETHER. The brochure and the spreadsheet are complementary, \
not alternatives: the spreadsheet carries exit options and updated program data, \
the brochure carries fees, eligibility, reservation, campuses and intake. A good \
answer combines everything the context offers on the question. Only when the two \
directly contradict each other on the same fact does the spreadsheet win \
(chunks labelled 'Updated Program Information', 'Program Summary Fact', 'Updated \
Program Catalogue' or 'Exit Qualification Pathway'). Never present one source as \
more "official" than the other.
- DSEU programs run four years with multiple exit points: a student may leave \
early and receive a lower qualification instead of continuing. So a qualification \
is OFFERED by DSEU if it appears as an exit option, even when no program carries \
that name. Never say DSEU does not offer a qualification that appears in an \
'Exit Qualification Pathway' chunk.
- When someone names a qualification or says what they want to study (for example \
"I want to do BCA", "how do I get a BBA"), give them the complete picture from \
whatever the context contains, as a short labelled list:
  * the ways to get in, as a numbered list under the heading "Routes", the direct \
route always first. An 'Exit Qualification Pathway' chunk labels them \
"ROUTE 1 - DIRECT" and "ROUTE 2 - EXIT OPTION"; keep that order and that split, \
but never copy those internal labels, or any wording from these instructions, \
into your answer. Write them as:
      "Option 1 - Direct: apply to the <name> program (<duration>)."
      "Option 2 - Exit option: enrol in <program> (4 years) and exit after Year \
<n>, receiving <qualification>."
    List every program under each option. If the context shows no directly-named \
program for that qualification, say so explicitly as Option 1 rather than \
silently dropping it, then give the exit route as Option 2. Present both options \
as equally valid choices and never call one the "official" one.
  * the fee: quote the 'Program Fee' chunk for that program -- the fee category \
AND the rupee amount with its billing period.
  * eligibility, duration and level, and campuses or intake if present.
  * which campuses offer it and the intake, from the 'Campus Availability' \
chunk for that program -- that chunk lists EVERY campus, so give all of them, \
not just one.
  Name explicitly anything they would want that the context does not cover.
- Never present a guess as a value. Do not write "assumed", "likely" or "probably" \
next to a fee, campus, intake or date. If the context does not give that value for \
the specific program asked about, say it is not listed in your documents.
- For any question about fees, always give the actual rupee amount, not just the \
fee category letter. A 'Program Fee' chunk already pairs the category with its \
amount; use it. Never guess an amount for a program whose fee chunk is absent.
- Each chunk begins with a bracketed header such as [Brochure page 42 · Fee \
Structure]. Use it to locate information, but do not quote the header back.
- Tables are given in Markdown. Read along the row and match the column heading \
carefully before quoting any number.
- If you only have partial information, give what you have and say plainly which \
part you could not find. Do not pad it out with guesses.
- If the answer is genuinely absent from the context, begin your reply with the \
exact tag {no_answer} on its own, then say you do not have that information in \
your current documents and suggest they check with the admissions office. Use the \
tag only when the context gave you nothing usable -- not when you found a partial \
answer.
- Never use knowledge from outside the context. Never invent fees, dates, \
percentages or eligibility rules.

CONTEXT DOCUMENTS:
{context}"""

# Citing pages under an "I don't know" is misleading, so the model flags that case
# explicitly. Matching on the prose instead would break the moment it rephrases.
NO_ANSWER_TAG = "[NO_ANSWER]"

NO_CONTEXT_ANSWER = (
    "I could not find anything about that in the admission brochure or the "
    "programs list. Could you rephrase it, or check with the DSEU admissions office?"
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: list[Message] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not OPENAI_API_KEY or OPENAI_API_KEY in PLACEHOLDER_KEYS:
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY in the .env file "
            "(see .env.example)."
        )

    app.state.pool = await asyncpg.create_pool(core.DSN, min_size=1, max_size=8)
    app.state.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Load the embedding model now so the first question is not slow.
    core.get_embedder()
    core.embed_query("warmup")

    async with app.state.pool.acquire() as conn:
        app.state.chunk_count = await conn.fetchval("SELECT count(*) FROM document_chunks")
    if not app.state.chunk_count:
        print("WARNING: document_chunks is empty -- run 'python backend/ingest.py' first.")
    else:
        print(f"Retrieval corpus: {app.state.chunk_count} chunks")

    # Verify the key and model up front, so a misconfiguration surfaces at boot
    # rather than as a 500 on every question.
    app.state.llm_error = None
    try:
        ids = {m.id async for m in (await app.state.client.models.list())}
        if OPENAI_MODEL not in ids:
            app.state.llm_error = (
                f"Model '{OPENAI_MODEL}' is not available to this API key. "
                f"Set OPENAI_MODEL in .env to one of, for example: "
                f"{', '.join(sorted(i for i in ids if i.startswith('gpt-4.1'))[:4])}"
            )
        else:
            print(f"LLM ready: {OPENAI_MODEL}")
    except Exception as exc:  # network down, bad key, revoked key
        app.state.llm_error = f"Could not reach the OpenAI API: {type(exc).__name__}: {exc}"
    if app.state.llm_error:
        print(f"WARNING: {app.state.llm_error}")

    try:
        yield
    finally:
        await app.state.pool.close()


app = FastAPI(title="DSEU Admissions Chatbot", lifespan=lifespan)


async def complete(client: AsyncOpenAI, messages: list[dict]) -> str:
    """Calls the model, retrying without params the chosen model rejects."""
    kwargs = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.1,  # grounded extraction, not creative writing
        "max_completion_tokens": 1200,
    }
    for _ in range(3):
        try:
            response = await client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()
        except APIStatusError as exc:
            detail = str(exc)
            dropped = next(
                (p for p in ("temperature", "max_completion_tokens")
                 if p in kwargs and p in detail and "unsupported" in detail.lower()),
                None,
            )
            if not dropped:
                raise
            kwargs.pop(dropped)
    raise RuntimeError("Could not find a parameter set this model accepts")


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if app.state.llm_error:
        raise HTTPException(status_code=503, detail=app.state.llm_error)

    async with app.state.pool.acquire() as conn:
        rows = await core.search(conn, request.query, top_k=TOP_K)

    if not rows:
        return {"answer": NO_CONTEXT_ANSWER, "sources": []}

    context, sources = core.build_context(rows)
    system = SYSTEM_PROMPT.format(context=context, no_answer=NO_ANSWER_TAG)
    messages = [{"role": "system", "content": system}]
    for message in request.history[-MAX_HISTORY_MESSAGES:]:
        role = "assistant" if message.role in ("model", "assistant") else "user"
        messages.append({"role": role, "content": message.content})
    messages.append({"role": "user", "content": request.query})

    try:
        answer = await complete(app.state.client, messages)
    except Exception as exc:
        print(f"Error during chat generation: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="The language model call failed.")

    if not answer:
        return {"answer": NO_CONTEXT_ANSWER, "sources": []}
    if answer.startswith(NO_ANSWER_TAG):
        return {"answer": answer[len(NO_ANSWER_TAG):].lstrip(), "sources": []}
    return {"answer": answer, "sources": sources}


@app.get("/api/health")
async def health():
    async with app.state.pool.acquire() as conn:
        by_source = await conn.fetch(
            "SELECT source, count(*) AS n FROM document_chunks GROUP BY source ORDER BY source"
        )
    return {
        "database": "ok",
        "chunks": {r["source"]: r["n"] for r in by_source},
        "total_chunks": sum(r["n"] for r in by_source),
        "embedding_model": core.EMBED_MODEL_NAME,
        "llm_model": OPENAI_MODEL,
        "llm": "ok" if not app.state.llm_error else app.state.llm_error,
    }


os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

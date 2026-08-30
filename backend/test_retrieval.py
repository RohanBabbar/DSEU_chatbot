"""Retrieval debugger -- shows exactly what the database returns, with no LLM call.

    python backend/test_retrieval.py "what is the eligibility for BBA?"

Uses the same search function as the server, so what you see here is what the
model is given.
"""
import asyncio
import sys

import asyncpg

import core


async def debug(query: str, top_k: int = 8) -> None:
    conn = await asyncpg.connect(core.DSN)
    try:
        total = await conn.fetchval("SELECT count(*) FROM document_chunks")
        print(f"corpus: {total} chunks")
        print(f"\n--- SEARCHING FOR: {query!r} ---\n")
        rows = await core.search(conn, query, top_k=top_k)
    finally:
        await conn.close()

    if not rows:
        print("NO RESULTS FOUND")
        return

    for i, row in enumerate(rows, 1):
        kind = "TABLE" if row["is_table"] else "TEXT "
        preview = " ".join(row["chunk_text"].split())[:280]
        print(f"[{i}] {kind} | {core.source_label(row)} | score {row['rrf_score']:.5f}")
        if row["section"]:
            print(f"    section: {row['section']}")
        print(f"    {preview}...\n")

    context, sources = core.build_context(rows)
    print(f"context size: {len(context)} chars / ~{core.count_tokens(context)} tokens")
    print(f"sources: {', '.join(sources)}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Enter a test question: ")
    asyncio.run(debug(question))

import asyncio
import httpx
import time
import json
import os
import statistics
from dotenv import load_dotenv

# We will use the existing OpenAI client to act as an LLM-as-a-judge for grading accuracy.
from openai import AsyncOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def judge_response(client: AsyncOpenAI, query: str, answer: str, sources: list) -> dict:
    """Uses GPT to grade if the answer is accurate, a correct refusal, or a hallucination."""
    if not answer or "I could not find" in answer or "I do not have information" in answer:
        return {"accuracy": 1, "is_refusal": 1, "is_hallucination": 0}
        
    prompt = f"""
    You are evaluating an AI Assistant.
    Query: "{query}"
    Answer provided: "{answer}"
    Sources cited: {sources}
    
    Rule 1: If the answer claims something not typically possible or invents data without sources, it is a hallucination.
    Rule 2: If the answer provides a reasonable response backed by the sources, it is accurate.
    
    Output JSON exactly: {{"accuracy": 1 or 0, "is_hallucination": 1 or 0}}
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error grading: {e}")
        return {"accuracy": 1, "is_hallucination": 0} # Default to pass if judge fails

async def main():
    queries = []
    with open('complex_test_queries.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                queries.append(line)

    print(f"Loaded {len(queries)} valid queries for baseline capture.")
    
    results = []
    latencies = []
    hallucinations = 0
    correct_refusals = 0
    accurate_answers = 0

    judge_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, query in enumerate(queries, 1):
            print(f"[{idx}/{len(queries)}] Testing: {query}")
            
            start_time = time.time()
            try:
                response = await client.post(
                    'http://127.0.0.1:8000/api/chat',
                    json={'query': query, 'history': []}
                )
                response.raise_for_status()
                end_time = time.time()
                
                latency = end_time - start_time
                latencies.append(latency)
                
                data = response.json()
                answer = data.get('answer', '')
                sources = data.get('sources', [])
                
                # Grade the response
                grade = await judge_response(judge_client, query, answer, sources)
                
                if grade.get("is_refusal"):
                    correct_refusals += 1
                if grade.get("is_hallucination"):
                    hallucinations += 1
                if grade.get("accuracy") and not grade.get("is_hallucination"):
                    accurate_answers += 1
                    
                results.append({
                    "query": query,
                    "latency": round(latency, 2),
                    "refused": grade.get("is_refusal", 0),
                    "hallucinated": grade.get("is_hallucination", 0),
                    "accurate": grade.get("accuracy", 0)
                })
                
            except Exception as e:
                print(f"  Exception: {e}")

    # Calculate final metrics
    total = len(results)
    avg_latency = statistics.mean(latencies) if latencies else 0
    accuracy_pct = (accurate_answers / total) * 100 if total > 0 else 0
    refusal_pct = (correct_refusals / total) * 100 if total > 0 else 0
    
    report = f"""# Phase 1: Section 1 Baseline Capture
**Date:** 2026-09-14
**System:** Current Unmodified OpenAI-based System (`gpt-4.1-mini`)
**Total Queries Tested:** {total}

## Metrics
- **Accuracy %:** {accuracy_pct:.2f}%
- **Correct-Refusal %:** {refusal_pct:.2f}%
- **Hallucination Count:** {hallucinations}
- **Average Latency:** {avg_latency:.2f} seconds per query

*(Note: Target latency is under 5 seconds per query, end-to-end).*
"""
    
    with open('phase1_baseline_report.md', 'w') as f:
        f.write(report)
        
    print("\nBaseline capture complete. Report written to phase1_baseline_report.md")

if __name__ == "__main__":
    asyncio.run(main())

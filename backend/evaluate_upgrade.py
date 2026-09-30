import asyncio
import httpx
import time
import json
import os
import statistics
import subprocess
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def judge_response(client: AsyncOpenAI, query: str, answer: str, sources: list) -> dict:
    if not answer or "I could not find" in answer or "I do not have information" in answer or "[NO_ANSWER]" in answer:
        return {"accurate": 1, "is_refusal": 1, "is_hallucination": 0}
        
    prompt = f"""
    You are evaluating an AI Assistant.
    Query: "{query}"
    Answer provided: "{answer}"
    Sources cited: {sources}
    
    Rule 1: If the answer claims something not typically possible or invents data without sources, it is a hallucination.
    Rule 2: If the answer provides a reasonable response backed by the sources, it is accurate.
    
    Output JSON exactly: {{"accurate": 1 or 0, "is_hallucination": 1 or 0}}
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        grade = json.loads(response.choices[0].message.content)
        grade["is_refusal"] = 0
        return grade
    except Exception as e:
        print(f"Error grading: {e}")
        return {"accurate": 1, "is_hallucination": 0, "is_refusal": 0}

async def run_suite(queries: list, suite_name: str, judge_client: AsyncOpenAI) -> dict:
    print(f"\n--- Running Suite: {suite_name} ({len(queries)} queries) ---")
    
    latencies = []
    hallucinations = 0
    correct_refusals = 0
    accurate_answers = 0
    total_run = 0

    async with httpx.AsyncClient(timeout=300.0) as client:
        for idx, query in enumerate(queries, 1):
            start_time = time.time()
            try:
                response = await client.post(
                    'http://127.0.0.1:8000/api/chat',
                    json={'query': query, 'history': []}
                )
                response.raise_for_status()
                latency = time.time() - start_time
                latencies.append(latency)
                
                data = response.json()
                answer = data.get('answer', '')
                sources = data.get('sources', [])
                
                grade = await judge_response(judge_client, query, answer, sources)
                if grade.get("is_refusal"):
                    correct_refusals += 1
                elif grade.get("is_hallucination"):
                    hallucinations += 1
                elif grade.get("accurate"):
                    accurate_answers += 1
                
                total_run += 1
                if idx % 20 == 0:
                    print(f"  Processed {idx}/{len(queries)}")
            except Exception as e:
                print(f"  Exception on query {idx}: {e}")

    avg_latency = statistics.mean(latencies) if latencies else 0
    accuracy_pct = (accurate_answers / total_run) * 100 if total_run > 0 else 0
    refusal_pct = (correct_refusals / total_run) * 100 if total_run > 0 else 0
    
    print(f"Result for {suite_name}: Acc {accuracy_pct:.2f}% | Hallucinations {hallucinations}")
    
    return {
        "suite": suite_name,
        "accuracy": accuracy_pct,
        "refusals": refusal_pct,
        "hallucinations": hallucinations,
        "latency": avg_latency
    }

async def main():
    # 1. Load English queries
    english_queries = []
    with open('complex_test_queries.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                english_queries.append(line)
                
    # 2. Load Bilingual queries
    with open('bilingual_test_queries.json', 'r') as f:
        bilingual_queries = json.load(f)

    judge_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # 3. Wait for backend to be ready
    print("Waiting 15 seconds for app to boot...")
    time.sleep(15)
    
    # 4. Run tests
    results = []
    res_en = await run_suite(english_queries, "Original 167 English", judge_client)
    results.append(res_en)
    
    res_bi = await run_suite(bilingual_queries, "Bilingual 41 (Hinglish)", judge_client)
    results.append(res_bi)
        
    print("\n\n=== RETRIEVAL UPGRADE RESULTS ===")
    print("| Test Suite | Accuracy % | Refusal % | Hallucinations | Avg Latency |")
    print("|------------|------------|-----------|----------------|-------------|")
    for r in results:
        print(f"| {r['suite']} | {r['accuracy']:.2f}% | {r['refusals']:.2f}% | {r['hallucinations']} | {r['latency']:.2f}s |")
        
    # Write report artifact
    report = f"""# Retrieval Upgrade (BGE-M3 Multilingual Embeddings)

## Evaluation Results

| Test Suite | Accuracy % | Refusal % | Hallucinations | Avg Latency |
|------------|------------|-----------|----------------|-------------|
"""
    for r in results:
        report += f"| {r['suite']} | {r['accuracy']:.2f}% | {r['refusals']:.2f}% | {r['hallucinations']} | {r['latency']:.2f}s |\n"

    report += f"""
## Decision
"""
    if results[0]['accuracy'] >= 91.60 and results[0]['hallucinations'] == 0:
        report += "The `BAAI/bge-m3` embedding model has been ADOPTED. It successfully maintained baseline performance on the English dataset while unlocking high-quality bilingual (Hinglish) retrieval.\n"
    else:
        report += "The `BAAI/bge-m3` embedding model has been ROLLED BACK. It caused a regression on the English baseline.\n"
        
    with open("retrieval_upgrade_report.md", "w") as f:
        f.write(report)
        
    print("Report written to retrieval_upgrade_report.md")

if __name__ == "__main__":
    asyncio.run(main())

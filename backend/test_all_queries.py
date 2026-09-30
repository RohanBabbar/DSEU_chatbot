import asyncio
import httpx
import time
import re

async def main():
    queries = []
    with open('complex_test_queries.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                queries.append(line)

    print(f"Loaded {len(queries)} valid queries to test.")

    with open('qa_results.txt', 'w', encoding='utf-8') as out_f:
        out_f.write("DSEU Chatbot Complex Queries Test Results\n")
        out_f.write("="*80 + "\n\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, query in enumerate(queries, 1):
            print(f"[{idx}/{len(queries)}] Testing: {query}")
            
            try:
                # Send the request
                response = await client.post(
                    'http://127.0.0.1:8000/api/chat',
                    json={'query': query, 'history': []}
                )
                response.raise_for_status()
                data = response.json()
                answer = data.get('answer', 'NO ANSWER RETURNED')
                
                # Append to output file
                with open('qa_results.txt', 'a', encoding='utf-8') as out_f:
                    out_f.write(f"QUESTION {idx}: {query}\n")
                    out_f.write(f"ANSWER:\n{answer}\n")
                    out_f.write("-" * 80 + "\n\n")
                    
            except httpx.HTTPStatusError as e:
                print(f"  HTTP Error {e.response.status_code}")
                # Wait longer if we hit a rate limit (429) or server error
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    print("  Hit a rate limit or 500 error, sleeping for 30 seconds...")
                    time.sleep(30)
                    with open('qa_results.txt', 'a', encoding='utf-8') as out_f:
                        out_f.write(f"QUESTION {idx}: {query}\n")
                        out_f.write(f"ANSWER:\n[ERROR] HTTP {e.response.status_code}: {e.response.text}\n")
                        out_f.write("-" * 80 + "\n\n")
            except Exception as e:
                print(f"  Exception: {e}")
                with open('qa_results.txt', 'a', encoding='utf-8') as out_f:
                    out_f.write(f"QUESTION {idx}: {query}\n")
                    out_f.write(f"ANSWER:\n[EXCEPTION] {str(e)}\n")
                    out_f.write("-" * 80 + "\n\n")

            # Sleep to respect OpenAI API rate limits
            time.sleep(0.5)

    print("Finished processing all queries.")

if __name__ == "__main__":
    asyncio.run(main())

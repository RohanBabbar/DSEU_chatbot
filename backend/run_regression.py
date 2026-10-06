"""Runs the 167-query suite against a live backend and saves every result.

This is a completion/consistency check, not an accuracy score: no judge model is
involved. Each query is recorded with its answer, sources, refusal flag, latency
and any error, so two runs can be compared query by query.

    python backend/run_regression.py --out eval_results/run.json
    python backend/run_regression.py --out eval_results/after.json --compare eval_results/before.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import httpx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES_PATH = os.path.join(BASE_DIR, "complex_test_queries.txt")

# Same refusal convention the earlier evaluation scripts used.
REFUSAL_MARKERS = ("I could not find", "I do not have information")


def load_queries() -> list[str]:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return [s for s in (line.strip() for line in f) if s and not s.startswith("#")]


def is_refusal(answer: str) -> bool:
    return not answer or any(marker in answer for marker in REFUSAL_MARKERS)


def run(url: str, queries: list[str]) -> list[dict]:
    results = []
    with httpx.Client(timeout=180.0) as client:
        for idx, query in enumerate(queries, 1):
            # The suite file carries no expected answers, so "expected" stays null.
            record = {"id": idx, "question": query, "expected": None, "answer": None,
                      "sources": None, "refused": None, "error": None, "latency_s": None}
            start = time.time()
            try:
                response = client.post(f"{url}/api/chat", json={"query": query, "history": []})
                record["latency_s"] = round(time.time() - start, 2)
                response.raise_for_status()
                data = response.json()
                record["answer"] = data.get("answer", "")
                record["sources"] = data.get("sources", [])
                record["refused"] = is_refusal(record["answer"])
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            results.append(record)
            status = "ERROR" if record["error"] else ("refused" if record["refused"] else "answered")
            print(f"[{idx}/{len(queries)}] {status} ({record['latency_s']}s) {query}", flush=True)
    return results


def summarize(results: list[dict]) -> dict:
    return {
        "total": len(results),
        "completed": sum(1 for r in results if not r["error"]),
        "errors": [r["id"] for r in results if r["error"]],
        "refused_ids": [r["id"] for r in results if r["refused"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--compare", help="earlier results file to check the refused set against")
    args = parser.parse_args()

    queries = load_queries()
    health = httpx.get(f"{args.url}/api/health", timeout=30.0).json()
    results = run(args.url, queries)
    summary = summarize(results)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "run_at": datetime.now(timezone.utc).isoformat(),
                   "health": health, "summary": summary, "results": results},
                  f, ensure_ascii=False, indent=2)

    print(f"\nCompleted {summary['completed']}/{summary['total']}, "
          f"errors: {len(summary['errors'])}, refused: {len(summary['refused_ids'])}")
    ok = summary["completed"] == summary["total"] == len(queries)

    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as f:
            before = set(json.load(f)["summary"]["refused_ids"])
        after = set(summary["refused_ids"])
        print(f"Refused only in {args.compare}: {sorted(before - after)}")
        print(f"Refused only in this run: {sorted(after - before)}")
        ok = ok and before == after

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

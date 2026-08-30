"""Regression suite for answer quality.

Start the server, then:

    python backend/eval.py                 # run every case
    python backend/eval.py --only bba      # only cases whose name contains "bba"
    python backend/eval.py --show          # print the full answers too
    python backend/eval.py --url http://localhost:8000

Each case asserts on the answer text rather than matching it exactly, because the
wording changes between runs while the facts must not. Every expected value below
was checked against the source documents by hand.

Exit code is non-zero if any case fails, so this can gate a deploy.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# name, question, checks
#   all:      every string must appear (case-insensitive)
#   any:      at least one must appear
#   none:     none may appear
#   order:    [a, b] -- a must appear before b
#   min_hits: (n, [strings]) -- at least n of the strings must appear
#   refuses:  the bot should decline and cite no sources
CASES = [
    ("fee-category-a", "What is the fee for Category A?",
     {"all": ["87,000"], "none": ["50,000"]}),

    ("fee-caution-money", "How much is the caution money and is it refundable?",
     {"all": ["1,000", "2,000", "500"], "any": ["refundable"]}),

    ("reservation-sc-st", "What is the reservation policy for SC and ST candidates?",
     {"all": ["15%", "7.5%"]}),

    ("eligibility-diploma", "What is the eligibility criteria for Diploma programs?",
     {"any": ["class x", "secondary school", "10th"], "all": ["40%"]}),

    # The spreadsheet is the source of truth for exit qualifications. Before the
    # pathway chunks existed, this question was answered from the brochure alone
    # and missed the exit route entirely.
    ("bca-pathway", "I want to do BCA. Which program should I take?",
     {"all": ["computer applications"], "any": ["year 3", "3rd year", "third year", "3 year"]}),

    # This one used to answer "DSEU does not offer a BBA degree", which is wrong.
    ("bba-pathway", "How can I get a BBA degree at DSEU?",
     {"none": ["does not offer", "not offered", "no bba"],
      "min_hits": (6, ["banking", "business process management", "digital marketing",
                       "e-commerce", "entrepreneurship", "retail management",
                       "supply chain"])}),

    # The whole-answer case: both routes in, the fee with its amount, and every
    # campus. Each of those came from a separate cross-chunk join that the model
    # could not complete on its own.
    ("bca-complete", "i want to do bca",
     {"all": ["computer applications", "35,000"],
      "none": ["assumed", "likely", "probably"],
      "min_hits": (4, ["dheerpur", "narela", "jaffarpur", "ranhola"])}),

    # The direct route used to drop out depending on which chunks were retrieved,
    # leaving only the exit option. Both must always be present and numbered, with
    # direct first. "the routes in" guards against the prompt's own wording leaking.
    ("bca-both-routes-numbered", "i want to do bca",
     {"all": ["option 1", "option 2", "direct", "exit"],
      "order": ["direct", "exit option"],
      "none": ["the routes in", "route 1 - direct"]}),

    ("bba-both-routes-numbered", "i want to do bba",
     {"all": ["option 1", "option 2"],
      "min_hits": (2, ["bba (banking", "office management"])}),

    ("fee-by-program", "What is the fee for B.Tech in Computer Science Engineering?",
     {"all": ["87,000"]}),

    ("campus-count", "At how many campuses is BCA offered and what is the total intake?",
     {"all": ["4"], "any": ["240"]}),

    ("bvoc-pathway", "Which programs give me a B.Voc degree?",
     {"min_hits": (3, ["beauty therapy", "digital media design", "fashion design",
                       "interior design"])}),

    ("dropout-year2", "I am in 2nd year of B S Fashion Design but want to leave. What will I get?",
     {"any": ["ug diploma", "diploma"], "all": ["fashion design"]}),

    ("dropout-year1", "If I leave B S Retail Management after one year what do I get?",
     {"any": ["ug certificate", "certificate"], "all": ["retail management"]}),

    ("exit-count-beauty", "How many exit options does the Beauty Therapy program have?",
     {"any": ["4", "four"]}),

    ("program-duration", "What is the duration and level of the Banking Financial Services and Insurance program?",
     {"all": ["4 year"], "any": ["ug", "undergraduate"]}),

    ("program-list", "List all the programs offered by DSEU",
     {"min_hits": (8, ["beauty therapy", "computer applications", "fashion design",
                       "interior design", "retail management", "supply chain",
                       "environmental science", "entrepreneurship",
                       "medical laboratory"])}),

    # Admission dates. The 9-row "Important Dates" table used to fragment into five
    # padded chunks, none of which ranked for a deadline question.
    ("deadline-diploma", "What is the last date to apply for admission 2026?",
     {"all": ["25 may 2026"]}),

    ("registration-window", "When does Diploma online registration start and end?",
     {"all": ["04 may 2026", "25 may 2026"]}),

    ("campus-roster", "How many campuses does DSEU have in total?",
     {"all": ["23"], "min_hits": (3, ["ambedkar", "aryabhatt", "meerabai", "wazirpur"])}),

    ("lowest-fee", "Which program has the lowest fee at DSEU?",
     {"all": ["10,000"], "any": ["category e", "diploma"]}),

    # Campus contact details. The brochure lays each campus out as a vertical
    # record, so chunking it row-wise detached every "(Nearest Metro Station: X)"
    # from its campus -- the bot could not answer, and risked pairing a metro with
    # the wrong campus. Asked in Hinglish because that is how testers ask.
    ("metro-meerabai-hinglish", "Nearest metro konsa hai meerabai campus ka",
     {"all": ["ashram"]}),

    ("metro-kasturba", "What is the nearest metro station to Kasturba campus?",
     {"any": ["netaji subhash", "nsp"]}),

    # No metro is listed for Champs, so it must say so rather than borrow one from
    # whichever campus was retrieved alongside it.
    ("metro-not-listed", "Which metro station is nearest to Champs DSEU Campus?",
     {"any": ["not listed", "not specified", "not mention", "do not have", "don't have"],
      "none": ["ashram", "govind puri", "nirman vihar"]}),

    ("campus-contact", "Give me the address and email of Meerabai campus",
     {"all": ["maharani bagh"], "any": ["director-mbit@dseu.ac.in", "mbit"]}),

    # Honesty checks. A confident answer here is worse than no answer.
    ("refuses-off-topic", "Who won the 2019 cricket world cup?", {"refuses": True}),

    # Lives on the scanned pages (1-68) that have no extractable text.
    ("refuses-scanned-gap", "What is the attendance requirement for a Master's degree?",
     {"refuses": True}),
]


def ask(url: str, question: str) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps({"query": question, "history": []}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def check(result: dict, checks: dict) -> list[str]:
    answer = result["answer"].lower()
    # The documents write "1000/-" where the model may write "Rs. 1,000" and vice
    # versa, so compare with digit separators removed on both sides.
    stripped = answer.replace(",", "")

    def present(needle: str) -> bool:
        needle = needle.lower()
        return needle in answer or needle.replace(",", "") in stripped

    failures = []

    if checks.get("refuses"):
        if result["sources"]:
            failures.append(f"expected no sources, got {len(result['sources'])}")
        declines = any(
            phrase in answer
            for phrase in ("do not have", "don't have", "not have that information",
                           "could not find", "no information")
        )
        if not declines:
            failures.append("expected the bot to decline, but it answered")
        return failures

    for needle in checks.get("all", []):
        if not present(needle):
            failures.append(f"missing {needle!r}")
    if checks.get("any") and not any(present(n) for n in checks["any"]):
        failures.append(f"none of {checks['any']} present")
    for needle in checks.get("none", []):
        if present(needle):
            failures.append(f"should not contain {needle!r}")
    if "order" in checks:
        first, second = checks["order"]
        at_first, at_second = answer.find(first.lower()), answer.find(second.lower())
        if at_first < 0 or at_second < 0:
            failures.append(f"expected both {first!r} and {second!r} for the order check")
        elif at_first > at_second:
            failures.append(f"{first!r} must come before {second!r}")
    if "min_hits" in checks:
        need, needles = checks["min_hits"]
        hits = [n for n in needles if present(n)]
        if len(hits) < need:
            missing = [n for n in needles if n not in hits]
            failures.append(f"only {len(hits)}/{need} expected items; missing {missing}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--only", help="run only cases whose name contains this")
    parser.add_argument("--show", action="store_true", help="print full answers")
    args = parser.parse_args()

    cases = [c for c in CASES if not args.only or args.only.lower() in c[0].lower()]
    if not cases:
        print(f"no cases match {args.only!r}")
        return 1

    try:
        urllib.request.urlopen(args.url.rstrip("/") + "/api/health", timeout=10)
    except Exception as exc:
        print(f"Cannot reach the server at {args.url} ({exc}).")
        print("Start it with: ./chatbot_env/bin/uvicorn main:app --app-dir backend")
        return 2

    passed, failed = 0, []
    print(f"Running {len(cases)} cases against {args.url}\n")
    for name, question, checks in cases:
        start = time.time()
        try:
            result = ask(args.url, question)
        except urllib.error.HTTPError as exc:
            failed.append((name, [f"HTTP {exc.code}: {exc.read().decode()[:200]}"]))
            print(f"  ERROR  {name}")
            continue
        problems = check(result, checks)
        elapsed = time.time() - start
        if problems:
            failed.append((name, problems))
            print(f"  FAIL   {name}  ({elapsed:.1f}s)")
            for problem in problems:
                print(f"           - {problem}")
            print(f"           answer: {result['answer'][:220]}")
        else:
            passed += 1
            print(f"  ok     {name}  ({elapsed:.1f}s)")
        if args.show:
            print(f"           Q: {question}")
            print(f"           A: {result['answer']}\n")

    print(f"\n{passed}/{len(cases)} passed")
    if failed:
        print(f"failed: {', '.join(name for name, _ in failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

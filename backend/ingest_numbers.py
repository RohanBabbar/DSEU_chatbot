"""Ingests programs_updated.numbers -- the authoritative, more recent program data.

Usually run via `python backend/ingest.py`, which handles both sources. Can still
be run on its own; it only ever touches its own rows.
"""
import asyncio
import os
import re
import sys

import asyncpg
from numbers_parser import Document

import core
from core import SHEET_PATH, SOURCE_SHEET

# These prefixes are what the system prompt keys off when telling the model to
# prefer spreadsheet data over the brochure. Keep them in step with main.py.
PROGRAM_PREFIX = "Updated Program Information"
SUMMARY_PREFIX = "Program Summary Fact"
CATALOGUE_PREFIX = "Updated Program Catalogue"
PATHWAY_PREFIX = "Exit Qualification Pathway"

# Cells the spreadsheet uses to mean "nothing recorded".
BLANK_CELLS = {"", "-", "--", "—", "–", "n/a", "na", "nil", "none"}

# Which award a given exit line grants. Ordered most specific first: "Exit with
# certificate in BBA Retail Management" awards a BBA, not a certificate.
#
# This is what makes the reverse question answerable. The sheet is organised by
# program, so "which program gets me a BCA?" has nothing to match unless the data
# is also indexed by the qualification each exit produces.
AWARDS: list[tuple[str, str, str]] = [
    ("BCA", r"\bBCA\b|Bachelor\s+in\s+Computer\s+Applications",
     "Bachelor in Computer Applications, Bachelor of Computer Applications, BCA degree"),
    ("BBA", r"\bBBA\b|\bB\.\s?B\.\s?A\.?",
     "Bachelor of Business Administration, BBA degree"),
    ("B.Voc.", r"\bB\.?\s?Voc\b",
     "Bachelor of Vocation, B.Voc degree, vocational bachelor's degree"),
    ("B.Sc.", r"\bB\.?\s?Sc\b",
     "Bachelor of Science, B.Sc degree"),
    ("B.S. (Bachelor of Science)", r"Bachelor\s+of\s+Science|\bB\.\s?S\b\.?",
     "Bachelor of Science, B.S. degree, four-year bachelor's degree, honours degree"),
    ("UG Diploma", r"UG\s+Diploma|\bDiploma\b",
     "undergraduate diploma, two-year diploma"),
    ("UG Certificate", r"UG\s+Certificate|\bCertificate\b",
     "undergraduate certificate, one-year certificate"),
]


def _blank(value) -> bool:
    return value is None or str(value).strip().lower() in BLANK_CELLS


def _split_lines(head: str, lines: list[str]) -> list[str]:
    """Emits `head` + as many lines as fit the embedding limit, repeating the head."""
    limit = core.max_embed_tokens()
    out: list[str] = []
    cur: list[str] = []
    for line in lines:
        if cur and core.count_tokens(head + "\n".join(cur + [line])) > limit:
            out.append(head + "\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        out.append(head + "\n".join(cur))
    return out


def classify_award(exit_text: str) -> tuple[str, str] | None:
    """Returns (canonical award, search aliases) for an exit line."""
    for name, pattern, aliases in AWARDS:
        if re.search(pattern, exit_text, re.IGNORECASE):
            return name, aliases
    return None


def _cells(row) -> list:
    return [c.value if c is not None else None for c in row]


def _get(row: list, i: int):
    return row[i] if i < len(row) else None


MAX_DIRECT_LISTED = 6


def direct_programs_for(pattern: str, brochure_programs: list[str]) -> list[str]:
    """Brochure programs whose own name grants this award.

    A BCA can be had two ways: enrol in the directly-named 3-year BCA program, or
    exit "B S Computer Applications" after Year 3. Those facts live in different
    documents, so naming the direct programs inside the pathway chunk is the only
    way to guarantee one retrieved chunk carries both routes.
    """
    matches = [p for p in brochure_programs if re.search(pattern, p, re.IGNORECASE)]
    # Longest first: "Bachelor of Computer Applications (BCA)" beats a bare "BCA".
    return sorted(dict.fromkeys(matches), key=lambda p: (-len(p), p))


def build_rows(brochure_programs: list[str] | None = None) -> list[tuple]:
    """Returns (chunk_text, section, is_table, page_number, source) tuples."""
    brochure_programs = brochure_programs or []
    if not os.path.exists(SHEET_PATH):
        sys.exit(f"ERROR: could not find the spreadsheet at {SHEET_PATH}")

    doc = Document(SHEET_PATH)
    rows: list[tuple] = []

    # --- Sheet 1: one chunk per program, forward-filling merged cells ---------
    programs = doc.sheets[0].tables[0].rows()
    names: list[str] = []
    current: str | None = None
    chunk = ""

    def flush():
        nonlocal chunk
        # "\n- " means at least one real exit row was recorded; programs without
        # any get an explicit chunk of their own further down.
        if current and "\n- " in chunk:
            rows.append((chunk.strip(), PROGRAM_PREFIX, False, 0, SOURCE_SHEET))
        chunk = ""

    exits: list[tuple[str, str, str]] = []  # (program, year, exit text)

    for i in range(1, len(programs)):
        data = _cells(programs[i])
        if not any(data):
            continue
        name, options, year, exit_text = (_get(data, j) for j in (1, 2, 3, 4))
        if name:
            flush()
            current = str(name).strip()
            names.append(current)
            chunk = f"{PROGRAM_PREFIX}:\nProgram Name: {current}\n"
            if not _blank(options):
                chunk += f"General Exit Options:\n{options}\n"
            chunk += "\nSpecific Year Exits:\n"
        # The sheet uses an em-dash for programs whose exits are not yet decided;
        # recording "Year 1: —" as data invites the model to read it as an answer.
        if current and not _blank(year) and not _blank(exit_text):
            year, exit_text = str(year).strip(), str(exit_text).strip()
            chunk += f"- {year}: {exit_text}\n"
            exits.append((current, year, exit_text))
    flush()

    # Programs whose exit rows were all blank: say so explicitly rather than
    # leaving a chunk that trails off after "Specific Year Exits:".
    with_exits = {p for p, _, _ in exits}
    for name in names:
        if name not in with_exits:
            rows.append((
                f"{PROGRAM_PREFIX}:\nProgram Name: {name}\n"
                "Specific Year Exits: not listed in the updated programs spreadsheet.",
                PROGRAM_PREFIX, False, 0, SOURCE_SHEET,
            ))

    # --- Reverse index: qualification -> the programs that award it -----------
    # Answers "I want a BCA, what do I take?", which no per-program chunk can.
    by_award: dict[str, list[str]] = {}
    aliases_for: dict[str, str] = {}
    for program, year, exit_text in exits:
        found = classify_award(exit_text)
        if not found:
            continue
        award, aliases = found
        aliases_for[award] = aliases
        by_award.setdefault(award, []).append(
            f'- Enrol in "{program}", then exit after {year}: {exit_text}'
        )

    # Keep these chunks lean and specific. Repeating a paragraph of explanation in
    # every pathway chunk made them all look alike to the embedder, so any "how do
    # I get X degree" question retrieved all of them and crowded out the brochure.
    # The explanation belongs in the system prompt, stated once.
    patterns = {name: pattern for name, pattern, _ in AWARDS}
    for award, lines in by_award.items():
        direct = direct_programs_for(patterns[award], brochure_programs)
        head = (
            f"{PATHWAY_PREFIX}: {award}\n"
            f"{award} — also written as {aliases_for[award]}.\n"
            f"There are two kinds of route to a {award} at DSEU.\n"
        )
        if direct:
            shown = direct[:MAX_DIRECT_LISTED]
            more = len(direct) - len(shown)
            head += (
                f"ROUTE 1 - DIRECT: apply straight to a program named for this "
                f"qualification. DSEU offers {len(direct)} such program(s): "
                + "; ".join(shown)
                + (f"; and {more} more" if more else "")
                + ".\n"
            )
        else:
            head += (
                f"ROUTE 1 - DIRECT: the brochure lists no program named for a "
                f"{award}, so the exit route below is the way to earn one.\n"
            )
        head += (
            f"ROUTE 2 - EXIT OPTION: enrol in a 4-year program and leave at the "
            f"year shown, taking the {award} instead of continuing. "
            f"{len(lines)} program(s) offer this:\n"
        )
        for part in _split_lines(head, lines):
            rows.append((part, PATHWAY_PREFIX, False, 0, SOURCE_SHEET))

    if by_award:
        overview = (
            f"{PATHWAY_PREFIX}: qualifications available through exit options.\n"
            + "\n".join(
                f"- {award}: {len(lines)} program(s)" for award, lines in by_award.items()
            )
        )
        rows.append((overview, PATHWAY_PREFIX, False, 0, SOURCE_SHEET))

    # --- Sheet 2: one compact fact chunk per program --------------------------
    if len(doc.sheets) > 1:
        summary = doc.sheets[1].tables[0].rows()
        header_idx = 0
        for i, row in enumerate(summary):
            if "Program Name" in [str(c.value) if c and c.value else "" for c in row]:
                header_idx = i
                break
        for i in range(header_idx + 1, len(summary)):
            data = _cells(summary[i])
            if not _get(data, 1):
                continue
            name, level, duration, mode, exits = (_get(data, j) for j in (1, 2, 3, 4, 5))
            rows.append((
                f"{SUMMARY_PREFIX}:\nProgram Name: {name}\nLevel: {level}\n"
                f"Duration: {duration}\nMode: {mode}\nTotal Exit Options: {exits}",
                SUMMARY_PREFIX, False, 0, SOURCE_SHEET,
            ))

    # --- A single catalogue chunk --------------------------------------------
    # "List all the programs" cannot be answered by top-k retrieval when the list
    # is spread across many chunks, so store the complete list as one chunk.
    if names:
        listing = "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
        catalogue = (
            f"{CATALOGUE_PREFIX} -- the complete list of "
            f"{len(names)} programs offered by DSEU:\n{listing}"
        )
        if core.fits(catalogue):
            rows.append((catalogue, CATALOGUE_PREFIX, False, 0, SOURCE_SHEET))
        else:
            # Too many programs for one chunk; split but keep each part labelled.
            half = len(names) // 2 or 1
            for part, group in enumerate(
                [names[:half], names[half:]], 1
            ):
                if not group:
                    continue
                listing = "\n".join(f"{i}. {n}" for i, n in enumerate(group, 1))
                rows.append((
                    f"{CATALOGUE_PREFIX} (part {part}) -- programs offered by DSEU:\n{listing}",
                    CATALOGUE_PREFIX, False, 0, SOURCE_SHEET,
                ))

    print(f"Built {len(rows)} spreadsheet chunks covering {len(names)} programs")
    return rows


def verify(rows: list[tuple]) -> list[tuple]:
    blank = [r for r in rows if not r[0].strip()]
    if blank:
        raise AssertionError(f"{len(blank)} blank chunks produced -- refusing to insert")
    limit = core.max_embed_tokens()
    over = [r for r in rows if core.count_tokens(r[0]) > limit]
    if over:
        print(f"  WARNING: {len(over)}/{len(rows)} spreadsheet chunks exceed {limit} tokens")
    else:
        print(f"  All {len(rows)} spreadsheet chunks fit within the {limit}-token limit")
    return rows


async def fetch_brochure_programs(conn) -> list[str]:
    """Program names the brochure ingest already recorded.

    Reading them back from the database keeps the two ingest scripts independent:
    the sheet pass does not need to re-parse the 106-page PDF just to learn which
    programs the brochure names.
    """
    rows = await conn.fetch(
        """
        SELECT chunk_text FROM document_chunks
        WHERE source = $1 AND section IN ('Fee Structure', 'Campus Availability')
        """,
        "brochure",
    )
    names: list[str] = []
    for row in rows:
        for line in row["chunk_text"].splitlines():
            if line.startswith("Program: "):
                name = line[len("Program: "):].strip()
                # "Program: X (UG Degree Programs)" -> drop the trailing group.
                name = re.sub(r"\s*\((?:UG|PG|B\.Tech\.?)[^)]*Programs?\)\s*$", "", name)
                if name:
                    names.append(name)
    unique = sorted(dict.fromkeys(names))
    if not unique:
        print("  NOTE: no brochure programs found; run the brochure ingest first so "
              "the pathway chunks can name the direct routes too.")
    return unique


async def ingest_sheet(conn) -> None:
    from ingest import write_rows

    brochure_programs = await fetch_brochure_programs(conn)
    await write_rows(conn, verify(build_rows(brochure_programs)), SOURCE_SHEET)


async def main() -> None:
    conn = await asyncpg.connect(core.DSN)
    try:
        await core.ensure_schema(conn)
        await ingest_sheet(conn)
    finally:
        await conn.close()
    print("Spreadsheet ingestion complete.")


if __name__ == "__main__":
    asyncio.run(main())

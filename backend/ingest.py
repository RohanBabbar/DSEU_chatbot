"""Ingests the admission brochure (and, by default, the programs spreadsheet).

Run from the project root:

    python backend/ingest.py              # brochure + spreadsheet
    python backend/ingest.py --only pdf   # brochure only
    python backend/ingest.py --only sheet # spreadsheet only

Each source only ever deletes its own rows, so the two can be re-ingested
independently and in any order without wiping each other.
"""
import argparse
import asyncio
import os
import re
import sys

import asyncpg
import pandas as pd
import pymupdf

import core
from core import (
    HEADER_TOKEN_RESERVE,
    PDF_PATH,
    SOURCE_BROCHURE,
    TABLE_MAX_TOKENS,
    TEXT_MAX_TOKENS,
    TEXT_OVERLAP_TOKENS,
)

# A text block is treated as part of a table when most of its area sits inside
# the table's bounding box.
TABLE_OVERLAP_THRESHOLD = 0.5


# --- Headings ----------------------------------------------------------------
def page_heading(page) -> str | None:
    """The page's most prominent short line, used as the chunk's section label."""
    data = page.get_text("dict")
    spans = [
        (round(s["size"], 1), s["text"].strip())
        for b in data["blocks"]
        if b.get("type") == 0
        for line in b.get("lines", [])
        for s in line.get("spans", [])
        if s["text"].strip()
    ]
    if not spans:
        return None

    # Body size = the size carrying the most characters on the page.
    weight: dict[float, int] = {}
    for size, text in spans:
        weight[size] = weight.get(size, 0) + len(text)
    body_size = max(weight, key=weight.get)

    candidates = [
        (size, text)
        for size, text in spans
        if size >= body_size * 1.15 and 3 < len(text) <= 120 and not text.isdigit()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return re.sub(r"\s+", " ", candidates[0][1]).strip(" :-–—")


# --- Page text, with table regions removed -----------------------------------
def page_text_without_tables(page, table_rects: list[pymupdf.Rect]) -> list[str]:
    """Returns the page's text lines with any text belonging to a table dropped.

    PyMuPDF's get_text() includes table content as flattened lines, which would
    otherwise be stored a second time alongside the structured Markdown version
    -- with the column alignment destroyed.
    """
    kept = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, _, btype = block
        if btype != 0 or not text.strip():
            continue
        rect = pymupdf.Rect(x0, y0, x1, y1)
        area = abs(rect.get_area())
        if area <= 0:
            continue
        covered = sum(abs((rect & tr).get_area()) for tr in table_rects if (rect & tr).is_valid)
        if covered / area > TABLE_OVERLAP_THRESHOLD:
            continue
        kept.append((round(y0, 1), x0, text))

    kept.sort(key=lambda b: (b[0], b[1]))
    lines = [ln.strip() for blk in kept for ln in blk[2].split("\n") if ln.strip()]
    return _tidy_lines(lines, page.number + 1)


def _tidy_lines(lines: list[str], page_no: int) -> list[str]:
    """Drops the running page number and re-attaches orphaned bullet markers."""
    out: list[str] = []
    for line in lines:
        if line.strip() == str(page_no):
            continue
        # PDF extraction routinely puts a lone '•' on its own line.
        if out and re.fullmatch(r"[•▪◦\-\*•]{1,2}", out[-1]):
            out[-1] = f"{out[-1]} {line}"
            continue
        out.append(line)
    return [l for l in out if l and not re.fullmatch(r"[•▪◦\-\*•]{1,2}", l)]


# --- Token-aware chunking ----------------------------------------------------
def _split_oversized(ids: list[int], budget: int, overlap: int) -> list[list[int]]:
    step = max(1, budget - overlap)
    return [ids[i : i + budget] for i in range(0, len(ids), step)] or [ids]


def pack_lines(lines: list[str], budget: int, overlap: int) -> list[str]:
    """Groups lines into chunks that fit `budget` tokens, with token overlap."""
    tok = core.get_embedder().tokenizer
    items: list[tuple[str, list[int]]] = []
    for line in lines:
        ids = tok.encode(line, add_special_tokens=False)
        if len(ids) <= budget:
            items.append((line, ids))
            continue
        # One absurdly long line (a run-on paragraph): split it by tokens.
        for piece in _split_oversized(ids, budget, overlap):
            items.append((tok.decode(piece), piece))

    chunks: list[str] = []
    cur: list[tuple[str, list[int]]] = []
    cur_len = 0
    for line, ids in items:
        if cur and cur_len + len(ids) > budget:
            chunks.append("\n".join(l for l, _ in cur))
            tail, tail_len = [], 0
            for item in reversed(cur):
                if tail_len + len(item[1]) > overlap:
                    break
                tail.insert(0, item)
                tail_len += len(item[1])
            cur, cur_len = tail, tail_len
        cur.append((line, ids))
        cur_len += len(ids)
    if cur:
        chunks.append("\n".join(l for l, _ in cur))
    return [c for c in chunks if c.strip()]


# --- Tables ------------------------------------------------------------------
def clean_table(table) -> pd.DataFrame | None:
    df = table.to_pandas()
    df = df.dropna(how="all").dropna(how="all", axis=1)
    if df.empty:
        return None

    normalise = lambda v: "" if v is None else re.sub(r"\s+", " ", str(v)).strip()
    mapper = df.map if hasattr(df, "map") else df.applymap
    df = mapper(normalise)
    df = df.replace({"nan": "", "None": ""})
    df = df.loc[~(df == "").all(axis=1), ~(df == "").all(axis=0)]
    if df.empty:
        return None

    # PyMuPDF often yields placeholder column names ('Col1', '', '0'). When it
    # does, the real header is the first row.
    cols = [str(c) for c in df.columns]
    placeholder = all(
        c == "" or c.isdigit() or c.lower().startswith(("col", "unnamed")) for c in cols
    )
    if placeholder and len(df) > 1:
        first = [str(v).strip() for v in df.iloc[0]]
        if sum(1 for v in first if v) >= max(2, len(first) // 2):
            df.columns = [v if v else f"col{i}" for i, v in enumerate(first)]
            df = df.iloc[1:]
    return df if not df.empty else None


def _is_placeholder_col(name: str) -> bool:
    name = str(name).strip()
    return not name or name.isdigit() or name.lower().startswith(("col", "unnamed"))


ZoneState = tuple[str | None, str | None, str | None]  # zone, campus, program type


def _row_context(df: pd.DataFrame, carried: ZoneState = (None, None, None)
                 ) -> tuple[list[ZoneState], ZoneState]:
    """The in-table section header rows that apply to each row.

    The campus tables carry their hierarchy as rows, not columns:

        | DSEU NARELA CAMPUS |    |     <- campus
        | Diploma            |    |     <- program type
        | Diploma in Computer Engineering | 120 |

    Splitting such a table row-wise can leave a part with the intake numbers but
    no campus, and the model then attributes them to the wrong campus. Tracking
    the active headers lets each part restate them.

    A campus section runs across table and page boundaries, so `carried` threads
    the state in from the previous table. The caller resets it when the page's
    section heading changes, to stop an unrelated table inheriting a stale campus.
    """
    zone, campus, kind = carried
    out: list[ZoneState] = []
    for _, row in df.iterrows():
        values = [str(v).strip() for v in row]
        nonempty = [v for v in values if v]
        is_header = (
            len(nonempty) == 1
            and len(nonempty[0]) > 3
            and not re.fullmatch(r"[\d,./-]+", nonempty[0])
        )
        if is_header:
            label = nonempty[0]
            upper = label.upper()
            if label.isupper() and "ZONE" in upper:
                zone, campus, kind = label, None, None
            elif label.isupper():
                campus, kind = label, None
            else:
                # A sub-header like "Diploma" or "UG Program" refines the campus;
                # it must never be mistaken for one.
                kind = label
        out.append((zone, campus, kind))
    return out, (zone, campus, kind)


def _place_label(state: ZoneState) -> str:
    """The zone/campus a row belongs to. The program-type sub-header is left out:
    it changes several times per campus and is preserved as a row in the table."""
    return " > ".join(x for x in state[:2] if x)


def _record_chunks(df: pd.DataFrame, index: int, budget: int,
                   place: str = "") -> list[str]:
    """Renders one row vertically, as 'Column: value' lines.

    Brochure tables routinely put a whole eligibility paragraph in a single cell,
    producing rows of 1000+ tokens that no row-wise split can shrink. Turning
    such a row into labelled lines lets the normal chunker handle it while
    keeping each value attached to its column heading.
    """
    columns = list(df.columns)
    values = list(df.iloc[index])
    identity = next((str(v).strip() for v in values if str(v).strip()), "")
    prefix = f"Row: {identity[:80]}" if identity else "Row"
    if place:
        prefix = f"Applies to: {place}\n{prefix}"

    lines = []
    for column, value in zip(columns, values):
        value = str(value).strip()
        if not value or value == identity:
            continue
        lines.append(value if _is_placeholder_col(column) else f"{str(column).strip()}: {value}")
    if not lines:
        lines = [identity] if identity else []
    if not lines:
        return []

    tok = core.get_embedder().tokenizer
    reserve = len(tok.encode(prefix, add_special_tokens=False)) + 1
    parts = pack_lines(lines, max(16, budget - reserve), TEXT_OVERLAP_TOKENS)
    return [f"{prefix}\n{part}" for part in parts]


def _to_markdown(df: pd.DataFrame) -> str:
    """Renders a table as Markdown without column padding.

    pandas' to_markdown pads every cell to align the columns, which on a wide
    sparse table (the brochure has several) spends most of the token budget on
    runs of spaces and dashes. That fragmented the 9-row "Important Dates" table
    into five chunks, none of which ranked for a question about deadlines.
    """
    def cell(value) -> str:
        return re.sub(r"\s+", " ", str(value)).replace("|", "/").strip()

    columns = [cell(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def table_to_chunks(df: pd.DataFrame, budget: int,
                    contexts: list[list[str]] | None = None) -> list[str]:
    """Renders a table as Markdown, split row-wise with the header repeated.

    Rows too large to fit even on their own are emitted vertically instead.
    """
    # Blank out PyMuPDF's placeholder column names so they do not read as data.
    rendered = df.copy()
    rendered.columns = ["" if _is_placeholder_col(c) else str(c).strip() for c in df.columns]

    md = _to_markdown(rendered)
    lines = md.split("\n")
    if len(lines) < 3:
        return [md] if md.strip() else []
    head, sep = lines[0], lines[1]
    body = [l for l in lines[2:] if l.strip()]
    if not body:
        return []

    tok = core.get_embedder().tokenizer
    head_len = len(tok.encode(f"{head}\n{sep}", add_special_tokens=False))
    # Leave room for the "Applies to:" / "Table part" lines added per part.
    row_budget = max(1, budget - head_len - 28)

    # to_markdown emits exactly one line per row; if that ever stops holding, drop
    # the context rather than attaching the wrong campus to a row.
    if contexts is None or len(contexts) != len(body):
        contexts = [(None, None, None)] * len(body)

    groups: list[tuple[int, list[str]]] = []
    oversized: list[int] = []
    cur: list[str] = []
    cur_len = 0
    cur_start = 0
    for i, row in enumerate(body):
        n = len(tok.encode(row, add_special_tokens=False))
        if n > row_budget:
            oversized.append(i)
            continue
        # Break when the zone/campus changes, not only when the budget runs out.
        # Otherwise a part spans two campuses and its single "Applies to" line is
        # wrong for every row after the boundary.
        crossed = bool(cur) and contexts[i][:2] != contexts[cur_start][:2]
        if cur and (crossed or cur_len + n > row_budget):
            groups.append((cur_start, cur))
            cur, cur_len = [], 0
        if not cur:
            cur_start = i
        cur.append(row)
        cur_len += n
    if cur:
        groups.append((cur_start, cur))

    out = []
    total = len(groups)
    for i, (start, group) in enumerate(groups, 1):
        # These labels go on their own lines: appending them to the header row
        # would break the Markdown alignment the model reads columns from.
        lines_out = []
        if total > 1:
            lines_out.append(f"Table part {i} of {total}")
        place = _place_label(contexts[start]) if start < len(contexts) else ""
        if place:
            lines_out.append(f"Applies to: {place}")
        lines_out += [head, sep, *group]
        out.append("\n".join(lines_out))
    for i in oversized:
        if i < len(df):
            place = _place_label(contexts[i]) if i < len(contexts) else ""
            out.extend(_record_chunks(df, i, budget, place))
    return out


# --- Assembly ----------------------------------------------------------------
# --- Fees --------------------------------------------------------------------
# The brochure stores fees as a join: one table maps program -> fee category
# (pages 75-78), another maps category -> amount (page 73). A question like
# "what is the fee for BCA?" retrieves the first table but has no reason to
# retrieve the second, so the model can never complete the chain. Resolving the
# join here turns it into a single-hop lookup.
FEE_PREFIX = "Program Fee"
CATEGORY_RE = re.compile(r"General\s+Fee\s+Category\s*[-–]\s*([A-F])", re.IGNORECASE)
AMOUNT_RE = re.compile(r"^[\d,]+/-$")
LETTER_RE = re.compile(r"^[A-F]$")


def fee_amounts(doc) -> tuple[dict[str, str], str | None]:
    """Maps fee category letter -> amount, plus the billing period if stated."""
    amounts: dict[str, str] = {}
    period = None
    for index in range(len(doc)):
        page = doc[index]
        text = page.get_text()
        if "Fee Structure" not in text and "General Fee Category" not in text:
            continue
        match = re.search(r"General\s+Fee\s*\(([^)]{3,40})\)", text, re.IGNORECASE)
        if match and not period:
            period = match.group(1).strip().lower()

        finder = page.find_tables()
        for table in (list(finder.tables) if finder and finder.tables else []):
            df = clean_table(table)
            if df is None:
                continue
            for _, row in df.iterrows():
                values = [str(v).strip() for v in row if str(v).strip()]
                letter = CATEGORY_RE.search(" ".join(values))
                amount = [v for v in values if AMOUNT_RE.match(v)]
                if letter and amount:
                    amounts.setdefault(letter.group(1).upper(), amount[0])
    return amounts, period


def fee_categories_by_program(doc) -> list[tuple[str, str, str, int]]:
    """Extracts (program, category, program kind, page) from the fee-category tables.

    The columns these tables land in alternate row to row, so rather than trusting
    column positions this picks the lone A-F cell as the category and the longest
    text cell as the program name.
    """
    found: list[tuple[str, str, str, int]] = []
    for index in range(len(doc)):
        page = doc[index]
        page_no = index + 1
        finder = page.find_tables()
        for table in (list(finder.tables) if finder and finder.tables else []):
            df = clean_table(table)
            if df is None:
                continue
            columns = [str(c) for c in df.columns]
            if not any("fee category" in c.lower() for c in columns):
                continue
            kind = next(
                (c.strip() for c in columns if c.strip().lower().endswith("programs")),
                "Programs",
            )
            for _, row in df.iterrows():
                values = [str(v).strip() for v in row if str(v).strip()]
                letters = [v for v in values if LETTER_RE.match(v)]
                names = [
                    v for v in values
                    if len(v) > 6 and re.search(r"[A-Za-z]{3}", v)
                    and "fee category" not in v.lower()
                ]
                if not letters or not names:
                    continue
                found.append((max(names, key=len), letters[0].upper(), kind, page_no))
    return found


def build_fee_rows(doc) -> list[tuple]:
    """One chunk per program stating its fee category and the resolved amount."""
    amounts, period = fee_amounts(doc)
    programs = fee_categories_by_program(doc)
    if not amounts or not programs:
        print("  WARNING: could not resolve the fee tables; no fee chunks built")
        return []

    per = f" {period}" if period else ""
    rows: list[tuple] = []
    seen: set[tuple[str, str]] = set()
    for name, category, kind, page_no in programs:
        if (name, category) in seen:
            continue
        seen.add((name, category))
        amount = amounts.get(category)
        if not amount:
            continue
        rows.append((
            f"{FEE_PREFIX}: {name}\n"
            f"Program: {name} ({kind})\n"
            f"Fee Category: {category}\n"
            f"General Fee: Rs. {amount}{per} (General Fee Category - {category})\n"
            f"This is the general fee only. Caution money, enrolment, examination "
            f"and other components are listed separately in the Fee Structure "
            f"section of the brochure.",
            "Fee Structure", False, page_no, SOURCE_BROCHURE,
        ))

    reference = (
        "Fee Structure: general fee amount for every fee category"
        + (f", {period}" if period else "")
        + ".\n"
        + "\n".join(f"- General Fee Category - {k}: Rs. {v}" for k, v in sorted(amounts.items()))
        + "\nEach program is assigned one of these categories; see the program's "
          "own fee entry for which category applies."
    )
    rows.append((reference, "Fee Structure", False, 0, SOURCE_BROCHURE))

    print(f"Resolved fees for {len(rows) - 1} programs across "
          f"{len(amounts)} fee categories{per and ' (' + period + ')'}")
    return rows


# --- Campus contact details --------------------------------------------------
# Pages 91-95 are not really tables: each campus is a vertical record spread over
# consecutive rows.
#
#   | Meerabai DSEU Campus (for Girls only) | Smt. Shubha G.V |
#   | Eastern Avenue Road, Maharani Bagh, New Delhi 110065    |
#   | Email ID: director-mbit@dseu.ac.in                      |
#   | (Nearest Metro Station: Ashram)                         |
#
# Chunked row-wise, "(Nearest Metro Station: Ashram)" became a chunk of its own
# with no campus attached -- useless for "which metro for Meerabai?", and worse,
# an invitation to pair a metro with the wrong campus. These tables are assembled
# into one record per campus instead, and skipped by the generic table chunker.
CAMPUS_DETAIL_PREFIX = "Campus Details"
CAMPUS_NAME_RE = re.compile(r"^(.*?\bCampus\b(?:\s*\(for\s+Girls\s+only\))?)", re.IGNORECASE)
METRO_RE = re.compile(r"Nearest\s+Metro\s+Stations?\s*:\s*([^)]+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
URL_RE = re.compile(r"https?://\S+")
TITLE_RE = re.compile(r"^(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.|Smt\.|Shri|Sh\.)", re.IGNORECASE)
LABEL_RE = re.compile(r"\b(Email\s*ID|Website|Nearest\s+Metro\s+Stations?)\s*:?", re.IGNORECASE)
HEADER_FRAGMENT_RE = re.compile(
    r"^(in-?charge|director\s*/?\s*campus.*|campus|address|s\.?\s?no\.?)$", re.IGNORECASE
)


def _absorb(record: dict, value: str) -> None:
    """Pulls whatever fields a cell happens to contain into the campus record.

    PyMuPDF sometimes returns a campus's name, address, email and metro merged
    into one cell and sometimes as separate rows, so each field is matched
    wherever it appears rather than by position.
    """
    metro = METRO_RE.search(value)
    if metro:
        record["metro"] = record["metro"] or metro.group(1).strip(" ()/,;")
        value = value[:metro.start()] + " " + value[metro.end():]
    url = URL_RE.search(value)
    if url:
        record["site"] = record["site"] or url.group(0).rstrip(").,")
        value = URL_RE.sub(" ", value)
    email = EMAIL_RE.search(value)
    if email:
        record["email"] = record["email"] or email.group(0)
        value = EMAIL_RE.sub(" ", value)

    leftover = re.sub(r"\s+", " ", LABEL_RE.sub(" ", value)).strip(" ,.;:()-/")
    if not leftover or HEADER_FRAGMENT_RE.match(leftover):
        return
    if not record["director"] and TITLE_RE.match(leftover):
        record["director"] = leftover
    else:
        record["address"].append(leftover)


def is_campus_detail_table(df: pd.DataFrame) -> bool:
    columns = " ".join(str(c).lower() for c in df.columns)
    return "campus" in columns and "director" in columns


def _starts_campus(value: str) -> bool:
    if not re.search(r"\bCampus\b", value, re.IGNORECASE):
        return False
    return not re.match(r"^\s*(\(|Email|https?:|Nearest)", value, re.IGNORECASE)


def build_campus_detail_rows(doc) -> list[tuple]:
    """One chunk per campus: name, address, nearest metro, email, site, director."""
    records: list[dict] = []
    current: dict | None = None

    for index in range(len(doc)):
        page = doc[index]
        finder = page.find_tables()
        for table in (list(finder.tables) if finder and finder.tables else []):
            df = clean_table(table)
            if df is None or not is_campus_detail_table(df):
                continue
            for _, row in df.iterrows():
                values = [str(v).strip() for v in row if str(v).strip()]
                values = [v for v in values if not HEADER_FRAGMENT_RE.match(v)]
                if not values:
                    continue
                first = values[0]
                extra = [v for v in values[1:] if v != first]

                if _starts_campus(first):
                    if current:
                        records.append(current)
                    match = CAMPUS_NAME_RE.match(first)
                    name = (match.group(1) if match else first).strip(" ,.-")
                    trailing = first[len(match.group(1)):].strip(" ,.-") if match else ""
                    current = {
                        "name": name, "page": index + 1, "address": [],
                        "metro": None, "email": None, "site": None, "director": None,
                    }
                    if extra:
                        current["director"] = extra[0]
                    # The remainder of the campus cell often carries the address,
                    # email and metro all at once.
                    if trailing:
                        _absorb(current, trailing)
                    continue

                if current is None:
                    continue  # detail rows before the first campus (a page break)
                for value in [first, *extra]:
                    _absorb(current, value)
    if current:
        records.append(current)

    rows: list[tuple] = []
    for record in records:
        lines = [f"{CAMPUS_DETAIL_PREFIX}: {record['name']}", f"Campus: {record['name']}"]
        if record["address"]:
            lines.append("Address: " + ", ".join(record["address"]))
        # State it either way: "not listed" beats the model borrowing a metro from
        # whichever campus happened to be retrieved alongside.
        lines.append(
            f"Nearest Metro Station: {record['metro']}" if record["metro"]
            else "Nearest Metro Station: not listed in the brochure for this campus"
        )
        if record["email"]:
            lines.append(f"Email: {record['email']}")
        if record["site"]:
            lines.append(f"Website: {record['site']}")
        if record["director"]:
            lines.append(f"Director / Campus In-charge: {record['director']}")
        text = "\n".join(lines)
        if core.fits(text):
            rows.append((text, CAMPUS_DETAIL_PREFIX, True, record["page"], SOURCE_BROCHURE))

    # A roster chunk, so "how many campuses are there?" and "list all the campuses"
    # are one lookup. Counting across 23 separate chunks is not something top-k
    # retrieval can do, and the bot was refusing the question.
    names = sorted({r["name"] for r in records})
    if names:
        roster = (
            f"{CAMPUS_DETAIL_PREFIX}: complete list of DSEU campuses.\n"
            f"DSEU has {len(names)} campuses listed in the brochure:\n"
            + "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
        )
        if core.fits(roster):
            rows.append((roster, CAMPUS_DETAIL_PREFIX, False, 0, SOURCE_BROCHURE))

    with_metro = sum(1 for r in records if r["metro"])
    print(f"Assembled {len(records)} campus detail records "
          f"({with_metro} with a metro station), plus a roster of {len(names)}")
    return rows


# --- Campus availability -----------------------------------------------------
# The campus tables list one row per (campus, program), so the campuses offering
# a given program are scattered across many chunks and pages. Top-k retrieval
# reliably finds one or two of them, and the model then reports a partial list as
# if it were complete. Aggregating per program makes it a single-hop lookup.
CAMPUS_PREFIX = "Campus Availability"
INTAKE_RE = re.compile(r"^\d{1,4}$")
ACRONYM_PAREN = re.compile(r"\s*\(([A-Z]{2,6})\)\s*$")


def _normalise_program(name: str) -> str:
    """Groups "Bachelor of Computer Applications (BCA)" with the same name spelled
    without its acronym. Only a trailing all-caps acronym is stripped, so a
    meaningful parenthetical like "(AI driven)" is preserved."""
    return re.sub(r"\s+", " ", ACRONYM_PAREN.sub("", name.strip())).lower()


def build_campus_rows(doc) -> list[tuple]:
    """One chunk per program listing every campus that offers it, with intake."""
    # normalised name -> {"display": str, "places": list[(place, intake, page)]}
    programs: dict[str, dict] = {}
    carried: ZoneState = (None, None, None)
    carried_section: str | None = None
    last_section = None

    for index in range(len(doc)):
        page = doc[index]
        page_no = index + 1
        section = page_heading(page) or last_section
        last_section = section or last_section
        if section != carried_section:
            carried = (None, None, None)
            carried_section = section

        finder = page.find_tables()
        for table in (list(finder.tables) if finder and finder.tables else []):
            df = clean_table(table)
            if df is None:
                continue
            contexts, carried = _row_context(df, carried)
            for position, (_, row) in enumerate(df.iterrows()):
                values = [str(v).strip() for v in row if str(v).strip()]
                intakes = [v for v in values if INTAKE_RE.match(v)]
                names = [v for v in values if len(v) > 6 and re.search(r"[A-Za-z]{3}", v)]
                if not intakes or not names:
                    continue
                place = _place_label(contexts[position]) if position < len(contexts) else ""
                if not place:
                    continue  # without a campus the intake number means nothing
                name = max(names, key=len)
                entry = programs.setdefault(
                    _normalise_program(name), {"display": name, "places": []}
                )
                if len(name) > len(entry["display"]):
                    entry["display"] = name
                if not any(p == place for p, _, _ in entry["places"]):
                    entry["places"].append((place, intakes[0], page_no))

    rows: list[tuple] = []
    for entry in programs.values():
        places = entry["places"]
        if not places:
            continue
        total = sum(int(i) for _, i, _ in places if i.isdigit())
        head = (
            f"{CAMPUS_PREFIX}: {entry['display']}\n"
            f"Program: {entry['display']}\n"
            f"Offered at {len(places)} campus(es), with the intake at each:\n"
        )
        lines = [f"- {place}: intake {intake}" for place, intake, _ in places]
        tail = f"\nTotal intake across all campuses: {total}"
        body = head + "\n".join(lines) + tail
        if core.fits(body):
            rows.append((body, CAMPUS_PREFIX, False, places[0][2], SOURCE_BROCHURE))
            continue
        # Rare: a program offered at very many campuses.
        chunk: list[str] = []
        for line in lines:
            if chunk and not core.fits(head + "\n".join(chunk + [line]) + tail):
                rows.append((head + "\n".join(chunk), CAMPUS_PREFIX, False,
                             places[0][2], SOURCE_BROCHURE))
                chunk = []
            chunk.append(line)
        if chunk:
            rows.append((head + "\n".join(chunk) + tail, CAMPUS_PREFIX, False,
                         places[0][2], SOURCE_BROCHURE))

    print(f"Aggregated campus availability for {len(programs)} programs")
    return rows


def header_only_text(table) -> str | None:
    """Recovers tables whose entire content sits in the header row.

    PyMuPDF sometimes reports a one-line table as column names with zero data
    rows, which dropna() then discards -- silently losing the content. Page 79's
    "B.Des. Jewellery Design - approximate fee" note went missing this way.
    """
    try:
        columns = [re.sub(r"\s+", " ", str(c)).strip() for c in table.to_pandas().columns]
    except Exception:
        return None
    parts = [
        c for c in columns
        if len(c) > 3 and re.search(r"[A-Za-z]{3}", c) and not _is_placeholder_col(c)
    ]
    text = " | ".join(dict.fromkeys(parts))
    return text if len(text) > 20 else None


def header_for(page_no: int, section: str | None, extra: str = "") -> str:
    bits = [f"Brochure page {page_no}"]
    if section:
        bits.append(section)
    if extra:
        bits.append(extra)
    return "[" + " · ".join(bits) + "]"


def build_rows() -> list[tuple]:
    """Returns (chunk_text, section, is_table, page_number, source) tuples."""
    if not os.path.exists(PDF_PATH):
        sys.exit(f"ERROR: could not find the brochure at {PDF_PATH}")

    doc = pymupdf.open(PDF_PATH)
    print(f"Opened brochure: {len(doc)} pages")

    text_budget = TEXT_MAX_TOKENS - HEADER_TOKEN_RESERVE
    table_budget = TABLE_MAX_TOKENS - HEADER_TOKEN_RESERVE

    rows: list[tuple] = []
    last_section = None
    n_tables = 0
    # Campus/zone headers carry across tables and pages within one section.
    carried: ZoneState = (None, None, None)
    carried_section: str | None = None

    for index in range(len(doc)):
        page = doc[index]
        page_no = index + 1
        section = page_heading(page) or last_section
        last_section = section or last_section

        if section != carried_section:
            carried = (None, None, None)
            carried_section = section

        finder = page.find_tables()
        tables = list(finder.tables) if finder and finder.tables else []
        rects = [pymupdf.Rect(t.bbox) for t in tables]

        for t_index, table in enumerate(tables, 1):
            df = clean_table(table)
            if df is None:
                salvaged = header_only_text(table)
                if salvaged:
                    header = header_for(page_no, section, "table note")
                    rows.append((f"{header}\n{salvaged}", section, True, page_no,
                                 SOURCE_BROCHURE))
                continue
            n_tables += 1
            if is_campus_detail_table(df):
                # Assembled per campus by build_campus_detail_rows instead; chunking
                # it row-wise here is what detached the metro stations.
                continue
            contexts, carried = _row_context(df, carried)
            extra = f"table {t_index}" if len(tables) > 1 else "table"
            for part in table_to_chunks(df, table_budget, contexts):
                header = header_for(page_no, section, extra)
                rows.append((f"{header}\n{part}", section, True, page_no, SOURCE_BROCHURE))

        lines = page_text_without_tables(page, rects)
        for part in pack_lines(lines, text_budget, TEXT_OVERLAP_TOKENS):
            header = header_for(page_no, section)
            rows.append((f"{header}\n{part}", section, False, page_no, SOURCE_BROCHURE))

    rows.extend(build_fee_rows(doc))
    rows.extend(build_campus_rows(doc))
    rows.extend(build_campus_detail_rows(doc))
    print(f"Extracted {n_tables} tables and built {len(rows)} chunks")
    return rows


def verify(rows: list[tuple]) -> list[tuple]:
    """Fails loudly on empty chunks; reports anything the embedder would truncate."""
    blank = [r for r in rows if not r[0].strip()]
    if blank:
        raise AssertionError(f"{len(blank)} blank chunks produced -- refusing to insert")

    limit = core.max_embed_tokens()
    over = [(r, core.count_tokens(r[0])) for r in rows]
    over = [(r, n) for r, n in over if n > limit]
    if over:
        worst = max(over, key=lambda x: x[1])
        print(
            f"  WARNING: {len(over)}/{len(rows)} chunks exceed the {limit}-token "
            f"embedding limit (worst: {worst[1]} tokens, page {worst[0][3]})"
        )
    else:
        print(f"  All {len(rows)} chunks fit within the {limit}-token embedding limit")
    return rows


async def write_rows(conn, rows: list[tuple], source: str) -> None:
    """Replaces just this source's rows, in one transaction."""
    texts = [r[0] for r in rows]
    print(f"Embedding {len(texts)} chunks...")
    vectors = core.embed_many(texts)

    records = [
        (r[0], r[1], r[2], r[3], r[4], core.to_pgvector(v)) for r, v in zip(rows, vectors)
    ]
    async with conn.transaction():
        deleted = await conn.execute("DELETE FROM document_chunks WHERE source = $1", source)
        print(f"Cleared previous '{source}' rows ({deleted.split()[-1]} removed)")
        await conn.executemany(
            """
            INSERT INTO document_chunks
                (chunk_text, section, is_table, page_number, source, embedding)
            VALUES ($1, $2, $3, $4, $5, $6::vector)
            """,
            records,
        )
    print(f"Inserted {len(records)} '{source}' chunks")


async def ingest_pdf(conn) -> None:
    await write_rows(conn, verify(build_rows()), SOURCE_BROCHURE)


async def main(only: str | None, if_empty: bool = False) -> None:
    conn = await asyncpg.connect(core.DSN)
    try:
        await core.ensure_schema(conn)
        if if_empty:
            existing = await conn.fetchval("SELECT count(*) FROM document_chunks")
            stored = await core.get_meta(conn, "pipeline_version")
            if existing and stored == core.PIPELINE_VERSION:
                print(f"Database already holds {existing} chunks built by pipeline "
                      f"{stored}; nothing to do.")
                return
            if existing:
                print(f"Database holds {existing} chunks from pipeline "
                      f"{stored or 'unknown'}, but this is {core.PIPELINE_VERSION}. "
                      f"Re-ingesting so the chunks match the code.")
        if only in (None, "pdf"):
            await ingest_pdf(conn)
        if only in (None, "sheet"):
            import ingest_numbers

            await ingest_numbers.ingest_sheet(conn)

        if only is None:
            # Only a full run guarantees every chunk came from this pipeline.
            await core.set_meta(conn, "pipeline_version", core.PIPELINE_VERSION)

        total = await conn.fetchval("SELECT count(*) FROM document_chunks")
        by_source = await conn.fetch(
            "SELECT source, count(*) FROM document_chunks GROUP BY source ORDER BY source"
        )
        print("\nDatabase now holds:")
        for row in by_source:
            print(f"  {row['source']:<12} {row['count']}")
        print(f"  {'TOTAL':<12} {total}")
    finally:
        await conn.close()
    print("\nIngestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["pdf", "sheet"], help="ingest a single source")
    parser.add_argument("--if-empty", action="store_true",
                        help="skip if the database already has chunks (used on container start)")
    args = parser.parse_args()
    asyncio.run(main(args.only, args.if_empty))

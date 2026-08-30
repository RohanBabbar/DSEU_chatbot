# DSEU Admissions Brochure Chatbot

A retrieval-augmented (RAG) chatbot that answers prospective students' questions about
DSEU admissions, grounded strictly in two source documents:

- `DSEU_Admission_Brochure_2026_updated 9.5.2026.pdf` — the official 2026 brochure
- `programs_updated.numbers` — the updated programs / exit-options spreadsheet, treated
  as the more authoritative of the two where they disagree

Embeddings are computed locally and for free; only the final answer is written by a
hosted model.

```
PDF + .numbers ──ingest──► Postgres (pgvector) ──hybrid search──► OpenAI ──► browser
                           document_chunks         vector + FTS,
                                                   fused with RRF
```

## Running it (for testers)

You need **Docker Desktop** and an **OpenAI API key**. Nothing else — no Python, no
virtualenv, no separate ingest step.

```bash
git clone https://github.com/RohanBabbar/DSEU_chatbot.git
cd DSEU_chatbot

cp .env.example .env       # open .env and paste the OpenAI key into OPENAI_API_KEY

docker compose up          # first run takes a few minutes to build
```

Then open <http://localhost:8000> and ask it something.

The first `up` builds the image, starts Postgres, parses the brochure and the
spreadsheet into ~628 searchable chunks, and starts the server. That happens once;
later runs start in seconds because the parsed data lives in a Docker volume.
`docker compose down` stops it, `docker compose down -v` also wipes the parsed data so
the next start re-ingests from scratch.

**After `git pull`, run `docker compose up --build`.** The build picks up the new code,
and if the chunking changed, the container notices its stored `PIPELINE_VERSION`
(`backend/core.py`) no longer matches and re-ingests automatically. That stops anyone
testing new code against chunks built by the old pipeline — which would look like a
bug in the code and waste your time.

The API key is **never** committed — `.env` is in `.gitignore`. Whoever shares the repo
sends the key separately.

If something looks wrong, `curl localhost:8000/api/health` reports the chunk counts and
whether the key and model validated.

### Running it without Docker

Useful if you are changing the code, since it gives you `--reload`:

```bash
python -m venv chatbot_env
./chatbot_env/bin/pip install -r requirements.txt

cp .env.example .env                      # add your OpenAI API key
docker compose up -d db                   # just Postgres
./chatbot_env/bin/python backend/ingest.py

./chatbot_env/bin/uvicorn main:app --app-dir backend --reload
```

`OPENAI_MODEL` defaults to `gpt-4.1-mini` and is validated when the server starts, so a
wrong key or model name fails at boot with a clear message rather than as a 500 on
every question.

## Reporting a bad answer

Please include:

1. the exact question you asked,
2. the answer you got, and the answer you expected,
3. the page of the brochure or the row of the spreadsheet that proves it.

Point 3 is what makes a report actionable — it separates "the bot could not find it"
from "the bot found it and read it wrong", which have different fixes. Bad answers get
added to `backend/eval.py` as permanent test cases.

## Layout

| Path | Purpose |
|---|---|
| [backend/core.py](backend/core.py) | Config, embedding, schema, and the hybrid search query — shared by everything else |
| [backend/ingest.py](backend/ingest.py) | Parses the brochure PDF; also runs the spreadsheet ingest by default |
| [backend/ingest_numbers.py](backend/ingest_numbers.py) | Parses the `.numbers` spreadsheet |
| [backend/main.py](backend/main.py) | FastAPI server: `/api/chat`, `/api/health`, serves the frontend |
| [backend/test_retrieval.py](backend/test_retrieval.py) | Retrieval debugger — shows what the model is given, no LLM call |
| [db/init.sql](db/init.sql) | Schema; applied on first volume creation |
| [frontend/index.html](frontend/index.html) | Single-file chat UI, no build step |

## Ingestion

```bash
python backend/ingest.py               # both sources
python backend/ingest.py --only pdf     # brochure only
python backend/ingest.py --only sheet   # spreadsheet only
```

Each source deletes only its own rows, so the two can be re-run independently, in any
order, without wiping each other.

Three properties matter for answer quality, and ingestion enforces all three:

- **Tables are stored once.** PyMuPDF's page text includes table content as flattened
  lines with the column alignment destroyed. Ingestion removes the table regions from
  the page text so each table exists only in its structured Markdown form.
- **Every chunk fits the embedding model.** `all-mpnet-base-v2` silently truncates past
  384 tokens; anything beyond that is invisible to vector search. Chunks are built
  token-aware and verified before insert.
- **No empty chunks.** An empty chunk still carries a unit-norm vector that competes in
  nearest-neighbour search. A `CHECK` constraint makes them impossible.

Wide table rows — the brochure often puts a whole eligibility paragraph in one cell —
are emitted vertically as `Column: value` records so no single row blows the limit.

### The spreadsheet is the source of truth

`programs_updated.numbers` is authoritative for programs, durations, levels and exit
options; the system prompt tells the model to follow it over the brochure wherever they
disagree.

DSEU programs run four years with an exit point each year, so a qualification can be
earned by *leaving early* — you get a BCA by enrolling in `B S Computer Applications`
and exiting after Year 3. Indexed by program alone, "which program gets me a BCA?" has
nothing to match, and the bot used to answer from the brochure and miss the exit route
entirely. Asked about a BBA it would even reply "DSEU does not offer a BBA degree",
when seven programs award one.

Ingestion therefore also builds a **reverse index**: one `Exit Qualification Pathway`
chunk per award (BCA, BBA, B.Voc., B.Sc., UG Diploma, UG Certificate, …), listing every
program that leads to it and the year to exit. Awards are recognised from the exit text
by [`AWARDS`](backend/ingest_numbers.py), most specific pattern first — "Exit with
certificate in BBA Retail Management" grants a BBA, not a certificate.

Keep those chunks lean. An earlier version repeated a paragraph of explanation in each
one, which made them all look alike to the embedder so every "how do I get X degree"
question retrieved all of them and crowded out the brochure. Shared explanation belongs
in the prompt, stated once.

## Retrieval

Hybrid search, fused with Reciprocal Rank Fusion (`1/(60+rank)` from each side):

1. Vector k-NN over `embedding` using cosine (`<=>`), matching the HNSW index's
   opclass so the index is actually usable
2. Postgres full-text search over the generated `tsv` column
3. Top 10 fused chunks go to the model

The full-text branch ORs the query's lexemes rather than using
`websearch_to_tsquery` directly. `websearch_to_tsquery` ANDs every term, so a real
question — "how do I get a BBA degree?" — required a chunk containing *want*, *get* and
*degree* as well as *BBA*, and matched **nothing**. The branch silently contributed
zero on every natural-language question. ORing the lexemes and letting `ts_rank` do the
discriminating restores it.

Debug what retrieval returns for any question, with no LLM call or cost:

```bash
python backend/test_retrieval.py "what is the eligibility for BBA?"
```

## Testing

`backend/eval.py` is a regression suite over facts checked by hand against the source
documents. Start the server, then:

```bash
python backend/eval.py                # all cases
python backend/eval.py --only bba     # just the BBA cases
python backend/eval.py --show         # print full answers
```

It exits non-zero on any failure, so it can gate a release. Cases assert on facts
(`87,000`, `15%`, `7.5%`, which programs award a BBA) rather than exact wording, since
phrasing varies run to run while the facts must not. Two cases assert the bot
*declines* — an off-topic question and one whose answer lives on the scanned pages —
because a confident answer there is worse than no answer.

Add a case whenever someone reports a bad answer: it takes one tuple in `CASES`, and it
stops that specific regression coming back.

To test by hand, `test_retrieval.py` is the fast way to localise a problem:

- fact not in the retrieved chunks → retrieval problem (chunking, or it's on a scanned page)
- fact is in the chunks but the answer is still wrong → prompt problem

## Known limitation: scanned pages

**Brochure pages 1–68 include roughly 35 pages that are scanned images with no
extractable text** (page 1, page 8, and most of 32–68). These hold academic
regulations, credit requirements, semester rules and examination policy.

The chatbot cannot see any of it. Asked about that material it will say it does not
have the information rather than inventing an answer — verified. All content from
**page 69 onward is fully extracted** (38 of 38 pages, no gaps), which covers programs,
eligibility, fees, reservation policy and campus details.

To close the gap, OCR those pages or obtain a text-based PDF from the source.

## Health check

```bash
curl localhost:8000/api/health
```

Reports chunk counts per source, the models in use, and whether the LLM key/model
validated at startup.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `RuntimeError: No API key found` at boot | `OPENAI_API_KEY` missing from `.env` |
| `503` with a model name in the message | `OPENAI_MODEL` isn't available to your key; the error lists valid alternatives |
| `WARNING: document_chunks is empty` | Run `python backend/ingest.py` |
| Bot says it has no information | Check `test_retrieval.py` first — if the chunk isn't retrieved it's a retrieval problem; if it is retrieved, it's a prompt problem. If the fact lives on pages 1–68, see the scanned-pages limitation above |
| Connection refused on :5432 | `docker compose up -d` |

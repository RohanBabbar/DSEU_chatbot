FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch. The default wheels bundle CUDA and add gigabytes we cannot use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so no tester waits on a 420MB download
# and the first question is not slow.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-mpnet-base-v2')"

COPY backend/ backend/
COPY frontend/ frontend/
COPY db/ db/
COPY DSEU_Admission_Brochure_2026_updated*.pdf ./
COPY programs_updated.numbers ./

EXPOSE 8000

# Ingest on first boot only, then serve. Re-running the container is cheap.
CMD ["sh", "-c", "python backend/ingest.py --if-empty && exec uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000"]

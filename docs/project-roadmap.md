# Project Roadmap: DSEU Admissions Chatbot

## 1. Project Goal
Develop an in-house admissions chatbot that answers queries strictly using official Delhi Skill and Entrepreneurship University (DSEU) data, eliminates recurring third-party API costs, and ultimately provides a voice interface.

## 2. Current Architecture
- **Retrieval-Augmented Generation (RAG)** pipeline.
- **Database & Search:** PostgreSQL with `pgvector` for hybrid search (vector k-NN combined with full-text search).
- **Backend Framework:** FastAPI running inside Docker.
- **Language Model:** Local `Qwen2.5` served via Ollama.

## 3. Planned Work (Planned, Not Yet Verified)
- Full OpenAI removal and comprehensive offline verification.
- Fine-tuning experimentation.
- Dedicated source-link index to handle direct document/URL references.
- Voice channel integration (speech-to-text and text-to-speech pipelines).

## 4. Known Issues Being Investigated
- Slow local inference latency on the development Mac.
- Test evaluation scripts requiring maintenance and bug fixes.

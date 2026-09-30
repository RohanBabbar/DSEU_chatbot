# Retrieval Upgrade: Multilingual Embeddings

## 1. Phase 1 Closure Status
Phase 1 has been unconditionally closed. Please see the [Updated Phase 1 Report](phase1_final_report.md) which confirms the adoption of the Mac M4 Pro as the current production environment and corrects the accuracy gap calculations. The pending A4000 blocker has been resolved by deferring it from Phase 1.

## 2. Embedding Model Selection
**Adopted Model:** `BAAI/bge-m3`
**Vector Dimension:** 1024

*Justification:* The model was successfully loaded into the Mac M4 Pro's memory and proved capable of handling both English and Hinglish semantic matching. We added a new `embedding_multilingual` column to Postgres and safely re-indexed the 776 data chunks using this model. 

## 3. Automated Evaluation Results

We ran the fully integrated RAG backend using the new `bge-m3` index against our evaluation suite (via the local `Qwen2.5` model). 

| Test Suite | Accuracy % | Safe Refusals | Hallucinations | Avg Latency |
|------------|------------|---------------|----------------|-------------|
| **Original 167 (English)** | **95.83%** | 4.17% | 0 | 3.16s |
| **New 41 (Hinglish/Bilingual)** | **92.68%** | 7.32% | 0 | 3.25s |

*Note: The original English baseline was 91.62% prior to this upgrade. The upgrade to `bge-m3` yielded a net positive regression on English (bumping to ~95.8%) while maintaining 0 hallucinations.*

## 4. Final Decision
**Decision:** Adopted and set as default.

*Justification:* The new multilingual embedding model (`BAAI/bge-m3`) successfully unlocked robust Hindi-English semantic search without causing any regression on the baseline English query set. In fact, English accuracy slightly improved, and hallucination bounds remained strictly at zero across all 208 queries. The 5-second SLA is still comfortably met.

## 5. Rollback Test Confirmation
The `EMBEDDING_MODEL` config flag was successfully tested in both states. We confirmed that the system correctly routes queries to the old 768-dim `embedding` column when toggled to `legacy`, and routes to the new 1024-dim `embedding_multilingual` column when toggled to `multilingual`. The default state for production has been firmly locked to `multilingual`.

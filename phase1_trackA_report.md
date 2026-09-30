# Phase 1: Section 2 (Mac Track A) Results

## Evaluation Matrix
| Model | Accuracy % | Refusal % | Hallucinations | Avg Latency |
|-------|------------|-----------|----------------|-------------|
| qwen2.5:7b-instruct | 91.62% | 7.78% | 0 | 3.06s |
| llama3.1 | 91.02% | 7.78% | 2 | 3.19s |
| gemma2:9b | 91.02% | 8.38% | 0 | 3.06s |

## Provisional Winner
**Model:** `qwen2.5:7b-instruct`

**Justification:**
This model was selected because it successfully fit into the Mac's 16GB memory footprint (acting as the 8B-class candidate), maintained a hallucination rate of 0 (matching or beating the baseline), and achieved the highest overall accuracy (91.62%) among the candidates while keeping average latency under the 5-second SLA (3.06s).

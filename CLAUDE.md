# RAG Retrieval & Re-Ranking Audit Tool

## Tech stack
- Python 3.10+
- Streamlit — web UI (`app.py`)
- Pydantic v2 — typed data models for audit results
- `requests` — DataForSEO Live SERP API + OpenAI chat completions
- `beautifulsoup4` — server-rendered HTML parsing / heading-bounded chunking

## Run commands
```bash
pip install -r requirements.txt
streamlit run app.py
```

No credentials are required to try the tool: leave the DataForSEO and OpenAI
fields in the sidebar blank and both stages run in deterministic simulated
mode (clearly flagged in the diagnostics panel).

## Files
- `auditor.py` — the audit engine. All business logic lives here; `app.py`
  only renders it. No Streamlit imports belong in `auditor.py`.
- `app.py` — Streamlit UI: sidebar for credentials/thresholds, a form for
  target URL + primary query, and a results dashboard.

## Core architecture: the K1 / K2 gate

This tool audits whether a page *can possibly* be cited by an LLM answer
engine, modeled as a two-stage funnel:

1. **K1 — Retrieval.** `DataForSEORetrievalAuditor` fans the primary query
   out into a handful of sub-queries (the way a RAG query planner would) and
   checks the target domain's rank for each via DataForSEO's live SERP API.
   K1 passes if the best rank across all queries is within the configured
   cutoff (default: top 20).

2. **K2 — Passage scoring.** `HTMLPassageChunker` fetches the page's raw
   server HTML (User-Agent spoofed to OAI-SearchBot — no JS execution,
   matching how most answer-engine crawlers see the page) and splits it into
   chunks bounded by `h1`/`h2`/`h3` headings. `PassageScorer` rates each
   chunk on three dimensions a cross-encoder re-ranker rewards:
   - **conclusion-first syntax** — does the chunk state its answer in the
     first sentence instead of building up to it?
   - **entity self-containment** — can the chunk be read in isolation
     without depending on pronouns that refer outside it?
   - **factual density** — concrete numbers, dates, and named entities per
     word.
   K2 passes if *any* chunk's composite score clears the configured
   threshold (default: 0.6).

3. **The gating rule.** Anderson's conditional probability equation is
   enforced exactly as written:

   ```
   P(Cited | Rank_R > K1 OR Rank_F > K2) = 0
   ```

   If retrieval misses the K1 cutoff **or** no passage clears K2, citation
   probability is forced to `0.0` — never averaged or partially credited.
   Only when both gates pass does the engine compute a non-zero citation
   probability from how far each signal clears its threshold.

## Fallback / simulation rules
- No DataForSEO credentials → `DataForSEORetrievalAuditor` returns
  deterministic simulated ranks (seeded by query + domain, so repeat runs
  are reproducible) instead of calling the live API. `RetrievalAudit.simulated`
  is set so the UI can flag it.
- No OpenAI key → `PassageScorer` skips the LLM refinement call and uses the
  heuristic (regex-based) scores directly. `PassageScore.scoring_method` is
  `"heuristic"` instead of `"llm"`.
- A failed HTML fetch (dead URL, timeout, non-200) is **never** simulated —
  it surfaces as `PassageAudit.fetch_error` and fails K2 honestly rather than
  fabricating page content.

## Conventions
- Keep all scoring/ranking logic in `auditor.py`, fully typed with Pydantic
  models — no bare dicts crossing module boundaries.
- Any new external API call must handle `requests.exceptions.RequestException`
  explicitly and degrade to a clearly-flagged fallback rather than raising
  into the Streamlit UI.

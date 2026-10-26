"""
Hybrid audit engine implementing Anderson's two-stage RAG retrieval and
re-ranking audit framework.

Stage K1 (Retrieval):   Does the target domain rank within the top-K1 cutoff
                        for the primary query and its fan-out sub-queries,
                        per a live Google SERP (DataForSEO)?

Stage K2 (Passage):     Among the page's heading-bounded passages, does any
                        chunk clear the K2 fact/structure score threshold
                        (conclusion-first syntax, entity self-containment,
                        factual density) as an LLM re-ranker would see it?

Anderson's gating rule, enforced in `HybridAuditEngine.run_audit`:

    P(Cited | Rank_R > K1 OR Rank_F > K2) = 0

In words: if the page fails retrieval (misses the K1 cutoff) OR fails
re-ranking (no passage clears K2), the probability of citation is forced to
zero regardless of any other signal. Both gates must pass for citation to
be possible at all.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DATAFORSEO_ENDPOINT = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_LOCATION_CODE = 2840  # United States
DEFAULT_LANGUAGE_CODE = "en"
REQUEST_TIMEOUT = 15
SERP_REQUEST_TIMEOUT = 60  # Live Advanced SERP tasks routinely take longer than a generic HTTP call.

SEARCHBOT_USER_AGENT = "Mozilla/5.0 (compatible; OAI-SearchBot/1.0; +https://openai.com/searchbot)"

FILLER_OPENERS = (
    "in this article", "in this post", "in this guide", "in this blog",
    "let's", "lets", "have you ever", "this section", "welcome to",
    "today we", "in this piece", "this guide will", "this article will",
    "before we", "so you want to",
)

ANSWER_PATTERNS = re.compile(
    r"\b(is|are|was|were|means|refers to|defined as|consists of|equals|"
    r"represents|includes|requires|costs|causes|results in)\b",
    re.IGNORECASE,
)

LEADING_PRONOUNS = re.compile(r"^\s*(it|this|that|these|those|they|he|she)\b", re.IGNORECASE)

NUMERIC_TOKEN = re.compile(r"\b\d+(\.\d+)?%?\b")
YEAR_TOKEN = re.compile(r"\b(19|20)\d{2}\b")
PROPER_NOUN_RUN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
HEADING_TAGS = ("h1", "h2", "h3")
SKIP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "svg", "form")


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------

class SubQueryRank(BaseModel):
    query: str
    is_primary: bool = False
    rank: Optional[int] = None
    found: bool = False


class RetrievalAudit(BaseModel):
    primary_query: str
    target_domain: str
    k1_cutoff: int
    sub_queries: list[SubQueryRank] = Field(default_factory=list)
    best_rank: Optional[int] = None
    passed: bool = False
    simulated: bool = False
    error: Optional[str] = None


class PassageChunk(BaseModel):
    index: int
    heading: str
    level: int
    text: str
    word_count: int


class PassageScore(BaseModel):
    chunk: PassageChunk
    conclusion_first: float
    entity_self_contained: float
    factual_density: float
    composite: float
    passed: bool
    scoring_method: str  # "llm" or "heuristic"


class PassageAudit(BaseModel):
    target_url: str
    k2_threshold: float
    chunks_evaluated: int = 0
    scores: list[PassageScore] = Field(default_factory=list)
    best_score: Optional[float] = None
    passed: bool = False
    simulated: bool = False
    fetch_error: Optional[str] = None


class AuditReport(BaseModel):
    target_url: str
    primary_query: str
    retrieval: RetrievalAudit
    passage: PassageAudit
    citation_probability: float
    overall_passed: bool
    diagnostics: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _domain_from_url(url: str) -> str:
    cleaned = re.sub(r"^https?://", "", url.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^www\.", "", cleaned, flags=re.IGNORECASE)
    return cleaned.split("/")[0].lower()


def _seeded_rng_value(*parts: str) -> int:
    """Deterministic pseudo-random int in [0, 10_000) derived from inputs,
    used only for simulation fallbacks so repeated runs are reproducible."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


# --------------------------------------------------------------------------
# K1: DataForSEO retrieval auditor
# --------------------------------------------------------------------------

class DataForSEORetrievalAuditor:
    """Checks whether a target domain ranks within the K1 cutoff for a
    primary query and its fan-out sub-queries, using DataForSEO's live SERP
    API. Falls back to a deterministic simulation when no credentials are
    supplied, so the tool remains usable for demos and offline review."""

    def __init__(self, login: str = "", password: str = "", k1_cutoff: int = 20):
        self.login = (login or "").strip()
        self.password = (password or "").strip()
        self.k1_cutoff = k1_cutoff

    @property
    def has_credentials(self) -> bool:
        return bool(self.login and self.password)

    # Lead-in phrases that would double up if prepended to a query already
    # phrased as a question (e.g. "what is X" -> "what is what is X").
    _REDUNDANT_LEADINS = ("what is", "what are", "how does", "how do", "why is", "why are", "best", "top")

    def generate_fanout_queries(self, primary_query: str, limit: int = 4) -> list[str]:
        """Expands the primary query into the sub-queries a RAG system's
        query planner would typically fan out to."""
        base = primary_query.strip().rstrip("?")
        base_lower = base.lower()
        already_has_leadin = base_lower.startswith(self._REDUNDANT_LEADINS)

        candidates: list[str] = []
        if not already_has_leadin:
            candidates.append(f"what is {base}")
            candidates.append(f"how does {base} work")
            candidates.append(f"best {base}")
        # Suffix templates never collide with an existing lead-in, so they
        # always apply.
        candidates.append(f"{base} benefits")
        candidates.append(f"{base} vs alternatives")

        seen: set[str] = set()
        fanout: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip().lower()
            if normalized == base_lower or normalized in seen:
                continue
            seen.add(normalized)
            fanout.append(candidate)
            if len(fanout) >= limit:
                break
        return fanout

    def _simulate_rank(self, query: str, target_domain: str) -> Optional[int]:
        """Deterministic stand-in for a live SERP lookup when no API
        credentials are configured."""
        value = _seeded_rng_value(query.lower(), target_domain)
        # Roughly 55% chance of ranking somewhere in 1-40, else "not found".
        if value % 100 < 55:
            return (value % 40) + 1
        return None

    def _query_serp(self, query: str, target_domain: str) -> tuple[Optional[int], Optional[str]]:
        """Issues one live SERP lookup. Returns (rank, error_message)."""
        payload = [{
            "keyword": query,
            "location_code": DEFAULT_LOCATION_CODE,
            "language_code": DEFAULT_LANGUAGE_CODE,
            "device": "desktop",
            "os": "windows",
            "depth": max(self.k1_cutoff, 20),
        }]
        try:
            response = requests.post(
                DATAFORSEO_ENDPOINT,
                auth=(self.login, self.password),
                json=payload,
                timeout=SERP_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            return None, f"DataForSEO request failed: {exc}"
        except ValueError as exc:
            return None, f"DataForSEO returned invalid JSON: {exc}"

        try:
            tasks = data.get("tasks") or []
            if not tasks or tasks[0].get("status_code") not in (20000,):
                status_message = tasks[0].get("status_message") if tasks else "no task returned"
                return None, f"DataForSEO task error: {status_message}"
            results = tasks[0].get("result") or []
            if not results:
                return None, None  # Valid response, zero results for this query.
            items = results[0].get("items") or []
        except (KeyError, IndexError, AttributeError) as exc:
            return None, f"DataForSEO response shape unexpected: {exc}"

        for item in items:
            item_domain = re.sub(r"^www\.", "", (item.get("domain") or "").lower())
            item_url = (item.get("url") or "").lower()
            if target_domain in item_domain or target_domain in item_url:
                rank = item.get("rank_absolute") or item.get("rank_group")
                return rank, None
        return None, None  # Not found in returned items.

    def audit_retrieval(self, primary_query: str, target_url: str) -> RetrievalAudit:
        target_domain = _domain_from_url(target_url)
        simulated = not self.has_credentials
        queries = [SubQueryRank(query=primary_query, is_primary=True)]
        for sub_query in self.generate_fanout_queries(primary_query):
            queries.append(SubQueryRank(query=sub_query))

        error: Optional[str] = None
        for entry in queries:
            if simulated:
                rank = self._simulate_rank(entry.query, target_domain)
            else:
                rank, query_error = self._query_serp(entry.query, target_domain)
                if query_error:
                    error = query_error
            entry.rank = rank
            entry.found = rank is not None

        found_ranks = [e.rank for e in queries if e.rank is not None]
        best_rank = min(found_ranks) if found_ranks else None
        passed = best_rank is not None and best_rank <= self.k1_cutoff

        return RetrievalAudit(
            primary_query=primary_query,
            target_domain=target_domain,
            k1_cutoff=self.k1_cutoff,
            sub_queries=queries,
            best_rank=best_rank,
            passed=passed,
            simulated=simulated,
            error=error,
        )


# --------------------------------------------------------------------------
# K2: HTML passage chunker
# --------------------------------------------------------------------------

class HTMLPassageChunker:
    """Fetches raw server-rendered HTML — simulating how OAI-SearchBot sees
    a page, i.e. no JavaScript execution — and splits it into passages
    bounded by h1/h2/h3 headings, the same unit a cross-encoder re-ranker
    would score independently."""

    def __init__(self, user_agent: str = SEARCHBOT_USER_AGENT, min_chunk_words: int = 20):
        self.user_agent = user_agent
        self.min_chunk_words = min_chunk_words

    def fetch_html(self, url: str) -> tuple[Optional[str], Optional[str]]:
        """Returns (html, error_message)."""
        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.text, None
        except requests.exceptions.RequestException as exc:
            return None, f"Could not fetch target URL: {exc}"

    def chunk_by_headings(self, html: str) -> list[PassageChunk]:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(SKIP_TAGS):
            tag.decompose()

        body = soup.body or soup
        chunks: list[PassageChunk] = []
        current_heading = "Untitled section"
        current_level = 0
        current_text_parts: list[str] = []

        def flush() -> None:
            text = " ".join(part.strip() for part in current_text_parts if part.strip())
            text = re.sub(r"\s+", " ", text).strip()
            word_count = len(text.split())
            if text and word_count >= self.min_chunk_words:
                chunks.append(PassageChunk(
                    index=len(chunks),
                    heading=current_heading,
                    level=current_level,
                    text=text,
                    word_count=word_count,
                ))

        for element in body.find_all(True, recursive=True):
            if element.name in HEADING_TAGS:
                flush()
                current_heading = element.get_text(strip=True) or "Untitled section"
                current_level = int(element.name[1])
                current_text_parts = []
            elif element.name in ("p", "li", "td", "blockquote", "figcaption"):
                text = element.get_text(" ", strip=True)
                if text:
                    current_text_parts.append(text)
        flush()

        if not chunks:
            # No headings found: fall back to one chunk from the whole body text.
            full_text = re.sub(r"\s+", " ", body.get_text(" ", strip=True))
            if len(full_text.split()) >= self.min_chunk_words:
                chunks.append(PassageChunk(
                    index=0,
                    heading="Full page (no headings found)",
                    level=0,
                    text=full_text,
                    word_count=len(full_text.split()),
                ))
        return chunks


# --------------------------------------------------------------------------
# K2: Passage scoring logic
# --------------------------------------------------------------------------

class PassageScorer:
    """Scores passages the way a cross-encoder re-ranker and an LLM citation
    pass would jointly favor: conclusion-first syntax, entity
    self-containment, and factual density. Uses an OpenAI chat completion as
    an optional refinement signal; falls back to pure heuristics when no key
    is supplied or the call fails."""

    WEIGHTS = {"conclusion_first": 0.4, "entity_self_contained": 0.3, "factual_density": 0.3}

    def __init__(self, openai_api_key: str = "", k2_threshold: float = 0.6):
        self.openai_api_key = (openai_api_key or "").strip()
        self.k2_threshold = k2_threshold

    @property
    def has_credentials(self) -> bool:
        return bool(self.openai_api_key)

    @staticmethod
    def _score_conclusion_first(text: str) -> float:
        sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return 0.0
        first = sentences[0].strip().lower()
        score = 0.5
        if any(first.startswith(opener) for opener in FILLER_OPENERS):
            score -= 0.4
        if ANSWER_PATTERNS.search(first):
            score += 0.3
        if len(first.split()) <= 25:
            score += 0.1
        if NUMERIC_TOKEN.search(first):
            score += 0.1
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_entity_self_containment(text: str, heading: str) -> float:
        sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return 0.0
        pronoun_leads = sum(1 for s in sentences if LEADING_PRONOUNS.match(s.strip()))
        pronoun_ratio = pronoun_leads / len(sentences)
        score = 1.0 - pronoun_ratio

        heading_terms = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", heading)]
        if heading_terms:
            body_lower = text.lower()
            mentioned = sum(1 for term in heading_terms if term in body_lower)
            coverage = mentioned / len(heading_terms)
            score = (score * 0.6) + (coverage * 0.4)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_factual_density(text: str) -> float:
        words = text.split()
        word_count = max(len(words), 1)
        numeric_hits = len(NUMERIC_TOKEN.findall(text))
        year_hits = len(YEAR_TOKEN.findall(text))
        proper_noun_hits = len(PROPER_NOUN_RUN.findall(text))
        raw_density = (numeric_hits + year_hits + (proper_noun_hits * 0.5)) / word_count
        # Scale so ~1 factual token per 15 words reaches the top of the range.
        return max(0.0, min(1.0, raw_density * 15))

    def _heuristic_scores(self, chunk: PassageChunk) -> dict[str, float]:
        return {
            "conclusion_first": self._score_conclusion_first(chunk.text),
            "entity_self_contained": self._score_entity_self_containment(chunk.text, chunk.heading),
            "factual_density": self._score_factual_density(chunk.text),
        }

    def _llm_refine_scores(self, chunk: PassageChunk, heuristic: dict[str, float]) -> tuple[dict[str, float], bool]:
        """Asks an LLM to rate the same three dimensions. Returns
        (scores, used_llm). On any failure, returns the heuristic scores
        unchanged and used_llm=False."""
        prompt = (
            "Rate this passage on three dimensions, each 0.0 to 1.0, as a "
            "citation re-ranker would:\n"
            "1. conclusion_first: does it state its answer/conclusion in the "
            "first sentence rather than building up to it?\n"
            "2. entity_self_contained: can this passage be read in isolation "
            "without relying on pronouns referring to text outside it?\n"
            "3. factual_density: how many concrete facts, numbers, or named "
            "entities does it pack in relative to its length?\n\n"
            f"Heading: {chunk.heading}\nPassage: {chunk.text[:1500]}\n\n"
            "Respond with ONLY a JSON object: "
            '{"conclusion_first": float, "entity_self_contained": float, "factual_density": float}'
        )
        try:
            response = requests.post(
                OPENAI_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            scores = {
                key: max(0.0, min(1.0, float(parsed[key])))
                for key in ("conclusion_first", "entity_self_contained", "factual_density")
            }
            return scores, True
        except (requests.exceptions.RequestException, KeyError, ValueError, TypeError, json.JSONDecodeError):
            return heuristic, False

    def score_chunk(self, chunk: PassageChunk) -> PassageScore:
        heuristic = self._heuristic_scores(chunk)
        if self.has_credentials:
            scores, used_llm = self._llm_refine_scores(chunk, heuristic)
        else:
            scores, used_llm = heuristic, False

        composite = sum(scores[key] * weight for key, weight in self.WEIGHTS.items())
        return PassageScore(
            chunk=chunk,
            conclusion_first=scores["conclusion_first"],
            entity_self_contained=scores["entity_self_contained"],
            factual_density=scores["factual_density"],
            composite=round(composite, 3),
            passed=composite >= self.k2_threshold,
            scoring_method="llm" if used_llm else "heuristic",
        )


# --------------------------------------------------------------------------
# Hybrid audit engine
# --------------------------------------------------------------------------

class HybridAuditEngine:
    """Ties the K1 retrieval auditor and K2 passage scorer together and
    enforces Anderson's gating equation: P(Cited | Rank_R > K1 OR Rank_F > K2) = 0."""

    def __init__(
        self,
        dataforseo_login: str = "",
        dataforseo_password: str = "",
        openai_api_key: str = "",
        k1_cutoff: int = 20,
        k2_threshold: float = 0.6,
    ):
        self.retrieval_auditor = DataForSEORetrievalAuditor(
            login=dataforseo_login, password=dataforseo_password, k1_cutoff=k1_cutoff,
        )
        self.chunker = HTMLPassageChunker()
        self.scorer = PassageScorer(openai_api_key=openai_api_key, k2_threshold=k2_threshold)

    def _audit_passages(self, target_url: str) -> PassageAudit:
        html, fetch_error = self.chunker.fetch_html(target_url)
        if fetch_error:
            return PassageAudit(
                target_url=target_url,
                k2_threshold=self.scorer.k2_threshold,
                fetch_error=fetch_error,
            )

        chunks = self.chunker.chunk_by_headings(html)
        if not chunks:
            return PassageAudit(
                target_url=target_url,
                k2_threshold=self.scorer.k2_threshold,
                fetch_error="Page fetched successfully but no scoreable text content was found.",
            )

        scores = [self.scorer.score_chunk(chunk) for chunk in chunks]
        best_score = max((s.composite for s in scores), default=None)
        return PassageAudit(
            target_url=target_url,
            k2_threshold=self.scorer.k2_threshold,
            chunks_evaluated=len(scores),
            scores=scores,
            best_score=best_score,
            passed=any(s.passed for s in scores),
            simulated=not self.scorer.has_credentials,
        )

    @staticmethod
    def _build_diagnostics(retrieval: RetrievalAudit, passage: PassageAudit) -> list[str]:
        diagnostics: list[str] = []

        if retrieval.passed:
            diagnostics.append(
                f"PASS K1: best rank #{retrieval.best_rank} is within the top {retrieval.k1_cutoff} "
                f"cutoff (across {len(retrieval.sub_queries)} queries)."
            )
        else:
            rank_desc = f"#{retrieval.best_rank}" if retrieval.best_rank else "not found in results"
            diagnostics.append(
                f"FAIL K1: best rank is {rank_desc}, outside the top {retrieval.k1_cutoff} cutoff "
                f"across {len(retrieval.sub_queries)} queries."
            )
        if retrieval.simulated:
            diagnostics.append("K1 ran in SIMULATED mode — no DataForSEO credentials were supplied.")
        if retrieval.error:
            diagnostics.append(f"K1 warning: {retrieval.error}")

        if passage.fetch_error:
            diagnostics.append(f"FAIL K2: {passage.fetch_error}")
        elif passage.passed:
            diagnostics.append(
                f"PASS K2: best passage score {passage.best_score:.2f} clears threshold "
                f"{passage.k2_threshold:.2f} ({passage.chunks_evaluated} chunks evaluated)."
            )
        else:
            best = f"{passage.best_score:.2f}" if passage.best_score is not None else "n/a"
            diagnostics.append(
                f"FAIL K2: no passage reached threshold {passage.k2_threshold:.2f} "
                f"(best score: {best}, {passage.chunks_evaluated} chunks evaluated)."
            )
        if passage.simulated and not passage.fetch_error:
            diagnostics.append("K2 ran in heuristic-only mode — no OpenAI key was supplied.")

        return diagnostics

    def run_audit(self, target_url: str, primary_query: str) -> AuditReport:
        retrieval = self.retrieval_auditor.audit_retrieval(primary_query, target_url)
        passage = self._audit_passages(target_url)

        # Anderson's gate: P(Cited | Rank_R > K1 OR Rank_F > K2) = 0
        gate_failed = (not retrieval.passed) or (not passage.passed)
        if gate_failed:
            citation_probability = 0.0
        else:
            # Both gates cleared: estimate probability from how far each
            # signal clears its threshold, as a simple confidence proxy.
            rank_margin = 1.0 - (retrieval.best_rank / max(retrieval.k1_cutoff, 1))
            score_margin = (passage.best_score - passage.k2_threshold) / max(1.0 - passage.k2_threshold, 1e-6)
            citation_probability = round(max(0.0, min(1.0, 0.5 + 0.25 * rank_margin + 0.25 * score_margin)), 3)

        diagnostics = self._build_diagnostics(retrieval, passage)
        if gate_failed:
            reason = "Rank_R > K1" if not retrieval.passed else "Rank_F > K2"
            diagnostics.append(f"GATE: citation probability forced to 0 ({reason}).")

        return AuditReport(
            target_url=target_url,
            primary_query=primary_query,
            retrieval=retrieval,
            passage=passage,
            citation_probability=citation_probability,
            overall_passed=not gate_failed,
            diagnostics=diagnostics,
        )

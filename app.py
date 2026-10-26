"""Streamlit front end for the RAG Retrieval & Re-Ranking Audit tool.

Implements Anderson's two-stage audit:
  K1 — retrieval rank cutoff (DataForSEO live SERP)
  K2 — cross-encoder passage scoring (heading-bounded HTML chunks)
"""

import streamlit as st

from auditor import AuditReport, HybridAuditEngine

st.set_page_config(page_title="RAG Retrieval Auditor", page_icon="🔎", layout="wide")


def _badge(passed: bool, label: str) -> str:
    color = "#1a7f37" if passed else "#cf222e"
    text = f"PASS · {label}" if passed else f"FAIL · {label}"
    return (
        f'<span style="background-color:{color};color:white;padding:4px 10px;'
        f'border-radius:12px;font-size:0.85rem;font-weight:600;">{text}</span>'
    )


def render_sidebar() -> dict:
    st.sidebar.header("API Credentials")
    dataforseo_login = st.sidebar.text_input("DataForSEO Login", value="", type="default")
    dataforseo_password = st.sidebar.text_input("DataForSEO Password", value="", type="password")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", value="", type="password")

    st.sidebar.caption(
        "Leave credentials blank to run in simulated mode — useful for demos "
        "or offline review. Real audits require valid credentials."
    )

    st.sidebar.header("Thresholds")
    k1_cutoff = st.sidebar.slider(
        "K1 Rank Cutoff (Top N)", min_value=1, max_value=100, value=20, step=1,
        help="Target must rank within this position for at least one query to pass K1.",
    )
    k2_threshold = st.sidebar.slider(
        "K2 Fact Score Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Minimum composite passage score (conclusion-first, self-containment, factual density) to pass K2.",
    )

    return {
        "dataforseo_login": dataforseo_login,
        "dataforseo_password": dataforseo_password,
        "openai_api_key": openai_api_key,
        "k1_cutoff": k1_cutoff,
        "k2_threshold": k2_threshold,
    }


def render_metrics(report: AuditReport) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rank_display = f"#{report.retrieval.best_rank}" if report.retrieval.best_rank else "Not found"
        st.metric("Best SERP Rank", rank_display, help="Lowest rank found across primary + fan-out queries.")
    with col2:
        score_display = f"{report.passage.best_score:.2f}" if report.passage.best_score is not None else "n/a"
        st.metric("Best Passage Score", score_display, help="Highest composite K2 score across all chunks.")
    with col3:
        st.metric("Chunks Evaluated", report.passage.chunks_evaluated)
    with col4:
        st.metric("Citation Probability", f"{report.citation_probability:.0%}")

    badge_col1, badge_col2, badge_col3 = st.columns(3)
    with badge_col1:
        st.markdown(_badge(report.retrieval.passed, "K1 Retrieval"), unsafe_allow_html=True)
    with badge_col2:
        st.markdown(_badge(report.passage.passed, "K2 Passage"), unsafe_allow_html=True)
    with badge_col3:
        st.markdown(_badge(report.overall_passed, "Overall Citable"), unsafe_allow_html=True)


def render_retrieval_detail(report: AuditReport) -> None:
    st.subheader("K1 — Retrieval Detail")
    if report.retrieval.simulated:
        st.info("Running in simulated mode (no DataForSEO credentials supplied).", icon="⚠️")
    if report.retrieval.error:
        st.warning(report.retrieval.error, icon="⚠️")

    for entry in report.retrieval.sub_queries:
        rank_text = f"Rank #{entry.rank}" if entry.found else "Not in results"
        label = "Primary query" if entry.is_primary else "Fan-out sub-query"
        icon = "✅" if entry.found and entry.rank and entry.rank <= report.retrieval.k1_cutoff else "⬜"
        st.write(f"{icon} **[{label}]** `{entry.query}` → {rank_text}")


def render_passage_tabs(report: AuditReport) -> None:
    st.subheader("K2 — Passage Inspection")
    if report.passage.fetch_error:
        st.error(report.passage.fetch_error, icon="🚫")
        return
    if report.passage.simulated:
        if report.passage.llm_error:
            st.warning(f"OpenAI key was rejected, scoring heuristic-only: {report.passage.llm_error}", icon="⚠️")
        else:
            st.info("Scoring heuristic-only (no OpenAI key supplied) — composite still enforces the K2 gate.", icon="⚠️")
    if not report.passage.scores:
        st.warning("No passages were available to score.")
        return

    tab_labels = [f"{'✅' if s.passed else '❌'} {s.chunk.heading[:40]}" for s in report.passage.scores]
    tabs = st.tabs(tab_labels)
    for tab, score in zip(tabs, report.passage.scores):
        with tab:
            st.markdown(_badge(score.passed, f"Composite {score.composite:.2f}"), unsafe_allow_html=True)
            st.caption(f"Scoring method: {score.scoring_method} · Word count: {score.chunk.word_count}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Conclusion-First", f"{score.conclusion_first:.2f}")
            m2.metric("Entity Self-Contained", f"{score.entity_self_contained:.2f}")
            m3.metric("Factual Density", f"{score.factual_density:.2f}")
            st.text_area("Passage text", score.chunk.text, height=150, key=f"chunk_{score.chunk.index}")


def render_diagnostics(report: AuditReport) -> None:
    st.subheader("Diagnostics")
    for line in report.diagnostics:
        if line.startswith("PASS"):
            st.success(line)
        elif line.startswith("FAIL") or line.startswith("GATE"):
            st.error(line)
        else:
            st.warning(line)


def main() -> None:
    st.title("🔎 RAG Retrieval & Re-Ranking Audit Tool")
    st.caption(
        "Two-stage audit framework: K1 checks whether a page is *retrieved* "
        "within the top-N SERP cutoff; K2 checks whether any passage on that "
        "page would clear a cross-encoder re-ranker's citation bar. "
        "Citation probability is forced to zero if either gate fails."
    )

    settings = render_sidebar()

    with st.form("audit_form"):
        target_url = st.text_input("Target URL", placeholder="https://example.com/some-page")
        primary_query = st.text_input("Primary Query", placeholder="e.g. best project management software")
        submitted = st.form_submit_button("Run Audit", type="primary")

    if not submitted:
        st.info("Enter a target URL and primary query, then run the audit.")
        return

    if not target_url.strip() or not primary_query.strip():
        st.error("Both Target URL and Primary Query are required.")
        return

    engine = HybridAuditEngine(
        dataforseo_login=settings["dataforseo_login"],
        dataforseo_password=settings["dataforseo_password"],
        openai_api_key=settings["openai_api_key"],
        k1_cutoff=settings["k1_cutoff"],
        k2_threshold=settings["k2_threshold"],
    )

    with st.spinner("Running K1 retrieval check and K2 passage scoring..."):
        report = engine.run_audit(target_url.strip(), primary_query.strip())

    st.divider()
    render_metrics(report)
    st.divider()
    render_retrieval_detail(report)
    st.divider()
    render_passage_tabs(report)
    st.divider()
    render_diagnostics(report)


if __name__ == "__main__":
    main()

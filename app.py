"""TriageSense: Streamlit UI only. Zero business logic lives here.

Every computation is delegated to classifier / workflows / database /
evaluation / redteam / business. This file only renders and wires them up.
"""
import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st

# Bridge Streamlit Cloud secrets into environment variables so the provider-agnostic
# classifier (which reads os.environ) works both locally (.env) and when deployed.
for _k in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEYS"):
    try:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

from agent import run_agent
from business import compute_service_tiers
from config import DATA_DIR, URGENCY_COLORS

# UI brand palette (kept local to the UI layer so the running dev server never
# depends on a hot-reload of config.py).
BRAND = {
    "navy": "#12284C",
    "coral": "#FF5C39",
    "teal": "#0EA5A4",
    "slate": "#64748B",
    "bg": "#F5F7FB",
    "border": "#E2E8F0",
}
from classifier import classify
from database import init_db, log_case, get_all_cases
from evaluation import run_evaluation, export_report
from redteam import run_redteam_suite
from workflows import remediate

st.set_page_config(page_title="TriageSense", page_icon="🩺", layout="wide")
init_db()


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        :root {
            --navy:#12284C; --coral:#FF5C39; --teal:#0EA5A4;
            --slate:#64748B; --border:#E6EAF2; --bg:#F5F7FB;
        }
        html, body, [class*="css"], .stMarkdown, .stApp {
            font-family:'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stApp { background:
            radial-gradient(1200px 400px at 100% -5%, #EEF2FB 0%, rgba(238,242,251,0) 60%),
            var(--bg); }
        /* tighten top padding, widen content */
        .block-container { padding-top:1.2rem; padding-bottom:3rem; max-width:1200px; }
        h1,h2,h3,h4 { color:var(--navy); letter-spacing:-0.01em; }

        /* Hero header */
        .ts-hero {
            background:linear-gradient(105deg, var(--navy) 0%, #1B3A6B 55%, #24507F 100%);
            border-radius:18px; padding:26px 30px; margin-bottom:22px; color:#fff;
            box-shadow:0 12px 30px -12px rgba(18,40,76,0.45);
            display:flex; align-items:center; justify-content:space-between; gap:16px;
        }
        .ts-hero h1 { color:#fff; font-size:1.7rem; font-weight:800; margin:0; }
        .ts-hero p { color:#C7D6EC; margin:6px 0 0; font-size:0.95rem; }
        .ts-hero .ts-badge {
            background:rgba(255,92,57,0.16); color:#FFB9A6; border:1px solid rgba(255,92,57,0.4);
            padding:6px 14px; border-radius:999px; font-size:0.78rem; font-weight:600; white-space:nowrap;
        }
        .ts-hero .ts-logo { font-size:2rem; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--border); }
        .stTabs [data-baseweb="tab"] {
            height:44px; padding:0 18px; color:var(--slate); font-weight:600;
            background:transparent; border-radius:10px 10px 0 0;
        }
        .stTabs [aria-selected="true"] { color:var(--navy) !important; }
        .stTabs [data-baseweb="tab-highlight"] { background:var(--coral); height:3px; }

        /* Metric cards */
        [data-testid="stMetric"] {
            background:#fff; border:1px solid var(--border); border-radius:14px;
            padding:16px 18px; box-shadow:0 1px 3px rgba(18,40,76,0.05);
        }
        [data-testid="stMetricLabel"] p { color:var(--slate); font-weight:600; font-size:0.8rem; }
        [data-testid="stMetricValue"] { color:var(--navy); font-weight:700; font-size:1.55rem; line-height:1.25; white-space:normal; overflow-wrap:anywhere; }

        /* Buttons */
        .stButton > button {
            border-radius:10px; font-weight:600; border:1px solid var(--border);
            padding:0.5rem 1.1rem; transition:all .15s ease;
        }
        .stButton > button[kind="primary"] {
            background:var(--coral); border:none; box-shadow:0 8px 18px -8px rgba(255,92,57,0.7);
        }
        .stButton > button[kind="primary"]:hover { background:#F0492B; }
        .stButton > button:hover { border-color:var(--coral); color:var(--coral); }

        /* Expanders */
        [data-testid="stExpander"] {
            border:1px solid var(--border); border-radius:12px; background:#fff; overflow:hidden;
        }

        /* Section label */
        .ts-section {
            font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;
            color:var(--coral); margin:22px 0 8px; padding-left:10px; border-left:3px solid var(--coral);
        }
        /* Result card */
        .ts-card {
            background:#fff; border:1px solid var(--border); border-radius:14px;
            padding:18px 20px; box-shadow:0 1px 3px rgba(18,40,76,0.05); margin-bottom:6px;
        }
        .ts-pill {
            display:inline-block; background:#EEF2FB; color:var(--navy); border:1px solid var(--border);
            padding:3px 12px; border-radius:999px; font-weight:600; font-size:0.82rem; margin-right:6px;
        }
        .ts-signal {
            display:inline-block; background:#F3F5FA; color:var(--slate); border:1px solid var(--border);
            padding:2px 10px; border-radius:8px; font-size:0.8rem; margin:2px 4px 2px 0;
        }
        /* Human review banner refinement */
        [data-testid="stAlert"] { border-radius:12px; }
        /* Sidebar */
        section[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--border); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero() -> None:
    st.markdown(
        """
        <div class="ts-hero">
          <div>
            <h1><span class="ts-logo">🩺</span>&nbsp; TriageSense</h1>
            <p>Agentic member-support triage with governed autonomy and underwritten outcomes.</p>
          </div>
          <div class="ts-badge">Governed Agentic Triage</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section(label: str) -> None:
    st.markdown(f'<div class="ts-section">{label}</div>', unsafe_allow_html=True)


def _style_fig(fig):
    fig.update_layout(
        font_family="Inter", font_color=BRAND["navy"],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=48, l=10, r=10, b=10), title_font_size=15,
    )
    return fig


_inject_theme()
_hero()

SAMPLE_REQUESTS = {
    "Billing: duplicate charge": "I was charged twice for the same claim on my last statement, please refund the extra amount.",
    "Billing: invoice error": "There's a billing error on invoice #77291, I'm being billed for a service I never received.",
    "Enquiry: claim status": "Can you check the status of the claim I submitted last week for my physical therapy sessions?",
    "Enquiry: coverage check": "I want to know if my upcoming MRI is covered under my current plan before I schedule it.",
    "Service: appointment": "I need to book a follow-up appointment with my primary care physician next week.",
    "Service: prior auth": "I need a prior authorization for an MRI scan scheduled next month, member ID 334455667.",
    "Escalation: repeated complaint": "This is the third time I've called about the same issue and nobody has helped me, I'm furious and considering legal action.",
    "Escalation: emergency": "I'm having severe chest pain right now, I don't know what to do.",
}


def _urgency_badge(urgency: str) -> str:
    color = URGENCY_COLORS.get(urgency, "#94a3b8")
    return (
        f'<span style="background-color:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">{urgency}</span>'
    )


with st.sidebar:
    st.title("🩺 TriageSense")
    st.caption("Healthcare Member Support Triage Engine")
    with st.expander("About", expanded=False):
        st.markdown(
            """
            TriageSense reads an incoming member request, works out what it is
            and how urgent it is, then runs the right multi-step workflow for
            that type. Billing disputes, coverage questions, prior-auth requests
            and complaints each follow their own path.

            When the classifier isn't confident, or the case is high-risk, it
            hands off to a human instead of acting on its own.

            It only does administrative triage. It never gives medical advice,
            and it masks personal details before writing anything to the log.
            """
        )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Triage a Request",
        "Batch Processing",
        "Analytics Dashboard",
        "Evaluation",
        "Red-Team / Robustness",
    ]
)

with tab1:
    st.subheader("Triage a Request")
    agentic = st.toggle("Agentic mode (governed)", value=True)
    sample_choice = st.selectbox("Sample requests", ["-- choose a sample --"] + list(SAMPLE_REQUESTS.keys()))
    default_text = SAMPLE_REQUESTS.get(sample_choice, "")
    text = st.text_area("Member request text", value=default_text, height=120)

    if st.button("Process", type="primary"):
        classification = classify(text)
        if agentic:
            result = run_agent(text, classification)
        else:
            result = remediate(text, classification)
        log_case(result, raw_text=text)

        if result.human_in_loop:
            st.error("HUMAN REVIEW REQUIRED: routed to a human expert for review", icon="🚨")

        _section("Classification")
        col1, col2, col3 = st.columns(3)
        col1.metric("Request Type", classification.request_type.value)
        col2.markdown(
            '<div style="background:#fff;border:1px solid #E6EAF2;border-radius:14px;'
            'padding:16px 18px;box-shadow:0 1px 3px rgba(18,40,76,0.05);">'
            '<div style="font-size:0.8rem;color:#64748B;font-weight:600;">URGENCY</div>'
            f'<div style="margin-top:10px;">{_urgency_badge(classification.urgency.value)}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        col3.metric("Confidence", f"{classification.confidence:.0%}")
        st.progress(classification.confidence)

        rationale_html = f'<div class="ts-card"><b>Rationale.</b> {classification.rationale}'
        if classification.signals:
            chips = "".join(f'<span class="ts-signal">{s}</span>' for s in classification.signals)
            rationale_html += f'<div style="margin-top:12px;">{chips}</div>'
        rationale_html += "</div>"
        st.markdown(rationale_html, unsafe_allow_html=True)

        if classification.extracted_fields:
            _section("Extracted fields")
            for _k, _v in classification.extracted_fields.items():
                st.markdown(f"- **{_k.replace('_', ' ').title()}:** {_v}")

        if agentic and (result.agent_reasoning or result.planned_tools):
            with st.expander("Agent reasoning", expanded=True):
                st.markdown(result.agent_reasoning or "_(fell back to the deterministic engine)_")
                if result.planned_tools:
                    st.markdown("**Planned tools (in order):**")
                    for i, tool in enumerate(result.planned_tools, start=1):
                        st.markdown(f"{i}. `{tool}`")

        _section("Remediation Steps")
        for i, step in enumerate(result.steps_executed, start=1):
            st.markdown(f"{i}. {step}")

        _section("Outputs")
        for _k, _v in result.outputs.items():
            if isinstance(_v, bool):
                _v = "Yes" if _v else "No"
            elif isinstance(_v, list):
                _v = "; ".join(str(x) for x in _v)
            if _k == "draft_response":
                st.markdown("**Draft response to member**")
                st.markdown(f'<div class="ts-card">{_v}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"**{_k.replace('_', ' ').title()}:** {_v}")

        st.caption(f"Case ID: {result.case_id} · {result.timestamp}")

with tab2:
    st.subheader("Batch Processing")
    _sample_csv = (DATA_DIR / "sample_batch_requests.csv").read_bytes()
    st.download_button(
        "Download a sample CSV to try", _sample_csv,
        file_name="sample_batch_requests.csv", mime="text/csv",
    )
    uploaded = st.file_uploader("Upload a CSV with a `text` column", type=["csv"])
    use_sample = st.checkbox("Use the built-in labeled set instead")

    batch_df = None
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
    elif use_sample:
        batch_df = pd.read_csv(DATA_DIR / "labeled_requests.csv")

    if batch_df is not None and "text" not in batch_df.columns:
        st.warning("CSV must contain a `text` column.")
        batch_df = None

    if batch_df is not None:
        st.caption(f"{len(batch_df)} rows ready to process.")
        if st.button("Run Batch", type="primary"):
            texts = [str(t) for t in batch_df["text"].fillna("")]
            total = len(texts)
            bar = st.progress(0.0, text=f"Processing 0/{total}…")
            rows = []
            for i, text_value in enumerate(texts):
                classification = classify(text_value)
                result = remediate(text_value, classification)
                log_case(result, raw_text=text_value)
                rows.append(
                    {
                        "text": text_value,
                        "request_type": classification.request_type.value,
                        "urgency": classification.urgency.value,
                        "confidence": classification.confidence,
                        "human_in_loop": result.human_in_loop,
                        "case_id": result.case_id,
                    }
                )
                bar.progress((i + 1) / total, text=f"Processing {i + 1}/{total}…")
            bar.empty()
            st.session_state["batch_results"] = pd.DataFrame(rows)

    if "batch_results" in st.session_state:
        results_df = st.session_state["batch_results"]
        _section("Results")
        st.dataframe(results_df, use_container_width=True)
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download results CSV",
            data=csv_buffer.getvalue(),
            file_name="triage_batch_results.csv",
            mime="text/csv",
        )

with tab3:
    st.subheader("Analytics Dashboard")
    log_df = get_all_cases()

    if len(log_df) == 0:
        st.info("No cases logged yet. Process some requests in the Triage or Batch tabs first.")
    else:
        tiers = compute_service_tiers(log_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cases", len(log_df))
        c2.metric("Avg Confidence", f"{log_df['confidence'].mean():.0%}")
        c3.metric("AI-Triaged", f"{tiers['ai_touched_pct']:.0%}")
        c4.metric("Fully Auto-Resolved", f"{tiers['auto_resolved_pct']:.0%}")

        _section("Service Tiers")
        st.caption(
            "Every request is triaged by the system. Cases split into three tiers by "
            "confidence and risk: full automation for low-risk work, AI-prepared "
            "human sign-off for cases needing approval, and escalation for high-risk cases."
        )
        t1, t2, t3 = st.columns(3)
        t1.metric("Auto-Resolved", tiers["auto_resolved"], f"{tiers['auto_resolved_pct']:.0%}")
        t2.metric("AI-Handled (human approves)", tiers["ai_handled"], f"{tiers['ai_handled_pct']:.0%}")
        t3.metric("Escalated to Human", tiers["escalated"], f"{tiers['escalated_pct']:.0%}")

        col_a, col_b = st.columns(2)
        with col_a:
            type_counts = log_df["request_type"].value_counts().reset_index()
            type_counts.columns = ["request_type", "count"]
            fig_type = px.bar(
                type_counts, x="request_type", y="count", title="Volume by Request Type",
                color_discrete_sequence=[BRAND["navy"]],
            )
            st.plotly_chart(_style_fig(fig_type), use_container_width=True)
        with col_b:
            urgency_counts = log_df["urgency"].value_counts().reset_index()
            urgency_counts.columns = ["urgency", "count"]
            fig_urgency = px.pie(
                urgency_counts, names="urgency", values="count", hole=0.55,
                title="Volume by Urgency", color="urgency", color_discrete_map=URGENCY_COLORS,
            )
            st.plotly_chart(_style_fig(fig_urgency), use_container_width=True)

        _section("Business Impact")
        daily_volume = st.slider("Projected daily volume", min_value=100, max_value=10000, value=1000, step=100)
        proj = compute_service_tiers(log_df, daily_volume=daily_volume)
        st.markdown(
            f"Every one of the **{daily_volume:,} requests/day** is auto-triaged. Counting full "
            f"automation for auto-resolved cases and partial savings where AI prepares a case for "
            f"human sign-off, that is about **{proj['projected_daily_hours_saved']:.1f} analyst-hours/day** "
            f"saved (**${proj['projected_daily_cost_saved']:,.2f}/day**). **{proj['escalated_pct']:.0%}** "
            f"of high-risk cases are always owned by a human."
        )
        b1, b2, b3 = st.columns(3)
        b1.metric("Hours Saved (logged cases)", f"{proj['weighted_hours_saved']:.1f}")
        b2.metric("Cost Saved (logged cases)", f"${proj['weighted_cost_saved']:,.2f}")
        b3.metric("AI-Prepared or Auto", proj["auto_resolved"] + proj["ai_handled"])

with tab4:
    st.subheader("Evaluation")
    st.caption(
        "Scores the classifier against the 71-example labeled set "
        "(`data/labeled_requests.csv`). This makes ~71 LLM calls, so it takes about a minute."
    )

    run_clicked = st.button("Run Evaluation", type="primary")
    if run_clicked:
        bar = st.progress(0.0, text="Classifying 0/71…")

        def _bar(done: int, total: int) -> None:
            bar.progress(done / total, text=f"Classifying {done}/{total}…")

        st.session_state["eval_results"] = run_evaluation(progress=_bar)
        bar.empty()

    if "eval_results" in st.session_state:
        results = st.session_state["eval_results"]
        st.metric("Overall Accuracy", f"{results['accuracy']:.1%}")

        _section("Per-class Precision / Recall / F1")
        st.dataframe(results["per_class"].round(3), use_container_width=True)

        _section("Confusion Matrix")
        cm = results["confusion_matrix"]
        fig_cm = px.imshow(
            cm.values,
            x=list(cm.columns),
            y=list(cm.index),
            text_auto=True,
            color_continuous_scale=["#EAF0FA", BRAND["navy"]],
            labels=dict(x="Predicted", y="True", color="Count"),
        )
        st.plotly_chart(_style_fig(fig_cm), use_container_width=True)

        col_x, col_y = st.columns(2)
        if col_x.button("Export report to eval_report.md"):
            export_report(results)
            st.success("Exported to eval_report.md")
        if col_y.button("Clear results"):
            del st.session_state["eval_results"]
            st.rerun()

with tab5:
    st.subheader("Red-Team / Robustness")
    st.caption("Runs the adversarial suite (`data/redteam_cases.json`) against the full pipeline.")

    if st.button("Run Red-Team Suite") or "redteam_results" in st.session_state:
        if "redteam_results" not in st.session_state:
            with st.spinner("Running red-team suite..."):
                st.session_state["redteam_results"] = run_redteam_suite()

        redteam_results = st.session_state["redteam_results"]
        passed = sum(1 for r in redteam_results if r["passed"])
        st.metric("Passed", f"{passed}/{len(redteam_results)}")

        for r in redteam_results:
            badge = "✅ PASS" if r["passed"] else "❌ FAIL"
            with st.expander(f"{badge}: {r['name']} ({r['category']})"):
                st.markdown(f"**Input:** {r['input'] or '_(empty)_'}")
                st.markdown(f"**Expected behavior:** {r['expected_behavior']}")
                if r["error"]:
                    st.error(f"Error: {r['error']}")
                elif r["classification"] is not None:
                    st.markdown(
                        f"**Classified as:** {r['classification'].request_type.value} "
                        f"({r['classification'].urgency.value}, "
                        f"confidence {r['classification'].confidence:.0%})"
                    )
                    st.markdown(f"**Human review required:** {r['remediation'].human_in_loop}")

        if st.button("Re-run red-team suite"):
            del st.session_state["redteam_results"]
            st.rerun()

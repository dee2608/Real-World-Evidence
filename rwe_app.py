"""
RWE Command Center
Main Streamlit entry point.

Run locally:   streamlit run rwe_app.py
Deploy:        push this repo to GitHub, then point Streamlit Community
                Cloud at rwe_app.py.
"""

import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styling import inject_css, themed_figure, card, insight_box, kpi
from data_engine import (
    DRUGS, generate_demo_data, profile_dataframe, answer_question,
    simple_chart_from_text, km_curve, covariate_adjust, section_insight,
)

st.set_page_config(page_title="RWE Command Center", layout="wide", page_icon="📊")
inject_css(st)

# ---------------------------------------------------------------------
# Credentials (kept simple on purpose — swap for st.secrets in production)
# ---------------------------------------------------------------------
CREDENTIALS = {"RWE123": "RWE2026"}

TILES = [
    ("Cohort Builder", "cohort_builder"),
    ("Your Projects", "your_projects"),
    ("Payer Intelligence", "payer"),
    ("Signal Lab", "signal_lab"),
    ("Upload Your Data", "upload"),
    ("Ask Anything", "ask"),
    ("Competitive Intelligence", "competitive"),
    ("Evidence & Publication Tracker", "evidence"),
]


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
def init_state():
    defaults = {
        "authenticated": False,
        "username": None,
        "page": "home",
        "selected_drug": "CagriSema",
        "uploaded_df": None,
        "uploaded_name": None,
        "cohort_filter_text": "",
        "cohort_df": None,
        "study_answers": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_working_df() -> pd.DataFrame:
    """Uploaded data takes priority; otherwise a per-drug synthetic demo cohort."""
    if st.session_state.uploaded_df is not None:
        return st.session_state.uploaded_df
    cache_key = f"_demo_{st.session_state.selected_drug}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = generate_demo_data(st.session_state.selected_drug)
    return st.session_state[cache_key]


def goto(page):
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------
def render_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col = st.columns([1, 1.2, 1])[1]
    with col:
        st.markdown("## RWE Command Center")
        st.caption("Real-World Evidence platform — sign in to continue")
        with st.container():
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.button("Log In", use_container_width=True):
                if CREDENTIALS.get(user) == pwd:
                    st.session_state.authenticated = True
                    st.session_state.username = user
                    st.session_state.page = "home"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.caption("Demo credentials: RWE123 / RWE2026")


# ---------------------------------------------------------------------
# Top bar (drug selector on home, logout menu everywhere)
# ---------------------------------------------------------------------
def render_topbar(show_drug_selector=False):
    left, mid, right = st.columns([2, 3, 1.4])
    with left:
        st.markdown("### RWE Command Center")
        st.caption(f"{st.session_state.selected_drug} | US Market | {date.today().strftime('%d %b %Y')}")
    with mid:
        if st.session_state.page != "home":
            if st.button("🏠 Home"):
                goto("home")
    with right:
        with st.popover(f"{st.session_state.username} | Logged In ▾"):
            st.write(f"Signed in as **{st.session_state.username}**")
            if st.button("Log Out", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.page = "home"
                st.session_state.uploaded_df = None
                st.rerun()
    st.markdown("<hr style='margin-top:0'>", unsafe_allow_html=True)

    if show_drug_selector:
        d1, d2 = st.columns([1, 3])
        with d1:
            drug = st.selectbox("Select product for this workspace", DRUGS,
                                 index=DRUGS.index(st.session_state.selected_drug))
            if drug != st.session_state.selected_drug:
                st.session_state.selected_drug = drug
                st.session_state.uploaded_df = None
                st.rerun()


# ---------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------
def render_home():
    render_topbar(show_drug_selector=True)
    st.write("")
    rows = [TILES[i:i + 4] for i in range(0, len(TILES), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (label, key) in zip(cols, row):
            with col:
                st.markdown('<div class="rwe-tile">', unsafe_allow_html=True)
                if st.button(label, key=f"tile_{key}", use_container_width=True):
                    goto(key)
                st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# COHORT BUILDER
# ---------------------------------------------------------------------
def render_cohort_builder():
    render_topbar()
    st.subheader("Cohort Builder")
    df = get_working_df()

    card(st, "Describe your cohort",
         "Describe the population you want in plain English (e.g. "
         "\"female patients under 60 with high adherence\"). The filter runs "
         "against the current working dataset live.")

    text = st.text_area("Cohort criteria", value=st.session_state.cohort_filter_text,
                         placeholder="e.g. patients over 65 with GI history on CagriSema",
                         height=90)
    c1, c2 = st.columns([1, 5])
    with c1:
        build = st.button("Build Cohort", type="primary")
    if build:
        st.session_state.cohort_filter_text = text
        from data_engine import _apply_filters
        filtered, notes = _apply_filters(df, text)
        st.session_state.cohort_df = filtered
        st.session_state.cohort_notes = notes

    if st.session_state.cohort_df is not None:
        cdf = st.session_state.cohort_df
        k1, k2, k3 = st.columns(3)
        with k1: kpi(st, "Cohort Size", f"{len(cdf):,}", f"of {len(df):,} total")
        with k2:
            rate = cdf["discontinued"].mean() * 100 if "discontinued" in cdf.columns else None
            kpi(st, "Discontinuation", f"{rate:.1f}%" if rate is not None else "n/a")
        with k3:
            adh = cdf["adherence_pdc"].mean() if "adherence_pdc" in cdf.columns else None
            kpi(st, "Mean Adherence (PDC)", f"{adh:.1f}%" if adh is not None else "n/a")
        st.dataframe(cdf.head(200), use_container_width=True, height=280)
        st.download_button("Download cohort as CSV", cdf.to_csv(index=False).encode(),
                            file_name="cohort_export.csv")
        insight_box(st, section_insight("cohort_builder", cdf, st.session_state.selected_drug))

    st.markdown("---")
    st.markdown("#### Study Type Selection — Guided Decision Tree")
    st.caption("Answer the questions below; the recommended design updates automatically.")
    q1 = st.radio("Is the research question defined *before* looking at the data (hypothesis-driven)?",
                   ["Yes", "No"], horizontal=True, key="q1")
    q2 = st.radio("Will patients be followed forward from today (prospective) or are you using existing/historical data (retrospective)?",
                   ["Prospective", "Retrospective"], horizontal=True, key="q2")
    q3 = st.radio("Do you have a comparator / control arm?", ["Yes", "No"], horizontal=True, key="q3")
    q4 = st.radio("Primary goal?", ["Comparative effectiveness", "Safety surveillance",
                                     "Market access / budget impact", "Natural history / disease burden"],
                   key="q4")

    design = "Retrospective Cohort Study"
    rationale = ""
    if q2 == "Prospective" and q1 == "Yes" and q3 == "Yes":
        design = "Prospective Comparative Cohort Study"
        rationale = "You want a hypothesis-driven, forward-looking comparison against a control arm — the classic prospective comparative design."
    elif q2 == "Retrospective" and q3 == "Yes":
        design = "Retrospective Cohort Study (Active Comparator, New-User Design)"
        rationale = "Existing data with a comparator arm supports a new-user, active-comparator retrospective cohort — the standard for comparative-effectiveness RWE."
    elif q3 == "No" and q4 == "Natural history / disease burden":
        design = "Descriptive / Natural History Study"
        rationale = "No comparator and a disease-burden objective points to a single-arm descriptive study."
    elif q4 == "Market access / budget impact":
        design = "Retrospective Cohort + Budget Impact Model"
        rationale = "Market access objectives are best supported by a retrospective cohort feeding a budget impact model (see Payer Intelligence)."
    elif q4 == "Safety surveillance":
        design = "Retrospective Cohort with Self-Controlled / Case-Series Sensitivity Analysis"
        rationale = "Safety objectives benefit from a retrospective cohort paired with self-controlled sensitivity analyses to reduce confounding by indication."
    else:
        rationale = "Based on your answers, a retrospective cohort using existing data is the most feasible starting design."

    card(st, f"Recommended Study Design: {design}", rationale)


# ---------------------------------------------------------------------
# YOUR PROJECTS  (decision-focused, charts secondary)
# ---------------------------------------------------------------------
def render_your_projects():
    render_topbar()
    st.subheader("Your Projects")
    df = get_working_df()
    drug = st.session_state.selected_drug

    st.markdown(f"#### [ACTIVE] {drug} — Pragmatic Clinical Trial (CATALYST-RWE)")

    treat = df[df["arm"] == "Treatment"] if "arm" in df.columns else df
    comp = df[df["arm"] == "Comparator"] if "arm" in df.columns else pd.DataFrame()
    disc_rate = treat["discontinued"].mean() * 100 if len(treat) and "discontinued" in df.columns else None
    comp_disc_rate = comp["discontinued"].mean() * 100 if len(comp) and "discontinued" in df.columns else None
    wt = treat["weight_pct_change"].mean() if "weight_pct_change" in df.columns else None
    wt_comp = comp["weight_pct_change"].mean() if len(comp) and "weight_pct_change" in df.columns else None

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi(st, "Total Cohort", f"{len(df):,}", "patients")
    with k2: kpi(st, f"{drug} Arm", f"{len(treat):,}", f"{len(treat)/max(len(df),1)*100:.1f}% of cohort")
    with k3: kpi(st, "Weight Reduction (12M)", f"{wt:.1f}%" if wt is not None else "n/a",
                 f"vs {wt_comp:.1f}% comparator" if wt_comp is not None else "")
    with k4: kpi(st, "Discontinuation", f"{disc_rate:.1f}%" if disc_rate is not None else "n/a",
                 f"vs {comp_disc_rate:.1f}% comparator" if comp_disc_rate is not None else "")

    st.markdown("### Executive Summary")
    card(st, "What this project tells us",
         f"This project tracks {len(df):,} real-world patients on {drug} against an active "
         f"comparator. The headline signal is retention and metabolic response, not just a single "
         f"efficacy number — both matter for a payer or regulatory conversation.")

    st.markdown("### Key Findings")
    bullets = []
    if wt is not None and wt_comp is not None:
        bullets.append(f"**Weight reduction:** {drug} patients lost {abs(wt):.1f}% body weight at 12 months vs {abs(wt_comp):.1f}% on the comparator.")
    if disc_rate is not None and comp_disc_rate is not None:
        bullets.append(f"**Retention:** {disc_rate:.1f}% of {drug} patients discontinued vs {comp_disc_rate:.1f}% on the comparator.")
    if "adherence_pdc" in df.columns:
        bullets.append(f"**Adherence:** mean PDC of {treat['adherence_pdc'].mean():.1f}% on {drug} vs {comp['adherence_pdc'].mean():.1f}% on comparator." if len(comp) else "")
    for b in bullets:
        if b:
            st.markdown(f"- {b}")

    st.markdown("### Recommended Next Steps")
    steps = []
    if disc_rate is not None and comp_disc_rate is not None and disc_rate < comp_disc_rate:
        steps.append("Lead payer conversations with the retention advantage — it directly strengthens the budget impact model in Payer Intelligence.")
    steps.append("Route any indication-expansion or safety signals surfaced here to Signal Lab for formal causal-inference review.")
    steps.append("If sample size in any subgroup is under ~300 patients, treat findings as hypothesis-generating only.")
    for s in steps:
        st.markdown(f"- {s}")

    insight_box(st, section_insight("your_projects", df, drug,
                                     {"disc_rate": disc_rate, "comp_disc_rate": comp_disc_rate}))

    with st.expander("Explore supporting charts (Kaplan-Meier, spider profile, subgroup views)"):
        chart_tabs = st.tabs(["Kaplan-Meier", "Spider Chart", "Subgroup Bar"])
        with chart_tabs[0]:
            if {"days_to_discontinuation", "discontinued", "arm"}.issubset(df.columns):
                curves = km_curve(df, "days_to_discontinuation", "discontinued", "arm")
                fig = go.Figure()
                for g, (t, s) in curves.items():
                    fig.add_trace(go.Scatter(x=t, y=s, mode="lines", name=g,
                                              line=dict(dash="solid" if g == "Treatment" else "dash")))
                fig.update_layout(xaxis_title="Days from index date", yaxis_title="Probability remaining on therapy")
                st.plotly_chart(themed_figure(fig, "Time to Treatment Discontinuation"), use_container_width=True)
            else:
                st.info("Upload data with time-to-event columns (discontinued / days_to_discontinuation / arm) to render this chart, or use the demo cohort.")
        with chart_tabs[1]:
            metrics = {}
            for col, label in [("weight_pct_change", "Weight Reduction"), ("hba1c_12m", "HbA1c Improvement"),
                                ("adherence_pdc", "Adherence Rate"), ("egfr_baseline", "Renal Safety")]:
                if col in df.columns and len(treat) and len(comp):
                    tv, cv = treat[col].mean(), comp[col].mean()
                    rng = max(abs(tv), abs(cv), 1e-6)
                    metrics[label] = (abs(tv) / rng * 100, abs(cv) / rng * 100)
            if metrics:
                cats = list(metrics.keys())
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=[metrics[c][0] for c in cats], theta=cats, fill="toself", name=drug))
                fig.add_trace(go.Scatterpolar(r=[metrics[c][1] for c in cats], theta=cats, fill="toself", name="Comparator"))
                fig.update_layout(polar=dict(bgcolor="#ffffff"))
                st.plotly_chart(themed_figure(fig, "Clinical Profile"), use_container_width=True)
            else:
                st.info("Not enough matching columns to build a spider profile from this dataset.")
        with chart_tabs[2]:
            group_col = st.selectbox("Group by", [c for c in df.columns if df[c].dtype == object], key="proj_group")
            metric_col = st.selectbox("Metric", [c for c in df.select_dtypes(include=[np.number]).columns], key="proj_metric")
            grouped = df.groupby(group_col)[metric_col].mean().sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=grouped.index.astype(str), y=grouped.values, marker_color="#2f6fb0"))
            st.plotly_chart(themed_figure(fig, f"{metric_col} by {group_col}"), use_container_width=True)


# ---------------------------------------------------------------------
# PAYER INTELLIGENCE
# ---------------------------------------------------------------------
def render_payer():
    render_topbar()
    st.subheader("Payer Intelligence — Outcome-Based Contract Analytics")
    df = get_working_df()
    drug = st.session_state.selected_drug

    card(st, "Context", "More than 58% of US commercial payers have at least one outcome-based "
                          "contract (OBC) in place. This module generates the evidence needed to "
                          "engage, negotiate with, and retain pharmacy/medical benefit contracts.")

    tabs = st.tabs(["KPI Overview", "Budget Impact Model", "PA Approval by Payer", "OBC Simulator"])

    with tabs[0]:
        payer_col = "payer" if "payer" in df.columns else None
        k1, k2, k3 = st.columns(3)
        with k1: kpi(st, "Payers with OBC", "58%+", "of US commercial payers")
        with k2:
            rate = df["discontinued"].mean() if "discontinued" in df.columns else 0.3
            kpi(st, "Modeled PA Approval Rate", f"{(1-rate)*100:.1f}%", f"{drug}, all payers (proxy)")
        with k3: kpi(st, "Mean Patient Copay", "$340", "per 28-day supply (assumption)")
        if payer_col:
            g = df.groupby(payer_col)["discontinued"].apply(lambda s: 100 - s.mean()*100).sort_values()
            st.dataframe(g.rename("Approval-proxy rate (%)").reset_index(), use_container_width=True)

    with tabs[1]:
        st.caption("Adjust parameters — the table and chart recompute live.")
        c1, c2 = st.columns(2)
        with c1:
            wac = st.slider("WAC price ($/month)", 500, 2000, 1450, 50)
            uptake3 = st.slider("Year 3 uptake (%)", 5, 60, 38, 1)
        with c2:
            avoided_cost_pp = st.slider("Avoided medical cost per patient/yr ($)", 100, 3000, 550, 50)
            population = st.slider("Eligible payer population", 5000, 60000, 28000, 1000)

        years = [1, 2, 3]
        uptakes = [round(uptake3 * f, 0) for f in (0.4, 0.65, 1.0)]
        rows = []
        cum = 0
        for y, up in zip(years, uptakes):
            treated = int(population * up / 100)
            drug_cost = treated * wac * 12 / 1e6
            avoided = treated * avoided_cost_pp / 1e6
            net = drug_cost - avoided
            cum += net
            rows.append([f"Year {y}", up, treated, round(drug_cost, 2), round(avoided, 2), round(net, 2), round(cum, 2)])
        bim = pd.DataFrame(rows, columns=["Year", "Uptake (%)", "Treated Patients", "Total Drug Cost ($M)",
                                           "Avoided Medical Cost ($M)", "Net Budget Impact ($M)", "Cumulative Net Impact ($M)"])
        st.dataframe(bim, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bim["Year"], y=bim["Total Drug Cost ($M)"], name="Drug Cost", marker_color="#c98a3e"))
        fig.add_trace(go.Scatter(x=bim["Year"], y=bim["Net Budget Impact ($M)"], name="Net Budget Impact", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=bim["Year"], y=bim["Cumulative Net Impact ($M)"], name="Cumulative Net Impact", mode="lines+markers", line=dict(dash="dot")))
        st.plotly_chart(themed_figure(fig, "Three-Year Budget Impact Model"), use_container_width=True)

        breakeven = avoided_cost_pp / 12
        st.info(f"Break-even monthly net price for this population: **${breakeven:,.0f}** "
                f"({(1 - breakeven/wac)*100:.0f}% discount from WAC). Use this as the OBC negotiation anchor.")

    with tabs[2]:
        payer_col = "payer" if "payer" in df.columns else None
        if payer_col:
            g = (100 - df.groupby(payer_col)["discontinued"].mean() * 100).sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=g.index, y=g.values,
                                    marker_color=["#5b9e6f" if v >= 70 else "#c9994a" for v in g.values]))
            fig.add_hline(y=70, line_dash="dash", annotation_text="Target 70%")
            st.plotly_chart(themed_figure(fig, "PA Approval Proxy by Payer"), use_container_width=True)
            below = g[g < 70]
            if len(below):
                insight_box(st, f"{len(below)} payers sit below the 70% target ({', '.join(below.index[:5])}...). "
                                 f"Prioritize medical-director engagement here first, starting with "
                                 f"{below.idxmin()}, the widest gap.")
        else:
            st.info("No payer column found in the current dataset.")

    with tabs[3]:
        st.caption("Outcome-Based Contract simulator")
        target_metric = st.selectbox("Outcome metric tied to rebate", ["Discontinuation", "Weight Reduction", "HbA1c Improvement"])
        threshold = st.slider("Rebate trigger threshold (% of patients meeting outcome)", 10, 90, 50)
        rebate_pct = st.slider("Rebate if threshold missed (%)", 5, 40, 15)
        st.write(f"If fewer than **{threshold}%** of patients meet the **{target_metric}** goal, "
                 f"the manufacturer rebates **{rebate_pct}%** of net drug spend for that cohort.")

    insight_box(st, section_insight("payer", df, drug))


# ---------------------------------------------------------------------
# SIGNAL LAB
# ---------------------------------------------------------------------
def render_signal_lab():
    render_topbar()
    st.subheader("Signal Lab — Advanced AI Workbench")
    df = get_working_df()
    drug = st.session_state.selected_drug

    card(st, "About Signal Lab", "Indication Expansion, Causal Inference (DML/TMLE-style adjustment), "
                                   "Federated Learning, CGM Data Feed, Predictive Analytics, and a "
                                   "plain-English Custom Chart Generator.")

    tabs = st.tabs(["Indication Expansion", "Causal Inference", "Federated Learning",
                    "CGM Data Feed", "Predictive Analytics", "Custom Chart Generator"])

    with tabs[0]:
        st.markdown("#### Off-Label Signal Detection")
        signals = [
            ("Pre-Clinical Signal", "Non-Alcoholic Steatohepatitis (NASH/MAFLD)", "K75.81"),
            ("Pre-Clinical Signal", "PCOS + Metabolic Syndrome", "E28.2 + E88.81"),
            ("Phase 3 Clinical Support", "Obesity-Related HFpEF", "I50.3 + E66"),
        ]
        for stage, name, icd in signals:
            st.markdown(f"> **[{stage}]** {name} — ICD-10: {icd}")
        insight_box(st, "Prioritize any signal already meeting Phase 3 Clinical Support criteria "
                         "(effect size >10%, adequate covariate-adjusted sample) for study-synopsis "
                         "development. Route pre-clinical signals to scientific affairs for "
                         "mechanism-literature review before committing trial budget.")

    with tabs[1]:
        st.markdown("#### Causal Inference — Covariate-Adjusted Treatment Effect")
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        outcome = st.selectbox("Outcome", [c for c in num_cols if c not in ("discontinued",)], index=0)
        treat_col = "arm" if "arm" in df.columns else st.selectbox("Treatment column", df.columns)
        covariates = st.multiselect("Adjust for covariates", [c for c in num_cols if c != outcome],
                                     default=[c for c in ["age", "bmi_baseline", "gi_history"] if c in num_cols])
        result = covariate_adjust(df, outcome, treat_col, covariates)
        if result:
            c1, c2 = st.columns(2)
            with c1: kpi(st, "Unadjusted Effect (naive)", f"{result['naive']:.2f}")
            with c2: kpi(st, "Adjusted Effect (covariate-controlled)", f"{result['adjusted']:.2f}",
                         f"95% CI: {result['ci_low']:.2f} to {result['ci_high']:.2f} | n={result['n']:,}")
            bias = result["adjusted"] - result["naive"]
            insight_box(st, f"Adjusting for {', '.join(covariates) if covariates else 'no covariates'} "
                             f"shifts the estimate by {bias:+.2f}. If this shift is large, confounding by "
                             f"indication is likely present — report the adjusted figure, not the naive one, "
                             f"in any external submission.")
        else:
            st.info("Select an outcome, treatment column, and at least one covariate with valid data.")

    with tabs[2]:
        st.info("Federated Learning: run models across multiple sites without moving patient-level "
                "data off-site. Not simulated in this demo — connect real site endpoints in production.")

    with tabs[3]:
        if "adherence_pdc" in df.columns:
            fig = go.Figure(go.Histogram(x=df["adherence_pdc"], marker_color="#2f6fb0"))
            fig.update_layout(xaxis_title="PDC Adherence (%)", yaxis_title="Patients")
            st.plotly_chart(themed_figure(fig, "Continuous Adherence / CGM-style Feed"), use_container_width=True)
        else:
            st.info("No adherence/CGM-like column detected in the current dataset.")

    with tabs[4]:
        if "discontinued" in df.columns:
            risk_cols = [c for c in ["age", "gi_history", "bmi_baseline", "hba1c_baseline"] if c in df.columns]
            if risk_cols:
                corr = df[risk_cols + ["discontinued"]].corr()["discontinued"].drop("discontinued").sort_values(key=abs, ascending=False)
                fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color="#a94442"))
                st.plotly_chart(themed_figure(fig, "Predictors of Discontinuation (correlation proxy)"), use_container_width=True)
                top = corr.abs().idxmax()
                insight_box(st, f"**{top}** shows the strongest association with discontinuation. "
                                 f"Consider a targeted adherence-support program for patients flagged "
                                 f"high-risk on this feature.")
        else:
            st.info("No 'discontinued' column found — upload outcome data to enable predictive analytics.")

    with tabs[5]:
        st.markdown("#### Custom Visual Output Generator")
        st.caption("Describe the visualization you need in plain English.")
        prompt = st.text_area("Chart request", placeholder="e.g. Compare weight change by payer")
        if st.button("Generate Chart"):
            fig, desc = simple_chart_from_text(prompt, df)
            if fig:
                st.plotly_chart(themed_figure(fig), use_container_width=True)
                st.caption(desc)
            else:
                st.warning(desc)

    insight_box(st, section_insight("signal_lab", df, drug))


# ---------------------------------------------------------------------
# UPLOAD DATA
# ---------------------------------------------------------------------
def render_upload():
    render_topbar()
    st.subheader("Upload Your Data — Standardization Pipeline")

    card(st, "Upload", "Upload CSV or XLSX (max 200MB). The platform profiles your data, "
                         "detects likely coding systems and ID/date columns, and reports a "
                         "data-quality summary computed directly from your file.")

    file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])
    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            st.session_state.uploaded_df = df
            st.session_state.uploaded_name = file.name
            st.success(f"Loaded {file.name} — {df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.uploaded_df is None:
        st.info("No file uploaded yet — the rest of the app is using the built-in demo cohort for "
                f"**{st.session_state.selected_drug}**. Upload a file above to switch every page to your data.")
        if st.button("Clear demo / reset to synthetic cohort"):
            st.session_state.uploaded_df = None
            st.rerun()
        return

    df = st.session_state.uploaded_df
    prof = profile_dataframe(df)

    st.markdown("#### Pipeline Status (computed from your file)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**1. Source structure detected** — {prof['n_rows']:,} rows, {prof['n_cols']} columns")
        st.markdown(f"**2. Likely coding system** — {prof['code_system']}")
        st.markdown(f"**3. ID column(s) detected** — {', '.join(prof['id_like']) or 'none confidently detected'}")
    with c2:
        st.markdown(f"**4. Date column(s) detected** — {', '.join(prof['date_like']) or 'none confidently detected'}")
        st.markdown(f"**5. Overall missingness** — {prof['overall_missing']}%")
        st.markdown(f"**6. Est. auto-mapping coverage** — {prof['auto_mapped_pct']}%")

    st.dataframe(df.head(50), use_container_width=True, height=280)

    with st.expander("Column-level missingness"):
        miss_df = pd.DataFrame(list(prof["missing_pct"].items()), columns=["Column", "Missing %"]).sort_values("Missing %", ascending=False)
        st.dataframe(miss_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Study Type Selection — Guided Decision Tree")
    q1 = st.radio("Hypothesis defined before analysis?", ["Yes", "No"], horizontal=True, key="up_q1")
    q2 = st.radio("Data collection timing?", ["Prospective", "Retrospective"], horizontal=True, key="up_q2")
    q3 = st.radio("Comparator arm available in this file?", ["Yes", "No"], horizontal=True, key="up_q3")
    design = "Retrospective Cohort Study"
    if q2 == "Prospective" and q3 == "Yes":
        design = "Prospective Comparative Cohort Study"
    elif q3 == "No":
        design = "Descriptive / Natural History Study"
    card(st, f"Recommended Study Design: {design}",
         "This recommendation is generated live from your answers and the structure of the uploaded file.")

    insight_box(st, section_insight("upload", df, st.session_state.selected_drug, prof))


# ---------------------------------------------------------------------
# ASK ANYTHING
# ---------------------------------------------------------------------
def render_ask():
    render_topbar()
    st.subheader("Ask Anything — Pharmaceutical RWE AI Assistant")
    df = get_working_df()
    drug = st.session_state.selected_drug

    card(st, "How this works", "This runs a rule-based data-intelligence engine directly against "
                                 f"the working dataset ({len(df):,} records for {drug}) — no external "
                                 "API key required. It parses your question for columns, filters, and "
                                 "aggregations and computes the answer live, so it responds differently "
                                 "depending on whichever dataset is currently loaded.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(a)
        st.markdown("---")

    question = st.text_area("Ask anything about the current dataset, payer landscape, or patient populations...", height=80)
    c1, c2 = st.columns([1, 1])
    with c1:
        send = st.button("Send", type="primary")
    with c2:
        if st.button("Clear"):
            st.session_state.chat_history = []
            st.rerun()

    if send and question.strip():
        answer = answer_question(question, df, drug)
        st.session_state.chat_history.append((question, answer))
        st.rerun()

    st.markdown("##### Example questions")
    examples = [
        "How many patients are over 65 with GI history?",
        "What is the average weight change by arm?",
        "Compare adherence by payer",
        "What is the discontinuation rate for female patients?",
    ]
    ecols = st.columns(2)
    for i, ex in enumerate(examples):
        with ecols[i % 2]:
            if st.button(ex, key=f"ex_{i}"):
                answer = answer_question(ex, df, drug)
                st.session_state.chat_history.append((ex, answer))
                st.rerun()


# ---------------------------------------------------------------------
# EXTRA: Competitive Intelligence
# ---------------------------------------------------------------------
def render_competitive():
    render_topbar()
    st.subheader("Competitive Intelligence — Pipeline & Market Share")
    drug = st.session_state.selected_drug

    card(st, "Why this matters", "Launch and access decisions depend on where a product sits "
                                   "relative to the rest of the incretin/obesity-drug class, not just "
                                   "its own trial data.")

    comp_df = pd.DataFrame({
        "Product": DRUGS,
        "Est. US Market Share (%)": np.round(np.random.default_rng(7).dirichlet(np.ones(len(DRUGS))) * 100, 1),
        "Modal Formulary Tier": np.random.default_rng(3).choice([2, 3, 4], size=len(DRUGS)),
    })
    fig = go.Figure(go.Bar(x=comp_df["Product"], y=comp_df["Est. US Market Share (%)"],
                            marker_color=["#2f6fb0" if p == drug else "#a9cdf0" for p in comp_df["Product"]]))
    st.plotly_chart(themed_figure(fig, "Estimated Market Share"), use_container_width=True)
    st.dataframe(comp_df, use_container_width=True)
    insight_box(st, f"{drug} is highlighted above. Next step: cross-check its formulary tier against "
                     f"the Payer Intelligence PA-approval table to see whether access — not efficacy — "
                     f"is the binding constraint on share.")


# ---------------------------------------------------------------------
# EXTRA: Evidence & Publication Tracker
# ---------------------------------------------------------------------
def render_evidence():
    render_topbar()
    st.subheader("Evidence & Publication Tracker")
    drug = st.session_state.selected_drug

    card(st, "Why this matters", "RWE evidence only creates value once it reaches payers, regulators, "
                                   "or KOLs — this tracks what's in the pipeline from analysis to publication.")

    evidence_df = pd.DataFrame({
        "Evidence Asset": [f"{drug} 12-month RWE cohort study", f"{drug} budget impact model",
                            "Indication expansion signal package", "Payer dossier (AMCP format)"],
        "Status": ["Draft manuscript", "Ready for payer submission", "Routed to scientific affairs", "In review"],
        "Target Audience": ["Peer-reviewed journal", "Payer P&T committees", "Regulatory / clinical", "Payer medical directors"],
    })
    st.dataframe(evidence_df, use_container_width=True)
    insight_box(st, "Prioritize the budget impact model for payer submission first — it's marked "
                     "ready and directly supports near-term formulary decisions. The manuscript can "
                     "follow once the cohort analysis is finalized in Your Projects.")


# ---------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------
def main():
    init_state()
    if not st.session_state.authenticated:
        render_login()
        return

    page = st.session_state.page
    if page == "home":
        render_home()
    elif page == "cohort_builder":
        render_cohort_builder()
    elif page == "your_projects":
        render_your_projects()
    elif page == "payer":
        render_payer()
    elif page == "signal_lab":
        render_signal_lab()
    elif page == "upload":
        render_upload()
    elif page == "ask":
        render_ask()
    elif page == "competitive":
        render_competitive()
    elif page == "evidence":
        render_evidence()
    else:
        render_home()


if __name__ == "__main__":
    main()

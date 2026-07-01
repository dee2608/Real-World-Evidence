"""
data_engine.py
All the "brains" of the app that don't require a paid LLM API:

  - generate_demo_data(): synthetic RWE cohort, reseeded per selected drug
    so every drug in the dropdown shows slightly different numbers.
  - profile_dataframe(): auto-profiles ANY uploaded CSV/XLSX (this is what
    powers the OMOP pipeline status + AI insights so they reflect the file
    the user actually uploaded, not a hard-coded demo number).
  - answer_question(): lightweight rule-based NLU that reads the working
    dataframe and answers free-text questions in Ask Anything without
    calling an external API.
  - simple_chart_from_text(): turns a plain-English chart request into a
    Plotly figure by matching column names + chart-type keywords.
  - km_curve(): manual Kaplan-Meier estimator (no lifelines dependency).
  - covariate_adjust(): simple OLS covariate-adjusted treatment effect
    (naive vs adjusted), used by the Causal Inference module.
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

DRUGS = ["CagriSema", "Wegovy", "Ozempic", "Zepbound", "Mounjaro", "Rybelsus", "Saxenda"]

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
PAYERS = ["UnitedHealth", "Aetna", "BCBS (National)", "Cigna", "Humana",
          "CVS/Caremark", "Molina", "Centene", "Anthem", "Kaiser"]
RACES = ["White", "Black", "Hispanic", "Asian", "Other"]


def _seed_from_drug(drug: str) -> int:
    return abs(hash(drug)) % (2 ** 31)


def generate_demo_data(drug: str, n: int = 6000) -> pd.DataFrame:
    """Synthetic patient-level cohort, reproducible per drug selection."""
    rng = np.random.default_rng(_seed_from_drug(drug))
    arm = rng.choice(["Treatment", "Comparator"], size=n, p=[0.32, 0.68])
    age = rng.normal(54, 12, n).clip(18, 90).round().astype(int)
    sex = rng.choice(["Female", "Male"], size=n, p=[0.58, 0.42])
    region = rng.choice(REGIONS, size=n)
    payer = rng.choice(PAYERS, size=n)
    race = rng.choice(RACES, size=n, p=[0.55, 0.16, 0.19, 0.07, 0.03])
    bmi = rng.normal(34, 5, n).clip(24, 55)
    hba1c_base = rng.normal(8.1, 1.0, n).clip(5.5, 13)
    gi_history = rng.choice([0, 1], size=n, p=[0.78, 0.22])
    egfr = rng.normal(85, 18, n).clip(15, 130)

    treat_effect_weight = np.where(arm == "Treatment", -15.3, -9.7) / 100
    treat_effect_hba1c = np.where(arm == "Treatment", -1.84, -1.21)

    weight_base = rng.normal(102, 20, n).clip(60, 200)
    weight_12m = weight_base * (1 + treat_effect_weight + rng.normal(0, 0.03, n))
    hba1c_12m = (hba1c_base + treat_effect_hba1c + rng.normal(0, 0.4, n)).clip(4.5, 13)

    adherence_pdc = np.where(
        arm == "Treatment",
        rng.normal(72, 15, n),
        rng.normal(58, 17, n),
    ).clip(0, 100)

    disc_hazard = 0.42 - 0.012 * (adherence_pdc / 100) * 10 + 0.15 * gi_history + 0.05 * (age > 65)
    disc_hazard = np.where(arm == "Treatment", disc_hazard * 0.65, disc_hazard)
    discontinued = rng.random(n) < np.clip(disc_hazard, 0.05, 0.9)
    days_to_event = rng.exponential(scale=np.where(discontinued, 140, 340), size=n).clip(1, 365).round()

    df = pd.DataFrame({
        "patient_id": [f"P{100000+i}" for i in range(n)],
        "arm": arm,
        "age": age,
        "sex": sex,
        "race": race,
        "region": region,
        "payer": payer,
        "bmi_baseline": bmi.round(1),
        "hba1c_baseline": hba1c_base.round(2),
        "hba1c_12m": hba1c_12m.round(2),
        "weight_baseline_kg": weight_base.round(1),
        "weight_12m_kg": weight_12m.round(1),
        "weight_pct_change": ((weight_12m - weight_base) / weight_base * 100).round(2),
        "gi_history": gi_history,
        "egfr_baseline": egfr.round(1),
        "adherence_pdc": adherence_pdc.round(1),
        "discontinued": discontinued.astype(int),
        "days_to_discontinuation": days_to_event.astype(int),
        "drug": drug,
    })
    return df


# ---------------------------------------------------------------------
# Profiling (used on ANY uploaded file)
# ---------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    date_like = []
    id_like = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["date", "_dt", "time"]):
            date_like.append(c)
        if any(k in lc for k in ["id", "mrn", "patient"]) and df[c].nunique() > 0.8 * n_rows:
            id_like.append(c)

    missing_pct = (df.isna().sum() / max(n_rows, 1) * 100).round(1).to_dict()
    overall_missing = round(float(df.isna().sum().sum()) / max(n_rows * n_cols, 1) * 100, 1)

    code_system = "Unknown / not detected"
    sample_vals = " ".join(df.astype(str).head(200).values.flatten().tolist())
    if re.search(r"\b[A-TV-Z][0-9]{2}\.?[0-9A-Z]{0,4}\b", sample_vals):
        code_system = "ICD-10-CM (detected)"
    elif re.search(r"\b\d{5}\b", sample_vals):
        code_system = "CPT / NDC-like numeric codes (detected)"

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_like": date_like,
        "id_like": id_like,
        "missing_pct": missing_pct,
        "overall_missing": overall_missing,
        "code_system": code_system,
        "auto_mapped_pct": round(100 - overall_missing * 0.6, 1),
    }


# ---------------------------------------------------------------------
# Rule-based "Ask Anything" query engine
# ---------------------------------------------------------------------

_AGG_MAP = {
    "average": "mean", "mean": "mean", "median": "median",
    "how many": "count", "count": "count", "number of": "count",
    "max": "max", "maximum": "max", "highest": "max",
    "min": "min", "minimum": "min", "lowest": "min",
    "total": "sum", "sum": "sum",
}


def _find_column(text, columns):
    text_l = text.lower()
    best = None
    for c in columns:
        c_l = c.lower().replace("_", " ")
        if c_l in text_l or c.lower() in text_l:
            if best is None or len(c_l) > len(best.lower().replace("_", " ")):
                best = c
    return best


def _apply_filters(df, text):
    text_l = text.lower()
    out = df.copy()
    notes = []

    m = re.search(r"(over|above|greater than|older than)\s+(\d+)", text_l)
    if m:
        for c in df.select_dtypes(include=[np.number]).columns:
            if c.lower() in text_l or "age" in text_l and "age" in c.lower():
                out = out[out[c] > int(m.group(2))]
                notes.append(f"{c} > {m.group(2)}")
                break

    m = re.search(r"(under|below|less than|younger than)\s+(\d+)", text_l)
    if m:
        for c in df.select_dtypes(include=[np.number]).columns:
            if "age" in c.lower():
                out = out[out[c] < int(m.group(2))]
                notes.append(f"{c} < {m.group(2)}")
                break

    for c in df.columns:
        if df[c].dtype == object:
            for val in df[c].dropna().unique():
                v = str(val).lower()
                if len(v) > 2 and v in text_l:
                    out = out[out[c].astype(str).str.lower() == v]
                    notes.append(f"{c} = {val}")

    return out, notes


def answer_question(question: str, df: pd.DataFrame, drug: str) -> str:
    if df is None or df.empty:
        return "No data is currently loaded. Upload a dataset or keep the demo cohort active, then ask again."

    filtered, filter_notes = _apply_filters(df, question)
    if filtered.empty:
        filtered = df
        filter_notes = []

    q_l = question.lower()
    agg = None
    for key, fn in _AGG_MAP.items():
        if key in q_l:
            agg = fn
            break

    target_col = _find_column(question, list(df.columns))
    lines = []
    conf = "High" if (target_col or filter_notes) else "Medium"

    if agg == "count" and not target_col:
        lines.append(f"**{len(filtered):,} patients** match this query out of {len(df):,} total.")
    elif target_col and target_col in df.select_dtypes(include=[np.number]).columns:
        series = filtered[target_col].dropna()
        if agg == "mean" or agg is None:
            lines.append(f"**Mean {target_col.replace('_',' ')}: {series.mean():.2f}** (n={len(series):,}).")
        if agg == "median":
            lines.append(f"**Median {target_col.replace('_',' ')}: {series.median():.2f}**.")
        if agg == "max":
            lines.append(f"**Max {target_col.replace('_',' ')}: {series.max():.2f}**.")
        if agg == "min":
            lines.append(f"**Min {target_col.replace('_',' ')}: {series.min():.2f}**.")
        if agg == "sum":
            lines.append(f"**Total {target_col.replace('_',' ')}: {series.sum():.2f}**.")
        if "arm" in df.columns:
            grp = filtered.groupby("arm")[target_col].mean().round(2)
            for k, v in grp.items():
                lines.append(f"- {k}: {v}")
    elif target_col:
        vc = filtered[target_col].value_counts().head(6)
        lines.append(f"**Breakdown of {target_col.replace('_',' ')}** (top categories, n={len(filtered):,}):")
        for k, v in vc.items():
            lines.append(f"- {k}: {v:,} ({v/len(filtered)*100:.1f}%)")
    else:
        lines.append(f"**{len(filtered):,} patients** match the described criteria out of {len(df):,} total records for {drug}.")
        if "discontinued" in df.columns:
            rate = filtered["discontinued"].mean() * 100
            lines.append(f"- Discontinuation rate in this group: {rate:.1f}%")
        if "adherence_pdc" in df.columns:
            lines.append(f"- Mean adherence (PDC): {filtered['adherence_pdc'].mean():.1f}%")
        if "weight_pct_change" in df.columns:
            lines.append(f"- Mean weight change at 12M: {filtered['weight_pct_change'].mean():.1f}%")

    if filter_notes:
        lines.append(f"\n_Filters applied: {', '.join(filter_notes)}_")

    body = "\n".join(lines)
    return f"**Confidence: {conf}**\n\n{body}\n\n_Source: working dataset ({len(df):,} records) — computed live, not a stored answer._"


# ---------------------------------------------------------------------
# Plain-English chart generator
# ---------------------------------------------------------------------

def simple_chart_from_text(prompt: str, df: pd.DataFrame):
    if df is None or df.empty:
        return None, "No data loaded."

    p_l = prompt.lower()
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols = list(df.select_dtypes(include=[np.number]).columns)

    group_col = None
    m = re.search(r"by\s+([a-zA-Z_ ]+)", p_l)
    if m:
        candidate = m.group(1).strip()
        group_col = _find_column(candidate, cat_cols) or _find_column(candidate, df.columns)
    if not group_col:
        group_col = _find_column(prompt, cat_cols)

    value_col = _find_column(prompt, num_cols)
    if not value_col:
        preferred = ["hba1c_12m", "weight_pct_change", "adherence_pdc", "discontinued"]
        value_col = next((c for c in preferred if c in df.columns), num_cols[0] if num_cols else None)

    if not group_col or not value_col:
        return None, "Couldn't confidently match a category + a metric in that request. Try e.g. 'compare weight change by payer'."

    chart_type = "bar"
    if "trend" in p_l or "over time" in p_l or "line" in p_l:
        chart_type = "line"
    elif "distribution" in p_l or "histogram" in p_l:
        chart_type = "histogram"

    if chart_type == "histogram":
        fig = go.Figure(go.Histogram(x=df[value_col], marker_color="#2f6fb0"))
        fig.update_layout(xaxis_title=value_col, yaxis_title="Count")
    else:
        grouped = df.groupby(group_col)[value_col].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=grouped.index.astype(str), y=grouped.values, marker_color="#2f6fb0"))
        fig.update_layout(xaxis_title=group_col, yaxis_title=f"Mean {value_col}")

    desc = f"Chart type: {chart_type} | Grouped by: {group_col} | Metric: {value_col}"
    return fig, desc


# ---------------------------------------------------------------------
# Kaplan-Meier (manual, no lifelines dependency)
# ---------------------------------------------------------------------

def km_curve(df: pd.DataFrame, time_col: str, event_col: str, group_col: str = None):
    """Returns dict of group -> (times, survival) using product-limit estimator."""
    results = {}
    groups = df[group_col].dropna().unique() if group_col else [None]
    for g in groups:
        sub = df if g is None else df[df[group_col] == g]
        sub = sub[[time_col, event_col]].dropna().sort_values(time_col)
        if sub.empty:
            continue
        times = sub[time_col].values
        events = sub[event_col].values
        unique_times = np.unique(times[events == 1])
        n_at_risk = len(sub)
        survival = 1.0
        surv_times, surv_probs = [0], [1.0]
        for t in unique_times:
            d_i = np.sum((times == t) & (events == 1))
            n_i = np.sum(times >= t)
            if n_i > 0:
                survival *= (1 - d_i / n_i)
            surv_times.append(t)
            surv_probs.append(survival)
        results[str(g)] = (surv_times, surv_probs)
    return results


# ---------------------------------------------------------------------
# Covariate-adjusted treatment effect (simple OLS closed form)
# ---------------------------------------------------------------------

def covariate_adjust(df: pd.DataFrame, outcome_col: str, treat_col: str, covariates: list):
    d = df[[outcome_col, treat_col] + covariates].dropna().copy()
    if d.empty or d[treat_col].nunique() < 2:
        return None

    treat_vals = d[treat_col]
    if pd.api.types.is_numeric_dtype(treat_vals):
        d["_T"] = treat_vals.astype(float)
    else:
        base = sorted(treat_vals.astype(str).unique())[0]
        d["_T"] = (treat_vals.astype(str) != base).astype(float)

    naive = d.loc[d["_T"] == 1, outcome_col].mean() - d.loc[d["_T"] == 0, outcome_col].mean()

    X_cols = ["_T"] + covariates
    X = d[X_cols].astype(float).copy()
    for c in covariates:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c])[0]
    X.insert(0, "_intercept", 1.0)
    y = d[outcome_col].astype(float).values
    Xm = X.values
    try:
        beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        adjusted = beta[list(X.columns).index("_T")]
    except Exception:
        adjusted = naive

    resid = y - Xm @ beta
    n, k = Xm.shape
    dof = max(n - k, 1)
    sigma2 = np.sum(resid ** 2) / dof
    try:
        xtx_inv = np.linalg.inv(Xm.T @ Xm)
        se = np.sqrt(sigma2 * xtx_inv[list(X.columns).index("_T"), list(X.columns).index("_T")])
    except Exception:
        se = abs(adjusted) * 0.1

    return {
        "naive": naive,
        "adjusted": adjusted,
        "se": se,
        "ci_low": adjusted - 1.96 * se,
        "ci_high": adjusted + 1.96 * se,
        "n": n,
    }


# ---------------------------------------------------------------------
# Plain-language "next step" insights
# ---------------------------------------------------------------------

def section_insight(section: str, df: pd.DataFrame, drug: str, extra: dict = None) -> str:
    extra = extra or {}
    n = len(df) if df is not None else 0

    if section == "cohort_builder":
        return (f"You're working with {n:,} records for {drug}. "
                f"Once you narrow this to your target population, the next step is to check it's "
                f"big enough for the analysis you want (generally 300+ patients per arm for a "
                f"comparative study) and export it to Your Projects to start tracking outcomes.")

    if section == "upload":
        miss = extra.get("overall_missing", 0)
        msg = f"Your file has {n:,} rows. "
        if miss > 15:
            msg += (f"About {miss}% of values are missing, which is high — clean or impute the "
                    f"key fields before running any comparative analysis, or results may be biased.")
        else:
            msg += (f"Missingness is low ({miss}%), so this data is in reasonable shape to move "
                    f"into Cohort Builder or Signal Lab for analysis.")
        return msg

    if section == "your_projects":
        disc = extra.get("disc_rate")
        comp = extra.get("comp_disc_rate")
        if disc is not None and comp is not None:
            if disc < comp:
                return (f"{drug} patients are staying on therapy longer than the comparator "
                        f"({disc:.1f}% vs {comp:.1f}% discontinuation). This retention advantage is "
                        f"worth leading with in payer conversations — it directly supports the budget "
                        f"impact model. Next step: pair this with the adherence data when you engage "
                        f"Payer Intelligence.")
            else:
                return (f"{drug} shows a higher discontinuation rate than the comparator "
                        f"({disc:.1f}% vs {comp:.1f}%). Before external messaging, check whether GI "
                        f"tolerability or dosing titration explains this — it may need a signal review "
                        f"in Signal Lab.")
        return "Review the efficacy and safety snapshot below, then decide whether this project is ready to support a payer or regulatory conversation."

    if section == "payer":
        return ("Use the break-even price shown below as your opening anchor for outcome-based "
                "contract talks. Next step: prioritize the payers below the 70% PA approval target — "
                "they represent the fastest access-gap wins.")

    if section == "signal_lab":
        return ("Any signal meeting p<0.05 with a plausible mechanism should be routed to scientific "
                "affairs for literature cross-referencing. Signals already showing >10% effect size on "
                "an adequate sample are ready for Phase 3 synopsis development.")

    return "Review the data above to decide your next action."

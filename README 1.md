# RWE Command Center

A Streamlit-based Real-World Evidence command center covering Cohort Builder,
Your Projects, Payer Intelligence, Signal Lab, Upload Your Data, Ask Anything,
plus two added modules: Competitive Intelligence and Evidence & Publication
Tracker.

## Files
- `rwe_app.py` — main app (routing, all pages, login, logout).
- `data_engine.py` — synthetic demo data generator, data profiler, rule-based
  "Ask Anything" query engine, plain-English chart generator, manual
  Kaplan-Meier estimator, and covariate-adjustment (causal inference) logic.
  None of this calls an external API — it's computed locally from whatever
  dataset is currently loaded (demo or your upload).
- `styling.py` — CSS overrides (white input/box backgrounds with light-blue
  borders instead of black boxes) and a shared Plotly theme with dark,
  visible axis lines.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — theme + upload size settings.
- `.devcontainer/devcontainer.json` — optional, for GitHub Codespaces / local
  dev containers.

## Login
Demo credentials: `RWE123` / `RWE2026` (edit `CREDENTIALS` at the top of
`rwe_app.py`, or move to `st.secrets` for production use).

## Deploy on Streamlit Community Cloud
1. Create a new GitHub repo and add all files in this folder to it
   (keep the `.streamlit/` folder — it controls the theme).
2. Go to https://share.streamlit.io, connect the repo, and set
   **Main file path** to `rwe_app.py`.
3. Deploy. No secrets/API keys are required — "Ask Anything" and all
   AI-insight text run on a local rule-based engine.

## Run locally
```bash
pip install -r requirements.txt
streamlit run rwe_app.py
```

## Notes on "AI" features
- **Ask Anything** parses your question for column names, filters (e.g.
  "over 65", "female"), and aggregation words (average/count/compare), then
  computes the answer directly from the active dataframe — so it responds
  differently depending on the dataset loaded, not from a fixed script.
- **Upload Your Data** profiles any CSV/XLSX you provide (row/column counts,
  missingness, likely ID/date columns, a rough coding-system guess) instead
  of showing fixed demo numbers.
- **Custom Chart Generator** matches plain-English requests ("compare X by
  Y", "trend of X") against your columns to build a Plotly chart on the fly.
- Kaplan-Meier and covariate-adjusted treatment effects are computed with
  plain NumPy/Pandas (no extra survival-analysis dependency), so they work
  on both the synthetic demo cohort and any uploaded dataset with matching
  columns.

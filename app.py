"""
app.py - Streamlit front end for the "Ask the Data Jobs database" Text-to-SQL app.

Flow of one question:
    user question -> Gemini writes SQL -> db.py validates + runs it on DuckDB
    -> result table / chart / CSV download shown on the page.

IMPORTANT Streamlit concept: Streamlit re-runs this whole file from top to bottom
every time the user clicks a button or changes a widget. Anything that must
survive between re-runs is kept in st.session_state (per visitor) or in
@st.cache_resource (shared by everyone, created once).
"""
import os

import streamlit as st
from google import genai  # Google's Gemini SDK (package: google-genai)

import db   # our file: DuckDB connection, SQL safety checks, query runner
import llm  # our file: builds the prompt and calls Gemini

# ---------------------------------------------------------------------------
# Page setup. Must be the first Streamlit call in the script.
# layout="wide" lets wide result tables use the full browser width.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Data Jobs Q&A", page_icon="📊", layout="wide")

# Per-visitor cap on questions. Protects the free Gemini quota from heavy use.
# Note: a page refresh resets it, so it slows heavy use rather than blocking it.
MAX_QUESTIONS_PER_SESSION = 20


def secret(name: str, default: str | None = None):
    """Read a setting from Streamlit secrets, falling back to an environment variable.

    - On Streamlit Cloud / locally: values come from .streamlit/secrets.toml.
    - Otherwise (e.g. Docker, CI): values come from environment variables.
    - If neither exists, return `default` so the app does not crash on optional settings.
    """
    try:
        return st.secrets[name]
    except Exception:  # secrets file missing or key not found
        return os.environ.get(name, default)


# Which Gemini model to call. Set GEMINI_MODEL in secrets (e.g. "gemini-3.1-flash-lite");
# the value below is only used if you did not set one.
MODEL = secret("GEMINI_MODEL", "gemini-2.5-flash")  # check AI Studio for current free models


# ---------------------------------------------------------------------------
# One-time setup, shared by all visitors.
# @st.cache_resource runs this function ONCE and reuses the returned objects on
# every re-run, so we do not reopen the database or rebuild the prompt each time.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing database (first run can take a minute)...")
def load_resources():
    # Open the read-only DuckDB file. If it is missing, db.py downloads it from
    # your Hugging Face repo (HF_DB_REPO), or rebuilds it from the original dataset.
    con = db.get_conn(secret("HF_DB_REPO"))
    # Read the live table schema and build the system prompt Gemini will receive.
    schema_prompt = llm.build_schema_prompt(con)
    # Gemini API client, authenticated with your key from secrets.
    client = genai.Client(api_key=secret("GEMINI_API_KEY"))
    return con, schema_prompt, client


def show_chart(df):
    """Offer a bar/line chart when the result looks plottable."""
    # Columns containing numbers (counts, medians, percentages...).
    num = df.select_dtypes("number").columns.tolist()
    # Skip charts for a single row, very long results (unreadable), or no numbers at all.
    if len(df) < 2 or len(df) > 60 or not num:
        return
    # Non-numeric columns are candidates for the x-axis (e.g. skill, job title).
    others = [c for c in df.columns if c not in num]
    # Use the first text column as x; if the result is all numbers, use the first column.
    x = others[0] if others else df.columns[0]
    # Everything numeric except the x column becomes a plotted series.
    ys = [c for c in num if c != x]
    if not ys:
        return
    # Let the user pick the chart type. The fixed key keeps the choice across re-runs.
    kind = st.radio("Chart", ["None", "Bar", "Line"], horizontal=True, key="chart_kind")
    if kind == "Bar":
        # sort=False keeps the SQL's ORDER BY (e.g. highest first) instead of re-sorting.
        st.bar_chart(df, x=x, y=ys, sort=False)
    elif kind == "Line":
        st.line_chart(df, x=x, y=ys)


# Run the cached setup (fast after the first call).
con, schema_prompt, client = load_resources()

# ---------------------------------------------------------------------------
# Per-visitor state. setdefault only sets a value the first time, so these
# survive across re-runs without being reset.
# ---------------------------------------------------------------------------
st.session_state.setdefault("count", 0)      # questions asked so far this session
st.session_state.setdefault("q", "")         # text currently in the question box
st.session_state.setdefault("result", None)  # last answer: {"sql", "df", "error"}

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("📊 Ask the Data Jobs database")
st.caption("Ask in plain English. Gemini writes the SQL, DuckDB runs it. "
           "Dataset: lukebarousse/data_jobs (2023 data-job postings).")

# ---------------------------------------------------------------------------
# Sidebar: clickable example questions + usage counter
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Try these")
    for ex in [
        "Top 10 most in-demand skills for Data Analysts",
        "Median salary by job role in the United States",
        "Which 10 companies post the most Data Engineer jobs?",
        "Percent of remote jobs per role",
        "Top 5 skills for Data Scientists in India",
    ]:
        # on_click copies the example into the question box (session key "q").
        # `e=ex` freezes the current example; without it every button would use the last one.
        st.button(ex, on_click=lambda e=ex: st.session_state.update(q=e))
    st.caption(f"Questions used: {st.session_state.count}/{MAX_QUESTIONS_PER_SESSION}")

# Question box. key="q" links it to st.session_state["q"], which is how the
# sidebar buttons can fill it in.
question = st.text_input("Your question", key="q")

# ---------------------------------------------------------------------------
# Handle a click on "Ask". The block only runs on the re-run triggered by the click.
# ---------------------------------------------------------------------------
if st.button("Ask", type="primary") and question.strip():
    # Stop early if this visitor used up their allowance.
    if st.session_state.count >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit reached for this session. Refresh to start over.")
        st.stop()  # nothing below this line runs on this re-run
    st.session_state.count += 1  # counted before the call, so failed attempts count too
    try:
        with st.spinner("Writing SQL..."):
            # Step 1: ask Gemini to turn the question into a SQL query.
            sql = llm.generate_sql(client, MODEL, schema_prompt, question)
            try:
                # Step 2: validate (SELECT only) and run it on DuckDB -> DataFrame.
                df = db.run_query(con, sql)
            except Exception as first_error:  # one self-correction retry
                # The SQL was invalid or rejected. Send Gemini the failed SQL and
                # the error message and ask for a corrected query, then run that.
                sql = llm.generate_sql(client, MODEL, schema_prompt, question,
                                       prev_sql=sql, error=str(first_error))
                df = db.run_query(con, sql)  # if this fails too, the outer except handles it
        # Save the outcome in session state so it survives the next re-run.
        st.session_state.result = {"sql": sql, "df": df, "error": None}
    except Exception as e:  # Gemini error, quota hit, or SQL failed twice
        st.session_state.result = {"sql": None, "df": None, "error": str(e)}

# ---------------------------------------------------------------------------
# Show the last result. This sits OUTSIDE the "Ask" block on purpose: choosing a
# chart type or clicking Download triggers a re-run where "Ask" is no longer
# clicked, and the answer would vanish if it were displayed only inside that block.
# ---------------------------------------------------------------------------
res = st.session_state.result  # kept in session state so widgets below don't wipe it
if res:
    if res["error"]:
        st.error(f"Sorry, that didn't work: {res['error']}")
    else:
        # Showing the generated SQL lets users check the answer.
        with st.expander("Generated SQL", expanded=True):
            st.code(res["sql"], language="sql")
        st.caption(f"{len(res['df'])} row(s)")
        st.dataframe(res["df"])  # interactive, sortable table
        show_chart(res["df"])    # optional bar / line chart
        st.download_button("Download CSV", res["df"].to_csv(index=False), "result.csv")

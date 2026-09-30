import os

import streamlit as st
from google import genai

import db
import llm

st.set_page_config(page_title="Data Jobs Q&A", page_icon="📊", layout="wide")

MAX_QUESTIONS_PER_SESSION = 20  # protects your free Gemini quota


def secret(name: str, default: str | None = None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


MODEL = secret("GEMINI_MODEL", "gemini-2.5-flash")  # check AI Studio for current free models


@st.cache_resource(show_spinner="Preparing database (first run can take a minute)...")
def load_resources():
    con = db.get_conn(secret("HF_DB_REPO"))
    schema_prompt = llm.build_schema_prompt(con)
    client = genai.Client(api_key=secret("GEMINI_API_KEY"))
    return con, schema_prompt, client


def show_chart(df):
    """Offer a bar/line chart when the result looks plottable."""
    num = df.select_dtypes("number").columns.tolist()
    if len(df) < 2 or len(df) > 60 or not num:
        return
    others = [c for c in df.columns if c not in num]
    x = others[0] if others else df.columns[0]
    ys = [c for c in num if c != x]
    if not ys:
        return
    kind = st.radio("Chart", ["None", "Bar", "Line"], horizontal=True, key="chart_kind")
    if kind == "Bar":
        st.bar_chart(df, x=x, y=ys, sort=False)
    elif kind == "Line":
        st.line_chart(df, x=x, y=ys)


con, schema_prompt, client = load_resources()

st.session_state.setdefault("count", 0)
st.session_state.setdefault("q", "")
st.session_state.setdefault("result", None)

st.title("📊 Ask the Data Jobs database")
st.caption("Ask in plain English. Gemini writes the SQL, DuckDB runs it. "
           "Dataset: lukebarousse/data_jobs (2023 data-job postings).")

with st.sidebar:
    st.header("Try these")
    for ex in [
        "Top 10 most in-demand skills for Data Analysts",
        "Median salary by job role in the United States",
        "Which 10 companies post the most Data Engineer jobs?",
        "Percent of remote jobs per role",
        "Top 5 skills for Data Scientists in India",
    ]:
        st.button(ex, on_click=lambda e=ex: st.session_state.update(q=e))
    st.caption(f"Questions used: {st.session_state.count}/{MAX_QUESTIONS_PER_SESSION}")

question = st.text_input("Your question", key="q")

if st.button("Ask", type="primary") and question.strip():
    if st.session_state.count >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit reached for this session. Refresh to start over.")
        st.stop()
    st.session_state.count += 1
    try:
        with st.spinner("Writing SQL..."):
            sql = llm.generate_sql(client, MODEL, schema_prompt, question)
            try:
                df = db.run_query(con, sql)
            except Exception as first_error:  # one self-correction retry
                sql = llm.generate_sql(client, MODEL, schema_prompt, question,
                                       prev_sql=sql, error=str(first_error))
                df = db.run_query(con, sql)
        st.session_state.result = {"sql": sql, "df": df, "error": None}
    except Exception as e:
        st.session_state.result = {"sql": None, "df": None, "error": str(e)}

res = st.session_state.result  # kept in session state so widgets below don't wipe it
if res:
    if res["error"]:
        st.error(f"Sorry, that didn't work: {res['error']}")
    else:
        with st.expander("Generated SQL", expanded=True):
            st.code(res["sql"], language="sql")
        st.caption(f"{len(res['df'])} row(s)")
        st.dataframe(res["df"])
        show_chart(res["df"])
        st.download_button("Download CSV", res["df"].to_csv(index=False), "result.csv")

# Chat to SQL: Data Jobs Q&A

Ask plain-English questions about 786,000 data-job postings and get the answer as a table, a chart and the SQL that produced it. No SQL knowledge needed.

> "Top 10 most in-demand skills for Data Analysts" → Gemini writes the SQL → DuckDB runs it → Streamlit shows the table.

## What it does

- Turns a natural-language question into a DuckDB `SELECT` query using Gemini Flash Lite.
- Runs the query on a local, read-only DuckDB database and shows the result as a table.
- Shows the generated SQL so you can check it, plus an optional bar or line chart and a CSV download.
- Retries once with the error message if the first query fails.

## Use cases

| User | Example question |
| --- | --- |
| Student choosing what to learn | Top 10 most in-demand skills for Data Analysts |
| Job seeker comparing offers | Median salary by job role in the United States |
| Career coach or analyst | Percent of postings that are remote, per role |

## Architecture

```
Question ─► Streamlit (rate limiter) ─► Gemini (schema + rules + examples) ─► SQL
                                                                              │
Table / chart / CSV ◄─ DuckDB (read-only) ◄─ Safety check (SELECT only, LIMIT 500)
                           ▲                           │ on error: retry once
                           │                           ▼
                  jobs.duckdb (built from Hugging Face dataset)
```

Gemini is the only AI step. DuckDB computes every number, so the model cannot invent results.

## Tech stack

| Tool | Role |
| --- | --- |
| Gemini Flash Lite (`gemini-3.1-flash-lite`) | Writes SQL from the question (free tier) |
| DuckDB | Local analytical SQL engine |
| Hugging Face | Source dataset (`lukebarousse/data_jobs`) and hosting for the prebuilt database |
| Streamlit | Web interface, hosted free on Streamlit Community Cloud |

## Project structure

```
├── app.py            # Streamlit UI, rate limiter, charts
├── llm.py            # Prompt construction and Gemini call
├── db.py             # Database build, read-only connection, SQL safety checks
├── upload_db.py      # One-time upload of jobs.duckdb to a Hugging Face repo
├── requirements.txt
└── .gitignore
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

1. **Get a Gemini API key** from [Google AI Studio](https://aistudio.google.com) and check the current free model name.
2. **Create `.streamlit/secrets.toml`** (never commit it):

   ```toml
   GEMINI_API_KEY = "your-key"
   GEMINI_MODEL = "gemini-3.1-flash-lite"
   # Optional, after step 4:
   HF_DB_REPO = "your-username/data-jobs-duckdb"
   ```

3. **Build the database** (a few minutes the first time):

   ```bash
   python db.py
   ```

4. **Optional: host the database on Hugging Face** so deployed apps start fast:

   ```bash
   export HF_TOKEN=hf_xxx            # token with write access
   python upload_db.py your-username/data-jobs-duckdb
   ```

5. **Run the app:**

   ```bash
   python -m streamlit run app.py
   ```

## Deploy for free

1. Push the code to GitHub. Check `git status` first: `.streamlit/secrets.toml` and `jobs.duckdb` must not be listed.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repo with `app.py` as the entry point.
3. Paste the contents of your `secrets.toml` into **Settings → Secrets**.

Free apps sleep when idle and wake on the next visit. With `HF_DB_REPO` set, waking only downloads one file instead of rebuilding the database.

## How the prompt is designed

- The schema is read from the live database, so column names and types are never stale.
- Short text columns list their allowed values (for example the ten job roles) to prevent near-miss filters.
- Notes explain the dataset's quirks: skills are a list (`UNNEST(job_skills)`), most salaries are NULL.
- Rules: output one DuckDB `SELECT` only, no explanation, temperature 0, and a fixed message if the question cannot be answered.
- Four worked question → SQL examples cover skills, salaries, percentages and dates.
- A failed query is sent back once with its error message.

## Safety

- Database opened `read_only`, with external file access disabled.
- Only a single `SELECT` or `WITH` statement is accepted; write and file-access keywords are rejected.
- Results capped at 500 rows.
- Shared rate limiter (12 calls per minute, 450 per day) and a 20-question session cap protect the free Gemini quota.

## Testing

Each sample is judged against a hand-written reference query run directly in DuckDB.

| Sample question | Good output |
| --- | --- |
| Top 10 most in-demand skills for Data Analysts | 10 rows, skill and count, highest first, counts equal the reference query |
| Median salary by job role in the United States | One row per role, NULL salaries excluded, plausible yearly values |
| What percent of postings are remote for each role? | Values from 0 to 100, one row per role, sorted high to low |

Safety checks: a "delete all rows" request must change nothing, an off-topic question must return the fixed "cannot answer" message, and exceeding the rate limit must show a friendly message. Full pass criteria and reference queries are in the project documentation.

## Limitations and next steps

- Text-to-SQL is usually right, not always. When a question fails, add it with its correct SQL to `EXAMPLES` in `llm.py`.
- Salary is missing for most postings, so medians over small groups can mislead.
- Questions are independent; there is no follow-up memory yet.
- Rate-limit counters live in app memory and reset on restart.
- Next: an automated test that compares each sample with its reference query, follow-up questions, and retrieval (RAG) over column descriptions if more tables are added.

## Credits

Dataset: [`lukebarousse/data_jobs`](https://huggingface.co/datasets/lukebarousse/data_jobs) on Hugging Face. Built with assistance from Claude (design, code and debugging).

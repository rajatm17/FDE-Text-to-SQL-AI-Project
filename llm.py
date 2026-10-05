"""Prompt construction and Gemini call. No training: schema + examples go in the prompt."""
# This file has three parts:
#   1. EXAMPLES            - worked question -> SQL pairs shown to Gemini
#   2. build_schema_prompt - assembles the instructions Gemini receives (run once at startup)
#   3. generate_sql        - sends a question to Gemini and returns clean SQL (run per question)
#
# NOTE: the text inside EXAMPLES and inside the prompt below is sent to Gemini word for
# word, so comments are kept OUTSIDE those strings on purpose. Changing that text
# changes how the model behaves.
import re

from google.genai import types  # configuration objects for the Gemini SDK

# Few-shot examples: the model copies the style and patterns it sees here, which is
# the cheapest way to improve accuracy. Each example teaches one query shape:
#   1) UNNEST on the skills list   2) median salary with a NULL filter
#   3) a percentage from a boolean 4) grouping by month
# When the app answers a question wrongly, add that question and its correct SQL here.
EXAMPLES = """
Q: Top 10 most in-demand skills for Data Analysts
SQL: SELECT skill, COUNT(*) AS demand_count
FROM (SELECT UNNEST(job_skills) AS skill FROM jobs WHERE job_title_short = 'Data Analyst')
GROUP BY skill ORDER BY demand_count DESC LIMIT 10

Q: Median yearly salary by job role in India
SQL: SELECT job_title_short, MEDIAN(salary_year_avg) AS median_salary, COUNT(*) AS postings
FROM jobs WHERE job_country = 'India' AND salary_year_avg IS NOT NULL
GROUP BY job_title_short ORDER BY median_salary DESC

Q: What percent of postings are remote, per role?
SQL: SELECT job_title_short, ROUND(100.0 * AVG(job_work_from_home::INT), 1) AS pct_remote
FROM jobs GROUP BY job_title_short ORDER BY pct_remote DESC

Q: Number of Data Scientist postings per month
SQL: SELECT MONTH(job_posted_date) AS month, COUNT(*) AS postings
FROM jobs WHERE job_title_short = 'Data Scientist' GROUP BY month ORDER BY month
"""


def build_schema_prompt(con) -> str:
    """Describe the table from the live DB, including values of low-cardinality columns."""
    # DESCRIBE returns one row per column: (name, type, nullable, key, default, extra).
    cols = con.execute("DESCRIBE jobs").fetchall()
    lines = []  # one bullet line per column, e.g. "- salary_year_avg (DOUBLE)"
    # Take the name and type; `*_` swallows the remaining fields we do not need.
    for name, dtype, *_ in cols:
        line = f"- {name} ({dtype})"
        # Only text columns can have a small set of allowed values worth listing.
        if dtype == "VARCHAR":
            # How many different values does this column hold? (quotes protect odd names)
            n = con.execute(f'SELECT COUNT(DISTINCT "{name}") FROM jobs').fetchone()[0]
            # 15 or fewer distinct values (e.g. the 10 job roles): list them all, so
            # the model writes 'Data Analyst' exactly instead of guessing a spelling.
            # Columns with many values (company names, locations) are not listed.
            if n <= 15:
                vals = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{name}" FROM jobs WHERE "{name}" IS NOT NULL ORDER BY 1'
                ).fetchall()]
                line += f" values: {vals}"
        lines.append(line)

    # The returned text is the "system instruction" sent with every question. Its parts:
    #   Columns  - the live schema built above (never out of date)
    #   Notes    - quirks of this dataset the model could not know on its own
    #   Rules    - output format and the safe fallback for off-topic questions
    #   Examples - the EXAMPLES block above
    # chr(10) is a newline character; it joins the column lines inside the f-string.
    return f"""You translate questions into DuckDB SQL for ONE table named `jobs`
(2023 data-related job postings worldwide, ~786k rows).

Columns:
{chr(10).join(lines)}

Notes:
- job_skills is a list of lowercase skills (e.g. 'python', 'sql', 'power bi'); use UNNEST(job_skills) to count skills.
- salary_year_avg / salary_hour_avg are NULL for most rows; filter IS NOT NULL when using salaries.
- job_country is the country of the job; job_location is free text (city, state).
- job_work_from_home is a BOOLEAN. job_posted_date is a TIMESTAMP.
- Use ILIKE for fuzzy text matching on job_title or company_name.

Rules:
- Output ONLY one DuckDB SELECT statement. No explanation, no markdown fences.
- Never modify data. Use readable column aliases. Add ORDER BY and LIMIT for rankings.
- If the question cannot be answered from this table, output exactly:
  SELECT 'Cannot answer this from the jobs dataset' AS message

Examples:
{EXAMPLES}"""


# Models sometimes wrap SQL in a markdown fence (```sql ... ```) even when told not to.
# This strips a fence at the start and/or end so the query can run. The pattern
# removes an optional leading ``` or ```sql, or a trailing ```; re.I ignores case.
def _clean(text: str) -> str:
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.I)
    return text.strip()


def generate_sql(client, model: str, schema_prompt: str, question: str,
                 prev_sql: str | None = None, error: str | None = None) -> str:
    # Normal request: just the question, ending in "SQL:" so the model continues
    # with the query, matching the Q:/SQL: pattern of the examples.
    contents = f"Q: {question}\nSQL:"
    # Retry request: only when BOTH a failed query and its error are supplied. The
    # model sees what it wrote and why it failed, so the second attempt is targeted.
    if prev_sql and error:
        contents = (f"Q: {question}\nYour previous SQL failed.\nSQL: {prev_sql}\n"
                    f"Error: {error}\nReturn a corrected SQL query.\nSQL:")
    resp = client.models.generate_content(
        model=model,        # e.g. "gemini-3.1-flash-lite", set in secrets by app.py
        contents=contents,  # the user's turn: question (and error details on a retry)
        config=types.GenerateContentConfig(
            # The schema, notes, rules and examples go here, separate from the question.
            # temperature=0 means the least random output, so the same question gives
            # (almost always) the same SQL, which is what you want for code.
            system_instruction=schema_prompt, temperature=0
        ),
    )
    # resp.text can be empty or None (for example if the reply was blocked); `or ""`
    # avoids a crash and lets validation report a clear error instead.
    return _clean(resp.text or "")

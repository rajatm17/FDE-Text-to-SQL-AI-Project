"""Prompt construction and Gemini call. No training: schema + examples go in the prompt."""
import re

from google.genai import types

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
    cols = con.execute("DESCRIBE jobs").fetchall()
    lines = []
    for name, dtype, *_ in cols:
        line = f"- {name} ({dtype})"
        if dtype == "VARCHAR":
            n = con.execute(f'SELECT COUNT(DISTINCT "{name}") FROM jobs').fetchone()[0]
            if n <= 15:
                vals = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{name}" FROM jobs WHERE "{name}" IS NOT NULL ORDER BY 1'
                ).fetchall()]
                line += f" values: {vals}"
        lines.append(line)

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


def _clean(text: str) -> str:
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.I)
    return text.strip()


def generate_sql(client, model: str, schema_prompt: str, question: str,
                 prev_sql: str | None = None, error: str | None = None) -> str:
    contents = f"Q: {question}\nSQL:"
    if prev_sql and error:
        contents = (f"Q: {question}\nYour previous SQL failed.\nSQL: {prev_sql}\n"
                    f"Error: {error}\nReturn a corrected SQL query.\nSQL:")
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=schema_prompt, temperature=0
        ),
    )
    return _clean(resp.text or "")
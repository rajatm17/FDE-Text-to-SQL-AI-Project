"""DuckDB ingestion, connection and SQL safety checks."""
# This file has three jobs:
#   1. BUILD   - turn the Hugging Face dataset into a local DuckDB file (build_db)
#   2. CONNECT - get that file onto disk and open it read-only (fetch_prebuilt, get_conn)
#   3. PROTECT - make sure only safe, read-only SQL ever runs (validate_sql, run_query)
import ast  # safely turns the text "['python', 'sql']" into a real Python list
import re   # regular expressions, used by the SQL safety checks
from pathlib import Path

import duckdb
import pyarrow as pa  # builds the typed columnar table that DuckDB loads

# ---------- settings ----------
DB_PATH = Path("jobs.duckdb")          # local database file (created next to the app)
HF_DATASET = "lukebarousse/data_jobs"  # source dataset on Hugging Face
MAX_ROWS = 500                         # hard cap on rows returned to the app


# ---------- 1. BUILD ----------
def _parse_skills(value):
    """'['python', 'sql', 'sql']' (string) -> ['python', 'sql'] (deduped list)."""
    # In the raw dataset, job_skills is a TEXT that merely looks like a list.
    # Anything that is not text (e.g. a missing value) becomes None (NULL in the database).
    if not isinstance(value, str):
        return None
    try:
        # literal_eval parses the text into a real list without running arbitrary code.
        # dict.fromkeys removes duplicates while keeping the original order.
        return list(dict.fromkeys(ast.literal_eval(value)))
    except (ValueError, SyntaxError):
        # Malformed text: store NULL instead of crashing the whole build.
        return None


def build_db(path: Path = DB_PATH) -> None:
    """Download the dataset from Hugging Face and write it to a DuckDB file."""
    # Imported here (not at the top) so the running app does not load these heavy
    # libraries unless it actually has to build the database.
    import pandas as pd
    from datasets import load_dataset

    # Download the data (about 786k rows) and convert it to a pandas DataFrame.
    df = load_dataset(HF_DATASET, split="train").to_pandas()
    df = df.drop(columns=["job_type_skills"])  # redundant nested string
    # Store dates as real timestamps so SQL functions like MONTH() work.
    df["job_posted_date"] = pd.to_datetime(df["job_posted_date"])
    # Add a simple unique id (1, 2, 3, ...) as the first column.
    df.insert(0, "job_id", range(1, len(df) + 1))

    # Turn the skills string into a real list column so SQL can UNNEST it.
    skills = pa.array(
        [_parse_skills(s) for s in df["job_skills"]], type=pa.list_(pa.string())
    )
    # Convert the rest of the DataFrame to an Arrow table (without the old text
    # skills column), then attach the new list-typed skills column at the end.
    table = pa.Table.from_pandas(
        df.drop(columns=["job_skills"]), preserve_index=False
    ).append_column("job_skills", skills)

    # Start from a clean file so an old, partial build is never reused.
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))   # creates the new database file
    con.register("t", table)          # expose the Arrow table to SQL under the name "t"
    con.execute("CREATE TABLE jobs AS SELECT * FROM t")  # copy it into a real table named jobs
    con.close()                       # closing flushes everything to disk


# ---------- 2. CONNECT ----------
def fetch_prebuilt(repo_id: str) -> bool:
    """Download jobs.duckdb from a (public) Hugging Face dataset repo. True on success."""
    try:
        # Imported inside the function so it is only needed when this path is used.
        from huggingface_hub import hf_hub_download

        # Saves the file into the current folder as jobs.duckdb.
        hf_hub_download(repo_id=repo_id, filename=DB_PATH.name,
                        repo_type="dataset", local_dir=".")
        return DB_PATH.exists()  # confirm the file really arrived
    except Exception:
        # Any problem (wrong repo name, private repo, no network) -> report failure
        # so the caller can fall back to building the database instead.
        return False


def get_conn(hf_db_repo: str | None = None) -> duckdb.DuckDBPyConnection:
    """Read-only connection. If the file is missing: try your HF repo (fast),
    then fall back to building from the original dataset (slow)."""
    if not DB_PATH.exists():
        # Missing file: try the quick download first (only if a repo name was given);
        # if that is not set up or fails, rebuild from the original dataset.
        if not (hf_db_repo and fetch_prebuilt(hf_db_repo)):
            build_db()
    # read_only=True: the app can query the data but never change it.
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET enable_external_access=false")  # no file/network reads from SQL
    return con


# ---------- 3. PROTECT ----------
# Words that must never appear in a query (outside quoted text): anything that
# writes data, changes the schema, reads files, loads extensions or changes settings.
# \b means "whole word only", and re.I makes the match case-insensitive.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|"
    r"install|load|pragma|call|set|truncate|read_csv|read_parquet|read_json|glob)\b",
    re.I,
)
# Matches a single-quoted string such as 'Data Analyst'. The (?:[^']|'') part also
# handles an escaped quote written as two quotes ('').
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


# Checks that a query is one read-only SELECT. Returns the cleaned query, or raises
# ValueError with a message the app shows to the user. (Safety layer 2 of 3: layer 1
# is the read-only connection, layer 3 is the row cap in run_query.)
def validate_sql(sql: str) -> str:
    # Trim spaces and one trailing semicolon, which models often add.
    sql = sql.strip().rstrip(";").strip()
    # Blank out the contents of quoted strings, so a value like 'drop; table' (or a
    # company name containing a keyword) is not mistaken for a command.
    bare = _STRING_LITERAL.sub("''", sql)  # ignore words inside quoted strings
    # Any semicolon left means a second statement is hiding in the text.
    if ";" in bare:
        raise ValueError("Only a single SQL statement is allowed.")
    # The query must start with SELECT, or WITH (a CTE that still ends in a SELECT).
    if not re.match(r"^(select|with)\b", bare, re.I):
        raise ValueError("Only SELECT queries are allowed.")
    # Reject any blocked keyword found anywhere in the query.
    if _FORBIDDEN.search(bare):
        raise ValueError("Query contains a forbidden keyword.")
    # Return the ORIGINAL cleaned query (with its string values intact), not `bare`.
    return sql


def run_query(con: duckdb.DuckDBPyConnection, sql: str):
    """Validate, cap the row count, execute, return a DataFrame."""
    sql = validate_sql(sql)  # raises ValueError if the query is not safe
    # Wrap the query in an outer SELECT ... LIMIT so no result can exceed MAX_ROWS,
    # even if the model forgot a LIMIT. A fresh cursor() per query keeps visitors'
    # queries from interfering with each other on the one shared connection.
    return con.cursor().execute(f"SELECT * FROM ({sql}) LIMIT {MAX_ROWS}").df()


if __name__ == "__main__":  # python db.py  -> pre-build the database locally
    # Runs only when you execute this file directly (not when app.py imports it).
    build_db()
    print("Built", DB_PATH)

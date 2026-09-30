"""DuckDB ingestion, connection and SQL safety checks."""
import ast
import re
from pathlib import Path

import duckdb
import pyarrow as pa

DB_PATH = Path("jobs.duckdb")
HF_DATASET = "lukebarousse/data_jobs"
MAX_ROWS = 500


def _parse_skills(value):
    """'['python', 'sql', 'sql']' (string) -> ['python', 'sql'] (deduped list)."""
    if not isinstance(value, str):
        return None
    try:
        return list(dict.fromkeys(ast.literal_eval(value)))
    except (ValueError, SyntaxError):
        return None


def build_db(path: Path = DB_PATH) -> None:
    """Download the dataset from Hugging Face and write it to a DuckDB file."""
    import pandas as pd
    from datasets import load_dataset

    df = load_dataset(HF_DATASET, split="train").to_pandas()
    df = df.drop(columns=["job_type_skills"])  # redundant nested string
    df["job_posted_date"] = pd.to_datetime(df["job_posted_date"])
    df.insert(0, "job_id", range(1, len(df) + 1))

    # Turn the skills string into a real list column so SQL can UNNEST it.
    skills = pa.array(
        [_parse_skills(s) for s in df["job_skills"]], type=pa.list_(pa.string())
    )
    table = pa.Table.from_pandas(
        df.drop(columns=["job_skills"]), preserve_index=False
    ).append_column("job_skills", skills)

    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    con.register("t", table)
    con.execute("CREATE TABLE jobs AS SELECT * FROM t")
    con.close()


def fetch_prebuilt(repo_id: str) -> bool:
    """Download jobs.duckdb from a (public) Hugging Face dataset repo. True on success."""
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=repo_id, filename=DB_PATH.name,
                        repo_type="dataset", local_dir=".")
        return DB_PATH.exists()
    except Exception:
        return False


def get_conn(hf_db_repo: str | None = None) -> duckdb.DuckDBPyConnection:
    """Read-only connection. If the file is missing: try your HF repo (fast),
    then fall back to building from the original dataset (slow)."""
    if not DB_PATH.exists():
        if not (hf_db_repo and fetch_prebuilt(hf_db_repo)):
            build_db()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET enable_external_access=false")  # no file/network reads from SQL
    return con


# ---------- safety ----------
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|"
    r"install|load|pragma|call|set|truncate|read_csv|read_parquet|read_json|glob)\b",
    re.I,
)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    bare = _STRING_LITERAL.sub("''", sql)  # ignore words inside quoted strings
    if ";" in bare:
        raise ValueError("Only a single SQL statement is allowed.")
    if not re.match(r"^(select|with)\b", bare, re.I):
        raise ValueError("Only SELECT queries are allowed.")
    if _FORBIDDEN.search(bare):
        raise ValueError("Query contains a forbidden keyword.")
    return sql


def run_query(con: duckdb.DuckDBPyConnection, sql: str):
    """Validate, cap the row count, execute, return a DataFrame."""
    sql = validate_sql(sql)
    return con.cursor().execute(f"SELECT * FROM ({sql}) LIMIT {MAX_ROWS}").df()


if __name__ == "__main__":  # python db.py  -> pre-build the database locally
    build_db()
    print("Built", DB_PATH)

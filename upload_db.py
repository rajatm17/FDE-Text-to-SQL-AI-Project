"""Run once, locally, after `python db.py`:

    export HF_TOKEN=hf_xxx          # a token with WRITE access (huggingface.co/settings/tokens)
    python upload_db.py your-username/data-jobs-duckdb
"""
import sys

from huggingface_hub import HfApi

from db import DB_PATH

if len(sys.argv) != 2:
    sys.exit("Usage: python upload_db.py <username>/<repo-name>")

repo_id = sys.argv[1]
api = HfApi()  # reads HF_TOKEN from the environment
api.create_repo(repo_id, repo_type="dataset", exist_ok=True)  # public by default
api.upload_file(path_or_fileobj=str(DB_PATH), path_in_repo=DB_PATH.name,
                repo_id=repo_id, repo_type="dataset")
print(f"Uploaded. Add this to Streamlit secrets:  HF_DB_REPO = \"{repo_id}\"")

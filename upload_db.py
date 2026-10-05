"""Run once, locally, after `python db.py`:

    export HF_TOKEN=hf_xxx          # a token with WRITE access (huggingface.co/settings/tokens)
    python upload_db.py your-username/data-jobs-duckdb
"""
# Purpose: copy the finished jobs.duckdb file to your own Hugging Face dataset repo.
# The deployed app then downloads this one ready-made file when it wakes up, instead
# of rebuilding the whole database from the original dataset (which takes minutes).
# This script is a one-time helper. The Streamlit app never runs it.
import sys  # gives access to the command-line arguments

from huggingface_hub import HfApi  # client for the Hugging Face Hub

from db import DB_PATH  # reuse the database file path defined in db.py (Path("jobs.duckdb"))

# The script needs exactly one argument: the repo name. sys.argv[0] is the script's own
# name, so a correct call has a length of 2. Otherwise print the usage hint and quit.
if len(sys.argv) != 2:
    sys.exit("Usage: python upload_db.py <username>/<repo-name>")

repo_id = sys.argv[1]  # e.g. "your-username/data-jobs-duckdb"
api = HfApi()  # reads HF_TOKEN from the environment
# Create the repo on Hugging Face. repo_type="dataset" makes it a dataset repo.
# exist_ok=True means re-running the script reuses the repo instead of failing.
api.create_repo(repo_id, repo_type="dataset", exist_ok=True)  # public by default
# Upload the local database file into the repo, keeping the same file name
# (jobs.duckdb) so db.py's download step finds it. Fails if the file does not exist
# yet, so run `python db.py` first. Uploading again replaces the old copy.
api.upload_file(path_or_fileobj=str(DB_PATH), path_in_repo=DB_PATH.name,
                repo_id=repo_id, repo_type="dataset")
# Remind you of the last setup step: tell the app which repo to download from.
print(f"Uploaded. Add this to Streamlit secrets:  HF_DB_REPO = \"{repo_id}\"")

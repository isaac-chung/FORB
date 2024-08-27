from __future__ import annotations

import glob

from huggingface_hub import create_repo, upload_file, upload_folder

repo_name = "isaacchung/forb_retrieval"
tok = ""

# create_repo(repo_name, repo_type="dataset", token=tok)

# upload_folder(repo_id=repo_name, folder_path="hf_files", repo_type="dataset", token=tok)
f = "hf_files/query-00000-of-00001.parquet"
upload_file(repo_id=repo_name, path_or_fileobj=f, path_in_repo=f.split("/")[-1], repo_type="dataset", token=tok)

# for f in glob.glob("hf_files/*.md"):
#     upload_file(
#         path_or_fileobj=f,
#         path_in_repo="README.md",
#         repo_id=repo_name,
#         repo_type="dataset",
#         token=tok,
#     )

# for f in glob.glob("hf_files/*.parquet"):
#     upload_file(
#         path_or_fileobj=f,
#         path_in_repo=f.split("/")[-1],
#         repo_id=repo_name,
#         repo_type="dataset",
#         token=tok,
#     )

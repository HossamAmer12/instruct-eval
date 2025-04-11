from huggingface_hub import HfApi
import re

# Function to extract numeric token value (e.g., 1007 from "token-1007B")
def extract_token_number(revision):
    match = re.search(r'token-(\d+)B', revision)
    return int(match.group(1)) if match else -1  # fallback to -1 if pattern not found


api = HfApi()
repo_id = "TinyLlama/tinyLlama-intermediate-checkpoints"

# List all revisions (branches and tags)
refs = api.list_repo_refs(repo_id=repo_id)

# Filter out branches or tags that follow the pattern you want
revisions = [r.name for r in refs.branches if r.name.startswith("step-")]
# print(revisions)
# print(len(revisions))

# Sort revisions by extracted token number
sorted_revisions = sorted(revisions, key=extract_token_number)

# print(sorted_revisions)
with open("test.txt", "w") as f:
    for l in sorted_revisions:
        f.write(l + "\n")
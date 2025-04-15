from huggingface_hub import HfApi
import re
import math
import transformers
import torch
import time
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
from safetensors.torch import save_file


# Function to extract numeric token value (e.g., 1007 from "token-1007B")
def extract_token_number(revision):
    match = re.search(r'token-(\d+)B', revision)
    return int(match.group(1)) if match else -1  # fallback to -1 if pattern not found


api = HfApi()
<<<<<<< Updated upstream
repo_id = "EleutherAI/pythia-1b-deduped"
repo_id = "EleutherAI/pythia-12b-deduped"
repo_id = "EleutherAI/pythia-6.9b-deduped"
=======
# repo_id = "EleutherAI/pythia-1b-deduped"
# repo_id = "EleutherAI/pythia-12b-deduped"
repo_id = "EleutherAI/pythia-2.8b-deduped"
>>>>>>> Stashed changes

# List all revisions (branches and tags)
refs = api.list_repo_refs(repo_id=repo_id)

# Filter out branches or tags that follow the pattern you want
revisions = [r.name for r in refs.branches if r.name.startswith("step")]
# print(revisions)
print(len(revisions))

# Sort revisions by extracted token number
sorted_revisions = sorted(revisions, key=extract_token_number)

# checkpoints_5day = [
#     "step-5k-token-10B",
#     "step-65k-token-136B",
#     "step-125k-token-262B",
#     "step-185k-token-388B",
#     "step-240k-token-503B",
#     "step-300k-token-629B",
#     "step-360k-token-755B",
#     "step-420k-token-881B",
#     "step-480k-token-1007B",
#     "step-540k-token-1132B",
#     "step-600k-token-1258B",
#     "step-660k-token-1384B"
# ]

# checkpoints_5day = [
#     "step1000" , "step20000", "step40000", "step80000", "step100000", "step120000", "step143000",
#     "main"
# ]
checkpoints_5day = [
    "step80000", "step143000"
    ]

checkpoints_5day = [
    "step80000", "step143000"
    ]

# Long run
# We take every 2nd or 3rd step in order, to reach ~60 samples:
# Indices: 0, 2, 4, ..., 118 (first 60 indices at interval ~2–3)

ts = time.time()
lt = time.localtime()

print(f"Loading Model {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}")

device = "cpu"

print("Loading")
print(len(checkpoints_5day))
for model_id in tqdm(checkpoints_5day):
    
    # path = f"/work/hossamamer/tinyllama/{model_id}"
    # path = f"/data00/dataset/pythia/{repo_id}/{model_id}"
    path = f"/work/hossamamer/pythia/{model_id}"
    tokenizer = AutoTokenizer.from_pretrained(repo_id,
           revision=model_id,
           cache_dir=path,)

    try:
        model_original = AutoModelForCausalLM.from_pretrained(repo_id,
            revision=model_id,
            cache_dir=path,
            )
    except:
        
        base_path = os.path.join(path, repo_id, model_id, "snapshots")
        snapshot_dirs = os.listdir(base_path)
        full_model_path = os.path.join(base_path, snapshot_dirs[0])  # usually just one

        
    lt = time.localtime()
    print(f"Model loaded in {math.floor((time.time() - ts)/60):02d}:{round((time.time() - ts)%60):02d} seconds at {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}")


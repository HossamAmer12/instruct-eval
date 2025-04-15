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
repo_id = "TinyLlama/tinyLlama-intermediate-checkpoints"

# List all revisions (branches and tags)
refs = api.list_repo_refs(repo_id=repo_id)

# Filter out branches or tags that follow the pattern you want
revisions = [r.name for r in refs.branches if r.name.startswith("step-")]
# print(revisions)
print(len(revisions))

# Sort revisions by extracted token number
sorted_revisions = sorted(revisions, key=extract_token_number)

checkpoints_5day = [
    "step-5k-token-10B",
    "step-65k-token-136B",
    "step-125k-token-262B",
    "step-185k-token-388B",
    "step-240k-token-503B",
    "step-300k-token-629B",
    "step-360k-token-755B",
    "step-420k-token-881B",
    "step-480k-token-1007B",
    "step-540k-token-1132B",
    "step-600k-token-1258B",
    "step-660k-token-1384B"
]


ts = time.time()
lt = time.localtime()

print(f"Loading Model {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}")

device = "cpu"

print(sorted_revisions)
exit(0)

print("Loading")
print(len(checkpoints_5day))
# for model_id in tqdm(checkpoints_5day):
for model_id in tqdm(sorted_revisions):
    
    if model_id in checkpoints_5day:
        continue
    print(model_id)
    # path = f"/work/hossamamer/tinyllama/{model_id}"
    path = f"/work/hossamamer/tinyllama/more_checkpoints/{model_id}"
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/tinyLlama-intermediate-checkpoints",
           revision=model_id,
           cache_dir=path,)

    try:
        model_original = AutoModelForCausalLM.from_pretrained("TinyLlama/tinyLlama-intermediate-checkpoints",
            revision=model_id,
            cache_dir=path,
            )
    except:
        
        base_path = os.path.join(path, "models--TinyLlama--tinyLlama-intermediate-checkpoints", "snapshots")
        snapshot_dirs = os.listdir(base_path)
        full_model_path = os.path.join(base_path, snapshot_dirs[0])  # usually just one

        # Load the pytorch weights
        pytorch = os.path.join(full_model_path, "pytorch_model.bin")

        # Load the weights from the step
        step_weights = torch.load(pytorch)

        # Load the working model
        model = AutoModelForCausalLM.from_pretrained("PY007/TinyLlama-1.1B-intermediate-step-480K-1T")

        # Load the weights into the model
        model.load_state_dict(step_weights)  # assuming x is a state_dict

        # Save the file
        save_file(model.state_dict(), os.path.join(full_model_path, "model.safetensors"))
    lt = time.localtime()
    print(f"Model loaded in {math.floor((time.time() - ts)/60):02d}:{round((time.time() - ts)%60):02d} seconds at {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}")


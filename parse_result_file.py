import sys
import os
import json
import re

def extract_fname_info(fname):
    # Extract step and training size (e.g., "480k", "1T")
    match = re.search(r"step-(\w+)-(\w+)_snapshots", fname)
    step = -1
    training_size= -1
    if match:
        step = match.group(1)
        training_size = match.group(2)
        # print(f"Step: {step}, Training size: {training_size}")
    else:
        print("Pattern not found.")
    return step, training_size

def extract_info_helper(filename):
    # Try pattern with "token-" first
    match = re.search(
        r"models--[^-]+--(?P<model>.+?)-step-(?P<step>[\w.]+)-token-(?P<tokens>[\w.]+)_snapshots", 
        filename
    )
    if not match:
        # Try hyphen-separated version
        match = re.search(
            r"models--[^-]+--(?P<model>.+?)-step-(?P<step>[\w.]+)-(?P<tokens>[\w.]+)_snapshots", 
            filename
        )
    
    if match:
        model_name = match.group("model")
        step = match.group("step")
        tokens = match.group("tokens")
        return model_name, step, tokens
    else:
        return "Tinyllama", "NA", "NA"

def extract_info(filename):
    # Case A: step-XXX-token-XXX
    match = re.search(
        r"step-(?P<step>[\w.]+)-token-(?P<tokens>[\w.]+)_models--[^-]+--(?P<model>.+?)_snapshots",
        filename
    )
    if match:
        return match.group("model"), match.group("step"), match.group("tokens")

    # Case B: step-XXX-XXX (tokens as second value)
    match = re.search(
        r"step-(?P<step>[\w.]+)-(?P<tokens>[\w.]+)_models--[^-]+--(?P<model>.+?)_snapshots",
        filename
    )
    if match:
        return match.group("model"), match.group("step"), match.group("tokens")

    # Case C: token prefix like _30B_models--...
    match = re.search(
        r"_(?P<tokens>\d+[kMGTB])_models--[^-]+--(?P<model>.+?)_snapshots",
        filename
    )
    if match:
        tokens = match.group("tokens")
        step = tokens.rstrip("kMGTB")  # Use numeric part as step placeholder
        return match.group("model"), step, tokens

    print(filename, "ERROR")
    match = re.search(
        r"models--TinyLlama--TinyLlama-1\.1B-intermediate-step-(?P<step>[\w.]+)-(?P<tokens>[\w.]+)_snapshots", 
        filename
    )
    if match:
        if "Tinyllama" in filename:
            return "TinyLlama", match.group("step"), match.group("tokens")
    # pythia case
    else:
        match = re.search(
        r"_step(?P<step>\d+)_models--[^-]+--(?P<model>pythia-\d+[a-zA-Z]*)[^_]*_snapshots", 
        filename
        )
        if match:
            step = int(match.group("step"))
            tokens_num = step * 2_097_152_000
            # Convert to readable token string (e.g., "167.8T")
            if tokens_num >= 1e12:
                tokens_str = f"{tokens_num / 1e12:.1f}T"
            elif tokens_num >= 1e9:
                tokens_str = f"{tokens_num / 1e9:.1f}B"
            else:
                tokens_str = str(tokens_num)
            return match.group("model"), str(step), tokens_str

    return extract_info_helper(filename=filename)
    # return "unknown_model", "NA", "NA"
    # return "Tinyllama", "NA", "NA"

# def parse_file(fname):
#     with open(fname) as f:
#         lines = f.readlines()
#         line = lines[-1]
#         data = json.loads(line)
#         data = data['pass_at_k']

#     result = []
#     for x in [1,2,4,8,16,32,64]:
#         y = data[f"pass@{x}"]*100.0
#         result.append(y)
#     return result

# fname = "humaneval__data00_maryam_models--TinyLlama--TinyLlama-1.1B-intermediate-step-480k-1T_snapshots_0e23ce8110cbb7e9afe5296bc9686e6dfdae47c2_predictions_pass@4.jsonl_results.jsonl"

# pass_results = parse_file(fname=fname)

# print(pass_results)
# step, training_tokens = extract_fname_info(fname=fname)

# print(step, training_tokens)


# def extract_fname_info(fname):
#     # Extract model name, step, and training size
#     model_match = re.search(r"models--(.*?)--.*?step-(\w+)-(\w+)_snapshots", fname)
#     model_name = "unknown_model"
#     step = "unknown_step"
#     training_size = "unknown_training_size"
#     if model_match:
#         model_name = model_match.group(1)
#         step = model_match.group(2)
#         training_size = model_match.group(3)
#     else:
#         print(f"Pattern not found in filename: {fname}")
#     return model_name, step, training_size

def parse_file(fname):
    try:
        with open(fname) as f:
            lines = f.readlines()
            line = lines[-1]
            data = json.loads(line)
            data = data['pass_at_k']

        result = []
        for x in [1, 2, 4, 8, 16, 32, 64]:
            y = data.get(f"pass@{x}", None)
            result.append(y*100.0)
        return result
    except Exception as e:
        print(f"Error reading file {fname}: {e}")
        return ["NA"] * 4

def tokens_to_number(token_str):
    match = re.match(r"([\d.]+)([kMGT])", token_str)
    if not match:
        return 0
    num = float(match.group(1))
    unit = match.group(2)
    multiplier = {"k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    return num * multiplier[unit]

# Directory containing files
directory = "human_eval_results"

print("Model\tStep\tTokens\tpass@1\tpass@2\tpass@4\tpass@8\tpass@16\tpass@32\tpass@64")
results = []

# Loop through files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jsonl") or filename.endswith(".jsonl_results.jsonl"):
        full_path = os.path.join(directory, filename)
        model_name, step, training_tokens = extract_info(filename)
        pass_results = parse_file(full_path)
        
        # print(f"Model: {model_name}, Step: {step}, Training Tokens: {training_tokens}, Pass@1/2/4: {pass_results}")

        token_value = tokens_to_number(training_tokens)
        results.append((token_value, model_name, step, training_tokens, pass_results))
        # print(f"{model_name}\t{step}\t{training_tokens}\t{pass_results}")
        # print(f"{model_name}\t{step}\t{training_tokens}\t" + "\t".join(str(p) for p in pass_results))


# Sort by token value
results.sort()

# Print header
print("Model\tStep\tTokens\tpass@1\tpass@2\tpass@4\tpass@8\tpass@16\tpass@32\tpass@64")
# Print sorted results
for _, model_name, step, training_tokens, pass_results in results:
    print(f"{model_name}\t{step}\t{training_tokens}\t" + "\t".join(str(p) for p in pass_results))
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

def parse_file(fname):
    with open(fname) as f:
        lines = f.readlines()
        line = lines[-1]
        data = json.loads(line)
        data = data['pass_at_k']

    result = []
    for x in [1,2,4]:
        y = data[f"pass@{x}"]
        result.append(y)
    return result

fname = "humaneval__data00_maryam_models--TinyLlama--TinyLlama-1.1B-intermediate-step-480k-1T_snapshots_0e23ce8110cbb7e9afe5296bc9686e6dfdae47c2_predictions_pass@4.jsonl_results.jsonl"

pass_results = parse_file(fname=fname)

print(pass_results)
step, training_tokens = extract_fname_info(fname=fname)

print(step, training_tokens)


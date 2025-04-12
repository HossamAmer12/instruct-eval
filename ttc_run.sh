
# https://huggingface.co/TinyLlama/tinyLlama-intermediate-checkpoints/tree/step-480k-token-1007B
# 6 checkpoints for tinyllama 
# Training tokens:
# 105B, 2.5T, 503B, 1T, 1.5T, 2T, 
MODEL_PATHS=(
    "/data00/maryam/models--TinyLlama--TinyLlama-1.1B-step-50K-105b/snapshots/ae666e2718bf72a193f12e082beb58acc284231d"
     "/data00/maryam/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1195k-token-2.5T/snapshots/706bc2851338c4a89bb212e96ff23f9cc1ebde1d"
    "/data00/maryam/models--TinyLlama--TinyLlama-1.1B-intermediate-step-240k-503b/snapshots/a016b960b941eb6eb7884e363b935370bafa5932"
    "/data00/maryam/models--TinyLlama--TinyLlama-1.1B-intermediate-step-480k-1T/snapshots/0e23ce8110cbb7e9afe5296bc9686e6dfdae47c2"
    "/data00/maryam/models--TinyLlama--TinyLlama-1.1B-intermediate-step-715k-1.5T/snapshots/19a81efa07bf28ec81dc4de327776fa00e34cf3f"
    "/data00/maryam/models--TinyLlama--TinyLlama-1.1B-intermediate-step-955k-token-2T/snapshots/195255c4e3e1a56ac89ceb95899c58f742a730dc"
   
)


# 45 mins for pass@1.. 
# GPU_ID=0
GPU_ID=5
model_path=${MODEL_PATHS[$GPU_ID]}

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py humaneval --model_name llama  --n_sample 64 --model_path $model_path


# CUDA_VISIBLE_DEVICES=7 python main.py humaneval --model_name llama  --n_sample 64 --model_path PY007/TinyLlama-1.1B-intermediate-step-480K-1T^C
# (ieval) hossam@lux-2-node-11:~/TTC_workspace/instruct-eval$ 
# {'pass@1': 0.03239329268292683, 'pass@2': 0.04780657181571816, 'pass@4': 0.06547018289458412, 'pass@8': 0.08565974816698098, 'pass@16': 0.10787609160566579, 'pass@32': 0.1289450432366897, 'pass@64': 0.1524390243902439}
# {'humaneval': 3.24}
# humaneval: 3.24
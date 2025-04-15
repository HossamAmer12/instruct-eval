
# https://huggingface.co/TinyLlama/tinyLlama-intermediate-checkpoints/tree/step-480k-token-1007B
# 6 checkpoints for tinyllama 
# Training tokens:
# 105B, 2.5T, 503B, 1T, 1.5T, 2T, 
# MODEL_PATHS=(
#     "/work/hossamamer/tinyllama/step-5k-token-10B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/feb60e43f33c2b8db5b122dbaef76b9dffd557c8"
#     "/work/hossamamer/tinyllama/step-65k-token-136B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/b68428d743c49097f905b9e31d0d27098e3cd44f"
#     "/work/hossamamer/tinyllama/step-125k-token-262B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/c4d3599fc7894e4e9e34154e1a84d165fcff016a"
#     "/work/hossamamer/tinyllama/step-185k-token-388B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/e58f49895fdf004fa8d0e5a57d42e17c8867b12c"
#     "/work/hossamamer/tinyllama/step-240k-token-503B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/cab66f52ac47ad5fc57e9510485c69c02cb99d09"
#     "/work/hossamamer/tinyllama/step-300k-token-629B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/5f9d394fc605f8027f88682e563b104c623fe4ed"
#     "/work/hossamamer/tinyllama/step-360k-token-755B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/223a4964750d14dd0265ce9d5a56b4f1c540b15b"
#     "/work/hossamamer/tinyllama/step-420k-token-881B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/87ac1fb103c323bafb34dc42b97a30aa4de225e1"
#     "/work/hossamamer/tinyllama/step-480k-token-1007B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/68c667a2869968a1d527edd85b578e54696434b2"
#     "/work/hossamamer/tinyllama/step-540k-token-1132B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/55ae337d54316b6817247b46c39b3bd99308786f"
#     "/work/hossamamer/tinyllama/step-600k-token-1258B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/f056976086771d075133feda7bad88047c1b7802"
#     "/work/hossamamer/tinyllama/step-660k-token-1384B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/91804715ca0e86a0bf67465a52383fe955462e54"
# )

# priotized
# MODEL_PATHS=(
#         "/data00/dataset/tinyllama/step-540k-token-1132B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/55ae337d54316b6817247b46c39b3bd99308786f"
#         "/data00/dataset/tinyllama/step-600k-token-1258B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/f056976086771d075133feda7bad88047c1b7802"
# )

# Finemath
MODEL_PATHS=(
    # 14,15,16,17
    "/data00/dataset/finemath/finemath-llama3b/40B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/a5327c94c99a795d0a48089253b8f9356ceed281/"
    "/data00/dataset/finemath/finemath-llama3b/10B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/f3be85d2df204cf454cfd06657b7b0c788ceedb1/"

    "/data00/dataset/finemath/finemath-llama3b/20B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/4ccc2949da259213552d6257a89c9448a666437e/"
    "/data00/dataset/finemath/finemath-llama3b/30B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/b60bc20540d30bc69efb0253a9ea1b4a77ac2054/"
    "/data00/dataset/finemath/finemath-llama3b/50B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/49c2b41df57e3e65368f7e2ccdcd50ec3fe88ba8/"
)


# 45 mins for pass@1.. 
# GPU_ID=0
GPU_ID=7
# model_path=${MODEL_PATHS[$GPU_ID]}

# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py humaneval --model_name llama  --n_sample 64 --model_path $model_path


for MODEL_ID in "${!MODEL_PATHS[@]}"; do
# for MODEL_ID in 1; do
    model_path=${MODEL_PATHS[$MODEL_ID]}

    echo "Running on GPU $MODEL_ID with model path: $model_path"

#     CUDA_VISIBLE_DEVICES=4 python main.py humaneval \
#         --model_name llama \
#         --n_sample 64 \
#         --model_path "$model_path"

     CUDA_VISIBLE_DEVICES=4 python main.py humaneval \
        --model_name causal \
        --n_sample 64 \
        --model_path "$model_path"
done



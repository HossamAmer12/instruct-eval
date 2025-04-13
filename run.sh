model_path="/data00/maryam/models--TinyLlama--TinyLlama-1.1B-intermediate-step-480k-1T/snapshots/0e23ce8110cbb7e9afe5296bc9686e6dfdae47c2"

model_path="/work/hossamamer/tinyllama/step-5k-token-10B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/feb60e43f33c2b8db5b122dbaef76b9dffd557c8/"

# model_path="/dataset/pythia_models/pythia-1b-deduped/step143000/models--EleutherAI--pythia-1b-deduped/snapshots/9f638c32a09e234bce2a2da4d37eb08211b816cb"

model_path="/dataset/pythia_models/pythia-1b-deduped/step80000/models--EleutherAI--pythia-1b-deduped/snapshots/2f1c9d5f26ade712e362b2e048156a64bc5b6d27"

# CUDA_VISIBLE_DEVICES=0 python main.py humaneval --model_name llama  --n_sample 1 --model_path $model_path

# 45 mins for one run for human eval
# 6.1%
# CUDA_VISIBLE_DEVICES=0 python main.py humaneval --model_name llama  --n_sample 1 --model_path $model_path

CUDA_VISIBLE_DEVICES=5 python main.py humaneval --model_name causal  --n_sample 1 --model_path $model_path

# 47 mins
# drop: 13.03%
# 13.03%
# CUDA_VISIBLE_DEVICES=1 python main.py drop --model_name llama  --n_sample 4 --model_path $model_path

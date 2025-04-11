model_path="/data00/maryam/models--TinyLlama--TinyLlama-1.1B-intermediate-step-480k-1T/snapshots/0e23ce8110cbb7e9afe5296bc9686e6dfdae47c2"

model_path="/work/hossamamer/tinyllama/step-5k-token-10B/models--TinyLlama--tinyLlama-intermediate-checkpoints/snapshots/feb60e43f33c2b8db5b122dbaef76b9dffd557c8/"

# CUDA_VISIBLE_DEVICES=0 python main.py humaneval --model_name llama  --n_sample 1 --model_path $model_path

# 45 mins for one run for human eval
# 6.1%
# CUDA_VISIBLE_DEVICES=0 python main.py humaneval --model_name llama  --n_sample 1 --model_path $model_path

CUDA_VISIBLE_DEVICES=7 python main.py humaneval --model_name llama  --n_sample 1 --model_path $model_path

# 47 mins
# drop: 13.03%
# 13.03%
# CUDA_VISIBLE_DEVICES=1 python main.py drop --model_name llama  --n_sample 4 --model_path $model_path

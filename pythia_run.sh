

model_path="/dataset/pythia-70m-deduped/step143000/models--EleutherAI--pythia-70m-deduped/snapshots/4ad6c938b037fd4762343dcc441ba1012a7401c8/"


CUDA_VISIBLE_DEVICES=0 python main.py humaneval --model_name causal  --n_sample 64 --model_path $model_path


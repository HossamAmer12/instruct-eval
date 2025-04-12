

model_path="/dataset/pythia-70m-deduped/step143000/models--EleutherAI--pythia-70m-deduped/snapshots/4ad6c938b037fd4762343dcc441ba1012a7401c8/"

model_path="/dataset/pythia-70m-deduped/step80000/models--EleutherAI--pythia-70m-deduped/snapshots/03866fcabb62cd47b2d281879cb3b56dc2ad9fb4/"

# model_path="/dataset/pythia-410m-deduped/step143000/models--EleutherAI--pythia-410m-deduped/snapshots/c0b6bef7dd1ec11d3baa07ee955de98a414dd464/"

model_path="/dataset/pythia-410m-deduped/step80000/models--EleutherAI--pythia-410m-deduped/snapshots/b77396893ccac5ec277aea65a323f0205c865ad4/"


CUDA_VISIBLE_DEVICES=3 python main.py humaneval --model_name causal  --n_sample 64 --model_path $model_path


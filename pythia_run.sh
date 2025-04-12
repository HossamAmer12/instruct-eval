

model_path="/dataset/pythia-70m-deduped/step143000/models--EleutherAI--pythia-70m-deduped/snapshots/4ad6c938b037fd4762343dcc441ba1012a7401c8/"

model_path="/dataset/pythia-70m-deduped/step80000/models--EleutherAI--pythia-70m-deduped/snapshots/03866fcabb62cd47b2d281879cb3b56dc2ad9fb4/"

# model_path="/dataset/pythia-410m-deduped/step143000/models--EleutherAI--pythia-410m-deduped/snapshots/c0b6bef7dd1ec11d3baa07ee955de98a414dd464/"

model_path="/dataset/pythia-410m-deduped/step80000/models--EleutherAI--pythia-410m-deduped/snapshots/b77396893ccac5ec277aea65a323f0205c865ad4/"

MODEL_PATHS=(
    "/dataset/pythia-70m-deduped/step143000/models--EleutherAI--pythia-70m-deduped/snapshots/4ad6c938b037fd4762343dcc441ba1012a7401c8/"
    "/dataset/pythia-70m-deduped/step80000/models--EleutherAI--pythia-70m-deduped/snapshots/03866fcabb62cd47b2d281879cb3b56dc2ad9fb4/"
   "/dataset/pythia-410m-deduped/step143000/models--EleutherAI--pythia-410m-deduped/snapshots/c0b6bef7dd1ec11d3baa07ee955de98a414dd464/"
   "/dataset/pythia-410m-deduped/step80000/models--EleutherAI--pythia-410m-deduped/snapshots/b77396893ccac5ec277aea65a323f0205c865ad4/"
)



# for MODEL_ID in "${!MODEL_PATHS[@]}"; do
for MODEL_ID in 0 1 2 3; do
    model_path=${MODEL_PATHS[$MODEL_ID]}
    echo "Running on GPU $MODEL_ID with model path: $model_path"

    CUDA_VISIBLE_DEVICES=4 python main.py humaneval --model_name causal  --n_sample 1 --model_path $model_path
done

# sampled_ckpt_list = [1000, 20000, 40000, 80000, 100000, 120000, 143000]

CUDA_VISIBLE_DEVICES=0 python probe.py \
  --per_label 500 \
  --batch_size 64 \
  --max_length 1024 \
  --positions first mean last \
  --few_shot_k 8 \
  --save_csv probe_results_16shots.csv
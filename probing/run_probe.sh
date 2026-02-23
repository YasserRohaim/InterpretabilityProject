CUDA_VISIBLE_DEVICES=0 python probe.py \
  --per_label 500 \
  --batch_size 64 \
  --max_length 128 \
  --positions first mean last
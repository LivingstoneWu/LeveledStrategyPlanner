python ../src/training/PPO_training.py \
    --task picking \
    --model_levels 3 \
    --model_start_hidden_size 512 \
    --sub-batch_size 25 \
    --num_epochs 4 \
    --total_fames 1e6 \
    --frames_per_batch 500 
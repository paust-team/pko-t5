#!/bin/bash

set -e

torchrun --nproc_per_node=4 --master_port=34321 -m pkot5.flan.train \
    --pretrained_model_name_or_path paust/pko-t5-large \
    --data_server_addr "$DATA_SERVER_ADDR" \
    --bf16 True \
    --output_dir ./pko-flan-t5-large \
    --overwrite_output_dir True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --optim='adafactor' \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True

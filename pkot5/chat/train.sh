#!/bin/bash

set -e

# 실험 #1
#torchrun --nproc_per_node=4 -m pkot5.chat.train \
#    --pretrained_model_name_or_path paust/pko-t5-large \
#    --data_path ./pkot5/chat/ko_alpaca_data.json \
#    --bf16 True \
#    --output_dir ./pko-t5-large-chat \
#    --overwrite_output_dir True \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 4 \
#    --evaluation_strategy "steps" \
#    --eval_steps 20 \
#    --save_strategy "no" \
#    --save_total_limit 1 \
#    --learning_rate 1e-4 \
#    --optim='adafactor' \
#    --warmup_ratio 0.0 \
#    --lr_scheduler_type "linear" \
#    --logging_steps 1 \
#    --tf32 True

# 실험 #2
#torchrun --nproc_per_node=4 -m pkot5.chat.train \
#    --pretrained_model_name_or_path paust/pko-t5-large \
#    --data_path ./pkot5/chat/ko_alpaca_data.json \
#    --bf16 True \
#    --output_dir ./pko-t5-large-chat \
#    --overwrite_output_dir True \
#    --num_train_epochs 3 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 4 \
#    --evaluation_strategy "steps" \
#    --eval_steps 20 \
#    --save_strategy "no" \
#    --save_total_limit 1 \
#    --learning_rate 1e-4 \
#    --optim='adafactor' \
#    --warmup_ratio 0.0 \
#    --lr_scheduler_type "linear" \
#    --logging_steps 1 \
#    --tf32 True

torchrun --nproc_per_node=4 -m pkot5.chat.train \
    --pretrained_model_name_or_path paust/pko-flan-t5-large \
    --data_path ./chat_t5_data.pkl \
    --bf16 True \
    --output_dir ./pko-t5-large-chat \
    --overwrite_output_dir True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --optim='adafactor' \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True
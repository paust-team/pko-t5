#!/bin/bash

nohup python3 -m pkot5.pretraining.tpu train \
  --data_grpc_endpoint="15.165.44.101:51051" \
  --model_name="paust/pko-t5-xl" \
  --seed=42 \
  --save_step=1000 \
  --max_steps=1000000 \
  --batch_size=64 \
  --max_gradient_accumulation_steps=1 \
  --load_checkpoint=0 \
  1> stdout.log 2> stderr.log
#!/bin/bash

set -e

MODEL_NAME_OR_PATH="./models/pko-t5/large"

python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.multitask train --model=$MODEL_NAME_OR_PATH
python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.multitask test --model_name_or_path=./models/klue_t5/checkpoint-2552

MODEL_NAME_OR_PATH="./models/t5-kr-base-bbpe"

python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.multitask train --model=$MODEL_NAME_OR_PATH
python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.multitask test --model_name_or_path=./models/klue_t5/checkpoint-2552

MODEL_NAME_OR_PATH="./models/t5-kr-small-bbpe"

python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.multitask train --model=$MODEL_NAME_OR_PATH
python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.multitask test --model_name_or_path=./models/klue_t5/checkpoint-2552
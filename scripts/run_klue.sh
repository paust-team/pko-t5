#!/bin/bash

set -e

MODEL_NAME_OR_PATH="./models/pko-t5/large"

#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=ynat --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=nli --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=sts --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=ner --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=dp --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=re --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=mrc --model=$MODEL_NAME_OR_PATH

MODEL_NAME_OR_PATH="./models/pko-t5/base"

#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=ynat --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=nli --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=sts --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=ner --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=dp --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=re --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=mrc --model=$MODEL_NAME_OR_PATH

MODEL_NAME_OR_PATH="./models/pko-t5/small"

python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=ynat --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=nli --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=sts --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=ner --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=dp --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=re --model=$MODEL_NAME_OR_PATH
#python3 -m torch.distributed.launch --logdir ./logs --use_env --nproc_per_node 8 -m pkot5.klue.finetuning --task=mrc --model=$MODEL_NAME_OR_PATH
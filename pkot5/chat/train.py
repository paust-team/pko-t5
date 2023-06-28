import dataclasses
import functools
import json
import logging
import os
import pickle
import random
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5TokenizerFast, T5ForConditionalGeneration, Trainer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq, PreTrainedTokenizer
from rouge_score import rouge_scorer
from accelerate import Accelerator


logger = logging.getLogger(__name__)


class ChatT5Dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, input_max_len=-1, target_max_len=-1):
        super().__init__()
        with open(data_path, "rb") as f:
            self.records = pickle.load(f)

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i) -> Dict[str, List[Union[int, bool]]]:
        record = self.records[i]
        input_ids = record['input_ids']
        target_ids = record['target_ids']

        if self.input_max_len > 0:
            input_ids = input_ids[:self.input_max_len]

        if self.target_max_len > 0:
            target_ids = target_ids[:self.target_max_len]

        attention_mask = [True] * len(input_ids)
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)


@dataclasses.dataclass
class ModelArguments:
    pretrained_model_name_or_path: str
    data_path: str


def train(model_args: ModelArguments, training_args: TrainingArguments):
    hf_auth_token = os.getenv("HF_AUTH_TOKEN", None)

    tokenizer = T5TokenizerFast.from_pretrained(model_args.pretrained_model_name_or_path, use_auth_token=hf_auth_token)
    dataset = ChatT5Dataset(model_args.data_path, target_max_len=1024)

    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16, device_map='cuda', use_auth_token=hf_auth_token)
    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8)
    )

    trainer.train()


def main():
    model_args: ModelArguments
    training_args: TrainingArguments
    model_args, training_args = HfArgumentParser((ModelArguments, TrainingArguments)).parse_args_into_dataclasses()
    train(model_args, training_args)


if __name__ == '__main__':
    main()

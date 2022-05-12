import logging
import os
import pickle

import fire
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
)
import torch.distributed as dist
import redis

from .args import get_config
from .processors import KLUE_PROCESSORS, Text2TextDataset


logger = logging.getLogger(__name__)


def train(model="./models/t5-kr-small-bbpe"):
    model_name_or_path = model
    local_rank = int(os.getenv("LOCAL_RANK", "-1"))
    args = Seq2SeqTrainingArguments(
        output_dir='./models/klue_t5',
        overwrite_output_dir=True,
        local_rank=local_rank,
        **get_config('multitask')
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    all_processors = {task: processor(tokenizer) for task, processor in KLUE_PROCESSORS.items()}

    all_train_entries, all_test_entries = [], []
    for processor in all_processors.values():
        all_train_entries += processor.process('train')
        all_test_entries += processor.process('validation')

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=Text2TextDataset(all_train_entries, max_length=1300),
        eval_dataset=Text2TextDataset(all_test_entries, max_length=1300),
        tokenizer=tokenizer,
    )

    trainer.train()


def test(model_name_or_path):
    r = redis.Redis(
        host='localhost',
        port=6379)

    local_rank = int(os.getenv('LOCAL_RANK', "-1"))
    assert local_rank >= 0
    dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    metrics = {}

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name_or_path).eval().cuda()

    metric_result = {}
    for task, processor_cls in KLUE_PROCESSORS.items():
        args = get_config(task)
        processor = processor_cls(tokenizer)
        all_dev_entries = processor.process('validation')
        sublist_size = len(all_dev_entries) // dist.get_world_size()
        if dist.get_rank() == dist.get_world_size() - 1:
            dev_entries = all_dev_entries[dist.get_rank() * sublist_size:]
        else:
            dev_entries = all_dev_entries[dist.get_rank() * sublist_size:(dist.get_rank() + 1) * sublist_size]
        dev_data = Text2TextDataset(dev_entries, max_length=512 if task != 'mrc' else 1300)
        dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args['per_device_eval_batch_size'], collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True))

        all_logits = []
        for data in tqdm(dev_dataloader, desc=f"Eval '{task}' .."):
            logits = model.generate(
                input_ids=data['input_ids'].cuda(),
                attention_mask=data['attention_mask'].cuda(),
                max_length=args['generation_max_length'],
                early_stopping=True,
            )
            assert logits.shape[0] == data['input_ids'].shape[0]
            all_logits += logits.tolist()
        r.set(f"{dist.get_rank()}", pickle.dumps(all_logits))
        dist.barrier()

        all_logits = []
        for i in range(dist.get_world_size()):
            logits = pickle.loads(r.get(f"{i}"))
            all_logits += logits
        metric = processor.compute_metrics(all_logits, all_dev_entries)
        if task not in metrics:
            metrics[task] = []
        metrics[task].append(metric)

        for name, value in metric.items():
            metric_result[f"eval_{task}_{name}"] = value
    print(metric_result)

    for task in KLUE_PROCESSORS.keys():
        best_metric = None
        for metric in metrics[task]:
            if best_metric is None:
                best_metric = metric
            elif any(metric[k] >= best_metric[k] for k in best_metric.keys()):
                best_metric = metric

        if local_rank == 0:
            with open(f'{model_name_or_path}/result_{task}.txt', 'wt') as f:
                for k, v in best_metric.items():
                    f.write(f"{k}={v}\n")


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
    })

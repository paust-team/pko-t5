import math
import os
import sys
from statistics import mean
import shelve

import fire
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm.std import tqdm
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Trainer, TrainerCallback, AutoTokenizer, Seq2SeqTrainer, Adafactor, \
    get_linear_schedule_with_warmup, TrainingArguments, T5TokenizerFast
from transformers.data.metrics import squad_metrics
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch_xla.utils import gcsfs

from ..generation_utils import beam_search

INPUT_MAX_LENGTH = 1024
OUTPUT_MAX_LENGTH = 32


def get_squad_metrics(tokenizer, predictions, labels):
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    em, f1 = [], []
    for a_pred, a_gold in zip(predictions, labels):
        em.append(float(squad_metrics.compute_exact(a_gold, a_pred)))
        f1.append(float(squad_metrics.compute_f1(a_gold, a_pred)))

    em = 100. * sum(em) / len(em)
    f1 = 100. * sum(f1) / len(f1)

    return em, f1


def prepare_data(tokenizer, target='train'):
    ds = load_dataset("squad_kor_v1")
    dataset = ds[target]

    all_inputs = [f"question: {row['question']} title: {row['title']} context: {row['context']}" for row in dataset]
    inputs = tokenizer(all_inputs, add_special_tokens=True, max_length=INPUT_MAX_LENGTH, truncation=True, padding=True, return_tensors='np')
    input_ids = inputs.input_ids.tolist()
    attention_mask = inputs.attention_mask.tolist()

    all_labels = [row['answers']['text'][0] for row in dataset]
    targets = tokenizer(all_labels, add_special_tokens=True, max_length=OUTPUT_MAX_LENGTH, truncation=True, padding=True, return_tensors='np')
    labels = targets.input_ids
    labels[~targets.attention_mask] = -100
    labels = labels.tolist()
    dataset = [{'input_ids': row[0], 'attention_mask': row[1], 'labels': row[2]} for row in zip(input_ids, attention_mask, labels)]

    return dataset


def train_per_device(proc_id, log_dir, model_name, output_path):
    input_max_length = INPUT_MAX_LENGTH
    output_max_length = OUTPUT_MAX_LENGTH

    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = open(f"{log_dir}/{proc_id}_stdout.txt", "w")
    sys.stderr = open(f"{log_dir}/{proc_id}_stderr.txt", "w")

    tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    train_data = prepare_data(tokenizer, target='train')
    valid_data = prepare_data(tokenizer, target='validation')

    args = Seq2SeqTrainingArguments(
        "pko-t5-korquad",
        overwrite_output_dir=True,
        optim='adafactor',
        learning_rate=7e-4,
        warmup_ratio=0.06,
        num_train_epochs=5,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,

        save_strategy='no',
        evaluation_strategy='epoch',

        generation_num_beams=4,
        generation_max_length=output_max_length,
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = xm.xla_device()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding='max_length', max_length=input_max_length)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.per_device_train_batch_size,
                                  sampler=DistributedSampler(train_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True, seed=args.seed),
                                  collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_data,
                                  batch_size=args.per_device_train_batch_size,
                                  sampler=DistributedSampler(valid_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False, seed=args.seed),
                                  collate_fn=data_collator)
    train_mp_deviceloader = pl.MpDeviceLoader(train_dataloader, device, loader_prefetch_size=1, device_prefetch_size=1)
    valid_mp_deviceloader = pl.MpDeviceLoader(valid_dataloader, device, loader_prefetch_size=1, device_prefetch_size=1)

    model = model.to(device)
    optimizer = Adafactor(model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
    num_train_epochs = int(args.num_train_epochs)
    num_training_steps = int(math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * num_train_epochs)
    num_warmup_steps = args.get_warmup_steps(num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    training_step = 0
    for epoch in range(num_train_epochs):
        if train_dataloader.sampler is not None and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        with tqdm(total=num_training_steps, desc="Train..", initial=training_step) as prog:
            model.train()
            grad_accumulation_step = 0
            optimizer.zero_grad()
            for data in train_mp_deviceloader:
                loss = model(**data)[0]
                loss.backward()
                grad_accumulation_step += 1
                if grad_accumulation_step == args.gradient_accumulation_steps:
                    xm.optimizer_step(optimizer)
                    xm.add_step_closure(lr_scheduler.step)
                    optimizer.zero_grad()
                    grad_accumulation_step = 0

                    training_step = prog.update(1)
            if grad_accumulation_step > 0:
                xm.optimizer_step(optimizer)
                xm.add_step_closure(lr_scheduler.step)
                xm.mark_step()

                training_step = prog.update(1)

        with torch.no_grad():
            losses, all_labels, all_logits = [], [], []
            model.eval()
            def _eval_update(loss_, logits_, labels_):
                losses.append(loss_.item())
                all_logits.extend(logits_.tolist())
                all_labels.extend(labels_.tolist())

            for data in tqdm(valid_mp_deviceloader, desc="Eval.."):
                loss = model(**data)[0]
                logits = beam_search(model, data['input_ids'], data['attention_mask'], num_beams=args.generation_num_beams, max_length=args.generation_max_length)
                labels = data['labels']
                labels = labels.masked_fill(labels < 0, 0)

                xm.add_step_closure(_eval_update, args=(loss, logits, labels))

            labels = all_labels
            predictions = all_logits

        torch.save([losses, labels, predictions], f"cached_{proc_id}.ptc")
        xm.rendezvous("reduce-eval_metrics")
        losses, labels, predictions = [], [], []
        for i in range(xm.xrt_world_size()):
            record = torch.load(f"cached_{i}.ptc")
            losses.extend(record[0])
            labels.extend(record[1])
            predictions.extend(record[2])

        eval_loss = mean(losses)
        eval_em, eval_f1 = get_squad_metrics(tokenizer, predictions, labels)
        print(f"[{epoch+1}/{num_train_epochs}] eval_loss={eval_loss:.4f} eval_em={eval_em:.4f} eval_f1={eval_f1:.4f}")

        with gcsfs.open(f"{output_path}/pytorch_model.bin", mode="wb") as f:
            xm.save(model.state_dict(), f)


def train(output_path="gs://pko_t5/base_finetuned_korquad", log_dir="./logs", model_name="paust/pko-t5-base", nprocs=8):
    xmp.spawn(train_per_device, args=(log_dir, model_name, output_path), nprocs=nprocs)


def test(model_name='paust/pko-t5-base'):
    dataset = load_dataset("squad_kor_v1")
    train_data = dataset['train']
    test_data = dataset['validation']

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    datasets = ([], [])
    for i, data in enumerate([train_data, test_data]):
        all_inputs = [f"question: {row['question']} title: {row['title']} context: {row['context']}" for row in data]
        all_input_ids = tokenizer(all_inputs,
                                  add_special_tokens=True, max_length=INPUT_MAX_LENGTH, truncation=True).input_ids
        all_labels = [row['answers']['text'][0] for row in data]
        all_label_ids = tokenizer(all_labels, add_special_tokens=True).input_ids
        for input_ids, label_ids in zip(all_input_ids, all_label_ids):
            datasets[i].append(dict(
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
                labels=label_ids,
            ))
    train_data, test_data = datasets

    res = dict(
        max_train_input_ids_len = max([len(t['input_ids']) for t in train_data]),
        max_train_labels_len = max([len(t['labels']) for t in train_data]),
        max_test_input_ids_len = max([len(t['input_ids']) for t in test_data]),
        max_test_labels_len = max([len(t['labels']) for t in test_data]),
    )
    print(res)


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
    })

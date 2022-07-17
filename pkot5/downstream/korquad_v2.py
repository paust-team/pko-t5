import os
import os
import pickle
import shutil
from pathlib import Path
import re
from typing import List

import fire
import torch
from datasets import load_dataset
from tqdm.std import tqdm
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Trainer, TrainerCallback, AutoTokenizer
from transformers.data.metrics import squad_metrics


CONTEXT_LEN = 512
WINDOW_LEN = 128


class MyTrainerCallback(TrainerCallback):
    def __init__(self, test_qas_ids: List[str]):
        super().__init__()
        self.test_qas_ids = test_qas_ids

    @torch.no_grad()
    def on_evaluate(self, args, state, control, eval_dataloader, metrics, tokenizer, model: T5ForConditionalGeneration, **kwargs):
        all_logits, all_scores, all_labels = [], [], []
        for data in tqdm(eval_dataloader, desc=f"evaluating.."):
            out = model.generate(
                input_ids=data['input_ids'].cuda(),
                attention_mask=data['attention_mask'].cuda(),
                max_length=32,
                num_beams=8,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True
            )

            all_logits += out.sequences.tolist()
            all_scores += out.sequences_scores.tolist()
            labels = data['labels']
            labels[labels < 0] = 0
            all_labels += labels.tolist()

        a_preds, a_golds = {}, {}
        predictions = tokenizer.batch_decode(all_logits, skip_special_tokens=True)
        labels = tokenizer.batch_decode(all_labels, skip_special_tokens=True)
        for qas_id, score, pred_text in zip(self.test_qas_ids, all_scores, predictions):
            if len(pred_text.strip()) == 0:
                continue
            a_pred = a_preds.get(qas_id, {'score': None, 'text': ''})
            if a_pred['score'] is None or score > a_pred['score']:
                a_pred = {'score': score, 'text': pred_text}
            a_preds[qas_id] = a_pred

        for qas_id, answer_text in zip(self.test_qas_ids, labels):
            if len(answer_text.strip()) == 0:
                continue
            a_gold = a_golds.get(qas_id, None)
            if a_gold is None:
                a_gold = answer_text
            else:
                assert a_gold == answer_text
            a_golds[qas_id] = a_gold

        predictions = [a_preds.get(id_, {'score': None, 'text': ''})['text'] for id_ in self.test_qas_ids]
        labels = [a_golds[id_] for id_ in self.test_qas_ids]

        em, f1 = [], []
        for a_pred, a_gold in zip(predictions, labels):
            em.append(float(squad_metrics.compute_exact(a_gold, a_pred)))
            f1.append(float(squad_metrics.compute_f1(a_gold, a_pred)))

        em = 100. * sum(em) / len(em)
        f1 = 100. * sum(f1) / len(f1)

        metrics['em'] = em
        metrics['f1'] = f1
        print(metrics)


def train(model_name):
    dataset = load_dataset("squad_kor_v1")
    train_data = dataset['train']
    test_data = dataset['validation']

    print(f"ex) {train_data[0]}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    datasets = ([], [])
    qas_ids = ([], [])
    for i, data in enumerate([train_data, test_data]):
        enc = tokenizer([row['context'] for row in data], return_offsets_mapping=True, add_special_tokens=False)
        all_ctx_offsets = enc.offset_mapping
        all_ctx_ids = enc.input_ids
        all_prefix_ids = tokenizer([f"question: {row['question']} title: {row['title']} context: " for row in data], add_special_tokens=False).input_ids
        all_label_ids = tokenizer([row['answers']['text'][0] for row in data], add_special_tokens=True).input_ids
        inputs, targets = [], []
        for row, ctx_ids, ctx_offsets, prefix_ids, label_ids in zip(data, all_ctx_ids, all_ctx_offsets, all_prefix_ids, all_label_ids):
            answer_start = row['answers']['answer_start'][0]
            for begin in range(0, len(ctx_ids), WINDOW_LEN):
                window_ctx_ids = ctx_ids[begin:begin + CONTEXT_LEN]
                window_ctx_offsets = ctx_offsets[begin:begin + CONTEXT_LEN]
                window_ctx_start = window_ctx_offsets[0][0]
                window_ctx_end = window_ctx_offsets[-1][1]
                targets.append(label_ids if window_ctx_start <= answer_start < window_ctx_end else [tokenizer.eos_token_id])
                inputs.append(prefix_ids + window_ctx_ids + [tokenizer.eos_token_id])
                qas_ids[i].append(row['id'])
        for input_ids, label_ids in zip(inputs, targets):
            datasets[i].append(dict(
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
                labels=label_ids,
            ))
    train_data, test_data = datasets

    print(f"train_data length: {len(train_data)}")

    args = Seq2SeqTrainingArguments(
        "pko-t5-korquad",
        overwrite_output_dir=True,
        learning_rate=7e-4,
        optim='adafactor',
        warmup_ratio=0.06,
        num_train_epochs=5,
        local_rank=int(os.getenv("LOCAL_RANK", "-1")),
        seed=42,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,

        save_strategy='no',
        evaluation_strategy='epoch',
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        callbacks=[MyTrainerCallback(qas_ids[1])],
    )

    trainer.train()


if __name__ == '__main__':
    fire.Fire(train)

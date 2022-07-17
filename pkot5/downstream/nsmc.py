import functools
import os
from statistics import mean

import fire
import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import T5TokenizerFast, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainingArguments, Trainer, EvalPrediction, \
    TrainerCallback, AutoTokenizer


class TrainerCallbackForEval(TrainerCallback):
    @torch.no_grad()
    def on_evaluate(self, args, state, control, **kwargs):
        model: T5ForConditionalGeneration = kwargs['model']
        tokenizer = kwargs['tokenizer']
        eval_dataloader = kwargs['eval_dataloader']
        metrics = kwargs['metrics']

        all_logits, all_labels = [], []
        for data in tqdm(eval_dataloader, desc="evaluating.."):
            data = data.convert_to_tensors('pt').to(device='cuda')
            logits = model.generate(input_ids=data['input_ids'],
                                    attention_mask=data['attention_mask'],
                                    max_length=10)
            all_logits += logits.tolist()
            labels = data['labels']
            labels[labels < 0] = 0
            all_labels += labels.tolist()

        predictions = all_logits
        label_ids = all_labels

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        acc = float(mean(1. if pred == label else 0. for pred, label in zip(predictions, labels)))
        metrics.update({'eval_accuracy': acc})
        tqdm.write(f"{metrics}")


def train(model_name):
    dataset = load_dataset("nsmc")
    train_data = dataset['train']
    test_data = dataset['test']

    print(f"ex) {train_data[0]}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dataset = []
    for data in [train_data, test_data]:
        all_input_ids = tokenizer([row['document'] for row in data], add_special_tokens=True, max_length=512, truncation=True).input_ids
        all_labels = tokenizer(['긍정적 댓글' if row['label'] == 1 else '부정적 댓글' for row in data], add_special_tokens=True).input_ids

        data = [
            dict(input_ids=input_ids, labels=labels)
            for input_ids, labels in zip(all_input_ids, all_labels)
        ]
        dataset.append(data)
    train_data, test_data = dataset

    # torch.distributed.init_process_group(backend="gloo")
    args = Seq2SeqTrainingArguments(
        "pko-t5-nsmc",
        overwrite_output_dir=True,
        learning_rate=1e-3,
        optim='adafactor',
        warmup_ratio=0.6,
        num_train_epochs=5,
        local_rank=int(os.getenv("LOCAL_RANK", "-1")),

        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,

        evaluation_strategy='epoch',
        save_strategy='no',
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()
    callback = TrainerCallbackForEval()
    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        callbacks=[callback]
    )

    trainer.train()


if __name__ == '__main__':
    fire.Fire(train)

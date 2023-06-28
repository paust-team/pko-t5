import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import grpc
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_metric
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, Trainer, T5TokenizerFast, T5ForConditionalGeneration, DataCollatorForSeq2Seq

from pkot5.flan.templates import TEMPLATES
from . import pb


@dataclass
class ModelArguments:
    data_server_addr: str
    pretrained_model_name_or_path: str = "paust/pko-t5-large"


class FlanDataset(Dataset):
    def __init__(self, data_server_channel):
        self.dataset = pb.FlanProcessingDatasetStub(data_server_channel)

    def __len__(self):
        resp = self.dataset.Metadata(pb.MetadataReq())
        return resp.total

    def __getitem__(self, index):
        resp: pb.GetResp = self.dataset.Get(pb.GetReq(index=index))
        input_ids = list(resp.input_ids)
        target_ids = list(resp.target_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': [True] * len(resp.input_ids),
            'labels': target_ids,
        }


def prepare_klue_sts(tokenizer):
    datasets = load_dataset('klue', 'sts')
    all_input_texts, all_target_texts = [], []
    for data in datasets['validation']:
        sentence1 = data['sentence1'].strip()
        sentence2 = data['sentence2'].strip()
        input_text = "{sentence1}\n\n{sentence2}\n\n위 두 문장은 같은 의미인가요?\n\n옵션:\n- 네\n- 아니오".format(sentence1=sentence1, sentence2=sentence2)
        target_text = "네" if data['labels']['binary-label'] == 1 else "아니오"
        all_input_texts.append(input_text)
        all_target_texts.append(target_text)

    encodings = tokenizer(all_input_texts + all_target_texts, add_special_tokens=True, return_attention_mask=True)
    input_ids = encodings.input_ids[:len(all_input_texts)]
    attention_mask = encodings.attention_mask[:len(all_input_texts)]
    labels = encodings.input_ids[len(all_input_texts):]

    return [
        dict(input_ids=row[0], attention_mask=row[1], labels=row[2])
        for row in zip(input_ids, attention_mask, labels)
    ]


def prepare_klue_ynat(tokenizer):
    datasets = load_dataset('klue', 'ynat')
    label2text = {
        0: "IT과학",
        1: "경제",
        2: "사회",
        3: "생활문화",
        4: "세계",
        5: "스포츠",
        6: "정치",
    }
    options_ = "옵션:\n" + '\n'.join([f"- {t}" for t in label2text.keys()])
    input_tpl = "{title}\n\n이 뉴스 제목은 어떤 주제에 관한 문서인가요?\n{options_}"
    all_input_texts, all_target_texts = [], []
    for data in datasets['validation']:
        title = data['title'].strip()
        target_text = label2text[data['label']]
        input_text = input_tpl.format(title=title, options_=options_)
        all_input_texts.append(input_text)
        all_target_texts.append(target_text)

    encodings = tokenizer(all_input_texts + all_target_texts, add_special_tokens=True, return_attention_mask=True)
    input_ids = encodings.input_ids[:len(all_input_texts)]
    attention_mask = encodings.attention_mask[:len(all_input_texts)]
    labels = encodings.input_ids[len(all_input_texts):]

    return [
        dict(input_ids=row[0], attention_mask=row[1], labels=row[2])
        for row in zip(input_ids, attention_mask, labels)
    ]


def prepare_klue_mrc(tokenizer):
    datasets = load_dataset('klue', 'mrc')

    input_tpl = "{title}:\n\n{context}\n\n이 문서에 대한 질문에 답변해 주세요. 답변할 수 없는 질문이면 \"답변할 수 없음\"이라고 말합니다. {question}"
    all_context_ids = tokenizer([data['context'].strip() for data in datasets['validation']], add_special_tokens=False, truncation=True, max_length=768).input_ids
    all_contexts = tokenizer.batch_decode(all_context_ids, skip_special_tokens=True)
    all_input_texts, all_target_texts = [], []
    for data, context in zip(datasets['validation'], all_contexts):
        question = data['question'].strip()
        title = data['title'].strip()
        is_impossible = bool(data['is_impossible'])
        if not is_impossible:
            answer = data['answers']['text'][0]
        else:
            answer = "답변 할 수 없음"

        all_input_texts.append(input_tpl.format(title=title, context=context, question=question))
        all_target_texts.append(answer)

    encodings = tokenizer(all_input_texts + all_target_texts, add_special_tokens=True, return_attention_mask=True)
    input_ids = encodings.input_ids[:len(all_input_texts)]
    attention_mask = encodings.attention_mask[:len(all_input_texts)]
    labels = encodings.input_ids[len(all_input_texts):]

    return [
        dict(input_ids=row[0], attention_mask=row[1], labels=row[2])
        for row in zip(input_ids, attention_mask, labels)
    ]


def prepare_klue_nli(tokenizer):
    datasets = load_dataset('klue', 'nli')

    options_ = "옵션:\n- 네\n- 아니오\n- 애매합니다."
    input_tpl = "여기에 전제가 있습니다:\n{premise}\n\n여기에 가설이 있습니다:\n{hypothesis}\n\n전제가 참이면 가설도 참이라는 결론을 내릴 수 있습니까?\n{options_}"
    all_input_texts, all_target_texts = [], []
    for data in datasets['validation']:
        premise = data['premise'].strip()
        hypothesis = data['hypothesis'].strip()
        label = data['label']
        if label == 0:
            answer = "네"
        elif label == 2:
            answer = "아니오"
        elif label == 1:
            answer = "애매합니다."
        else:
            raise RuntimeError()

        all_input_texts.append(input_tpl.format(premise=premise, hypothesis=hypothesis, options_=options_))
        all_target_texts.append(answer)

    encodings = tokenizer(all_input_texts + all_target_texts, add_special_tokens=True, return_attention_mask=True)
    input_ids = encodings.input_ids[:len(all_input_texts)]
    attention_mask = encodings.attention_mask[:len(all_input_texts)]
    labels = encodings.input_ids[len(all_input_texts):]

    return [
        dict(input_ids=row[0], attention_mask=row[1], labels=row[2])
        for row in zip(input_ids, attention_mask, labels)
    ]


class MyTrainer(Trainer):
    def __init__(self, dataset_lengths, **kwargs):
        super().__init__(**kwargs)
        self.dataset_lengths = dataset_lengths

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        chunk_size = len(eval_dataset) // self.args.world_size
        if self.args.local_rank == self.args.world_size - 1:
            eval_dataset = eval_dataset[chunk_size * self.args.local_rank:]
        else:
            eval_dataset = eval_dataset[chunk_size * self.args.local_rank:chunk_size * (self.args.local_rank + 1)]

        eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.per_device_eval_batch_size, shuffle=False, collate_fn=DataCollatorForSeq2Seq(self.tokenizer, self.model))
        all_losses, all_predictions, all_labels = [], [], []
        with torch.no_grad():
            if self.args.local_rank == 0:
                eval_dataloader = tqdm(eval_dataloader)
            for batch in eval_dataloader:
                batch = batch.to(device='cuda')
                loss = self.model(**batch).loss
                logits = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=64,
                    early_stopping=True,
                )
                all_losses.append(loss.item())
                all_labels.append(batch['labels'].tolist())
                all_predictions.append(logits.tolist())
        losses = all_losses
        predictions = sum(all_predictions, [])
        labels = sum(all_labels, [])

        with open(f"{self.args.local_rank}.pkl", "wb") as f:
            pickle.dump({'losses': losses, 'predictions': predictions, 'labels': labels}, f)
        dist.barrier()
        losses, predictions, labels = [], [], []
        for i in range(self.args.world_size):
            with open(f"{i}.pkl", 'rb') as f:
                data = pickle.load(f)
                losses += data['losses']
                predictions += data['predictions']
                labels += data['labels']
        loss = np.mean(losses)
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = [
            [i if i != -100 else self.tokenizer.pad_token_id for i in ids]
            for ids in labels
        ]
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        metrics = {'eval_loss': loss}
        for i, eval_fn in enumerate([eval_klue_ynat, eval_klue_nli, eval_klue_mrc, eval_klue_sts]):
            size = self.dataset_lengths[i]
            metrics.update(eval_fn(predictions[:size], labels[:size]))
            predictions = predictions[size:]
            labels = labels[size:]

        metrics['step'] = self.state.global_step
        self.log(metrics)

        return metrics


def eval_klue_ynat(predictions, labels):
    label2text = {
        0: "IT과학",
        1: "경제",
        2: "사회",
        3: "생활문화",
        4: "세계",
        5: "스포츠",
        6: "정치",
    }
    text2label = {v:k for k, v in label2text.items()}
    y_pred = [text2label.get(t.strip(), -1) for t in predictions]
    y_true = [text2label[t.strip()] for t in labels]

    f1 = f1_score(y_true, y_pred, average='macro')
    return {'ynat_f1': f1}


def eval_klue_nli(predictions, labels):
    def text2label(t_):
        if t_ == '네':
            return 0
        elif t_ == '아니오':
            return 2
        elif t_.startswith('애매'):
            return 1
        else:
            return -1
    y_pred = [text2label(t.strip()) for t in predictions]
    y_true = [text2label(t.strip()) for t in labels]

    acc = accuracy_score(y_true, y_pred)
    return {'nli_acc': acc}


def eval_klue_mrc(predictions, labels):
    squad_metric = load_metric('squad')

    predictions = [{'prediction_text': t.strip(), 'id': f"{i}"} for i, t in enumerate(predictions)]
    references = [{'answers': {'text': [t.strip()], 'answer_start': [0]}, 'id': f"{i}"} for i, t in enumerate(labels)]
    results = squad_metric.compute(predictions=predictions, references=references)
    return {'mrc_em': results['exact_match'], 'mrc_f1': results['f1']}


def eval_klue_sts(predictions, labels):
    text2label = {
        '네': 1,
        '아니오': 0,
    }
    y_pred = [text2label.get(t.strip(), 0) for t in predictions]
    y_true = [text2label[t.strip()] for t in labels]
    f1 = f1_score(y_true, y_pred)
    return {"sts_f1": f1}


def train():
    training_args: TrainingArguments
    model_args: ModelArguments
    training_args, model_args = HfArgumentParser([TrainingArguments, ModelArguments]).parse_args_into_dataclasses()

    tokenizer = T5TokenizerFast.from_pretrained(model_args.pretrained_model_name_or_path)

    dataset_klue_ynat = prepare_klue_ynat(tokenizer)[:100]
    dataset_klue_mrc = prepare_klue_mrc(tokenizer)[:100]
    max_len = max([len(data['labels']) for data in dataset_klue_mrc])
    print(f"mrc output max length: {max_len}")
    dataset_klue_sts = prepare_klue_sts(tokenizer)[:100]
    dataset_klue_nli = prepare_klue_nli(tokenizer)[:100]

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    )
    with grpc.insecure_channel(model_args.data_server_addr) as channel:
        trainer = MyTrainer(
            args=training_args,
            model=model,
            tokenizer=tokenizer,
            train_dataset=FlanDataset(channel),
            eval_dataset=dataset_klue_ynat + dataset_klue_nli + dataset_klue_mrc + dataset_klue_sts,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model, padding=True),
            dataset_lengths=[len(dataset_klue_ynat), len(dataset_klue_nli), len(dataset_klue_mrc), len(dataset_klue_sts)]
        )

        trainer.train(resume_from_checkpoint=True)


if __name__ == '__main__':
    train()

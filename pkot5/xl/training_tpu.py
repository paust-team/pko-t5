import functools
import logging
import math
import os
import random
import time
from argparse import ArgumentParser
from statistics import mean
from typing import NamedTuple

import grpc
import numpy as np
import torch.nn.utils
from datasets import load_dataset
from transformers import BatchEncoding, T5Config, T5TokenizerFast, Adafactor, get_linear_schedule_with_warmup
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.utils import gcsfs
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

from pkot5.data.dataset_pb2_grpc import LargeCorpusDatasetStub
from pkot5.data.dataset_pb2 import OpenRequest, ReadNextRequest, CloseRequest
from pkot5.utils import fill_in_the_blank
from pkot5.xl.modeling_t5_tpu import T5ForConditionalGeneration, merge_parallel_weights


logger = logging.getLogger(__name__)

_HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", None)


class TrainingArguments(NamedTuple):
    output_path: str
    seed: int
    data_grpc_endpoint: str
    model_name: str
    load_checkpoint: int
    max_steps: int
    save_step: int
    max_input_length: int
    max_target_length: int
    max_gradient_accumulation_steps: int
    batch_size: int


def save_checkpoint(model_path, step, model, optimizer):
    rank = xm.get_ordinal()
    ckpt_path = f"{model_path}/step_{step:07d}/pp_{rank:02d}"

    with gcsfs.open(f"{ckpt_path}/pytorch_model.bin", mode="wb") as f:
        xm.save(model.state_dict(), f, master_only=False)
    if optimizer is not None:
        with gcsfs.open(f"{ckpt_path}/optimizer.pt", mode="wb") as f:
            xm.save(optimizer.state_dict(), f, master_only=False)


def load_checkpoint(model_path, step, load_optimizer=True):
    rank = xm.get_ordinal()
    ckpt_path = f"{model_path}/step_{step:07d}/pp_{rank:02d}"

    with gcsfs.open(f"{ckpt_path}/pytorch_model.bin", mode="rb") as f:
        model_state_dict = torch.load(f)
    if load_optimizer:
        with gcsfs.open(f"{ckpt_path}/optimizer.pt", mode="rb") as f:
            optimizer_state_dict = torch.load(f)
        return model_state_dict, optimizer_state_dict
    else:
        return model_state_dict


def read_large_corpus(tokenizer, args: TrainingArguments, rank, num_replicas, seed, extra_token_ids, eos_token_id, bos_token_id):
    rng = random.Random(seed)
    channel = grpc.insecure_channel(
        args.data_grpc_endpoint,
        options=[
            ('grpc.max_message_length', 256 * 1024 * 1024),
            ('grpc.max_receive_message_length', 256 * 1024 * 1024),
            ('grpc.client_idle_timeout_ms', 2147483647)
        ]
    )
    stub = LargeCorpusDatasetStub(channel)

    resp = stub.Open(OpenRequest(rank=rank, num_replicas=num_replicas, seed=args.seed))
    session_id: str = resp.session_id
    try:
        batch, all_tokens = [], []

        while True:
            resp = stub.ReadNext(ReadNextRequest(session_id=session_id), timeout=3600)
            texts = [t.strip() for t in resp.texts]
            if len(texts) == 0:
                break

            for ids in tokenizer(texts, add_special_tokens=False).input_ids:
                all_tokens += ids
                while len(all_tokens) > args.max_input_length - 1:
                    tokens = all_tokens[:args.max_input_length - 1]
                    record = fill_in_the_blank(rng, tokens, extra_token_ids)
                    input_ids = record['inputs'] + [eos_token_id]
                    attention_mask = [1] * len(input_ids)
                    decoder_input_ids = [bos_token_id] + record['targets']
                    labels = record['targets'] + [eos_token_id]

                    assert len(input_ids) == len(attention_mask)
                    assert len(labels) <= args.max_target_length
                    assert len(input_ids) <= args.max_input_length
                    batch.append(dict(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels,
                    ))
                    if len(batch) == args.batch_size:
                        yield batch
                        batch = []
                    all_tokens = all_tokens[args.max_input_length - 1:]
    finally:
        stub.Close(CloseRequest(session_id=session_id))
        channel.close()


def collate_for_seq2seq(batch, max_input_length=None, max_target_length=None, label_pad_index=-100, pad_index=0):
    if max_input_length is None:
        max_input_length = max(t['input_ids'] for t in batch)
    if max_target_length is None:
        max_target_length = max(t['labels'] for t in batch)

    input_ids, attention_mask, decoder_input_ids, labels = tuple([] for _ in range(4))
    for row in batch:
        input_ids.append(row['input_ids'] + [pad_index] * (max_input_length - len(row['input_ids'])))
        attention_mask.append(row['attention_mask'] + [0] * (max_input_length - len(row['attention_mask'])))
        if 'decoder_input_ids' in row:
            decoder_input_ids.append(row['decoder_input_ids'] + [pad_index] * (max_target_length - len(row['decoder_input_ids'])))
        labels.append(row['labels'] + [label_pad_index] * (max_target_length - len(row['labels'])))

    return BatchEncoding(data={
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids if len(decoder_input_ids) > 0 else None,
        'labels': labels,
    })


def get_device_map(num_layers: int, num_groups: int):
    range_start = 0
    range_end = num_layers - 1

    group_size = (range_end - range_start + 1) // num_groups
    reserved = (range_end - range_start + 1) % num_groups
    groups = []

    begin = 0
    while begin < range_end + 1:
        end = begin + group_size
        if reserved > 0:
            end += 1
        end = min(end, range_end + 1)

        groups.append(list(range(begin, end)))
        begin = end
        reserved -= 1

    groups = {i: g for i, g in enumerate(groups)}
    return groups


def build_training_state(args: TrainingArguments):
    step = 0
    device = xm.xla_device()

    config = T5Config.from_pretrained(args.model_name, use_auth_token=_HF_AUTH_TOKEN,
                                      dropout_rate=0.,
                                      num_decoder_layers=24,
                                      num_layers=24,
                                      vocab_size=50500)
    tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(args.model_name, unk_token="<pad>", extra_ids=200, use_auth_token=_HF_AUTH_TOKEN)
    device_map = get_device_map(config.num_layers, xm.xrt_world_size())
    logger.debug(f"device_map={device_map}")
    model: T5ForConditionalGeneration = T5ForConditionalGeneration(config, device_map=device_map)
    if args.load_checkpoint > 0:
        step = args.load_checkpoint
        model_state_dict, optimizer_state_dict = load_checkpoint(args.output_path, step)
        model.load_state_dict(model_state_dict, strict=True)
        model = model.to(device)
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, weight_decay=0.01)
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        model = model.to(device)
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, weight_decay=0.01)
    model.train()
    return step, tokenizer, model, optimizer


def train_loop(args: TrainingArguments, model, optimizer, batch, gradient_accumulation_step, training_callback):
    device = xm.xla_device()
    model.train()
    if gradient_accumulation_step == 1:
        optimizer.zero_grad()
    batch = batch.to(device=device)
    loss = model(**batch)[0]
    if xm.get_ordinal() == 0:
        loss_0 = loss / args.max_gradient_accumulation_steps
        loss_0.backward()

    if gradient_accumulation_step == args.max_gradient_accumulation_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    xm.mark_step()
    xm.add_step_closure(functools.partial(training_callback, loss=loss))


def init_logging(level="INFO"):
    rank = xm.get_ordinal()
    os.makedirs("logs", exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=f"logs/{rank}.log", format='%(asctime)s %(levelname)-8s %(name)+15s --- %(message)s', level=level, force=True)
    xm.rendezvous('init')
    logger.info("=== START on TPU ===")


def train(rank, args: TrainingArguments):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    init_logging(level="INFO")

    elapsed_time = time.time()
    step, tokenizer, model, optimizer = build_training_state(args)
    elapsed_time = time.time() - elapsed_time
    logger.info(f"Initialize time={elapsed_time:.4f}s")
    extra_token_ids = tokenizer.convert_tokens_to_ids([f"<extra_id_{i}>" for i in range(200)])
    for i, token_id in enumerate(extra_token_ids):
        assert token_id != tokenizer.unk_token_id, f"{i}'th extra_token_id is None"
    gradient_accumulation_step = 0
    elapsed_time = [time.time()]
    while step < args.max_steps:
        for batch in read_large_corpus(tokenizer, args, 0, 1, args.seed + step, extra_token_ids, tokenizer.eos_token_id, model.config.decoder_start_token_id):
            gradient_accumulation_step += 1
            is_gradient_accumulated = gradient_accumulation_step == args.max_gradient_accumulation_steps
            if is_gradient_accumulated:
                step += 1
            if step >= args.max_steps:
                break

            batch = collate_for_seq2seq(batch, args.max_input_length, args.max_target_length, label_pad_index=-100, pad_index=model.config.pad_token_id)
            batch = batch.convert_to_tensors('pt')
            def _train_loop_callback(loss):
                elapsed_time[0] = time.time() - elapsed_time[0]
                logger.info(f"step={step} loss={loss:.4f} time={elapsed_time[0]:.4f}s grad_accum={is_gradient_accumulated}")
                elapsed_time[0] = time.time()
            train_loop(args, model, optimizer, batch, gradient_accumulation_step=gradient_accumulation_step, training_callback=_train_loop_callback)
            if is_gradient_accumulated:
                gradient_accumulation_step = 0

            if is_gradient_accumulated and step % args.save_step == 0 and step > 1:
                save_checkpoint(args.output_path, step, model, optimizer)


def run_parallel(func, args):
    xmp.spawn(func, args=(args,), nprocs=args.nprocs, start_method="fork")


def merge_weights(args):
    config = T5Config.from_pretrained(args.model_name, use_auth_token=_HF_AUTH_TOKEN)
    state_dict = merge_parallel_weights(config, args.output_path, args.num_pp)
    t5 = T5ForConditionalGeneration(config)
    t5.load_state_dict(state_dict, strict=True)


def finetune_korquad(rank, args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    init_logging(level="INFO")

    korquad_dataset = load_dataset("squad_kor_v1")

    tokenizer = T5TokenizerFast.from_pretrained(args.model_name, use_auth_token=_HF_AUTH_TOKEN)
    config = T5Config.from_pretrained(args.model_name, use_auth_token=_HF_AUTH_TOKEN)
    attention_mask, decoder_input_ids, input_ids, label_ids = extract_korquad_features(config, korquad_dataset, "train", tokenizer)

    indices = list(range(len(input_ids)))
    device = xm.xla_device()
    device_map = get_device_map(config.num_layers, xm.xrt_world_size())
    model = T5ForConditionalGeneration(config, device_map).to(device)
    model_state_dict = load_checkpoint(args.model_path, args.load_checkpoint, load_optimizer=False)
    model.load_state_dict(model_state_dict)

    optimizer = Adafactor(model.parameters(), lr=args.learning_rate, scale_parameter=False, warmup_init=False, relative_step=False, weight_decay=0.01)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, math.ceil(len(indices) / args.batch_size) * args.num_train_epochs)

    model.train()
    train_step = 0
    for epoch in range(args.num_train_epochs):
        random.Random(args.seed + epoch).shuffle(indices)
        accumulation_step = 0
        losses = []
        start_time = time.time()
        for begin in range(0, len(indices), args.batch_size):
            batch_indices = indices[begin:begin+args.batch_size]
            accumulation_step += 1
            if accumulation_step == 1:
                optimizer.zero_grad()
            loss = model(
                input_ids=input_ids[batch_indices, :].to(device),
                attention_mask=attention_mask[batch_indices, :].to(device),
                decoder_input_ids=decoder_input_ids[batch_indices, :].to(device),
                labels=label_ids[batch_indices, :].to(device),
            )[0]
            loss.backward()
            if accumulation_step == args.accumulation_step or ((begin + args.batch_size) >= len(indices)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                train_step += 1
                accumulation_step = 0
            def _on_train():
                losses.append(loss.cpu().detach().item())
            xm.add_step_closure(_on_train)
            xm.mark_step()
        train_loss = mean(losses)

        save_checkpoint(args.output_path, train_step, model, optimizer=None)
        elapsed_time = time.time() - start_time
        logger.info(f"step={train_step} loss={train_loss:.4f} time={elapsed_time:.1f}seconds")


def extract_korquad_features(config, korquad_dataset, target, tokenizer, max_length=1024):
    input_texts, label_texts = [], []
    for row in korquad_dataset[target]:
        input_texts.append(f"question: {row['question']} title: {row['title']} context: {row['context']}")
        label_texts.append(f"{row['answers']['text'][0]}")
    inputs = tokenizer(input_texts, truncation=True, max_length=max_length, return_tensors='pt', padding=True, add_special_tokens=True, return_attention_mask=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    labels = tokenizer(label_texts, padding=True, return_tensors='pt', add_special_tokens=True, return_attention_mask=True)
    label_ids = labels.input_ids
    label_ids[~labels.attention_mask] = -100
    with torch.no_grad():
        decoder_input_ids = torch.cat([torch.tensor([[config.decoder_start_token_id]] * len(labels.input_ids)), labels.input_ids[:, :-1]], dim=1)
    return attention_mask, decoder_input_ids, input_ids, label_ids


@torch.no_grad()
def test_korquad(rank, args):
    init_logging(level="INFO")

    config = T5Config.from_pretrained(args.model_name, use_auth_token=_HF_AUTH_TOKEN)
    tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(args.model_name, use_auth_token=_HF_AUTH_TOKEN)
    korquad_dataset = load_dataset("squad_kor_v1")
    attention_mask, decoder_input_ids, input_ids, label_ids = extract_korquad_features(config, korquad_dataset, "validation", tokenizer)

    device = xm.xla_device()
    device_map = get_device_map(config.num_layers, list(range(xm.xrt_world_size())))
    model = T5ForConditionalGeneration(config, device_map).to(device)

    model.eval()
    for train_step in [15102, 30204, 45306, 60408, 75510]:
        model_state_dict = load_checkpoint(args.model_path, train_step, load_optimizer=False)
        model.load_state_dict(model_state_dict)

        start_time = time.time()
        predictions, golds = [], []
        for begin in range(0, len(input_ids), args.batch_size):
            end = begin + args.batch_size
            logits = model.beam_search(
                input_ids=input_ids[begin:end].to(device),
                attention_mask=attention_mask[begin:end].to(device),
                num_beams=8,
                max_length=32,
            )
            xm.mark_step()
            predictions += logits.cpu().detach().tolist()
            golds += label_ids[begin:end].tolist()

        predictions = [[label for label in seq] for seq in predictions]
        golds = [[max(label, 0) for label in seq] for seq in golds]
        logger.debug(f"predictions={predictions} golds={golds}")

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        golds = tokenizer.batch_decode(golds, skip_special_tokens=True)

        em = mean([float(compute_exact(a_gold, a_pred)) for a_gold, a_pred in zip(golds, predictions)])
        f1 = mean([float(compute_f1(a_gold, a_pred)) for a_gold, a_pred in zip(golds, predictions)])
        elapsed_time = time.time() - start_time

        logger.info(f"step={train_step} em={em:.4f} f1={f1:.4f} time={elapsed_time*60.:.1f}minutes")


def add_command(subparser):
    parser = subparser.add_parser("train")
    parser.add_argument("--output_path", type=str, default="gs://pko_t5_xl")
    parser.add_argument("--data_grpc_endpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="paust/pko-t5-xl")
    parser.add_argument("--load_checkpoint", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--save_step", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--max_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--nprocs", type=int, default=8)
    parser.set_defaults(func=functools.partial(run_parallel, train))

    parser = subparser.add_parser("merge_weights")
    parser.add_argument("--model_name", type=str, default="paust/pko-t5-xl")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_pp", type=int, default=8)
    parser.set_defaults(func=merge_weights)

    parser = subparser.add_parser("finetune_korquad")
    parser.add_argument("--model_name", type=str, default="paust/pko-t5-xl")
    parser.add_argument("--model_path", type=str, default="gs://pko_t5_xl")
    parser.add_argument("--load_checkpoint", type=int, required=True)
    parser.add_argument("--output_path", type=str, default="gs://pko_t5_xl/finetuned_korquad")
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulation_step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--nprocs", type=int, default=8)
    parser.set_defaults(func=functools.partial(run_parallel, finetune_korquad))

    parser = subparser.add_parser("test_korquad")
    parser.add_argument("--model_name", type=str, default="paust/pko-t5-xl")
    parser.add_argument("--model_path", type=str, default="gs://pko_t5_xl/finetuned_korquad")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--nprocs", type=int, default=8)
    parser.set_defaults(func=functools.partial(run_parallel, test_korquad))


def main():
    parser = ArgumentParser()
    add_command(parser.add_subparsers())

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

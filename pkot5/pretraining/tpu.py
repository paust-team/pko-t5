import functools
import logging
import os
import random
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple, Optional

import fire
import grpc
import numpy as np
import torch.nn.utils
from transformers import BatchEncoding, T5Config, T5TokenizerFast, Adafactor
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.utils import gcsfs
from transformers.utils.model_parallel_utils import get_device_map

from ..data.dataset_pb2_grpc import LargeCorpusDatasetStub
from ..data.dataset_pb2 import ReadRequest
from ..utils import fill_in_the_blank
from ..modeling_t5_tpu import T5ForConditionalGeneration, merge_parallel_weights


logger = logging.getLogger(__name__)


class TrainingArguments(NamedTuple):
    output_path: str
    seed: int
    data_grpc_endpoint: str
    model_name: str
    load_checkpoint: Optional[int]
    max_steps: int
    save_step: int
    max_input_length: int
    max_target_length: int
    max_gradient_accumulation_steps: int
    batch_size: int


def read_large_corpus(tokenizer, args: TrainingArguments, rank, num_replicas, seed, extra_token_ids, eos_token_id, bos_token_id):
    rng = random.Random(seed)
    channel = grpc.insecure_channel(args.data_grpc_endpoint, options=[('grpc.max_message_length', 256 * 1024 * 1024), ('grpc.max_receive_message_length', 256 * 1024 * 1024)])
    stub = LargeCorpusDatasetStub(channel)

    try:
        batch, all_tokens = [], []

        for resp in stub.Read(ReadRequest(rank=rank, num_replicas=num_replicas, seed=args.seed)):
            texts = [t.strip() for t in resp.texts]
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
        decoder_input_ids.append(row['decoder_input_ids'] + [pad_index] * (max_target_length - len(row['decoder_input_ids'])))
        labels.append(row['labels'] + [label_pad_index] * (max_target_length - len(row['labels'])))

    return BatchEncoding(data={
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,
    })


def build_training_state(args: TrainingArguments):
    step = 0
    device = xm.xla_device()

    config = T5Config.from_pretrained(args.model_name)
    tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(args.model_name)
    device_map = get_device_map(config.num_layers, list(range(xm.xrt_world_size())))
    model: T5ForConditionalGeneration = T5ForConditionalGeneration(config, device_map=device_map).to(device)

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, weight_decay=0.01)
    return step, tokenizer, model, optimizer


def train_loop(model, optimizer, batch, gradient_accumulation, training_callback):
    device = xm.xla_device()
    batch = batch.to(device=device)
    loss = model(**batch)[0]
    loss.backward()
    if gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.zero_grad()
    xm.add_step_closure(functools.partial(training_callback, loss=loss))
    xm.mark_step()


def save_checkpoint(args: TrainingArguments, step, model, optimizer):
    rank = xm.get_ordinal()
    ckpt_path = f"{args.output_path}/step_{step:07d}/pp_{rank:02d}"

    with gcsfs.open(f"{ckpt_path}/pytorch_model.bin", mode="wb") as f:
        xm.save(model.state_dict(), f, master_only=False)
    with gcsfs.open(f"{ckpt_path}/optimizer.pt", mode="wb") as f:
        xm.save(optimizer.state_dict(), f, master_only=False)


def init_logging(level="INFO"):
    rank = xm.get_ordinal()
    os.makedirs("logs", exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=f"logs/{rank}.log", format='%(asctime)s %(levelname)-8s %(name)+15s --- %(message)s', level=level, force=True)
    xm.rendezvous('init')


def train(rank, args: TrainingArguments):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    init_logging(level="INFO")

    step, tokenizer, model, optimizer = build_training_state(args)
    extra_token_ids = tokenizer.convert_tokens_to_ids([f"<extra_id_{i}>" for i in range(200)])
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
                if is_gradient_accumulated:
                    elapsed_time[0] = time.time() - elapsed_time[0]
                    logger.info(f"step={step} loss={loss:.4f} time={elapsed_time[0]:.4f}s")
                    elapsed_time[0] = time.time()
            train_loop(model, optimizer, batch, gradient_accumulation=is_gradient_accumulated, training_callback=_train_loop_callback)
            if is_gradient_accumulated:
                gradient_accumulation_step = 0

            if is_gradient_accumulated and step % args.save_step == 0 and step > 0:
                save_checkpoint(args, step, model, optimizer)


def run_train(args):
    xmp.spawn(train, args=(args,), nprocs=args.nprocs, start_method="fork")


class WeightsMergingArguments(NamedTuple):
    model_name: str
    output_path: str
    num_pp: int


def merge_weights(args: WeightsMergingArguments):
    config = T5Config.from_pretrained(args.model_name)
    state_dict = merge_parallel_weights(config, args.output_path, args.num_pp)
    t5 = T5ForConditionalGeneration(config)
    t5.load_state_dict(state_dict, strict=True)


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
    parser.set_defaults(func=run_train)

    parser = subparser.add_parser("merge_weights")
    parser.add_argument("--model_name", type=str, default="paust/pko-t5-xl")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_pp", type=int, default=8)
    parser.set_defaults(func=merge_weights)


def main():
    parser = ArgumentParser()
    add_command(parser.add_subparsers())

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

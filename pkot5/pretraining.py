import os
import random
import time
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, TrainingArguments, T5Config, Adafactor

from .args import ARGS, CONFIGS
from .data import LargeCorpusDatasetFromServerV2, DataCollatorForSeq2Seq, NUM_EXTRA_IDS, LargeCorpusDatasetFromServer, DataCollatorForT5MLM


def train(model_size: str, tokenizer_path: str, grpc_endpoint: str, resume_checkpoint: Optional[int] = None, version: Optional[str] = None):
    local_rank = int(os.getenv("LOCAL_RANK", "-1"))
    model_size = model_size.lower()
    args = TrainingArguments(
        output_dir=f'./models/pko-t5/{model_size}',
        local_rank=local_rank,
        **ARGS[model_size]
    )

    torch.cuda.set_device(local_rank)

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, unk_token='<pad>', extra_ids=NUM_EXTRA_IDS)
    config = T5Config(**CONFIGS[model_size])
    config.dropout_rate = 0.0
    model = T5ForConditionalGeneration(config).cuda()

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True)
    scaler = amp.GradScaler()
    step = 0

    if resume_checkpoint is not None:
        step = resume_checkpoint
        ckpt_dir = Path(args.output_dir + f'/checkpoint-{step}')
        model.load_state_dict(torch.load(ckpt_dir / "pytorch_model.bin", map_location='cpu'))
        optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location='cpu'))

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    if version == "1":
        train_data = LargeCorpusDatasetFromServer(grpc_endpoint, seed=args.data_seed)
        train_loader = DataLoader(train_data, batch_size=args.per_device_train_batch_size, collate_fn=DataCollatorForT5MLM(tokenizer, prefix="fill: "))
    else:
        train_data = LargeCorpusDatasetFromServerV2(tokenizer, grpc_endpoint, seed=args.data_seed)
        train_loader = DataLoader(train_data, batch_size=args.per_device_train_batch_size, collate_fn=DataCollatorForSeq2Seq(tokenizer))
    print(f"Start pretraining of t5-{model_size}")
    while step < args.max_steps:
        total_loss = 0
        dt = time.time()
        gradient_accumulation_step = 0
        optimizer.zero_grad()
        for data in train_loader:
            if step >= args.max_steps:
                break

            data = data.convert_to_tensors('pt').to(device='cuda')
            with amp.autocast():
                loss = ddp_model(**data).loss
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward(retain_graph=True)
            gradient_accumulation_step += 1
            if gradient_accumulation_step == args.gradient_accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                dt = time.time() - dt

                step += 1
                total_loss = loss.detach().item() + total_loss
                print(f"step={step} loss={total_loss:.4f} time={dt:.4f}s")

                if step % args.save_steps == 0:
                    ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
                    ckpt_dir.mkdir(exist_ok=True, parents=True)
                    config.dropout_rate = 0.1
                    if local_rank == 0:
                        config.save_pretrained(ckpt_dir)
                        torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
                        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

                gradient_accumulation_step = 0
                total_loss = 0
                dt = time.time()
            else:
                total_loss = loss.detach().item() + total_loss

    print(f"End of pretraining t5-{model_size}")


if __name__ == '__main__':
    fire.Fire(train)

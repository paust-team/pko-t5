import copy
import functools
import pickle
import queue
import random
import threading
import time
from collections import deque
from pathlib import Path
from multiprocessing.pool import ThreadPool
from typing import List, Dict, Optional

import fire
import grpc
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, BatchEncoding, PreTrainedTokenizerFast, PreTrainedTokenizer
from torch import distributed as dist

from .dataset_pb2_grpc import LargeCorpusDatasetStub
from . import dataset_pb2 as pb


NUM_EXTRA_IDS = 256


class DataCollatorForT5MLM(object):
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            noise_density: float = 0.15,
            mean_noise_span_length: float = 3.0,
            input_length: int = 512,
            target_length: int = 114,
            max_sentinel_ids: int = 100,
            prefix: Optional[str] = "fill: "
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.max_sentinel_ids = max_sentinel_ids
        self.prefix_ids = tokenizer([prefix], add_special_tokens=False, return_tensors='np').input_ids if prefix is not None else None

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel, prefix_ids=self.prefix_ids)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel, prefix_ids=None)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - self.max_sentinel_ids + sentinel_ids - 1), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids, prefix_ids=None):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        concat_list = [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)]
        if prefix_ids is not None:
            concat_list = [prefix_ids.repeat(batch_size, axis=0)] + concat_list
        input_ids = np.concatenate(concat_list, axis=-1)
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


class LargeCorpusDatasetFromServer(IterableDataset):
    def __init__(self, jobname, grpc_endpoint, seed):
        channel = grpc.insecure_channel(grpc_endpoint)
        self.stub = LargeCorpusDatasetStub(channel)

        self.context = queue.Queue(maxsize=10)
        self.stop_worker = False
        self.jobname = jobname

        if dist.get_rank() == 0:
            self.stub.Init(pb.InitRequest(world_size=dist.get_world_size(), seed=seed, session_id=jobname))
        dist.barrier()

    def _get_from_server(self):
        request = pb.ReadRequest(rank=dist.get_rank(), session_id=self.jobname)
        for response in self.stub.Read(request):
            if self.stop_worker:
                break
            input_ids = response.input_ids
            while not self.stop_worker:
                try:
                    self.context.put_nowait(input_ids)
                    break
                except queue.Full:
                    time.sleep(0.01)
                    continue

    def __iter__(self):
        max_len = 564
        self.stop_worker = False
        worker = threading.Thread(target=self._get_from_server)
        worker.start()
        try:
            preserved_ids = []
            while worker.is_alive():
                try:
                    ids = self.context.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
                    continue

                preserved_ids += ids
                while len(preserved_ids) > max_len:
                    input_ids = preserved_ids[:max_len]
                    yield {
                        'input_ids': np.array(input_ids),
                        'attention_mask': np.array([1] * 512),
                    }
                    preserved_ids = preserved_ids[max_len:]
        finally:
            self.stop_worker = True

    def __del__(self):
        self.stop_worker = True


class LargeCorpusDatasetFromServerV2(IterableDataset):
    def __init__(self, jobname, tokenizer, grpc_endpoint, seed):
        channel = grpc.insecure_channel(grpc_endpoint)
        self.stub = LargeCorpusDatasetStub(channel)

        self.context = queue.Queue(maxsize=50)
        self.stop_worker = False
        self.jobname = jobname

        self.prefix = tokenizer("fill: ", add_special_tokens=False).input_ids
        self.extra_ids = [tokenizer.convert_tokens_to_ids(f'<extra_id_{i}') for i in range(NUM_EXTRA_IDS)]
        self.eos_token_id = tokenizer.eos_token_id

        if dist.get_rank() == 0:
            self.stub.Init(pb.InitRequest(world_size=dist.get_world_size(), seed=seed, session_id=jobname))
        dist.barrier()

    def _get_from_server(self):
        request = pb.ReadRequest(rank=dist.get_rank(), session_id=self.jobname)
        for response in self.stub.Read(request):
            if self.stop_worker:
                break
            input_ids = response.input_ids
            while not self.stop_worker:
                try:
                    self.context.put_nowait(input_ids)
                    break
                except queue.Full:
                    time.sleep(0.01)
                    continue

    def __iter__(self):
        max_len = 564

        self.stop_worker = False
        worker = threading.Thread(target=self._get_from_server)
        worker.start()
        try:
            token_ids = []
            while worker.is_alive():
                try:
                    ids = self.context.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
                    continue

                token_ids += ids
                while len(token_ids) > max_len:
                    data = self._fill_in_the_blank(token_ids[:max_len])
                    yield {
                        'input_ids': self.prefix + data['inputs'] + [self.eos_token_id],
                        'attention_mask': [1] * len(data),
                        'labels': data['targets'] + [self.eos_token_id],
                    }
                    token_ids = token_ids[max_len:]
        finally:
            self.stop_worker = True

    def __del__(self):
        self.stop_worker = True

    def _fill_in_the_blank(self, words: List[int]):
        mask_id = -1

        min_prob = 1 / (len(words) + 1)
        max_prob = 1 / 2
        inputs = copy.deepcopy(words)
        targets = words
        for i in range(len(words)):
            prob = random.random()
            if min_prob < prob < max_prob:
                inputs[i] = mask_id
            else:
                targets[i] = mask_id

        def merge_mask(words_):
            mask_spans = []
            begin, end = None, None
            for i, w in enumerate(words_):
                if w == mask_id:
                    if begin is None:
                        begin = i
                    end = i + 1
                else:
                    if end is not None:
                        mask_spans.append((begin, end))
                        begin, end = None, None
            if begin is not None and end is not None:
                mask_spans.append((begin, end))

            new_words_ = []
            last_offset = 0
            assert len(mask_spans) <= len(self.extra_ids), f"mask_spans={len(mask_spans)} is over length of extra_ids"
            for i, (begin, end) in enumerate(mask_spans):
                new_words_ += words_[last_offset:begin]
                new_words_.append(self.extra_ids[i])
                last_offset = end
            new_words_ += words_[last_offset:]

            return new_words_

        inputs = merge_mask(inputs)
        targets = merge_mask(targets)

        return {'inputs': inputs, 'targets': targets}


class DataCollatorForSeq2Seq(object):
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.label_pad_token_id = -100

    def __call__(self, features):
        labels = [feature['labels'] for feature in features] if 'labels' in features[0] else None
        input_ids = [feature['input_ids'] for feature in features] if 'input_ids' in features[0] else None
        attention_mask = [feature['attention_mask'] for feature in features] if 'attention_mask' in features[0] else None

        max_len_inputs = max(len(ids) for ids in input_ids)
        max_len_labels = max(len(ids) for ids in labels)

        for arr in input_ids:
            arr.extend([self.pad_token_id] * (max_len_inputs - len(arr)))
        for arr in attention_mask:
            arr.extend([0] * (max_len_inputs - len(arr)))
        for arr in labels:
            arr.extend([self.label_pad_token_id] * (max_len_labels - len(arr)))

        return BatchEncoding(data={'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels})


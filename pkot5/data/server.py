import queue
import random
import threading
import time
from concurrent import futures
from pathlib import Path

import fire
import grpc
import pandas as pd
from transformers import T5TokenizerFast

from .dataset_pb2_grpc import LargeCorpusDatasetServicer, add_LargeCorpusDatasetServicer_to_server
from . import dataset_pb2 as pb


class LargeCorpusReader(object):
    def __init__(self, tokenizer, corpus_dir):
        self.tokenizer = tokenizer
        self.corpus_dir = corpus_dir

        self.rng = None
        self.world_size = None
        self.corpus_files = None

    def reinit(self, seed, world_size):
        self.rng = random.Random(seed)
        self.world_size = world_size

        corpus_dir = Path(self.corpus_dir)
        self.corpus_files = list(corpus_dir.glob("*.json"))
        self.rng.shuffle(self.corpus_files)

    def start(self, rank):
        assert self.rng is not None
        assert self.world_size is not None
        assert self.corpus_files is not None

        corpus_files = self.corpus_files[rank::self.world_size]
        self.rng.shuffle(corpus_files)

        for corpus_file in corpus_files:
            texts = []
            assert corpus_file.exists(), f"corpus_file={corpus_file}"
            corpus_df = pd.read_json(corpus_file, orient='records', lines=True)
            if 'text' not in corpus_df.keys():
                continue
            texts += list(corpus_df['text'])
            self.rng.shuffle(texts)

            all_ids = self.tokenizer(texts, add_special_tokens=False).input_ids
            yield from all_ids


class LargeCorpusDatasetServicerImpl(LargeCorpusDatasetServicer):
    def __init__(self, model_path, corpus_dir):
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        self.corpus_dir = corpus_dir

        self.reader = LargeCorpusReader(self.tokenizer, self.corpus_dir)

    def Init(self, request, context):
        print(f"Connected world_size={request.world_size}")
        self.reader.reinit(request.seed, request.world_size)
        return pb.InitResponse()

    def Read(self, request, context):
        print(f"Start reading rank={request.rank}")
        for input_ids in self.reader.start(request.rank):
            yield pb.ReadResponse(input_ids=input_ids)


def serve(model_path, corpus_dir, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(10))
    add_LargeCorpusDatasetServicer_to_server(LargeCorpusDatasetServicerImpl(model_path, corpus_dir), server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"Start corpus dataset serving on {port}")
    server.start()
    server.wait_for_termination()

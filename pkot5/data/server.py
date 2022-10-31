import queue
import random
import threading
import time
import uuid
from concurrent import futures
from pathlib import Path

import fire
import grpc
import pandas as pd
from transformers import T5TokenizerFast

from .dataset_pb2_grpc import LargeCorpusDatasetServicer, add_LargeCorpusDatasetServicer_to_server
from . import dataset_pb2 as pb


def read_large_corpus(corpus_dir, seed, rank, num_replicas):
    rng = random.Random(seed)
    corpus_dir = Path(corpus_dir)
    corpus_files = list(corpus_dir.glob("part-*.json"))
    rng.shuffle(corpus_files)

    corpus_files = corpus_files[rank::num_replicas]

    for corpus_file in corpus_files:
        assert corpus_file.exists(), f"corpus_file={corpus_file}"
        corpus_df = pd.read_json(corpus_file, orient='records', lines=True)
        if 'text' not in corpus_df.keys():
            continue
        documents = corpus_df['text'].tolist()
        rng.shuffle(documents)
        yield documents


class LargeCorpusDatasetServicerImpl(LargeCorpusDatasetServicer):
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir

    def Read(self, req, context):
        print(f"Start reading rank={req.rank}")
        for documents in read_large_corpus(self.corpus_dir, req.seed, req.rank, req.num_replicas):
            yield pb.ReadResponse(texts=documents)


def serve(corpus_dir, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(10))
    add_LargeCorpusDatasetServicer_to_server(LargeCorpusDatasetServicerImpl(corpus_dir), server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    print(f"Start corpus dataset serving on {port}")
    server.start()
    server.wait_for_termination()

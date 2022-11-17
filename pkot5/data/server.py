import logging
import queue
import random
import sys
import threading
import time
import uuid
from concurrent import futures
from pathlib import Path
from typing import Dict

import fire
import grpc
import pandas as pd
from transformers import T5TokenizerFast

from .dataset_pb2_grpc import LargeCorpusDatasetServicer, add_LargeCorpusDatasetServicer_to_server
from . import dataset_pb2 as pb


logger = logging.getLogger(__name__)


class Session(object):
    def __init__(self, seed, rank, num_replicas, corpus_dir):
        self.session_id = str(uuid.uuid4())
        self.offset = 0

        self.rng = random.Random(seed)
        corpus_dir = Path(corpus_dir)
        corpus_files = list(corpus_dir.glob("part-*.json"))
        self.rng.shuffle(corpus_files)

        self.corpus_files = corpus_files[rank::num_replicas]

    def read_next(self):
        while True:
            if self.offset >= len(self.corpus_files):
                return []
            corpus_file = self.corpus_files[self.offset]
            self.offset += 1

            assert corpus_file.exists(), f"corpus_file={corpus_file}"
            corpus_df = pd.read_json(corpus_file, orient='records', lines=True)
            if 'text' not in corpus_df.keys():
                continue

            documents = corpus_df['text'].tolist()
            if len(documents) == 0:
                continue

            self.rng.shuffle(documents)
            return documents

    @property
    def id(self) -> str:
        return self.session_id


class LargeCorpusDatasetServicerImpl(LargeCorpusDatasetServicer):
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.sessions: Dict[str, Session] = {}

    def Open(self, req, context):
        logger.info(f"Start reading rank={req.rank}")
        session = Session(req.seed, req.rank, req.num_replicas, self.corpus_dir)
        self.sessions[session.id] = session
        return pb.OpenResponse(session_id=session.id)

    def ReadNext(self, req, context):
        assert req.session_id in self.sessions
        sess = self.sessions[req.session_id]
        documents = sess.read_next()
        logger.info(f"Read offset={sess.offset}")
        return pb.ReadNextResponse(texts=documents)

    def Close(self, req, context):
        assert req.session_id in self.sessions
        self.sessions.pop(req.session_id)
        return pb.CloseResponse()


def serve(corpus_dir, port=50051):
    logging.basicConfig(level="DEBUG", format='%(asctime)s %(levelname)-8s %(name)+15s --- %(message)s', force=True)
    server = grpc.server(futures.ThreadPoolExecutor(10), options=[('grpc.keepalive_time_ms', 2147483647), ('grpc.keepalive_timeout_ms', 2147483647), ('grpc.client_idle_timeout_ms', 2147483647)])
    add_LargeCorpusDatasetServicer_to_server(LargeCorpusDatasetServicerImpl(corpus_dir), server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    logger.info(f"Start corpus dataset serving on {port}")
    server.start()
    server.wait_for_termination()

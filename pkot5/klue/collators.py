from typing import Optional, Sized, Iterator

import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class DistributedSamplerForEval(Sampler[int]):
    def __init__(self, data_source: Optional[Sized]):
        super().__init__(data_source)

        self.indices = list(range(len(data_source)))

        r = dist.get_rank()
        s = dist.get_world_size()
        each_total = len(self.indices) // s
        if r == s - 1:
            self.indices = self.indices[each_total * r:]
        else:
            self.indices = self.indices[each_total * r:each_total * (r + 1)]

    def __iter__(self) -> Iterator[int]:
        yield from self.indices

    def __len__(self):
        return len(self.indices)

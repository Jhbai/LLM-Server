import uuid
import time
import torch
import config
import asyncio
import numpy as np
from enum import Enum
import torch.nn as nn
from typing import List, Optional
from transformers import DynamicCache
from dataclasses import dataclass, field

# ----- Define the request status ----- #
class RequestStatus(Enum):
    WAITING = "WAITING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    FINISH = "FINISH"
    CANCELL = "CANCELL"
    ERROR = "ERROR"

@dataclass
class Request:
    # ----- uuid of requests ----- #
    request_id: str = field(default_factory = lambda: str())

    # ----- ids -----
    input_ids: torch.Tensor
    total_ids: List[int] = field(default_factory=list)

    # ----- status ----- #
    status: RequestStatus = RequestStatus.WAITING

    # ----- for scheduling ----- #
    arrival_time: float = field(default_factory=time.time)

    # ----- Token Counter ----- #
    tidx = 0

    # ----- cache assignment ----- #
    kv_cache: DynamicCache = field(default_factory=DynamicCache)

    # ----- storage ----- #
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def __post_init__(self):
        self.total_ids += self.input_ids.squeeze().tolist()

    def get_ids(self):
        self.status = RequestStatus.PREFILLING
        st = self.tidx
        ed = np.min(self.tidx+config.PREFILL_TOKEN_SIZE, self.get_sequence_length()-1)
        if self.tidx == self.get_sequence_length()-1: 
            self.status = RequestStatus.DECODING
            self.tidx = ed
            return self.input_ids[-1:]

        ids = self.input_ids[st:ed]
        self.tidx = ed
        return ids

    def get_sequence_length(self):
        return len(self.total_ids)
    
    async def stream_token(self, token_str: str):
        await self.output_queue.put(token_str)

    def finish(self):
        self.status = RequestStatus.FINISH

    def cancel(self):
        if self.status not in {RequestStatus.FINISH, RequestStatus.ERROR}:
            self.status = RequestStatus.CANCELL






import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import uuid
import time
import torch
import asyncio
import numpy as np
from enum import Enum
import torch.nn as nn
from typing import List, Optional
from transformers import DynamicCache
from dataclasses import dataclass, field
from src.main.python.config import config

# ----- Define the request status ----- #
class RequestStatus(Enum):
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"

@dataclass
class Request:
    input_ids: torch.Tensor
    tidx: int = 0
    seq_len: int = 0
    status: RequestStatus = RequestStatus.PREFILLING
    request_id: str = field(default_factory = lambda: "")
    kv_cache: DynamicCache = field(default_factory=DynamicCache)

    def __post_init__(self):
        # ----- q0為PREFILLING，且定義 δ(PREFILLING, Prefilling_token) 的終點 ----- #
        self.seq_len = self.input_ids.shape[1] # (n_batch, seq_len)

    def get_ids(self):
        # ----- 檢查 Prefilling 狀態 ----- #
        st = self.tidx
        ed = int(np.min([self.tidx+config.PREFILL_TOKEN_SIZE, self.get_sequence_length()-1]))

        # ----- δ(PREFILLING, Prefilling_Complete) ----- #
        if self.tidx == self.get_sequence_length()-1: 
            self.status = RequestStatus.DECODING
            self.tidx = ed
            return self.input_ids[:, -1:]

        # ------ 回傳並更新 ----- #
        ids = self.input_ids[:, st:ed]
        self.tidx = ed
        return ids
    
    def get_remain_length(self):
        return self.seq_len - self.tids

    def get_sequence_length(self):
        return self.seq_len
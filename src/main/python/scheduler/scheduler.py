import gc
import torch
from typing import List, Dict, Any
from collections import OrderedDict
from src.main.python.schema import model 
from src.main.python.config import config
from src.main.python.engine import cache_manager
from src.main.python.engine import decode, prefill
from transformers import DynamicCache, Gemma3ForConditionalGeneration

class RequestManager:
    def __init__(self):
        self.PrefillList: Dict[model.Request.request_id, model.Request] = OrderedDict()
        self.DecodeList: Dict[model.Request.request_id, model.Request] = OrderedDict()

    def add_request(self, request: model.Request):
        if request.status is model.RequestStatus.PREFILL:
            self.PrefillList[request.request_id] = request
        elif request.status is model.RequestStatus.DECODE:
            self.DecodeList[request.request_id] = request

    def step(self):
        self.idx = list()
        self.jobs = list()
        # ----- 先取input_ids(同時更新狀態)，再放回List中 ----- #
        for _ in range(min([config.BATCH_SIZE, len(self.DecodeList)])):
            key, value = self.DecodeList.popitem(last=False)
            self.jobs += [(value.get_ids(), value.kv_cache)]
            self.idx += [(value.status, key)]
            self.DecodeList[key] = value
        for _ in range(config.BATCH_SIZE - len(self.jobs)):
            if len(self.PrefillList) == 0:
                break
            key, value = self.PrefillList.popitem(last=False)
            self.jobs += [(value.get_ids(), value.kv_cache)]
            self.idx += [(value.status, key)]
            if value.status is model.RequestStatus.DECODING:
                self.DecodeList[key] = value
            else:
                self.PrefillList[key] = value
        return self.jobs
    
    def update(self, caches: List[DynamicCache]):
        # ----- 更新cache ----- #
        for i, status, _id in enumerate(self.idx):
            if status is model.RequestStatus.PREFILLING:
                self.PrefillList[_id].kv_cache = caches[i]
            elif status is model.RequestStatus.DECODING:
                self.DecodeList[_id].kv_cache = caches[i]
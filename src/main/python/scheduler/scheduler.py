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
        if request.status is model.RequestStatus.PREFILLING:
            self.PrefillList[request.request_id] = request
        elif request.status is model.RequestStatus.DECODING:
            self.DecodeList[request.request_id] = request

    def step(self):
        self.idx = list() # 之後拿來更新cache
        d_input_ids = list()
        d_caches = list()
        p_input_ids = list()
        p_caches = list()
        # ----- 先取input_ids(同時更新狀態)，再放回List中 ----- #

        # -Decoding- #
        for _ in range(min([config.BATCH_SIZE, len(self.DecodeList)])):
            key, value = self.DecodeList.popitem(last=False)
            d_input_ids += [value.get_ids()]
            d_caches += [value.kv_cache]
            self.idx += [(value.status, key)]
            self.DecodeList[key] = value
        
        # -Prefilling- #
        for _ in range(min([config.BATCH_SIZE - len(d_input_ids), len(self.PrefillList)])):
            if len(self.PrefillList) == 0:
                break
            key, value = self.PrefillList.popitem(last=False)
            ids = value.get_ids()
            if len(ids) < config.PREFILL_TOKEN_SIZE:
                zeros = torch.zeros((1, config.PREFILL_TOKEN_SIZE - ids.shape[1]), dtype=torch.long)
                ids = torch.cat([ids, zeros], dim=1)
            p_input_ids += [ids]
            p_caches += [value.kv_cache]
            self.idx += [(value.status, key)]
            if value.status is model.RequestStatus.PREFILLING:
                self.PrefillList[key] = value
            elif value.status is model.RequestStatus.DECODING:
                self.DecodeList[key] = value
        return d_input_ids, d_caches, p_input_ids, p_caches
    
    def update(self, caches: List[DynamicCache]):
        # ----- 更新cache ----- #
        i = 0
        for status, _id in self.idx:
            if status is model.RequestStatus.PREFILLING:
                self.PrefillList[_id].kv_cache = caches[i]
            elif status is model.RequestStatus.DECODING:
                self.DecodeList[_id].kv_cache = caches[i]
            i += 1
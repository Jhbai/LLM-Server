import gc
import torch
from typing import List, Dict, Any
from src.main.python.schema import model 
from src.main.python.config import config
from src.main.python.engine import cache_manager
from transformers import DynamicCache, Gemma3ForConditionalGeneration

def infer(model: Gemma3ForConditionalGeneration,  
          input_ids: List[torch.Tensor], 
          kv_caches: List[DynamicCache]):
    if len(input_ids) == 0:
        return None, []
    try:
        cache = None

        # ----- 整合所有input ----- #
        _input_ids = torch.cat(input_ids, dim=0)
        cache = cache_manager.KVCache_merge(kv_caches)

        # ----- Prefilling過程 ----- #
        with torch.no_grad():
            model(
                input_ids=torch.LongTensor(_input_ids).to(model.device),
                use_cache=True,
                past_key_values=cache,
                )
        # print("Prefill Cache Shape:", cache.key_cache[0].shape)
        # ----- 拆解cache ----- #
        eds = []
        for ids in input_ids:
            _list = torch.where(ids==0)[1].tolist()
            if _list:
                eds += [_list[0] - ids.shape[1]]
            else:
                eds += [None]
        caches = cache_manager.KVCache_split(cache,eds)
    finally:
        for item in ("input_ids", ):
            exec(f"del {item}")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return None, caches

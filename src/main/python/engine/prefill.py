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
    try:
        cache = None
        input_ids = list()
        attn_mask = list()
        seq_len = config.PREFILL_TOKEN_SIZE

        # ----- 整合所有input ----- #
        _input_ids = torch.cat(input_ids)
        cache = cache_manager.KVCache_merge(kv_caches)
        for i in range(len(input_ids)):
            curr_seq_len = len(input_ids[i][0])
            diff_seq_len = seq_len - curr_seq_len
            attn_mask + [[1]*curr_seq_len + [0]*diff_seq_len]

        # ----- Prefilling過程
        model(input_ids=torch.FloatTensor(input_ids).to(model.device),
            use_cache=True,
            past_key_values=cache,
            attention_mask=torch.LongTensor(attn_mask).to(model.device)
            )
        
        # ----- 拆解cache ----- #
        caches = cache_manager(cache)
    finally:
        for item in ("input_ids", ):
            exec(f"del {item}")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return None, caches

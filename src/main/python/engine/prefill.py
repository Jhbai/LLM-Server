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
        return None, None
    try:
        cache = None
        attn_mask = list()
        seq_len = config.PREFILL_TOKEN_SIZE

        # ----- 整合所有input ----- #
        _input_ids = torch.cat(input_ids, dim=0)
        cache = cache_manager.KVCache_merge(kv_caches)
        for i in range(len(_input_ids)):
            sample_cache = cache.key_cache[0]
            if sample_cache is not None:
                sample_tensor = sample_cache[i:i+1] # (1, n_heads, seq_len, head_dim)
                sum_abs = torch.abs(sample_tensor).sum(dim=(1, 3)).squeeze(0)
                non_zero_indices = torch.where(sum_abs > 1e-6)[0] # (seq_len, )
                front_padding = [0]*non_zero_indices[0].item()
            else:
                front_padding = []
            curr_seq_len = torch.where(_input_ids[i] == 0)[0].tolist()
            if curr_seq_len:
                curr_seq_len = curr_seq_len[0]
            else:
                curr_seq_len = _input_ids.shape[1]
            diff_seq_len = seq_len - curr_seq_len
            attn_mask += [front_padding + [1]*curr_seq_len + [0]*diff_seq_len]

        # ----- Prefilling過程 ----- #
        with torch.no_grad():
            model(
                input_ids=torch.LongTensor(_input_ids).to(model.device),
                use_cache=True,
                past_key_values=cache,
                attention_mask=torch.LongTensor(attn_mask).to(model.device)
                )
        
        # ----- 拆解cache ----- #
        caches = cache_manager.KVCache_split(cache)
    finally:
        for item in ("input_ids", ):
            exec(f"del {item}")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return None, caches

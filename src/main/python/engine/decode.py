import gc
import torch
from typing import List, Dict, Any
from src.main.python.schema import model 
from src.main.python.config import config
from src.main.python.engine import cache_manager
from transformers import DynamicCache, Gemma3ForConditionalGeneration
MAX_NEW_TOKENS_SIZE = 16

def infer(model: Gemma3ForConditionalGeneration, 
          input_ids: List[torch.Tensor], 
          kv_cache: List[DynamicCache]):
    if len(input_ids) == 0:
        return None, None
    try:
        # ----- 宣告物件 ----- #
        device = model.device
        n_batch = len(kv_cache)
        eos_token_ids = [1, 106] # processor.tokenizer.eos_token_id == 1
        unfinished_sequences = torch.ones(n_batch, dtype=torch.long, device=device)
        generated_ids = [[] for _ in range(n_batch)]
        merged_cache = cache_manager.KVCache_merge(kv_cache)

        # ----- Decoding Loop ----- #
        for step in range(MAX_NEW_TOKENS_SIZE):
            # ----- 全部都做完了 ----- #
            if unfinished_sequences.max() == 0:
                break # Stop Decoding
            
            # ----- 將input做合併 ----- #
            input_ids_tensor = torch.cat(input_ids, dim=0).to(device)
            cache_len = merged_cache.get_seq_length(layer_idx=0)
            position_ids = torch.tensor([[cache_len]], device=device).expand(n_batch, -1)

            # ----- 生成tokens ----- #
            with torch.no_grad(): 
                outputs = model(
                    input_ids=input_ids_tensor,
                    past_key_values=merged_cache,
                    position_ids=position_ids,
                    use_cache=True)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            next_token = next_token * unfinished_sequences + eos_token_ids[0] * (1 - unfinished_sequences)
            for i in range(n_batch):
                if unfinished_sequences[i]:
                    generated_ids[i].append(next_token[i].item())
            is_eos = torch.isin(next_token, torch.tensor(eos_token_ids, device=device))
            unfinished_sequences.mul_(~is_eos) # in-place更新
        # ----- Cache更新 ----- #
        new_caches_list = cache_manager.KVCache_split(merged_cache)
 
    finally:
        for item in ("input_ids", "outputs", "logits", "next_token", "token_id"):
            try:
                exec(f"del {item}")
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return generated_ids, new_caches_list
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from transformers import DynamicCache

def KVCache_merge(caches: List[DynamicCache]):
    results = DynamicCache()

    # ----- 取出layer ----- #
    seq_len = max(c.get_seq_length(layer_idx=0) for c in caches)
    n_layers = len(caches[0].key_cache)
    
    # ----- 建立cache ----- #
    for i in range(n_layers):
        # ----- 依照不同layer去建立cache ----- #
        keys, values = list(), list()
        for c in caches:
            key_tensor = c.key_cache[i]
            value_tensor = c.value_cache[i]

            # ----- 過長的部分做padding ----- # 
            curr_seq_len = key_tensor.shape[2]
            if curr_seq_len < seq_len:
                padding_to_add = seq_len - curr_seq_len
                key_tensor = F.pad(key_tensor, (0, 0, 0, padding_to_add), "constant", 0)
                value_tensor = F.pad(value_tensor, (0, 0, 0, padding_to_add), "constant", 0)

            keys += [key_tensor]
            values += [value_tensor]
        
        # ----- merge tensor ----- #
        key_batch = torch.cat(keys, dim=0)
        value_batch = torch.cat(values, dim=0)

        # ----- update cache ----- #
        results.update(key_states=key_batch, value_states=value_batch, layer_idx=i)
    results.seen_tokens = seq_len
    return results

def KVCache_split(cache: DynamicCache):
    # ----- 把cache的layer跟數量定義出來 ----- #
    batch_size = cache.key_cache[0].shape[0]
    n_layers = len(cache.key_cache)

    # ----- return的結果 ----- #
    results: List[DynamicCache] = [DynamicCache() for _ in range(batch_size)]

    # ----- by batch操作
    for i in range(batch_size):
        # ----- cache的原始長度，只要用第0層來找即可 ----- #
        """
        因為padding是用0填充，所以如果 hid_dim 和 n_head 都是0，那該位必定padding
        最終找到最後一個非零位置
        """
        sample_key_tensor = cache.key_cache[0][i:i+1] # (1, n_heads, seq_len, hid_dim)
        sum_abs = torch.abs(sample_key_tensor).sum(dim=(1, 3)).squeeze(0)
        non_zero_indices = torch.where(sum_abs > 1e-6)[0] # (seq_len, )

        # ----- seq_len 儲存長度計算 ----- #
        if len(non_zero_indices) == 0: original_seq_len = 0
        else: original_seq_len = non_zero_indices.max().item() + 1
            
        for layer_idx in range(n_layers):
            key_slice = cache.key_cache[layer_idx][i:i+1]
            value_slice = cache.value_cache[layer_idx][i:i+1]

            truncated_key = key_slice[:, :, :original_seq_len, :]
            truncated_value = value_slice[:, :, :original_seq_len, :]
            
            results[i].update(
                key_states=truncated_key,
                value_states=truncated_value,
                layer_idx=layer_idx
            )
        
        # 6. 更新這個 cache 的 seen_tokens
        results[i].seen_tokens = original_seq_len

    return results
    
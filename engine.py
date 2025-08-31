import gc
import torch
import config
import torch.nn.functional as F
from PIL import Image, ImageGrab
from typing import List, Dict, Any
from model import Request, RequestStatus
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration, DynamicCache

def KVCache_merge(caches: List[DynamicCache]):
    results = DynamicCache()

    # ----- Use layer0 to find seq len and use first cache to see n_layers ----- #
    seq_len = max(c.get_seq_length(layer_idx=0) for c in caches)
    n_layers = len(caches[0].key_cache)
    
    # ----- Cache building ----- #
    for i in range(n_layers):
        # ----- by layers building ----- #
        keys, values = list(), list()
        for c in caches:
            key_tensor = c.key_cache[i]
            value_tensor = c.value_cache[i]

            # ----- padding ----- #
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

def KVCache_Split(cache: DynamicCache):
    


class InferenceEngine:
    def __init__(self, model_path: str = "D://LLM//gemma//gemma3_4b", max_batch_size: int = config.PREFILL_TOKEN_SIZE):
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
            )
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    @torch.inference_mode()
    def inference(self, requests: List[Request]):
        # ----- get ids ------ #
        _input_ids = [r.get_ids() for r in requests]
        
        # ----- prefilling ----- #
        input_ids = list()
        attn_mask = list()
        caches = list()
        seq_len = config.PREFILL_TOKEN_SIZE
        prefilling_requests = list()
        for i, r in enumerate(requests):
            if r.status == RequestStatus.PREFILLING:
                curr_seq_len = _input_ids[i]
                diff = curr_seq_len - seq_len
                if diff != 0:
                    input_ids += [_input_ids[i] + [0]*(diff)]
                    attn_mask += [[1]*curr_seq_len + [0]*diff]
                else:
                    input_ids += [_input_ids[i]]
                    attn_mask += [[1]*curr_seq_len]
                caches += [r.kv_cache]
                prefilling_requests += [r]
        cache = KVCache_merge(caches)
        self.model(input_ids=torch.FloatTensor(input_ids).to(self.model.device)
                   use_cache=True
                   past_key_values=cache,
                   attention_mask=torch.LongTensor(attn_mask).to(self.model.device)
                   )
        
        # ----- update kv cache ----- #
        
        # ----- decoding ----- #
        eos_token_ids = [self.processor.tokenizer.eos_token_id, 106]
        decode_requests = [r for i, r in enumerate(requests) if r.status == RequestStatus.DECODING]
        caches = [r.kv_cache for r in decode_requests]
        results = list()
        for i, r in decode_requests:
            try:
                res = list()
                input_ids = _input_ids[i]
                ed = r.get_sequence_length()-1
                for _ in range(128000):
                    cache_position = torch.arange(ed-1, ed, dtype=torch.long, device = self.model.device)
                    outputs = self.model(input_ids=input_ids, 
                                    use_cache=True, 
                                    past_key_values=caches[i], 
                                    cache_position=cache_position)
                    logits = outputs.logits
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    token_id = next_token.item()
                    input_ids = next_token
                    if token_id in eos_token_ids:
                        break
                    res += [self.processor.decode(token_id)]
            finally:
                for item in ("input_ids", "outputs", "logits", "next_token", "token_id"):
                    exec(f"del {item}")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        results += [res]

        # ----- return processing ----- #
        idx = 0
        answer = list()
        for r in requests:
            if r.status == RequestStatus.DECODING:
                answer += [results[idx]]
                idx += 1
            else:
                answer += [None]
        return answer


    def tokenize(self, prompt):
        input_ids = self.processor(text=prompt)[0]
        return input_ids